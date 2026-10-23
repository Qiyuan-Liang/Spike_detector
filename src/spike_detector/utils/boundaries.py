"""Recording-wide boundary exclusions, distinct from spike-specific masks."""
import copy
import numpy as np
from scipy.ndimage import median_filter, percentile_filter
from .processing import apply_frame_processing


def observation_bounds(n, fs, initial_ms=150.0, final_ms=50.0):
    values = np.asarray([fs, initial_ms, final_ms], dtype=float)
    if not np.all(np.isfinite(values)) or fs <= 0 or initial_ms < 0 or final_ms < 0:
        raise ValueError('Boundary blanks must be finite and nonnegative; sampling rate must be positive.')
    start = min(n, int(np.ceil(initial_ms * fs / 1000 - 1e-9)))
    stop = max(0, n - int(np.ceil(final_ms * fs / 1000 - 1e-9)))
    return start, max(start, stop)


def observation_mask(n, fs, params):
    start, stop = observation_bounds(n, fs, params.get('INITIAL_BLANK_MS', 150.0),
                                      params.get('FINAL_BLANK_MS', 50.0))
    valid = np.zeros(n, dtype=bool)
    valid[start:stop] = True
    return valid


def smooth_blanked_display(raw, processed, valid, frames=0, mode='Rolling average'):
    """Smooth excluded segments independently for display; never cross a mask edge."""
    result = np.asarray(processed, dtype=float).copy()
    if mode != 'Rolling average' or int(frames) <= 1:
        return result
    raw = np.asarray(raw, dtype=float)
    excluded = ~np.asarray(valid, dtype=bool)
    edges = np.diff(np.r_[False, excluded, False].astype(int))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        segment = raw[start:stop]
        count = min(int(frames), len(segment))
        if count > 1:
            padded = np.pad(segment, (count//2, (count-1)//2), mode='reflect')
            result[start:stop] = np.convolve(padded, np.ones(count)/count, mode='valid')
    return result


def visualization_window(time_ms, window_ms=1000.0, center_fraction=0.0,
                         valid=None, start_at_observation=False):
    """Default to first observed timestamp; explicit navigation covers the full trace."""
    t = np.asarray(time_ms, dtype=float)
    first, last = float(t[0]), float(t[-1])
    width = float(window_ms)
    indices = np.flatnonzero(valid) if valid is not None else np.arange(len(t))
    if start_at_observation and indices.size:
        start = float(t[indices[0]])
        return start, min(last, start + width)
    center = first + float(center_fraction) * (last-first)
    start = max(first, min(center-width/2, last-width))
    return start, min(last, start+width)


def protected_preprocess(raw, fs, params, baseline_params, frames=0, mode='Rolling average'):
    """Fit/process observed samples only; smooth blanked segments separately for display."""
    raw = np.asarray(raw, dtype=float)
    valid = observation_mask(len(raw), fs, params)
    processed = smooth_blanked_display(raw, raw, valid, frames, mode)
    baseline = np.zeros_like(raw)
    indices = np.flatnonzero(valid)
    if not indices.size:
        return processed, baseline, valid
    start, stop = indices[0], indices[-1] + 1
    segment = raw[start:stop]
    count = min(int(frames), len(segment))
    if mode == 'Rolling average' and count > 1:
        padded = np.pad(segment, (count//2, (count-1)//2), mode='reflect')
        clean = np.convolve(padded, np.ones(count)/count, mode='valid')
    else:
        clean = apply_frame_processing(segment, frames=count, mode=mode)
    size = max(5, min(int(baseline_params['window_ms'] * fs / 1000),
                       max(5, min(len(clean)//2, 5000))))
    method = baseline_params['method']
    if method == 'Median':
        base = median_filter(clean, size=size)
    elif method == 'Percentile':
        base = percentile_filter(clean, baseline_params['percentile'], size=size)
    else:
        base = np.zeros_like(clean)
    processed[start:stop] = clean
    baseline[:] = np.pad(base, (start, len(raw)-stop), mode='edge')
    return processed, baseline, valid


def embed_observed_result(result, n, start, stop):
    """Restore original sample coordinates after detecting on the valid interval."""
    result = copy.deepcopy(result)
    size = stop - start
    for key, value in list(result.items()):
        if (key.endswith('_trace') or key == 'detrended' or key == 'baseline') and isinstance(value, np.ndarray) and value.shape == (size,):
            result[key] = np.pad(value, (start, n-stop), mode='edge')
        elif key.endswith('_exclusion_mask') and isinstance(value, np.ndarray) and value.shape == (size,):
            result[key] = np.pad(value, (start, n-stop), constant_values=True)
    valid = np.zeros(n, bool); valid[start:stop] = True
    result['recording_exclusion_mask'] = ~valid
    if 'stats_source_trace' in result:
        result['stats_source_trace'][~valid] = np.nan
    for key in ('cs_peaks', 'ss_peaks', 'amplitude_snr_indices_cs', 'amplitude_snr_indices_ss'):
        if key in result:
            result[key] = np.asarray(result[key], dtype=int) + start
    seen = set()
    for key in ('cs_width_candidates', 'ss_width_candidates', 'event_widths_cs', 'event_widths_ss'):
        for row in result.get(key, []):
            if id(row) in seen:
                continue
            seen.add(id(row))
            for field in ('candidate_index', 'peak_index', 'left_index', 'right_index'):
                if row.get(field) is not None:
                    row[field] += int(start)
    for kind in ('cs', 'ss'):
        quality = result.get('combo', {}).get(kind, {})
        for key in ('seed_indices', 'clean_seed_indices'):
            if key in quality:
                quality[key] = (np.asarray(quality[key], dtype=int) + start).tolist()
    result['observation_start_index'], result['observation_stop_index'] = int(start), int(stop)
    return result


def protected_filter(trace, fs, params, **options):
    from .processing import apply_filter
    trace = np.asarray(trace, dtype=float)
    valid = observation_mask(len(trace), fs, params)
    indices = np.flatnonzero(valid)
    if indices.size < 128:
        return trace.copy()
    start, stop = indices[0], indices[-1]+1
    result = apply_filter(trace[start:stop], fs, **options)
    return np.pad(result, (start, len(trace)-stop), mode='edge')
