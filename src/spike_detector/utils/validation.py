"""Explicit input and effective-setting validation for detection."""
import numpy as np


# The highest supported band-pass order needs up to 123 samples for
# zero-phase Butterworth padding. Reject shorter inputs before detection.
MIN_TRACE_SAMPLES = 128


def validate_time(time_ms, fs=None):
    t = np.asarray(time_ms, dtype=float)
    if t.ndim != 1 or t.size < 2 or not np.all(np.isfinite(t)):
        raise ValueError('Time must contain at least two finite samples; missing rows are not removed.')
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError('Time must be strictly increasing, without duplicates.')
    step = float(np.median(dt))
    if not np.allclose(dt, step, rtol=1e-3, atol=max(1e-9, step * 1e-6)):
        raise ValueError('Irregular time spacing or dropped frames: resample explicitly before detection (tolerance 0.1%).')
    inferred = 1000.0 / step
    if fs is not None and (not np.isfinite(fs) or fs <= 0 or not np.isclose(fs, inferred, rtol=1e-3)):
        raise ValueError(f'Sampling rate {fs} Hz disagrees with timestamps ({inferred:g} Hz).')
    return inferred


def validate_session(data):
    fs = validate_time(data['time_ms'], float(data['fs']))
    raw = np.asarray(data['raw_data'], dtype=float)
    if raw.ndim != 2 or raw.shape[0] != len(data['time_ms']) or raw.shape[1] < 1:
        raise ValueError('Trace dimensions must match time samples and contain at least one cell.')
    if raw.shape[0] < MIN_TRACE_SAMPLES:
        raise ValueError(f'Recording needs at least {MIN_TRACE_SAMPLES} time samples for filtering; got {raw.shape[0]}.')
    if len(data['cell_names']) != raw.shape[1]:
        raise ValueError('Cell names do not match trace columns.')
    if not np.all(np.isfinite(raw)):
        raise ValueError('Trace contains missing/nonfinite samples; repair or exclude them explicitly before detection.')
    return fs


def validate_band(low, high, fs, order):
    vals = [0.0 if v is None else float(v) for v in (low, high)]
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('Sampling rate must be finite and positive.')
    if any(not np.isfinite(v) or v < 0 or v >= fs / 2 for v in vals):
        raise ValueError(f'Filter cutoffs must be 0 (off) or below Nyquist ({fs/2:g} Hz).')
    if vals[0] and vals[1] and vals[0] >= vals[1]:
        raise ValueError('Filter low cutoff must be smaller than high cutoff.')
    if not np.isfinite(order) or int(order) != order or not 1 <= order <= 20:
        raise ValueError('Filter order must be an integer from 1 to 20.')
    return vals


def mask_windows(params):
    """Preserve legacy centered total-width settings unless explicit sides exist."""
    half = float(params.get('SS_BLANK_MS', 18.0)) / 2.0
    return float(params.get('SS_MASK_PRE_MS', half)), float(params.get('SS_MASK_POST_MS', half))


def effective_settings(params, baseline, frames, fs, n):
    out = {'fs_hz': float(fs), 'samples': int(n), 'filters': {}}
    if params.get('TEMPLATE_MATCH_METHOD', 'LLR Probability Vector') not in ('LLR Probability Vector', 'Normalized Similarity', 'Burst-aware LLR'):
        raise ValueError('Unknown template matching method.')
    for key in ('TEMPLATE_CS_SIMILARITY', 'TEMPLATE_SS_SIMILARITY'):
        val = float(params.get(key, 0.90 if 'CS_' in key else 0.80))
        if not np.isfinite(val) or not 0 < val <= 1:
            raise ValueError(f'{key} must be in (0, 1].')
    val = float(params.get('TEMPLATE_MIN_RESPONSE_SIGMA', 2.2))
    if not np.isfinite(val) or val <= 0:
        raise ValueError('TEMPLATE_MIN_RESPONSE_SIGMA must be positive.')
    cs_peak = float(params.get('TEMPLATE_CS_MIN_FILTERED_PEAK_SIGMA', 3.0))
    if not np.isfinite(cs_peak) or cs_peak < 0:
        raise ValueError('TEMPLATE_CS_MIN_FILTERED_PEAK_SIGMA must be finite and nonnegative.')
    out['cs_min_filtered_peak_sigma'] = cs_peak
    group_count = params.get('TEMPLATE_PARALLEL_GROUPS', 3)
    components = params.get('TEMPLATE_PARALLEL_COMPONENTS', 2)
    for name, value in (('TEMPLATE_PARALLEL_GROUPS', group_count),
                        ('TEMPLATE_PARALLEL_COMPONENTS', components)):
        if not isinstance(value, (int, np.integer)) or not 1 <= value <= 8:
            raise ValueError(f'{name} must be an integer from 1 to 8.')
    for kind in ('CS', 'SS'):
        chosen = params.get(f'TEMPLATE_{kind}_SELECTED_GROUPS')
        if chosen is not None and (not isinstance(chosen, (list, tuple)) or
                                   len(chosen) == 0 or
                                   any(not isinstance(index, (int, np.integer)) or
                                       not 1 <= index <= group_count for index in chosen) or
                                   len(set(chosen)) != len(chosen)):
            raise ValueError(f'TEMPLATE_{kind}_SELECTED_GROUPS must contain unique group numbers from 1 to {group_count}.')
    out['parallel_template_groups'] = int(group_count)
    out['parallel_template_components'] = int(components)
    val = float(params.get('TEMPLATE_SS_LOWPASS_HZ', 700.0))
    if not np.isfinite(val) or val < 0:
        raise ValueError('TEMPLATE_SS_LOWPASS_HZ must be nonnegative.')
    out['template_ss_lowpass_hz_used'] = (min(val, 0.45 * fs)
        if val > 0 and params.get('DETECTION_METHOD') == 'Template Matching' else 0.0)
    for kind in ('CS', 'SS'):
        low, high = validate_band(params.get(f'{kind}_LOW_CUT_HZ', 0),
                                  params.get(f'{kind}_HIGH_CUT_HZ', 150 if kind == 'CS' else 0),
                                  fs, params.get(f'{kind}_FILTER_ORDER', 4))
        out['filters'][kind] = {'low_hz': low, 'high_hz': high,
                                'order': int(params.get(f'{kind}_FILTER_ORDER', 4))}
        for key in (f'{kind}_MIN_DIST_MS', f'{kind}_THRESHOLD_SIGMA', f'TEMPLATE_{kind}_SIGMA'):
            val = float(params.get(key, 1))
            if not np.isfinite(val) or val < 0 or ('SIGMA' in key and val == 0):
                raise ValueError(f'{key} must be finite and {"positive" if "SIGMA" in key else "nonnegative"}.')
        out[f'{kind.lower()}_min_distance_samples'] = max(1, int(np.ceil(params.get(f'{kind}_MIN_DIST_MS', 1)*fs/1000)))
    pre, post = mask_windows(params)
    for key, val in [('SS pre-CS mask', pre), ('SS post-CS mask', post),
                     ('Initial blank', params.get('INITIAL_BLANK_MS', 0)),
                     ('CS min FWHM', params.get('CS_MIN_FWHM_MS', 4)),
                     ('SS max FWHM', params.get('SS_MAX_FWHM_MS', 4.5))]:
        if not np.isfinite(val) or val < 0:
            raise ValueError(f'{key} must be finite and nonnegative.')
    if params.get('SS_MAX_FWHM_FILTER_ENABLED', True) and params.get('SS_MAX_FWHM_MS', 4.5) <= 0:
        raise ValueError('Enabled SS maximum width must be positive.')
    if not np.isfinite(frames) or int(frames) != frames or frames < 0 or frames > n:
        raise ValueError('Frame averaging must be an integer between zero and the recording length.')
    method = baseline.get('method', 'Median')
    if method not in ('Disable', 'Median', 'Percentile'):
        raise ValueError('Unknown baseline method.')
    window = float(baseline.get('window_ms', 30))
    pct = float(baseline.get('percentile', 20))
    if not np.isfinite(window) or window <= 0 or not np.isfinite(pct) or not 0 <= pct <= 100:
        raise ValueError('Baseline window must be positive and percentile between 0 and 100.')
    size = min(max(5, int(window*fs/1000)), max(5, min(n//2, 5000)))
    out.update(baseline_method=method, baseline_window_samples=size if method != 'Disable' else 0,
               baseline_window_ms=size*1000/fs if method != 'Disable' else 0,
               frame_samples=int(frames), ss_mask_pre_ms=pre, ss_mask_post_ms=post,
               initial_blank_samples=int(np.ceil(float(params.get('INITIAL_BLANK_MS', 0))*fs/1000)),
               ss_mask_pre_samples=int(np.ceil(pre*fs/1000)), ss_mask_post_samples=int(np.ceil(post*fs/1000)))
    if params.get('FRAME_PROCESSING_MODE', 'Rolling average') not in ('Rolling average', 'Downsampling frames'):
        raise ValueError('Unknown frame processing mode.')
    if params.get('DENOISE_ENABLED', False):
        low = float(params.get('DENOISE_F_MIN_HZ', 3))
        high = float(params.get('DENOISE_F_MAX_HZ', 1000))
        if not np.isfinite(low) or not np.isfinite(high) or low < .5 or high <= low or high > .95*fs/2:
            raise ValueError(f'Denoise band must satisfy 0.5 <= low < high <= {0.95*fs/2:g} Hz.')
        out['denoise_band_hz'] = [low, high]
    for key in ('LOCAL_BASELINE_CS_MS', 'LOCAL_BASELINE_SS_MS'):
        val = float(params.get(key, 200 if 'CS' in key else 50))
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f'{key} must be finite and positive.')
        out[key.lower() + '_samples'] = min(n, max(5, int(val*fs/1000)))
    return out
