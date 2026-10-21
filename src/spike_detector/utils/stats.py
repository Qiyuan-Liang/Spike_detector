import numpy as np


def mean_std_count(arr):
    a = np.array(arr) if len(arr) > 0 else np.array([])
    a = a[np.isfinite(a)]
    if a.size > 0:
        return float(np.nanmean(a)), float(np.nanstd(a)), int(a.size)
    return np.nan, np.nan, 0


def autocorrelogram_counts(event_times_ms, window_ms, bin_ms):
    """Symmetric ACG pair counts for one cell in one recording.

    Each distinct, positive-lag pair is counted once and mirrored to negative
    lag. Self-pairs and duplicate timestamps at zero lag are excluded. Bins on
    the positive side are (left, right], including a possibly shorter last bin.
    Call separately for each cell/session before pooling counts.
    """
    window_ms, bin_ms = float(window_ms), float(bin_ms)
    if not np.isfinite(window_ms) or not np.isfinite(bin_ms) or window_ms <= 0 or bin_ms <= 0:
        raise ValueError('ACG window and bin width must be finite and positive.')
    positive_edges = np.arange(0.0, window_ms, bin_ms)
    positive_edges = np.r_[positive_edges, window_ms]
    positive_counts = np.zeros(len(positive_edges) - 1, dtype=np.int64)
    times = np.asarray(event_times_ms, dtype=float).ravel()
    times = np.sort(times[np.isfinite(times)])
    for i, start in enumerate(times[:-1]):
        stop = np.searchsorted(times, start + window_ms, side='right')
        lags = times[i + 1:stop] - start
        lags = lags[lags > 0]
        indices = np.searchsorted(positive_edges, lags, side='left') - 1
        positive_counts += np.bincount(indices, minlength=positive_counts.size)
    edges = np.r_[-positive_edges[:0:-1], positive_edges]
    return edges, np.r_[positive_counts[::-1], positive_counts]


def recording_intervals_ms(time_ms, fs, exclusion_mask=None):
    """Observed sample intervals, excluding masked samples, with an exclusive end."""
    time = np.asarray(time_ms, dtype=float)
    if time.ndim != 1 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError('Recording timestamps must be finite and strictly increasing.')
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('Sampling rate must be finite and positive.')
    if not time.size:
        return np.empty((0, 2))
    excluded = np.zeros(time.size, dtype=bool) if exclusion_mask is None else np.asarray(exclusion_mask, dtype=bool)
    if excluded.shape != time.shape:
        raise ValueError('ACG exclusion mask must match the recording timestamps.')
    boundaries = np.flatnonzero(np.diff(np.r_[False, ~excluded, False]))
    sample_edges = np.r_[time, time[-1] + 1000.0 / fs]
    return sample_edges[boundaries].reshape(-1, 2)


def autocorrelogram_components(event_times_ms, window_ms, bin_ms, observation_intervals_ms):
    """Return lag edges, within-recording pair counts and reference exposure (s).

    Rate = counts / exposure. Each reference spike contributes only the observed
    part of each shifted lag bin, correcting recording edges and masked gaps.
    Sum counts AND exposure across independent cells/sessions before dividing;
    never form spike pairs across those recordings. Zero-lag self-pairs are
    excluded by autocorrelogram_counts. Zero exposure means an undefined rate.
    """
    intervals = np.asarray(observation_intervals_ms, dtype=float).reshape(-1, 2)
    if (not np.all(np.isfinite(intervals)) or
            np.any(intervals[:, 1] <= intervals[:, 0]) or
            np.any(intervals[1:, 0] < intervals[:-1, 1])):
        raise ValueError('Observation intervals must be finite, positive, sorted and disjoint.')
    times = np.asarray(event_times_ms, dtype=float).ravel()
    times = times[np.isfinite(times)]
    if len(intervals):
        indices = np.searchsorted(intervals[:, 0], times, side='right') - 1
        observed = (indices >= 0) & (times < intervals[np.maximum(indices, 0), 1])
        times = times[observed]
    else:
        times = times[:0]
    edges, counts = autocorrelogram_counts(times, window_ms, bin_ms)
    exposure_ms = np.zeros_like(counts, dtype=float)
    if not len(intervals) or not times.size:
        return edges, counts, exposure_ms
    lengths = intervals[:, 1] - intervals[:, 0]
    prefix = np.r_[0.0, np.cumsum(lengths)]

    def observed_time_at(values):
        indices = np.searchsorted(intervals[:, 0], values, side='right') - 1
        safe = np.maximum(indices, 0)
        cumulative = prefix[safe] + np.clip(values - intervals[safe, 0], 0, lengths[safe])
        return np.where(indices >= 0, cumulative, 0.0)

    for start in range(0, times.size, 1024):
        shifted_edges = times[start:start+1024, None] + edges
        exposure_ms += np.maximum(np.diff(observed_time_at(shifted_edges), axis=1), 0).sum(axis=0)
    return edges, counts, exposure_ms / 1000.0
