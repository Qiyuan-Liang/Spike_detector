import numpy as np


def mean_std_count(arr):
    a = np.array(arr) if len(arr) > 0 else np.array([])
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
