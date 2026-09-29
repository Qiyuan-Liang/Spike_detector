"""Candidate-anchored half-height widths with explicit uncertainty reasons."""
import numpy as np
from scipy.signal import find_peaks


def measure_width(trace, peak, fs, window_ms=50, neighbors=(), align_ms=0):
    x = np.asarray(trace, dtype=float).ravel()
    p = int(peak)
    result = {'candidate_index': p, 'peak_index': p, 'fwhm_ms': np.nan,
              'status': 'uncertain', 'reason': 'invalid_candidate'}
    if not np.isfinite(fs) or fs <= 0 or not 0 <= p < len(x):
        return result
    half = max(2, int(np.ceil(window_ms*fs/2000)))
    lo, hi = max(0, p-half), min(len(x)-1, p+half)
    others = np.asarray(neighbors, dtype=int)
    left = others[others < p]
    right = others[others > p]
    if left.size:
        lo = max(lo, int(np.floor((p+left.max())/2))+1)
    if right.size:
        hi = min(hi, int(np.ceil((p+right.min())/2))-1)
    if (lo == 0 and p-lo < 3) or (hi == len(x)-1 and hi-p < 3):
        result['reason'] = 'clipped'
        return result
    if hi-lo < 4:
        result['reason'] = 'overlapping_events'
        return result
    if not np.all(np.isfinite(x[lo:hi+1])):
        result['reason'] = 'nonfinite_waveform'
        return result
    # A positive alignment window must include at least the nearest native
    # sample, even when its duration is slightly longer than align_ms.
    radius = int(np.ceil(align_ms*fs/1000 - 1e-9)) if align_ms > 0 else 0
    if radius:
        local, _ = find_peaks(x[lo:hi+1])
        local += lo
        local = local[np.abs(local-p) <= radius]
        if local.size == 0:
            result['reason'] = 'no_aligned_peak'
            return result
        p = int(local[np.argmin(np.abs(local-p))])
        result['peak_index'] = p
    if p <= lo or p >= hi or x[p] < x[p-1] or x[p] < x[p+1]:
        result['reason'] = 'not_a_local_peak'
        return result
    # A short robust baseline at the left boundary, never beyond the candidate.
    count = min(max(2, int(.1*(hi-lo+1))), p-lo, hi-p)
    # Use the quieter boundary when an adjacent event contaminates one side.
    baseline = float(min(np.median(x[lo:lo+count]), np.median(x[hi-count+1:hi+1])))
    height = float(x[p]-baseline)
    if height <= 0:
        result['reason'] = 'nonpositive_amplitude'
        return result
    level = baseline + height/2
    crossings = []
    for indices, direction in ((range(p-1, lo-1, -1), -1), (range(p, hi), 1)):
        crossing = None
        for i in indices:
            y0, y1 = x[i], x[i+1]
            found = y0 <= level <= y1 if direction < 0 else y0 >= level >= y1
            if found and y1 != y0:
                crossing = i + (level-y0)/(y1-y0)
                break
        crossings.append(crossing)
    if None in crossings:
        result['reason'] = ('clipped' if lo == 0 or hi == len(x)-1 else
                            'overlapping_events' if left.size or right.size else 'missing_half_height_crossing')
        return result
    result.update(fwhm_ms=float((crossings[1]-crossings[0])*1000/fs),
                  status='measured', reason='', left_index=crossings[0], right_index=crossings[1])
    return result


def measure_widths(trace, peaks, fs, window_ms=50, neighbors=(), align_ms=0):
    all_peaks = np.unique(np.r_[np.asarray(peaks, dtype=int), np.asarray(neighbors, dtype=int)])
    rows = []
    for p in peaks:
        index = np.searchsorted(all_peaks, p)
        adjacent = all_peaks[max(0, index-1):index+2]
        rows.append(measure_width(trace, p, fs, window_ms, adjacent, align_ms))
    return rows
