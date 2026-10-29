"""Polarity-aligned event amplitudes and independent-session time courses."""
from pathlib import Path
import re
import numpy as np


def event_dff_percent(trace, baseline, peaks, fs, negative_going=True, valid=None):
    """Missing-field fallback matching compare.ipynb's event_extractor.

    Use a 30-ms event window and its central +/-5-ms raw extremum (right end
    exclusive), then sign*100*(F-F0)/abs(F0). No redetection; one slot per event.
    Saved amplitude arrays, including NaNs, must take precedence in callers.
    """
    trace, baseline = np.asarray(trace, float).copy(), np.asarray(baseline, float)
    peaks = np.asarray(peaks, int)
    if trace.ndim != 1 or baseline.shape != trace.shape or not np.isfinite(fs) or fs <= 0:
        raise ValueError('Event fluorescence/baseline must match; sampling rate must be positive.')
    if valid is not None:
        valid = np.asarray(valid, bool)
        if valid.shape != trace.shape:
            raise ValueError('Amplitude observation mask must match fluorescence.')
        trace[~valid] = np.nan
    half = max(2, int(round(30*fs/2000)))
    local_half = max(2, half//3)
    sign = -1 if negative_going else 1
    values = np.full(peaks.size, np.nan)
    for i, peak in enumerate(peaks):
        if peak-half < 0 or peak+half+1 > trace.size:
            continue
        start, stop = max(0, peak-local_half), min(trace.size, peak+local_half)
        local = trace[start:stop]
        if not local.size or not np.all(np.isfinite(local)):
            continue
        refined = start + (np.argmin(local) if negative_going else np.argmax(local))
        f0 = baseline[refined]
        if np.isfinite(f0) and abs(f0) > 1e-12:
            values[i] = sign*100*(trace[refined]-f0)/abs(f0)
    return values


def peth_edges(pre_ms, post_ms, bin_ms):
    pre, post, width = map(float, (pre_ms, post_ms, bin_ms))
    if not np.all(np.isfinite([pre,post,width])) or min(pre,post) < 0 or pre+post <= 0 or width <= 0:
        raise ValueError('PETH range must be positive on at least one side and bin width positive.')
    left = -np.r_[np.arange(0,pre,width),pre][::-1] if pre else np.array([0.])
    right = np.r_[np.arange(0,post,width),post][1:] if post else np.array([])
    return np.r_[left,right]


def natural_key(name):
    return tuple((1,int(part)) if part.isdigit() else (0,part.casefold())
                 for part in re.split(r'(\d+)', str(name)))


def session_time_courses(sessions, records, bin_s):
    """Join acquisition durations in natural-name order within ONE source folder.

    records are independent cells: pooled rate=sum counts/sum observed cell-time,
    dF/F=mean of finite event amplitudes. No intervals, smoothing or bins cross
    sessions. Sessions without a selected cell retain their time slot. Unknown
    inter-session wall-clock gaps are not invented. Multiple folders are unsafe.
    """
    bin_s = float(bin_s)
    if not np.isfinite(bin_s) or bin_s <= 0:
        raise ValueError('Trend window must be finite and positive.')
    if not sessions:
        return dict(reason='No selected sessions', segments=[], boundaries_s=[])
    folders = {str(Path(s['folder']).expanduser().resolve()) for s in sessions if s.get('folder')}
    if len(folders) != 1 or any(not s.get('folder') for s in sessions):
        return dict(reason='Select one Path to concatenate sessions', segments=[], boundaries_s=[])
    ordered = sorted(sessions,key=lambda s:natural_key(s['name']))
    result = dict(reason=None,segments=[],boundaries_s=[],names=[],duration_s=0.)
    offset = 0.
    for session in ordered:
        duration = float(session['duration_s'])
        if not np.isfinite(duration) or duration <= 0:
            return dict(reason='Recording duration unavailable',segments=[],boundaries_s=[])
        edges = np.r_[np.arange(0,duration,bin_s),duration]
        counts = np.zeros(edges.size-1, int)
        exposure = np.zeros(edges.size-1)
        amplitude_sum, amplitude_n = np.zeros(edges.size-1), np.zeros(edges.size-1, int)
        for record in records:
            if record['session_key'] != session['key']:
                continue
            times = np.asarray(record['times_s'],float)
            amp = np.asarray(record['dff_percent'],float)
            intervals = np.asarray(record['intervals_s'],float).reshape(-1,2)
            if times.shape != amp.shape:
                raise ValueError('Amplitude values must retain event order.')
            if not len(intervals):
                continue
            if (not np.all(np.isfinite(intervals)) or np.any(intervals[:,1] <= intervals[:,0]) or
                    np.any(intervals[1:,0] < intervals[:-1,1])):
                raise ValueError('Observation intervals must be sorted and disjoint.')
            index = np.searchsorted(intervals[:,0],times,side='right')-1
            valid = np.isfinite(times)&(index>=0)&(times<intervals[np.maximum(index,0),1])&(times>=0)&(times<duration)
            counts += np.histogram(times[valid],edges)[0]
            finite = valid&np.isfinite(amp)
            amplitude_sum += np.histogram(times[finite],edges,weights=amp[finite])[0]
            amplitude_n += np.histogram(times[finite],edges)[0]
            lengths = intervals[:,1]-intervals[:,0]
            prefix = np.r_[0.,np.cumsum(lengths)]
            idx = np.searchsorted(intervals[:,0],edges,side='right')-1
            safe = np.maximum(idx,0)
            cumulative = np.where(idx>=0,prefix[safe]+np.clip(edges-intervals[safe,0],0,lengths[safe]),0)
            exposure += np.maximum(np.diff(cumulative),0)
        rate = np.divide(counts,exposure,out=np.full(exposure.shape,np.nan),where=exposure>0)
        mean = np.divide(amplitude_sum,amplitude_n,out=np.full(exposure.shape,np.nan),where=amplitude_n>0)
        result['segments'].append(dict(name=session['name'],time_s=offset+(edges[:-1]+edges[1:])/2,
            rate_hz=rate,dff_percent=mean,counts=counts,exposure_s=exposure,amplitude_n=amplitude_n))
        result['names'].append(session['name'])
        offset += duration
        result['boundaries_s'].append(offset)
    result['boundaries_s']=result['boundaries_s'][:-1]
    result['duration_s']=offset
    return result
