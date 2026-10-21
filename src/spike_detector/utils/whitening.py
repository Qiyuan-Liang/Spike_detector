"""Regularized, finite-support spectral whitening for template matching.

Estimate spectra from contiguous background stretches; never concatenate gaps.
The symmetric FIR bounds edge/mask influence and is shared by trace and kernels.
This approximates whitening, not a calibration of detection probabilities.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve, find_peaks, firwin2, periodogram


def _sigma(x):
    center = np.median(x)
    return float(1.4826 * np.median(np.abs(x - center)))


def fit_whitening(trace, fs, exclusion, kind):
    x = np.asarray(trace, dtype=float)
    excluded = np.asarray(exclusion, dtype=bool).copy() | ~np.isfinite(x)
    info = dict(applied=False, reason='', method='contiguous-background median Welch / symmetric FIR',
                psd_floor_fraction=.01, max_spectral_gain=10.0)
    state = dict(fir=np.ones(1), noise_mask=excluded, info=info)
    clean = x[~excluded]
    if clean.size < 64:
        info['reason'] = 'insufficient background samples'
        return state
    sigma = _sigma(clean)
    if not np.isfinite(sigma) or sigma <= 0:
        info['reason'] = 'background MAD is zero'
        return state
    # Conservative, fixed seed cutoff, independent of the final score cutoff.
    # Masks are used only for estimating noise, not to reject final candidates.
    peaks, _ = find_peaks(x, height=float(np.median(clean)) + 3 * sigma)
    peaks = peaks[~excluded[peaks]]
    pre_ms, post_ms = (20., 80.) if kind == 'cs' else (2., 4.)
    pre, post = int(np.ceil(pre_ms * fs / 1000)), int(np.ceil(post_ms * fs / 1000))
    for peak in peaks:
        excluded[max(0, peak-pre):min(len(x), peak+post+1)] = True
    state['noise_mask'] = excluded
    transitions = np.diff(np.r_[False, ~excluded, False].astype(np.int8))
    starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    longest = int(np.max(ends-starts)) if starts.size else 0
    info.update(clean_samples=int(np.sum(~excluded)), noise_seed_peaks=int(peaks.size))
    if longest < 32:
        info['reason'] = 'no contiguous background stretch of 32 samples'
        return state
    nperseg = 2 ** int(np.floor(np.log2(min(1024, longest))))
    windows = [i for start, end in zip(starts, ends)
               for i in range(int(start), int(end)-nperseg+1, nperseg//2)]
    if len(windows) < 4:
        info['reason'] = 'fewer than four clean spectral windows'
        return state
    # Bound memory/time on long recordings, spread windows over the recording.
    if len(windows) > 512:
        windows = [windows[i] for i in np.linspace(0, len(windows)-1, 512, dtype=int)]
    segments = np.stack([x[i:i+nperseg] for i in windows])
    frequencies, spectra = periodogram(segments, fs=fs, window='hann',
                                      detrend='constant', axis=-1)
    psd = np.exp(gaussian_filter1d(np.log(np.maximum(np.median(spectra, axis=0),
                                                    np.finfo(float).tiny)), 1.0))
    reference = float(np.max(psd))
    if not np.isfinite(reference) or reference <= np.finfo(float).tiny:
        info['reason'] = 'invalid noise spectrum'
        return state
    gain = np.sqrt(reference / np.maximum(psd, reference * .01))
    # Five ms on either side; shorten when the available spectrum is coarse.
    radius = min(nperseg//4, max(2, int(round(.005*fs))))
    fir = firwin2(2*radius+1, frequencies, gain, fs=fs, window='hamming')
    info.update(applied=True, reason='', nperseg=nperseg, spectral_windows=len(windows),
                fir_half_ms=radius*1000/fs, fir_coefficients=fir.tolist(),
                noise_seed_cutoff_mad=3.0, seed_exclusion_ms=[pre_ms, post_ms])
    state['fir'] = fir
    return state


def whiten_trace(trace, state):
    x = np.asarray(trace, dtype=float)
    fir = state['fir']
    clean = x[~state['noise_mask']]
    center = float(np.median(clean)) if clean.size else 0.0
    radius = len(fir)//2
    if not radius:
        return x.copy()
    padded = np.pad(x-center, radius, mode='reflect')
    return fftconvolve(padded, fir, mode='valid')


def whiten_kernel(kernel, state):
    """Zero outside the event waveform; retain the FIR's complete support."""
    return fftconvolve(kernel, state['fir'], mode='full')
