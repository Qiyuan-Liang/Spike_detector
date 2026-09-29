import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, fftconvolve, peak_widths
from scipy.ndimage import percentile_filter
from scipy.stats import median_abs_deviation

TEMPLATE_TARGET_FS = 5000.0


def estimate_noise_mad(trace):
    try:
        mad = median_abs_deviation(trace, scale='normal')
        if np.isfinite(mad) and mad > 0:
            return mad
    except Exception:
        mad = None
    try:
        std = float(np.nanstd(trace))
        if np.isfinite(std) and std > 0:
            return std * 0.6745
    except Exception:
        pass
    return 1e-6


def estimate_noise_mad_local(trace, window_samples):
    """Compute a local MAD-based noise estimate using a sliding window.

    Returns an array of the same length as *trace* where each element is the
    MAD (scaled to normal) computed within a centred window of
    *window_samples*.  Uses a fast rolling-median approach for efficiency.
    """
    from scipy.ndimage import median_filter
    n = len(trace)
    if window_samples < 5:
        window_samples = 5
    if window_samples > n:
        window_samples = n
    # Ensure odd window for symmetry
    if window_samples % 2 == 0:
        window_samples += 1
    local_median = median_filter(trace, size=window_samples, mode='reflect')
    abs_dev = np.abs(trace - local_median)
    local_mad = median_filter(abs_dev, size=window_samples, mode='reflect')
    # Scale factor for MAD → σ (normal distribution)
    scale = 1.4826
    out = local_mad * scale
    # Ensure a minimum floor to avoid zero-threshold
    out[out < 1e-9] = 1e-9
    return out


def detrend_trace(trace, fs, window_sec=0.05, percentile=20):
    window_samples = int(window_sec * fs)
    if window_samples < 5:
        window_samples = 5
    baseline = percentile_filter(trace, percentile, size=window_samples)
    return trace - baseline, baseline


def butter_bandpass(lowcut, highcut, fs, order=3):
    from .validation import validate_band
    low, high = validate_band(lowcut, highcut, fs, order)
    if not low and not high:
        return None
    if not low:
        return butter(order, high, btype='low', fs=fs, output='sos')
    if not high:
        return butter(order, low, btype='high', fs=fs, output='sos')
    return butter(order, [low, high], btype='band', fs=fs, output='sos')


def apply_filter(trace, fs, low=None, high=None, order=3):
    sos = butter_bandpass(low, high, fs, order=order)
    if sos is None:
        return trace
    return sosfiltfilt(sos, trace)


def apply_frame_processing(trace, frames=0, mode='Rolling average'):
    arr = np.asarray(trace, dtype=float).ravel()
    try:
        n_frames = int(frames)
    except Exception:
        n_frames = 0
    if n_frames <= 0 or arr.size <= 1:
        return arr.copy()

    mode_txt = str(mode)
    if 'Downsampling' in mode_txt:
        step = max(1, n_frames)
        n_blocks = int(np.ceil(arr.size / float(step)))
        if n_blocks <= 1:
            return arr.copy()
        ds = np.empty(n_blocks, dtype=float)
        x_ds = np.empty(n_blocks, dtype=float)
        for i in range(n_blocks):
            s = i * step
            e = min(arr.size, s + step)
            blk = arr[s:e]
            ds[i] = float(np.mean(blk)) if blk.size > 0 else float(arr[min(s, arr.size - 1)])
            x_ds[i] = s + 0.5 * max(1, (e - s) - 1)
        x_full = np.arange(arr.size, dtype=float)
        return np.interp(x_full, x_ds, ds, left=float(ds[0]), right=float(ds[-1]))

    win = int(max(1, n_frames))
    kern = np.ones(win, dtype=float) / float(win)
    return np.convolve(arr, kern, mode='same')


def _resample_template_to_fs(template, template_fs, target_fs):
    tpl = np.asarray(template, dtype=float).ravel()
    if tpl.size <= 3:
        return tpl
    try:
        src_fs = float(template_fs) if template_fs is not None else float(target_fs)
        dst_fs = float(target_fs)
    except Exception:
        return tpl
    if not np.isfinite(src_fs) or src_fs <= 0 or not np.isfinite(dst_fs) or dst_fs <= 0:
        return tpl
    if abs(src_fs - dst_fs) < 1e-9:
        return tpl
    duration_s = (tpl.size - 1) / src_fs
    n_new = int(round(duration_s * dst_fs)) + 1
    n_new = max(4, n_new)
    x_old = np.linspace(0.0, duration_s, tpl.size)
    x_new = np.linspace(0.0, duration_s, n_new)
    return np.interp(x_new, x_old, tpl)


def _resample_to_length(arr, n_out):
    x = np.asarray(arr, dtype=float).ravel()
    n_out = int(max(1, n_out))
    if x.size == 0:
        return np.zeros(n_out, dtype=float)
    if x.size == n_out:
        return x.copy()
    if x.size == 1:
        return np.full(n_out, float(x[0]), dtype=float)
    x_old = np.linspace(0.0, 1.0, x.size)
    x_new = np.linspace(0.0, 1.0, n_out)
    return np.interp(x_new, x_old, x)


def _resample_trace_to_fs(trace, src_fs, target_fs):
    x = np.asarray(trace, dtype=float).ravel()
    if x.size <= 1:
        return x.copy()
    try:
        src = float(src_fs)
        dst = float(target_fs)
    except Exception:
        return x.copy()
    if not np.isfinite(src) or not np.isfinite(dst) or src <= 0 or dst <= 0:
        return x.copy()
    if abs(src - dst) < 1e-9:
        return x.copy()
    n_out = int(np.floor((x.size - 1) * dst / src + 1e-9)) + 1
    return np.interp(np.arange(n_out)/dst, np.arange(x.size)/src, x)


def _orient_template_peak_positive(template):
    tpl = np.asarray(template, dtype=float).ravel()
    if tpl.size <= 3:
        return tpl
    x = tpl - np.nanmean(tpl)
    n = x.size
    center = n // 2
    half_w = max(1, int(round(0.1 * n)))
    s = max(0, center - half_w)
    e = min(n, center + half_w + 1)
    try:
        center_mean = float(np.nanmean(x[s:e]))
    except Exception:
        center_mean = np.nan
    if np.isfinite(center_mean):
        if center_mean < 0:
            return -x
        return x
    try:
        if abs(float(np.nanmin(x))) > abs(float(np.nanmax(x))):
            return -x
    except Exception:
        pass
    return x


def _build_template_distribution(template_bank, fs_bank, target_fs, force_peak_positive=False):
    rows = []
    if template_bank is None:
        return None, None
    for k, tpl in enumerate(template_bank):
        try:
            tpl_fs = fs_bank[k] if fs_bank is not None and k < len(fs_bank) else target_fs
        except Exception:
            tpl_fs = target_fs
        try:
            arr = np.asarray(tpl, dtype=float).ravel()
            if arr.size <= 3:
                continue
            if force_peak_positive:
                arr = _orient_template_peak_positive(arr)
            arr_rs = _resample_template_to_fs(arr, tpl_fs, target_fs)
            if arr_rs.size > 3 and np.all(np.isfinite(arr_rs)):
                rows.append(arr_rs)
        except Exception:
            continue
    if len(rows) == 0:
        return None, None
    lengths = [int(np.asarray(r).size) for r in rows]
    m = int(np.median(lengths))
    m = max(4, m)
    stack = np.vstack([_resample_to_length(np.asarray(r, dtype=float).ravel(), m) for r in rows])
    mu_signal = np.nanmean(stack, axis=0)
    return mu_signal, stack


def _llr_probability_vector(trace, mu_signal, sigma_signal, mu_noise, sigma_noise):
    x = np.asarray(trace, dtype=float).ravel()
    mu_s = np.asarray(mu_signal, dtype=float).ravel()
    sig_s = np.asarray(sigma_signal, dtype=float).ravel()
    if x.size == 0 or mu_s.size <= 3 or sig_s.size != mu_s.size:
        return np.zeros_like(x)

    eps = 1e-9
    sig_b = float(max(abs(float(sigma_noise)), eps))
    mu_b = float(mu_noise)
    sig_s = np.maximum(np.abs(sig_s), eps)
    m = int(mu_s.size)

    x2 = x * x
    ones = np.ones(m, dtype=float)

    sum_x = fftconvolve(x, ones, mode='same')
    sum_x2 = fftconvolve(x2, ones, mode='same')

    inv_var_s = 1.0 / (sig_s * sig_s)
    w1 = inv_var_s
    w2 = mu_s * inv_var_s
    w3 = (mu_s * mu_s) * inv_var_s

    term_signal = (
        -0.5 * float(np.sum(np.log(2.0 * np.pi * sig_s * sig_s)))
        -0.5 * fftconvolve(x2, w1[::-1], mode='same')
        + fftconvolve(x, w2[::-1], mode='same')
        -0.5 * float(np.sum(w3))
    )

    term_noise = (
        -0.5 * m * float(np.log(2.0 * np.pi * sig_b * sig_b))
        -0.5 * (sum_x2 - 2.0 * mu_b * sum_x + m * (mu_b * mu_b)) / (sig_b * sig_b)
    )

    return np.asarray(term_signal - term_noise, dtype=float)


def _compute_llr_from_template_bank(trace, template_bank, fs_bank, fs, force_peak_positive=False, noise_mask=None):
    x = np.asarray(trace, dtype=float).ravel()
    if x.size == 0:
        return x
    mu_signal, stack = _build_template_distribution(template_bank, fs_bank, fs, force_peak_positive=force_peak_positive)
    if mu_signal is None or stack is None:
        return np.zeros_like(x)

    noise = x if noise_mask is None else x[~noise_mask]
    if noise.size < 3:
        return np.zeros_like(x)
    mu_noise = float(np.nanmedian(noise))
    sigma_noise = float(estimate_noise_mad(noise))
    sigma_noise = max(abs(sigma_noise), 1e-9)

    n_templates = int(stack.shape[0])
    if n_templates > 9:
        sigma_signal = np.nanstd(stack, axis=0, ddof=1)
    else:
        sigma_signal = np.full(mu_signal.shape, sigma_noise, dtype=float)
    sigma_signal = np.maximum(np.asarray(sigma_signal, dtype=float), max(1e-9, sigma_noise * 1e-3))

    return _llr_probability_vector(x, mu_signal, sigma_signal, mu_noise, sigma_noise)


def _similarity_score(trace, template_bank, fs_bank, fs, short_core=False):
    """Pearson similarity and fitted positive response for an event-centred template.

    A short SS core permits truncated recovery tails in rapid bursts. The fitted
    response is kept separately so a high correlation with tiny noise cannot
    become a spike by itself.
    """
    mu, _ = _build_template_distribution(template_bank, fs_bank, fs, force_peak_positive=True)
    x = np.asarray(trace, dtype=float).ravel()
    if mu is None:
        return np.zeros_like(x), np.zeros_like(x), 0
    if short_core:
        length = min(mu.size, max(5, int(round(0.003 * fs)) | 1))
        center = mu.size // 2
        lo = min(max(0, center - length // 2), mu.size - length)
        mu = mu[lo:lo + length]
    m = mu.size
    kernel = mu - np.mean(mu)
    norm = float(np.linalg.norm(kernel))
    if norm < 1e-9:
        return np.zeros_like(x), np.zeros_like(x), m // 2
    kernel /= norm
    response = fftconvolve(x, kernel[::-1], mode='same')
    sums = fftconvolve(x, np.ones(m), mode='same')
    sums2 = fftconvolve(x*x, np.ones(m), mode='same')
    energy = np.sqrt(np.maximum(1e-12, sums2 - sums*sums/m))
    similarity = np.clip(response / energy, -1.0, 1.0)
    return similarity, response, m // 2


def _amplitude_fitted_llr_score(trace, template_bank, fs_bank, fs, noise_mask, short_core=False):
    """Positive-amplitude Gaussian GLRT gain for a bank's mean waveform.

    Fitting the amplitude prevents pooled bright-cell templates from rejecting
    same-shaped events in dim cells. A short SS core tolerates burst overlap.
    """
    similarity, response, support = _similarity_score(trace, template_bank, fs_bank, fs, short_core)
    sigma = max(float(_masked_sigma(trace, noise_mask)), 1e-9)
    z = np.maximum(response, 0.0) / sigma
    return 0.5 * z * z, response, similarity, support


def _positive_core_ss_score(trace, template_bank, fs_bank, fs, noise_mask):
    """Fit a positive SS core without imposing a neighboring-event prior.

    The baseline-corrected trace has zero as its null mean. Clipping negative
    template weights prevents nearby downward oscillations from increasing the
    matched response to a small positive deflection.
    """
    mu, _ = _build_template_distribution(template_bank, fs_bank, fs, force_peak_positive=True)
    x = np.asarray(trace, dtype=float).ravel()
    if mu is None:
        return np.zeros_like(x), np.zeros_like(x), 0
    length = min(mu.size, max(5, int(round(0.003 * fs)) | 1))
    center = mu.size // 2
    start = min(max(0, center - length // 2), mu.size - length)
    core = np.maximum(mu[start:start + length], 0.0)
    norm = float(np.linalg.norm(core))
    if norm < 1e-9:
        return np.zeros_like(x), np.zeros_like(x), length // 2
    kernel = core / norm
    response = fftconvolve(x, kernel[::-1], mode='same')
    sigma = max(float(_masked_sigma(x, noise_mask)), 1e-9)
    score = 0.5 * (np.maximum(response, 0.0) / sigma) ** 2
    return score, response, length // 2


def _kmeans_points(points, k, max_iter=40):
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros(0, dtype=int)
    n = x.shape[0]
    k = int(max(1, min(int(k), n)))
    init_idx = np.linspace(0, n - 1, k).round().astype(int)
    centers = x[init_idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(int(max_iter)):
        d2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(k):
            mask = labels == ci
            if np.any(mask):
                centers[ci] = np.mean(x[mask], axis=0)
    return labels


def _build_parallel_template_banks(template_bank, fs_bank, target_fs, force_peak_positive=False,
                                   max_use_types=3, n_components=2, selected_groups=None):
    if template_bank is None or len(template_bank) == 0:
        return []
    rows = []
    for k, tpl in enumerate(template_bank):
        try:
            tpl_fs = fs_bank[k] if fs_bank is not None and k < len(fs_bank) else target_fs
        except Exception:
            tpl_fs = target_fs
        try:
            arr = np.asarray(tpl, dtype=float).ravel()
            if arr.size <= 3:
                continue
            if force_peak_positive:
                arr = _orient_template_peak_positive(arr)
            # Keep group identities stable across recording sampling rates.
            arr_rs = _resample_template_to_fs(arr, tpl_fs, TEMPLATE_TARGET_FS)
            if arr_rs.size > 3 and np.all(np.isfinite(arr_rs)):
                rows.append(np.asarray(arr_rs, dtype=float).ravel())
        except Exception:
            continue
    if len(rows) == 0:
        return []

    lengths = [r.size for r in rows]
    m = max(4, int(np.median(lengths)))
    stack = np.vstack([_resample_to_length(r, m) for r in rows])

    # Cluster waveform shape, not brightness; the detector fits amplitude later.
    normalized = stack - np.mean(stack, axis=1, keepdims=True)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-9)
    centered = normalized - np.mean(normalized, axis=0, keepdims=True)
    if centered.shape[0] > 1 and centered.shape[1] > 1:
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            n_pc = max(1, min(int(n_components), vt.shape[0]))
            feats = centered @ vt[:n_pc].T
        except Exception:
            feats = centered[:, :1]
    else:
        feats = centered[:, :1]

    n_templates = stack.shape[0]
    k_clusters = max(1, min(int(max_use_types), n_templates))
    labels = _kmeans_points(feats, k_clusters)

    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    chosen = [int(unique[idx]) for idx in order]

    out = []
    active = None if selected_groups is None else set(int(group) for group in selected_groups)
    for group_number, cid in enumerate(chosen, start=1):
        if active is not None and group_number not in active:
            continue
        mask = labels == cid
        if not np.any(mask):
            continue
        bank = [_resample_template_to_fs(stack[i], TEMPLATE_TARGET_FS, target_fs)
                for i in np.where(mask)[0]]
        fs_list = [float(target_fs)] * len(bank)
        out.append((bank, fs_list))
    return out


def _filter_peaks_by_reference(peaks, ref_peaks, tol_samples):
    p = np.asarray(peaks, dtype=int).ravel()
    r = np.asarray(ref_peaks, dtype=int).ravel()
    if p.size == 0 or r.size == 0:
        return np.array([], dtype=int)
    tol = max(0, int(tol_samples))
    r_sorted = np.sort(r)
    keep = []
    for pk in p:
        i = np.searchsorted(r_sorted, pk)
        ok = False
        if i < r_sorted.size and abs(int(r_sorted[i]) - int(pk)) <= tol:
            ok = True
        if i > 0 and abs(int(r_sorted[i - 1]) - int(pk)) <= tol:
            ok = True
        if ok:
            keep.append(int(pk))
    if len(keep) == 0:
        return np.array([], dtype=int)
    return np.asarray(sorted(set(keep)), dtype=int)


def _filter_peaks_min_fwhm(signal, peaks, fs_hz, min_fwhm_ms):
    p = np.asarray(peaks, dtype=int).ravel()
    if p.size == 0:
        return np.array([], dtype=int)
    try:
        min_ms = float(min_fwhm_ms)
    except Exception:
        return np.sort(np.unique(p))
    try:
        fs_val = float(fs_hz)
    except Exception:
        fs_val = np.nan
    if (not np.isfinite(min_ms)) or min_ms <= 0 or (not np.isfinite(fs_val)) or fs_val <= 0:
        return np.sort(np.unique(p))
    try:
        widths_samples, _, _, _ = peak_widths(np.asarray(signal, dtype=float), p, rel_height=0.5)
        fwhm_ms = (widths_samples / fs_val) * 1000.0
        keep = p[fwhm_ms > min_ms]
        if keep.size == 0:
            return np.array([], dtype=int)
        return np.asarray(sorted(set(keep.tolist())), dtype=int)
    except Exception:
        return np.sort(np.unique(p))


def exclusion_mask(n, fs, cs_peaks=(), pre_ms=0, post_ms=0, initial_ms=0):
    """True marks excluded samples; intervals include the CS sample when enabled."""
    mask = np.zeros(n, dtype=bool)
    mask[:min(n, int(np.ceil(initial_ms*fs/1000)))] = True
    before, after = int(np.ceil(pre_ms*fs/1000)), int(np.ceil(post_ms*fs/1000))
    if before or after:
        for p in cs_peaks:
            mask[max(0, int(p)-before):min(n, int(p)+after+1)] = True
    return mask


def _masked_sigma(trace, mask):
    clean = np.asarray(trace)[~mask]
    return estimate_noise_mad(clean) if clean.size >= 3 else np.inf


def _masked_local_sigma(trace, mask, window):
    # Compute robust residual noise without allowing excluded samples into windows.
    import pandas as pd
    x = pd.Series(np.where(mask, np.nan, trace))
    size = max(5, min(len(x), int(window)))
    median = x.rolling(size, center=True, min_periods=3).median()
    mad = (x-median).abs().rolling(size, center=True, min_periods=3).median()*1.4826
    floor = _masked_sigma(trace, mask)
    return np.maximum(mad.fillna(floor).to_numpy(), max(1e-9, floor*.1))


def _select_peaks(trace, threshold, distance, mask):
    # Remove invalid/excluded candidates before refractory competition.
    peaks, _ = find_peaks(trace, height=threshold)
    peaks = peaks[~mask[peaks]]
    return _suppress_candidates(trace, peaks, distance)


def _suppress_candidates(trace, peaks, distance):
    peaks = np.asarray(peaks, dtype=int)
    kept = []
    blocked = np.zeros(len(trace), dtype=bool)
    for p in peaks[np.argsort(-np.asarray(trace)[peaks], kind='stable')]:
        if not blocked[p]:
            kept.append(int(p))
            blocked[max(0, p-distance+1):min(len(trace), p+distance)] = True
    return np.asarray(sorted(kept), dtype=int)


def _cs_width_filter(trace, peaks, fs, limit, align_ms=0):
    from .widths import measure_widths
    rows = measure_widths(trace, peaks, fs, 100, align_ms=align_ms)
    keep = []
    for row in rows:
        measured = np.isfinite(row['fwhm_ms'])
        row['decision'] = ('pass' if row['fwhm_ms'] > limit or limit <= 0 else 'fail') if measured else 'uncertain'
        if row['decision'] != 'fail':
            keep.append(row['candidate_index'])
    return np.asarray(keep, dtype=int), rows


def _prepare_detection(raw, fs, negative_going, use_preprocessed, pre_detrended, pre_baseline,
                       pre_detrended_cs, pre_detrended_ss):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1 or raw.size < 3 or not np.all(np.isfinite(raw)):
        raise ValueError('Detection requires a finite one-dimensional trace with at least three samples.')
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('Sampling rate must be finite and positive.')
    if use_preprocessed and pre_detrended is not None and pre_baseline is not None:
        detrended, baseline = np.asarray(pre_detrended), np.asarray(pre_baseline)
    else:
        detrended, baseline = detrend_trace(-raw if negative_going else raw, fs)
    cs = np.asarray(pre_detrended_cs) if pre_detrended_cs is not None else detrended
    ss = np.asarray(pre_detrended_ss) if pre_detrended_ss is not None else detrended
    for array in (detrended, baseline, cs, ss):
        if array.shape != raw.shape or not np.all(np.isfinite(array)):
            raise ValueError('Preprocessed trace/baseline must be finite and match the raw trace.')
    return detrended, baseline, cs, ss


def process_cell_simple(raw_trace, fs, negative_going=True,
                        cs_low_cut=0.0, cs_high_cut=150.0, cs_thresh_sigma=6.0, cs_min_dist_ms=25,
                        cs_min_fwhm_ms=4.0, ss_low_cut=0.0, ss_high_cut=0.0, ss_thresh_sigma=2.5,
                        ss_min_dist_ms=2, ss_blank_ms=15, ss_min_width_ms=1, ss_max_width_ms=6,
                        use_preprocessed=False, pre_detrended=None, pre_baseline=None,
                        pre_detrended_cs=None, pre_detrended_ss=None,
                        initial_blank_ms=0.0, cs_order=3, ss_order=3,
                        local_baseline=False, local_baseline_cs_ms=200.0, local_baseline_ss_ms=50.0,
                        ss_mask_pre_ms=None, ss_mask_post_ms=None):
    detrended, baseline, cs, ss = _prepare_detection(raw_trace, fs, negative_going, use_preprocessed,
        pre_detrended, pre_baseline, pre_detrended_cs, pre_detrended_ss)
    cs = apply_filter(cs, fs, cs_low_cut, cs_high_cut, cs_order)
    ss = apply_filter(ss, fs, ss_low_cut, ss_high_cut, ss_order)
    cs_mask = exclusion_mask(len(cs), fs, initial_ms=initial_blank_ms)
    sigma_cs = _masked_sigma(cs, cs_mask)
    cs_threshold = cs_thresh_sigma * (_masked_local_sigma(cs, cs_mask, local_baseline_cs_ms*fs/1000)
                                     if local_baseline else sigma_cs)
    cs_peaks = _select_peaks(cs, cs_threshold, max(1, int(np.ceil(cs_min_dist_ms*fs/1000))), cs_mask)
    cs_peaks, cs_widths = _cs_width_filter(cs, cs_peaks, fs, cs_min_fwhm_ms)
    pre = ss_blank_ms/2 if ss_mask_pre_ms is None else ss_mask_pre_ms
    post = ss_blank_ms/2 if ss_mask_post_ms is None else ss_mask_post_ms
    ss_mask = exclusion_mask(len(ss), fs, cs_peaks, pre, post, initial_blank_ms)
    sigma_ss = _masked_sigma(ss, ss_mask)
    ss_threshold = ss_thresh_sigma * (_masked_local_sigma(ss, ss_mask, local_baseline_ss_ms*fs/1000)
                                     if local_baseline else sigma_ss)
    ss_peaks = _select_peaks(ss, ss_threshold, max(1, int(np.ceil(ss_min_dist_ms*fs/1000))), ss_mask)
    result = dict(detrended=detrended, baseline=baseline, cs_trace=cs, ss_trace=ss,
                  cs_peaks=cs_peaks, ss_peaks=ss_peaks, sigma_cs=sigma_cs, sigma_ss=sigma_ss,
                  raw_sigma=estimate_noise_mad(detrended), det_method='Threshold',
                  threshold_mode='Sigma x MAD (local)' if local_baseline else 'Sigma x MAD',
                  local_baseline=bool(local_baseline), cs_threshold_used=cs_thresh_sigma*sigma_cs,
                  ss_threshold_used=ss_thresh_sigma*sigma_ss, cs_width_candidates=cs_widths,
                  cs_exclusion_mask=cs_mask, ss_exclusion_mask=ss_mask,
                  ss_mask_pre_ms_used=pre, ss_mask_post_ms_used=post)
    if local_baseline:
        result.update(cs_threshold_trace=cs_threshold, ss_threshold_trace=ss_threshold)
    return result


def process_cell_template_matching(raw_trace, fs,
                                   template_cs_bank=None, template_ss_bank=None,
                                   template_cs_fs_bank=None, template_ss_fs_bank=None,
                                   negative_going=True,
                                   cs_low_cut=0.0, cs_high_cut=150.0, cs_thresh_sigma=6.0, cs_min_dist_ms=25,
                                   cs_min_fwhm_ms=4.0, ss_low_cut=0.0, ss_high_cut=0.0, ss_thresh_sigma=3.0,
                                   ss_min_dist_ms=2, ss_blank_ms=15,
                                   template_match_method='LLR Probability Vector', parallel_match=False,
                                   use_preprocessed=False, pre_detrended=None, pre_baseline=None,
                                   pre_detrended_cs=None, pre_detrended_ss=None,
                                   initial_blank_ms=0.0, cs_order=3, ss_order=3,
                                   ss_mask_pre_ms=None, ss_mask_post_ms=None,
                                   cs_similarity_threshold=0.90, ss_similarity_threshold=0.80,
                                   similarity_min_response_sigma=2.2,
                                   template_ss_lowpass_hz=0.0,
                                   fixed_exclusion_masks=None,
                                   parallel_groups=3, parallel_components=2,
                                   cs_selected_groups=None, ss_selected_groups=None,
                                   cs_min_filtered_peak_sigma=3.0):
    if template_match_method not in ('LLR Probability Vector', 'Normalized Similarity', 'Burst-aware LLR'):
        raise ValueError('Unsupported template match method.')
    for val in (cs_similarity_threshold, ss_similarity_threshold):
        if not np.isfinite(val) or not 0 < val <= 1:
            raise ValueError('Similarity thresholds must be between 0 and 1.')
    if not np.isfinite(similarity_min_response_sigma) or similarity_min_response_sigma <= 0:
        raise ValueError('Minimum similarity response must be positive.')
    if not np.isfinite(cs_min_filtered_peak_sigma) or cs_min_filtered_peak_sigma < 0:
        raise ValueError('Minimum CS filtered-trace peak must be finite and nonnegative.')
    if not np.isfinite(template_ss_lowpass_hz) or template_ss_lowpass_hz < 0:
        raise ValueError('Template SS low-pass must be nonnegative.')
    detrended, baseline, cs, ss = _prepare_detection(raw_trace, fs, negative_going, use_preprocessed,
        pre_detrended, pre_baseline, pre_detrended_cs, pre_detrended_ss)
    cs = apply_filter(cs, fs, cs_low_cut, cs_high_cut, cs_order)
    ss = apply_filter(ss, fs, ss_low_cut, ss_high_cut, ss_order)
    lowpass_used = 0.0
    if template_ss_lowpass_hz > 0:
        lowpass_used = min(float(template_ss_lowpass_hz), 0.45 * fs)
        ss = apply_filter(ss, fs, high=lowpass_used, order=ss_order)
    sim_fs = max(fs, TEMPLATE_TARGET_FS)
    cs_sim, ss_sim = _resample_trace_to_fs(cs, fs, sim_fs), _resample_trace_to_fs(ss, fs, sim_fs)
    fixed_masks = None
    if fixed_exclusion_masks is not None:
        fixed_masks = {}
        for kind in ('cs', 'ss'):
            native = np.asarray(fixed_exclusion_masks[kind], dtype=bool)
            if native.ndim != 1 or native.size != len(cs):
                raise ValueError(f'Fixed {kind.upper()} mask must match the native trace length.')
            sample_map_sim = np.minimum(np.rint(np.arange(len(cs_sim))*fs/sim_fs).astype(int), len(cs)-1)
            fixed_masks[kind] = native[sample_map_sim]
    cs_mask = (fixed_masks['cs'] if fixed_masks is not None else
               exclusion_mask(len(cs_sim), sim_fs, initial_ms=initial_blank_ms))

    def score_and_detect(trace, bank, rates, sigma, similarity_threshold, spacing, mask, short_core,
                         selected_groups, mask_is_safe=False):
        if bank is None or len(bank) == 0:
            return np.zeros_like(trace), np.array([], dtype=int), np.nan, mask, dict(
                score_peaks=0, masked=0, response_rejected=0, peak_rejected=0,
                refractory_rejected=0, accepted=0, no_templates=True)
        for i, tpl in enumerate(bank):
            arr = np.asarray(tpl, dtype=float)
            rate = rates[i] if rates is not None and i < len(rates) else sim_fs
            if arr.ndim != 1 or arr.size < 4 or not np.all(np.isfinite(arr)) or not np.isfinite(rate) or rate <= 0:
                raise ValueError('Templates need finite waveforms and positive sampling rates.')
        banks = (_build_parallel_template_banks(bank, rates, sim_fs, force_peak_positive=True,
                 max_use_types=parallel_groups, n_components=parallel_components,
                 selected_groups=selected_groups)
                 if parallel_match else [(bank, rates)])
        if not banks:
            return np.zeros_like(trace), np.array([], dtype=int), np.nan, mask, dict(
                score_peaks=0, masked=0, response_rejected=0, peak_rejected=0,
                refractory_rejected=0, accepted=0, no_active_groups=True)
        scores, candidates, thresholds = [], [], []
        diagnostics = dict(score_peaks=0, masked=0, response_rejected=0, peak_rejected=0,
                           refractory_rejected=0, accepted=0)
        # Score windows touching excluded data are excluded too; never zero the trace.
        support = max(len(_resample_template_to_fs(tpl, rates[i] if rates is not None and i < len(rates) else sim_fs, sim_fs))
                      for i, tpl in enumerate(bank))//2
        if short_core:
            support = min(support, max(5, int(round(0.003 * sim_fs)) | 1)//2)
        from scipy.ndimage import maximum_filter1d
        if mask_is_safe:
            safe_mask = mask.copy()
        else:
            safe_mask = maximum_filter1d(mask.astype(np.uint8), size=2*support+1, mode='constant') > 0
            safe_mask[:support] = True
            if support:
                safe_mask[-support:] = True
        response_floor = max(similarity_min_response_sigma, 3.0 if not short_core else 0.0) * _masked_sigma(trace, safe_mask)
        for bk, br in banks:
            if template_match_method == 'Normalized Similarity':
                score, response, _ = _similarity_score(trace, bk, br, sim_fs, short_core)
                threshold = similarity_threshold
            elif template_match_method == 'Burst-aware LLR' and short_core:
                score, response, _ = _positive_core_ss_score(
                    trace, bk, br, sim_fs, safe_mask)
                threshold = 0.5 * float(sigma)**2
            else:
                score, response, _, _ = _amplitude_fitted_llr_score(
                    trace, bk, br, sim_fs, safe_mask, short_core)
                # Map the user-selected matched-response ratio to its GLRT gain.
                # This is not a calibrated false-positive probability.
                threshold = 0.5 * float(sigma)**2
            scores.append(score)
            thresholds.append(threshold)
            found, _ = find_peaks(score, height=threshold)
            diagnostics['score_peaks'] += len(found)
            diagnostics['masked'] += int(np.count_nonzero(safe_mask[found]))
            found = found[~safe_mask[found]]
            diagnostics['response_rejected'] += int(np.count_nonzero(response[found] < response_floor))
            found = found[response[found] >= response_floor]
            if not short_core and cs_min_filtered_peak_sigma > 0:
                # A long CS template can match slow baseline structure. Require
                # an actual local deflection as well as integrated evidence.
                cs_height_floor = cs_min_filtered_peak_sigma * _masked_sigma(trace, safe_mask)
                diagnostics['peak_rejected'] += int(np.count_nonzero(trace[found] < cs_height_floor))
                found = found[trace[found] >= cs_height_floor]
            candidates.extend(found)
        score = np.maximum.reduce([s - t for s, t in zip(scores, thresholds)]) if len(scores) > 1 else scores[0]
        display_threshold = 0.0 if len(scores) > 1 else thresholds[0]
        # Apply the Advanced Settings spacing once, across all template groups.
        candidates = np.unique(candidates).astype(int)
        candidates = candidates[~safe_mask[candidates]]
        peaks = _suppress_candidates(score, candidates, max(1, int(np.ceil(spacing*sim_fs/1000))))
        diagnostics['refractory_rejected'] += len(candidates) - len(peaks)
        diagnostics['accepted'] = len(peaks)
        return score, peaks, display_threshold, safe_mask, diagnostics

    cs_score, cs_candidates, cs_thr, cs_mask, cs_diagnostics = score_and_detect(cs_sim, template_cs_bank,
        template_cs_fs_bank, cs_thresh_sigma, cs_similarity_threshold, cs_min_dist_ms, cs_mask, False,
        cs_selected_groups, mask_is_safe=fixed_masks is not None)
    cs_peaks = np.unique(np.clip(np.rint(cs_candidates*fs/sim_fs).astype(int), 0, len(cs)-1))
    cs_peaks, cs_widths = _cs_width_filter(cs, cs_peaks, fs, cs_min_fwhm_ms, align_ms=2.0)
    pre = ss_blank_ms/2 if ss_mask_pre_ms is None else ss_mask_pre_ms
    post = ss_blank_ms/2 if ss_mask_post_ms is None else ss_mask_post_ms
    ss_mask = (fixed_masks['ss'] if fixed_masks is not None else exclusion_mask(
        len(ss_sim), sim_fs, np.rint(cs_peaks*sim_fs/fs).astype(int), pre, post, initial_blank_ms))
    ss_score, ss_candidates, ss_thr, ss_mask, ss_diagnostics = score_and_detect(ss_sim, template_ss_bank,
        template_ss_fs_bank, ss_thresh_sigma, ss_similarity_threshold, ss_min_dist_ms, ss_mask, True,
        ss_selected_groups, mask_is_safe=fixed_masks is not None)
    ss_peaks = np.unique(np.clip(np.rint(ss_candidates*fs/sim_fs).astype(int), 0, len(ss)-1))
    sample_map = np.minimum(np.rint(np.arange(len(cs))*sim_fs/fs).astype(int), len(cs_sim)-1)
    cs_mask, ss_mask = cs_mask[sample_map], ss_mask[sample_map]
    cs_peaks, ss_peaks = cs_peaks[~cs_mask[cs_peaks]], ss_peaks[~ss_mask[ss_peaks]]
    return dict(detrended=detrended, baseline=baseline, cs_trace=cs, ss_trace=ss,
                cs_peaks=cs_peaks, ss_peaks=ss_peaks,
                raw_sigma=estimate_noise_mad(detrended),
                sigma_cs=_masked_sigma(cs, cs_mask), sigma_ss=_masked_sigma(ss, ss_mask),
                cs_threshold_used=cs_thr, ss_threshold_used=ss_thr,
                det_method=f'Template Matching ({"Positive-core LLR" if template_match_method == "Burst-aware LLR" else template_match_method})',
                threshold_mode=('Pearson r + minimum response' if template_match_method == 'Normalized Similarity'
                    else 'Positive-core LLR gain (response / per-sample MAD)² / 2' if template_match_method == 'Burst-aware LLR'
                    else 'Amplitude-fitted LLR gain (response / per-sample MAD)² / 2'),
                parallel_match=bool(parallel_match), cs_width_candidates=cs_widths,
                cs_candidate_diagnostics=cs_diagnostics,
                ss_candidate_diagnostics=ss_diagnostics,
                cs_similarity_trace=np.interp(np.arange(len(cs))/fs, np.arange(len(cs_score))/sim_fs, cs_score),
                ss_similarity_trace=np.interp(np.arange(len(ss))/fs, np.arange(len(ss_score))/sim_fs, ss_score),
                cs_exclusion_mask=cs_mask, ss_exclusion_mask=ss_mask,
                ss_mask_pre_ms_used=pre, ss_mask_post_ms_used=post,
                template_mask_includes_support=True, width_alignment_ms=2.0,
                template_ss_lowpass_hz_used=lowpass_used,
                cs_min_filtered_peak_sigma_used=float(cs_min_filtered_peak_sigma))


def get_interpolated_wave(wave, fs, upscale_factor=10):
    n_points = len(wave)
    if n_points <= 1:
        return np.arange(n_points), wave
    x_new = np.linspace(0, n_points - 1, int(n_points * upscale_factor))
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(np.arange(n_points), wave)
        return x_new, cs(x_new)
    except Exception:
        return x_new, np.interp(x_new, np.arange(n_points), wave)


def get_wave_stats(wave, time_axis_ms):
    if len(wave) == 0:
        return np.nan, np.nan
    arr = np.asarray(wave, dtype=float).ravel()
    tx = np.asarray(time_axis_ms, dtype=float).ravel()
    if arr.size == 0 or tx.size != arr.size:
        return np.nan, np.nan
    finite = np.isfinite(arr) & np.isfinite(tx)
    if not np.any(finite):
        return np.nan, np.nan
    arr = arr[finite]
    tx = tx[finite]
    if arr.size == 0:
        return np.nan, np.nan
    peak_idx = int(np.nanargmax(arr))
    n_base = max(5, int(round(0.10 * arr.size)))
    n_base = min(n_base, max(1, peak_idx)) if peak_idx > 0 else min(n_base, arr.size)
    baseline = float(np.nanmedian(arr[:n_base])) if n_base > 0 else 0.0
    y = arr - baseline
    amp = float(y[peak_idx])
    if not np.isfinite(amp) or amp <= 0:
        return amp, np.nan
    half_height = 0.5 * amp

    def _cross_time(i0, i1):
        y0 = float(y[i0] - half_height)
        y1 = float(y[i1] - half_height)
        t0 = float(tx[i0])
        t1 = float(tx[i1])
        denom = y1 - y0
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            return t0
        frac = float(np.clip(-y0 / denom, 0.0, 1.0))
        return t0 + frac * (t1 - t0)

    left_t = np.nan
    for i in range(peak_idx - 1, -1, -1):
        if y[i] <= half_height <= y[i + 1]:
            left_t = _cross_time(i, i + 1)
            break
    right_t = np.nan
    for i in range(peak_idx, arr.size - 1):
        if y[i] >= half_height >= y[i + 1]:
            right_t = _cross_time(i, i + 1)
            break
    fwhm = np.nan
    if np.isfinite(left_t) and np.isfinite(right_t) and right_t > left_t:
        fwhm = float(right_t - left_t)
    else:
        try:
            widths, _, left_ips, right_ips = peak_widths(y, [peak_idx], rel_height=0.5)
            if widths.size > 0 and np.isfinite(widths[0]) and widths[0] > 0:
                sample_axis = np.arange(arr.size, dtype=float)
                lt = float(np.interp(float(left_ips[0]), sample_axis, tx))
                rt = float(np.interp(float(right_ips[0]), sample_axis, tx))
                if rt > lt:
                    fwhm = rt - lt
        except Exception:
            pass
    return amp, fwhm


def _select_event_bank(peaks, max_per_cell=None):
    try:
        arr = np.array(peaks, dtype=int)
        if arr.size <= 0:
            return arr
        if max_per_cell is None:
            return arr
        max_n = int(max_per_cell)
        if max_n <= 0 or arr.size <= max_n:
            return arr
        idx = np.linspace(0, arr.size - 1, max_n, dtype=int)
        return arr[idx]
    except Exception:
        try:
            arr = np.array(peaks, dtype=int)
            if max_per_cell is None:
                return arr
            return arr[:int(max_per_cell)]
        except Exception:
            return np.array([], dtype=int)


def compute_event_snrs(res, spike_type='CS', fs=1000.0, window_ms=100, max_per_cell=None, trace_override=None):
    """Compute per-event SNR using robust local noise windows with spike masking.

    Parameters kept for compatibility:
    - window_ms is currently unused by design (type-specific windows are fixed).
    - trace_override allows callers to enforce a unified waveform source trace.
    """
    snr_list = []
    if res is None:
        return snr_list

    def _ms_to_samples(ms):
        try:
            return int(max(1, round(float(ms) * float(fs) / 1000.0)))
        except Exception:
            return 1

    def _robust_sigma(arr):
        x = np.asarray(arr, dtype=float).ravel()
        if x.size <= 0:
            return np.nan
        x = x[np.isfinite(x)]
        if x.size <= 0:
            return np.nan
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        sig = 1.4826 * mad
        return sig if np.isfinite(sig) and sig > 0 else np.nan

    def _mark_exclusion(mask, peak_arr, pre_ms, post_ms, n_total):
        pks = np.asarray(peak_arr, dtype=int).ravel()
        if pks.size <= 0:
            return
        pre = _ms_to_samples(abs(pre_ms))
        post = _ms_to_samples(abs(post_ms))
        for q in pks:
            if not np.isfinite(q):
                continue
            qq = int(q)
            s = max(0, qq - pre)
            e = min(n_total, qq + post + 1)
            if e > s:
                mask[s:e] = False

    try:
        st = str(spike_type).upper()
        ss_peaks = np.array(res.get('ss_peaks', []), dtype=int)
        cs_peaks = np.array(res.get('cs_peaks', []), dtype=int)
        peaks = cs_peaks if st == 'CS' else ss_peaks

        if trace_override is not None:
            trace = np.asarray(trace_override, dtype=float).ravel()
        elif st == 'CS':
            trace = np.asarray(res.get('cs_trace', np.array([])), dtype=float).ravel()
        else:
            trace = np.asarray(res.get('ss_trace', res.get('detrended', np.array([]))), dtype=float).ravel()

        if trace.size <= 0 or peaks.size == 0:
            return snr_list

        n = int(trace.size)
        exclusion = np.asarray(res.get(st.lower() + '_exclusion_mask', np.zeros(n, dtype=bool)), dtype=bool)
        non_spike_mask = ~exclusion.copy()
        if st == 'SS':
            _mark_exclusion(non_spike_mask, ss_peaks, pre_ms=2.0, post_ms=4.0, n_total=n)
            _mark_exclusion(non_spike_mask, cs_peaks, pre_ms=8.0, post_ms=30.0, n_total=n)
            base_pre_ms, base_post_ms = 5.0, 1.0
            noise_pre_ms, noise_post_ms = 50.0, 5.0
        else:
            _mark_exclusion(non_spike_mask, ss_peaks, pre_ms=3.0, post_ms=5.0, n_total=n)
            _mark_exclusion(non_spike_mask, cs_peaks, pre_ms=20.0, post_ms=80.0, n_total=n)
            base_pre_ms, base_post_ms = 20.0, 5.0
            noise_pre_ms, noise_post_ms = 150.0, 20.0

        global_sigma = _robust_sigma(trace[non_spike_mask])
        if not np.isfinite(global_sigma) or global_sigma <= 0:
            global_sigma = _robust_sigma(trace[~exclusion])
        if not np.isfinite(global_sigma) or global_sigma <= 0:
            return snr_list

        chosen = _select_event_bank(peaks, max_per_cell=max_per_cell)
        base_pre = _ms_to_samples(base_pre_ms)
        base_post = _ms_to_samples(base_post_ms)
        noise_pre = _ms_to_samples(noise_pre_ms)
        noise_post = _ms_to_samples(noise_post_ms)
        min_clean = max(20, _ms_to_samples(5.0))

        for p in chosen:
            pi = int(p)
            if pi < 0 or pi >= n:
                continue

            b0 = max(0, pi - base_pre)
            b1 = max(0, pi - base_post)
            if b1 <= b0:
                continue
            baseline_seg = trace[b0:b1]
            if baseline_seg.size <= 0:
                continue
            baseline_i = float(np.median(baseline_seg[np.isfinite(baseline_seg)])) if np.any(np.isfinite(baseline_seg)) else np.nan
            if not np.isfinite(baseline_i):
                continue
            amplitude_i = float(trace[pi] - baseline_i)

            n0 = max(0, pi - noise_pre)
            n1 = max(0, pi - noise_post)
            if n1 <= n0:
                continue
            local_vals = trace[n0:n1]
            local_mask = non_spike_mask[n0:n1]
            x_clean = local_vals[local_mask]
            x_clean = x_clean[np.isfinite(x_clean)]

            if x_clean.size >= min_clean:
                sigma_i = _robust_sigma(x_clean)
            else:
                sigma_i = np.nan
            if not np.isfinite(sigma_i) or sigma_i <= 0:
                sigma_i = float(global_sigma)
            if not np.isfinite(sigma_i) or sigma_i <= 0:
                continue

            snr_i = amplitude_i / sigma_i
            if np.isfinite(snr_i):
                snr_list.append(float(snr_i))
    except Exception:
        pass
    return snr_list
