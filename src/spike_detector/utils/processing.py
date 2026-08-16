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
    nyq = 0.5 * fs
    if lowcut is None and highcut is None:
        return None

    def _norm(val):
        if val is None:
            return None
        try:
            v = float(val)
        except Exception:
            return None
        if not np.isfinite(v) or v <= 0.0:
            return None
        return v / float(nyq)

    low_n = _norm(lowcut)
    high_n = _norm(highcut)
    if low_n is None and high_n is None:
        return None
    if high_n is not None and high_n >= 1.0:
        high_n = 1.0 - 1e-3
    if low_n is not None and high_n is not None and low_n >= high_n:
        return None
    try:
        if low_n is None:
            return butter(order, high_n, btype='low', output='sos')
        if high_n is None:
            return butter(order, low_n, btype='high', output='sos')
        return butter(order, [low_n, high_n], btype='band', output='sos')
    except ValueError:
        return None


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
    n_out = int(round(x.size * dst / src))
    return _resample_to_length(x, max(2, n_out))


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


def _compute_llr_from_template_bank(trace, template_bank, fs_bank, fs, force_peak_positive=False):
    x = np.asarray(trace, dtype=float).ravel()
    if x.size == 0:
        return x
    mu_signal, stack = _build_template_distribution(template_bank, fs_bank, fs, force_peak_positive=force_peak_positive)
    if mu_signal is None or stack is None:
        return np.zeros_like(x)

    mu_noise = float(np.nanmedian(x))
    sigma_noise = float(estimate_noise_mad(x))
    sigma_noise = max(abs(sigma_noise), 1e-9)

    n_templates = int(stack.shape[0])
    if n_templates > 9:
        sigma_signal = np.nanstd(stack, axis=0, ddof=1)
    else:
        sigma_signal = np.full(mu_signal.shape, sigma_noise, dtype=float)
    sigma_signal = np.maximum(np.asarray(sigma_signal, dtype=float), max(1e-9, sigma_noise * 1e-3))

    return _llr_probability_vector(x, mu_signal, sigma_signal, mu_noise, sigma_noise)


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


def _build_parallel_template_banks(template_bank, fs_bank, target_fs, force_peak_positive=False, max_use_types=3):
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
            arr_rs = _resample_template_to_fs(arr, tpl_fs, target_fs)
            if arr_rs.size > 3 and np.all(np.isfinite(arr_rs)):
                rows.append(np.asarray(arr_rs, dtype=float).ravel())
        except Exception:
            continue
    if len(rows) == 0:
        return []

    lengths = [r.size for r in rows]
    m = max(4, int(np.median(lengths)))
    stack = np.vstack([_resample_to_length(r, m) for r in rows])

    centered = stack - np.mean(stack, axis=0, keepdims=True)
    if centered.shape[0] > 1 and centered.shape[1] > 1:
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            n_pc = max(1, min(2, vt.shape[0]))
            feats = centered @ vt[:n_pc].T
        except Exception:
            feats = centered[:, :1]
    else:
        feats = centered[:, :1]

    n_templates = stack.shape[0]
    k_clusters = max(1, min(max(4, max_use_types), n_templates))
    labels = _kmeans_points(feats, k_clusters)

    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    chosen = [int(unique[idx]) for idx in order[:max(1, min(int(max_use_types), unique.size))]]

    out = []
    for cid in chosen:
        mask = labels == cid
        if not np.any(mask):
            continue
        bank = [stack[i].copy() for i in np.where(mask)[0]]
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


def process_cell_template_matching(raw_trace, fs,
                                   template_cs_bank=None, template_ss_bank=None,
                                   template_cs_fs_bank=None, template_ss_fs_bank=None,
                                   negative_going=True,
                                   cs_low_cut=0.0, cs_high_cut=150.0, cs_thresh_sigma=6.0, cs_min_dist_ms=25,
                                   cs_min_fwhm_ms=4.0,
                                   ss_low_cut=0.0, ss_high_cut=0.0, ss_thresh_sigma=2.5,
                                   ss_min_dist_ms=2, ss_blank_ms=15,
                                   template_match_method='LLR Probability Vector',
                                   parallel_match=False,
                                   use_preprocessed=False, pre_detrended=None, pre_baseline=None,
                                   pre_detrended_cs=None, pre_detrended_ss=None,
                                   initial_blank_ms=0.0, cs_order=3, ss_order=3):
    working = raw_trace * -1 if negative_going else raw_trace
    if use_preprocessed and pre_detrended is not None and pre_baseline is not None:
        detrended = pre_detrended
        baseline = pre_baseline
    else:
        detrended, baseline = detrend_trace(working, fs, window_sec=0.05, percentile=20)

    detr_for_detection = detrended
    detr_for_detection_cs = np.asarray(pre_detrended_cs, dtype=float) if pre_detrended_cs is not None else detr_for_detection
    detr_for_detection_ss = np.asarray(pre_detrended_ss, dtype=float) if pre_detrended_ss is not None else detr_for_detection

    global_sigma = estimate_noise_mad(detrended)

    sim_fs = float(max(float(fs), TEMPLATE_TARGET_FS))
    cs_base = apply_filter(detr_for_detection_cs, fs, low=cs_low_cut, high=cs_high_cut, order=cs_order)
    cs_base_sim = _resample_trace_to_fs(cs_base, fs, sim_fs)
    if bool(parallel_match):
        cs_banks = _build_parallel_template_banks(template_cs_bank, template_cs_fs_bank, sim_fs,
                                                  force_peak_positive=bool(negative_going),
                                                  max_use_types=3)
        cs_scores = []
        cs_union_candidates = []
        cs_thr_each = []
        for bk, bk_fs in cs_banks:
            s = _compute_llr_from_template_bank(cs_base_sim, bk, bk_fs, sim_fs, force_peak_positive=False)
            cs_scores.append(s)
            try:
                sig_i = estimate_noise_mad(s)
                thr_i = float(cs_thresh_sigma * sig_i)
                cs_thr_each.append(thr_i)
                p_i, _ = find_peaks(s, height=thr_i, distance=max(1, int((cs_min_dist_ms / 1000.0) * sim_fs)))
                if p_i.size > 0:
                    cs_union_candidates.append(np.asarray(p_i, dtype=int))
            except Exception:
                pass
        if len(cs_scores) > 0:
            cs_trace_sim = np.maximum.reduce(cs_scores)
        else:
            cs_trace_sim = _compute_llr_from_template_bank(cs_base_sim, template_cs_bank, template_cs_fs_bank, sim_fs, force_peak_positive=bool(negative_going))
    else:
        cs_trace_sim = _compute_llr_from_template_bank(cs_base_sim, template_cs_bank, template_cs_fs_bank, sim_fs, force_peak_positive=bool(negative_going))
    sigma_cs = estimate_noise_mad(cs_trace_sim)
    cs_dist = max(1, int((cs_min_dist_ms / 1000.0) * sim_fs))
    cs_has_templates = template_cs_bank is not None and len(template_cs_bank) > 0
    if cs_has_templates:
        cs_threshold_used = float(cs_thresh_sigma * sigma_cs)
        cs_candidates_sim, _ = find_peaks(cs_trace_sim, height=cs_threshold_used, distance=cs_dist)
        if bool(parallel_match):
            extra = []
            try:
                if len(cs_union_candidates) > 0:
                    extra.append(np.concatenate(cs_union_candidates))
            except Exception:
                pass
            if len(extra) > 0:
                cs_candidates_sim = np.unique(np.concatenate([cs_candidates_sim] + extra).astype(int))
            try:
                if len(cs_thr_each) > 0:
                    cs_threshold_used = float(min([cs_threshold_used] + cs_thr_each))
            except Exception:
                pass
    else:
        cs_threshold_used = np.nan
        cs_candidates_sim = np.array([], dtype=int)
    cs_candidates_sim = _filter_peaks_min_fwhm(cs_trace_sim, cs_candidates_sim, sim_fs, cs_min_fwhm_ms)
    if initial_blank_ms is not None and initial_blank_ms > 0:
        init_blank_samples = int((initial_blank_ms / 1000.0) * sim_fs)
        cs_candidates_sim = cs_candidates_sim[cs_candidates_sim >= init_blank_samples]
    if cs_candidates_sim.size > 0:
        cs_peaks = np.unique(np.clip(np.round(cs_candidates_sim * (fs / sim_fs)).astype(int), 0, len(raw_trace) - 1))
    else:
        cs_peaks = np.array([], dtype=int)

    ss_base = apply_filter(detr_for_detection_ss, fs, low=ss_low_cut, high=ss_high_cut, order=ss_order)
    ss_base_sim = _resample_trace_to_fs(ss_base, fs, sim_fs)
    ss_base_clean_sim = ss_base_sim.copy()
    blank_samples = max(0, int((ss_blank_ms / 1000.0) * sim_fs))
    for cs_idx in cs_candidates_sim:
        start = max(0, cs_idx - blank_samples // 2)
        end = min(len(ss_base_clean_sim), start + blank_samples)
        ss_base_clean_sim[start:end] = 0

    if bool(parallel_match):
        ss_banks = _build_parallel_template_banks(template_ss_bank, template_ss_fs_bank, sim_fs,
                                                  force_peak_positive=bool(negative_going),
                                                  max_use_types=3)
        ss_scores = []
        ss_union_candidates = []
        ss_thr_each = []
        for bk, bk_fs in ss_banks:
            s = _compute_llr_from_template_bank(ss_base_clean_sim, bk, bk_fs, sim_fs, force_peak_positive=False)
            ss_scores.append(s)
            try:
                sig_i = estimate_noise_mad(s)
                thr_i = float(ss_thresh_sigma * sig_i)
                ss_thr_each.append(thr_i)
                p_i, _ = find_peaks(s, height=thr_i, distance=max(1, int((ss_min_dist_ms / 1000.0) * sim_fs)))
                if p_i.size > 0:
                    ss_union_candidates.append(np.asarray(p_i, dtype=int))
            except Exception:
                pass
        if len(ss_scores) > 0:
            ss_trace_sim = np.maximum.reduce(ss_scores)
        else:
            ss_trace_sim = _compute_llr_from_template_bank(ss_base_clean_sim, template_ss_bank, template_ss_fs_bank, sim_fs, force_peak_positive=bool(negative_going))
    else:
        ss_trace_sim = _compute_llr_from_template_bank(ss_base_clean_sim, template_ss_bank, template_ss_fs_bank, sim_fs, force_peak_positive=bool(negative_going))

    sigma_ss = estimate_noise_mad(ss_trace_sim)
    ss_dist = max(1, int((ss_min_dist_ms / 1000.0) * sim_fs))
    ss_has_templates = template_ss_bank is not None and len(template_ss_bank) > 0
    if ss_has_templates:
        ss_threshold_used = float(ss_thresh_sigma * sigma_ss)
        ss_candidates_sim, _ = find_peaks(ss_trace_sim, height=ss_threshold_used, distance=ss_dist)
        if bool(parallel_match):
            extra = []
            try:
                if len(ss_union_candidates) > 0:
                    extra.append(np.concatenate(ss_union_candidates))
            except Exception:
                pass
            if len(extra) > 0:
                ss_candidates_sim = np.unique(np.concatenate([ss_candidates_sim] + extra).astype(int))
            try:
                if len(ss_thr_each) > 0:
                    ss_threshold_used = float(min([ss_threshold_used] + ss_thr_each))
            except Exception:
                pass
    else:
        ss_threshold_used = np.nan
        ss_candidates_sim = np.array([], dtype=int)
    if initial_blank_ms is not None and initial_blank_ms > 0:
        init_blank_samples = int((initial_blank_ms / 1000.0) * sim_fs)
        ss_candidates_sim = ss_candidates_sim[ss_candidates_sim >= init_blank_samples]
    if ss_candidates_sim.size > 0:
        ss_peaks = np.unique(np.clip(np.round(ss_candidates_sim * (fs / sim_fs)).astype(int), 0, len(raw_trace) - 1))
    else:
        ss_peaks = np.array([], dtype=int)

    cs_similarity_trace = _resample_to_length(cs_trace_sim, len(raw_trace))
    ss_similarity_trace = _resample_to_length(ss_trace_sim, len(raw_trace))
    ss_base_clean = _resample_to_length(ss_base_clean_sim, len(raw_trace))

    return {
        'detrended': detrended,
        'baseline': baseline,
        'cs_trace': cs_base,
        'cs_peaks': cs_peaks,
        'ss_trace': ss_base_clean,
        'ss_peaks': ss_peaks,
        'sigma_cs': sigma_cs,
        'sigma_ss': sigma_ss,
        'raw_sigma': global_sigma,
        'det_method': f'Template Matching ({template_match_method})',
        'threshold_mode': 'Sigma x MAD',
        'parallel_match': bool(parallel_match),
        'cs_min_fwhm_ms_used': float(cs_min_fwhm_ms),
        'cs_threshold_used': cs_threshold_used,
        'ss_threshold_used': ss_threshold_used,
        'cs_similarity_trace': cs_similarity_trace,
        'ss_similarity_trace': ss_similarity_trace,
    }


def process_cell_simple(raw_trace, fs, negative_going=True,
                        cs_low_cut=0.0, cs_high_cut=150.0, cs_thresh_sigma=6.0, cs_min_dist_ms=25,
                        cs_min_fwhm_ms=4.0,
                        ss_low_cut=0.0, ss_high_cut=0.0, ss_thresh_sigma=2.5,
                        ss_min_dist_ms=2, ss_blank_ms=15, ss_min_width_ms=1, ss_max_width_ms=6,
                        use_preprocessed=False, pre_detrended=None, pre_baseline=None,
                        pre_detrended_cs=None, pre_detrended_ss=None,
                        initial_blank_ms=0.0, cs_order=3, ss_order=3,
                        local_baseline=False, local_baseline_cs_ms=200.0,
                        local_baseline_ss_ms=50.0):
    working = raw_trace * -1 if negative_going else raw_trace
    if use_preprocessed and pre_detrended is not None and pre_baseline is not None:
        detrended = pre_detrended
        baseline = pre_baseline
    else:
        detrended, baseline = detrend_trace(working, fs, window_sec=0.05, percentile=20)

    detr_for_detection = detrended
    try:
        if isinstance(baseline, np.ndarray) and np.allclose(baseline, 0):
            try:
                detr_for_detection = apply_filter(detrended, fs, low=1.0, high=None, order=3)
            except Exception:
                detr_for_detection = detrended
    except Exception:
        detr_for_detection = detrended
    detr_for_detection_cs = np.asarray(pre_detrended_cs, dtype=float) if pre_detrended_cs is not None else detr_for_detection
    detr_for_detection_ss = np.asarray(pre_detrended_ss, dtype=float) if pre_detrended_ss is not None else detr_for_detection
    global_sigma = estimate_noise_mad(detrended)

    cs_trace = apply_filter(detr_for_detection_cs, fs, low=cs_low_cut, high=cs_high_cut, order=cs_order)
    sigma_cs = estimate_noise_mad(cs_trace)
    cs_dist = int((cs_min_dist_ms / 1000.0) * fs)
    if cs_dist < 1:
        cs_dist = 1

    # CS threshold: local or global
    if local_baseline:
        cs_win_samples = max(5, int((local_baseline_cs_ms / 1000.0) * fs))
        cs_local_sigma = estimate_noise_mad_local(cs_trace, cs_win_samples)
        cs_threshold_trace = cs_thresh_sigma * cs_local_sigma
        cs_candidates, _ = find_peaks(cs_trace, distance=cs_dist)
        # filter by local threshold
        cs_candidates = cs_candidates[cs_trace[cs_candidates] >= cs_threshold_trace[cs_candidates]]
    else:
        cs_threshold_trace = None
        cs_candidates, _ = find_peaks(cs_trace, height=cs_thresh_sigma * sigma_cs, distance=cs_dist)

    cs_candidates = _filter_peaks_min_fwhm(cs_trace, cs_candidates, fs, cs_min_fwhm_ms)

    if initial_blank_ms is not None and initial_blank_ms > 0:
        init_blank_samples = int((initial_blank_ms / 1000.0) * fs)
        cs_peaks = cs_candidates[cs_candidates >= init_blank_samples]
    else:
        cs_peaks = cs_candidates

    ss_trace = apply_filter(detr_for_detection_ss, fs, low=ss_low_cut, high=ss_high_cut, order=ss_order)
    ss_trace_clean = ss_trace.copy()
    blank_samples = int((ss_blank_ms / 1000.0) * fs)
    for cs_idx in cs_peaks:
        start = max(0, cs_idx - blank_samples // 2)
        end = min(len(raw_trace), start + blank_samples)
        ss_trace_clean[start:end] = 0

    try:
        nonzero = ss_trace_clean[ss_trace_clean != 0]
        if nonzero.size > 0:
            sigma_ss_filtered = estimate_noise_mad(nonzero)
        else:
            sigma_ss_filtered = estimate_noise_mad(ss_trace_clean)
    except Exception:
        sigma_ss_filtered = estimate_noise_mad(ss_trace_clean)
    ss_dist = int((ss_min_dist_ms / 1000.0) * fs)
    ss_dist = max(1, ss_dist)

    # SS threshold: local or global
    if local_baseline:
        ss_win_samples = max(5, int((local_baseline_ss_ms / 1000.0) * fs))
        ss_local_sigma = estimate_noise_mad_local(ss_trace_clean, ss_win_samples)
        ss_threshold_trace = ss_thresh_sigma * ss_local_sigma
        try:
            ss_candidates, _ = find_peaks(ss_trace_clean, distance=ss_dist)
            ss_candidates = ss_candidates[ss_trace_clean[ss_candidates] >= ss_threshold_trace[ss_candidates]]
        except Exception:
            ss_candidates = np.array([], dtype=int)
    else:
        ss_threshold_trace = None
        try:
            ss_candidates, _ = find_peaks(
                ss_trace_clean,
                height=ss_thresh_sigma * sigma_ss_filtered,
                distance=ss_dist,
            )
        except Exception:
            ss_candidates = np.array([], dtype=int)

    if initial_blank_ms is not None and initial_blank_ms > 0:
        init_blank_samples = int((initial_blank_ms / 1000.0) * fs)
        ss_peaks = ss_candidates[ss_candidates >= init_blank_samples]
    else:
        ss_peaks = ss_candidates

    result = {
        'detrended': detrended,
        'baseline': baseline,
        'cs_trace': cs_trace,
        'cs_peaks': cs_peaks,
        'ss_trace': ss_trace_clean,
        'ss_peaks': ss_peaks,
        'sigma_cs': sigma_cs,
        'sigma_ss': sigma_ss_filtered,
        'raw_sigma': global_sigma,
        'det_method': 'Threshold',
        'threshold_mode': 'Sigma x MAD (local)' if local_baseline else 'Sigma x MAD',
        'cs_min_fwhm_ms_used': float(cs_min_fwhm_ms),
        'cs_threshold_used': float(cs_thresh_sigma * sigma_cs),
        'ss_threshold_used': float(ss_thresh_sigma * sigma_ss_filtered),
        'local_baseline': bool(local_baseline),
        'ss_width_filter_enabled': False,
        'ss_min_dist_ms_used': float(ss_min_dist_ms),
        'ss_blank_ms_used': float(ss_blank_ms),
        'initial_blank_ms_used': float(initial_blank_ms) if initial_blank_ms is not None else 0.0,
    }
    if cs_threshold_trace is not None:
        result['cs_threshold_trace'] = cs_threshold_trace
    if ss_threshold_trace is not None:
        result['ss_threshold_trace'] = ss_threshold_trace
    return result


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
        non_spike_mask = np.ones(n, dtype=bool)
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
            global_sigma = _robust_sigma(trace)
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
