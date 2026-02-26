import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, fftconvolve
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
        return None if val is None else float(val) / float(nyq)

    low_n = _norm(lowcut)
    high_n = _norm(highcut)
    eps = 1e-6
    if low_n is not None and low_n <= 0.0:
        low_n = eps
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


def process_cell_template_matching(raw_trace, fs,
                                   template_cs_bank=None, template_ss_bank=None,
                                   template_cs_fs_bank=None, template_ss_fs_bank=None,
                                   negative_going=True,
                                   cs_high_cut=150.0, cs_thresh_sigma=6.0, cs_min_dist_ms=25,
                                   ss_low_cut=50.0, ss_high_cut=700.0, ss_thresh_sigma=2.0,
                                   ss_min_dist_ms=2, ss_blank_ms=8,
                                   template_match_method='LLR Probability Vector',
                                   parallel_match=False,
                                   use_preprocessed=False, pre_detrended=None, pre_baseline=None,
                                   initial_blank_ms=0.0, cs_order=3, ss_order=3):
    working = raw_trace * -1 if negative_going else raw_trace
    if use_preprocessed and pre_detrended is not None and pre_baseline is not None:
        detrended = pre_detrended
        baseline = pre_baseline
    else:
        detrended, baseline = detrend_trace(working, fs, window_sec=0.05, percentile=20)

    detr_for_detection = detrended

    global_sigma = estimate_noise_mad(detrended)

    sim_fs = float(max(float(fs), TEMPLATE_TARGET_FS))
    cs_base = apply_filter(detr_for_detection, fs, low=None, high=cs_high_cut, order=cs_order)
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
    if initial_blank_ms is not None and initial_blank_ms > 0:
        init_blank_samples = int((initial_blank_ms / 1000.0) * sim_fs)
        cs_candidates_sim = cs_candidates_sim[cs_candidates_sim >= init_blank_samples]
    if cs_candidates_sim.size > 0:
        cs_peaks = np.unique(np.clip(np.round(cs_candidates_sim * (fs / sim_fs)).astype(int), 0, len(raw_trace) - 1))
    else:
        cs_peaks = np.array([], dtype=int)

    ss_base = apply_filter(detr_for_detection, fs, low=ss_low_cut, high=ss_high_cut, order=ss_order)
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
        'cs_threshold_used': cs_threshold_used,
        'ss_threshold_used': ss_threshold_used,
        'cs_similarity_trace': cs_similarity_trace,
        'ss_similarity_trace': ss_similarity_trace,
    }


def process_cell_simple(raw_trace, fs, negative_going=True,
                        cs_high_cut=150.0, cs_thresh_sigma=6.0, cs_min_dist_ms=25,
                        ss_low_cut=50.0, ss_high_cut=700.0, ss_thresh_sigma=2.0,
                        ss_min_dist_ms=2, ss_blank_ms=8, ss_min_width_ms=1, ss_max_width_ms=6,
                        use_preprocessed=False, pre_detrended=None, pre_baseline=None,
                        initial_blank_ms=0.0, cs_order=3, ss_order=3):
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
    global_sigma = estimate_noise_mad(detrended)

    cs_trace = apply_filter(detr_for_detection, fs, low=None, high=cs_high_cut, order=cs_order)
    sigma_cs = estimate_noise_mad(cs_trace)
    cs_dist = int((cs_min_dist_ms / 1000.0) * fs)
    if cs_dist < 1:
        cs_dist = 1
    cs_candidates, _ = find_peaks(cs_trace, height=cs_thresh_sigma * sigma_cs, distance=cs_dist)
    if initial_blank_ms is not None and initial_blank_ms > 0:
        init_blank_samples = int((initial_blank_ms / 1000.0) * fs)
        cs_peaks = cs_candidates[cs_candidates >= init_blank_samples]
    else:
        cs_peaks = cs_candidates

    ss_trace = apply_filter(detr_for_detection, fs, low=ss_low_cut, high=ss_high_cut, order=ss_order)
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
    w_min = (ss_min_width_ms / 1000.0) * fs
    w_max = (ss_max_width_ms / 1000.0) * fs
    try:
        ss_candidates, _ = find_peaks(ss_trace_clean, height=ss_thresh_sigma * sigma_ss_filtered, distance=ss_dist, width=(w_min, w_max), rel_height=0.5)
    except Exception:
        ss_candidates = np.array([], dtype=int)
    if initial_blank_ms is not None and initial_blank_ms > 0:
        init_blank_samples = int((initial_blank_ms / 1000.0) * fs)
        ss_peaks = ss_candidates[ss_candidates >= init_blank_samples]
    else:
        ss_peaks = ss_candidates

    return {
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
        'threshold_mode': 'Sigma x MAD',
        'cs_threshold_used': float(cs_thresh_sigma * sigma_cs),
        'ss_threshold_used': float(ss_thresh_sigma * sigma_ss_filtered),
    }


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
    peak_idx = int(np.argmax(wave))
    baseline = np.mean(wave[:5]) if len(wave) > 5 else 0.0
    amp = wave[peak_idx] - baseline
    half_height = baseline + (amp / 2.0)
    crossings = np.where(np.diff(np.sign(wave - half_height)))[0]
    fwhm = np.nan
    if len(crossings) >= 2:
        left = crossings[crossings < peak_idx]
        right = crossings[crossings >= peak_idx]
        if len(left) > 0 and len(right) > 0:
            fwhm = time_axis_ms[right[0]] - time_axis_ms[left[-1]]
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


def compute_event_snrs(res, spike_type='CS', fs=1000.0, window_ms=100, max_per_cell=None):
    snr_list = []
    if res is None:
        return snr_list
    try:
        half_win = int((window_ms / 2.0) * fs / 1000.0)
        if spike_type == 'CS':
            peaks = np.array(res.get('cs_peaks', []), dtype=int)
            trace = res.get('cs_trace', None)
            sigma = float(res.get('sigma_cs', 1.0))
        else:
            peaks = np.array(res.get('ss_peaks', []), dtype=int)
            trace = res.get('ss_trace', None)
            if trace is None:
                trace = res.get('detrended', None)
            sigma = float(res.get('raw_sigma', 1.0))

        if trace is None or peaks.size == 0:
            return snr_list

        chosen = _select_event_bank(peaks, max_per_cell=max_per_cell)
        for p in chosen:
            s = int(p - half_win)
            e = int(p + half_win)
            if s < 0 or e >= len(trace):
                continue
            wave = trace[s:e]
            if len(wave) != (2 * half_win):
                continue
            baseline = np.mean(wave[:5]) if len(wave) > 5 else 0.0
            amp = float(np.max(np.abs(wave - baseline)))
            if sigma == 0:
                continue
            snr = amp / float(sigma)
            snr_list.append(snr)
    except Exception:
        pass
    return snr_list
