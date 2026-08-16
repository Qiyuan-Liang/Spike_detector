import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


CMOR_WAVELET = 'cmor1.5-1.0'


def default_denoise_config():
    return {
        'enabled': False,
        'f_min_hz': 3.0,
        'f_max_hz': 1000.0,
        'n_freqs': 72,
        'max_clusters': 20,
        'min_clusters': 2,
        'max_pca_components': 20,
        'threshold_sigma': 1.5,
        'attenuation_min': 0.25,
        'soft_threshold': False,
        'moving_cycles': 1.0,
        'max_timepoints_for_clustering': 10000,
        'event_refine_enabled': True,
        'event_refine_window_ms': 20.0,
        'event_refine_sigma': 2.0,
        'event_refine_pc1_z_cutoff': -0.5,
        'event_refine_attenuation': 0.6,
    }


def _robust_mad(x):
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return 1e-9
    med = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - med)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 1e-12:
        s = float(np.nanstd(arr))
    if not np.isfinite(s) or s <= 1e-12:
        return 1e-9
    return s


def _match_length(y, n):
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size == n:
        return arr
    if arr.size == 0:
        return np.zeros(int(n), dtype=float)
    return np.interp(
        np.linspace(0.0, 1.0, int(n)),
        np.linspace(0.0, 1.0, int(max(2, arr.size))),
        arr if arr.size > 1 else np.repeat(arr, 2),
    )


def _icwt_fallback(coeffs, scales):
    c = np.asarray(coeffs)
    s = np.asarray(scales, dtype=float)
    if c.ndim != 2 or s.size == 0:
        return np.array([], dtype=float)
    weights = (1.0 / np.sqrt(np.maximum(s, 1e-9)))[:, None]
    return np.real(np.sum(c * weights, axis=0))


def _inverse_cwt(coeffs, scales, sampling_period):
    try:
        import pywt
        if hasattr(pywt, 'icwt'):
            rec = pywt.icwt(coeffs, scales, CMOR_WAVELET, sampling_period=sampling_period)
            return np.asarray(np.real(rec), dtype=float)
    except Exception:
        pass
    return _icwt_fallback(coeffs, scales)


def _choose_elbow_n(explained, max_components):
    ev = np.asarray(explained, dtype=float).ravel()
    ev = ev[np.isfinite(ev)]
    if ev.size == 0:
        return 1
    n = int(min(max(1, max_components), ev.size))
    if n <= 2:
        return n
    y = np.cumsum(ev[:n])
    if y[-1] > 0:
        y = y / y[-1]
    x = np.linspace(0.0, 1.0, n)
    line = y[0] + (y[-1] - y[0]) * x
    idx = int(np.argmax(y - line))
    return int(max(1, idx + 1))


def _cluster_frequency_profiles(freq_profiles, cfg):
    n_freq, n_time = freq_profiles.shape
    if n_freq <= 1:
        return np.zeros(n_freq, dtype=int), 1, np.nan, {
            'backend': 'degenerate',
            'warning': 'single_frequency_bin',
            'pca_components_used': 0,
            'pca_elbow_components': 0,
            'explained_variance_ratio': np.array([], dtype=float),
        }

    max_tp = int(max(50, cfg.get('max_timepoints_for_clustering', 6000)))
    stride = max(1, int(np.ceil(float(n_time) / float(max_tp))))
    X = np.asarray(freq_profiles[:, ::stride], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    pca_feats = X
    pca_used = 0
    pca_elbow = 0
    explained = np.array([], dtype=float)
    pca_warning = None
    try:
        from sklearn.decomposition import PCA  # type: ignore[reportMissingImports]
        max_comp = int(min(max(1, cfg.get('max_pca_components', 20)), X.shape[0], X.shape[1]))
        pca = PCA(n_components=max_comp, random_state=0)
        all_feats = pca.fit_transform(X)
        explained = np.asarray(pca.explained_variance_ratio_, dtype=float)
        pca_elbow = _choose_elbow_n(explained, max_comp)
        pca_feats = all_feats[:, :pca_elbow]
        pca_used = int(max_comp)
    except Exception:
        pca_feats = X
        pca_warning = 'pca_unavailable_or_failed'

    max_k_cfg = int(max(1, cfg.get('max_clusters', 20)))
    max_k = int(min(max_k_cfg, 20, n_freq))
    min_k = int(min(max(1, cfg.get('min_clusters', 2)), max_k))
    if max_k <= 1:
        return np.zeros(n_freq, dtype=int), 1, np.nan, {
            'backend': 'degenerate',
            'warning': 'max_clusters_le_1',
            'pca_components_used': int(pca_used),
            'pca_elbow_components': int(pca_elbow),
            'pca_warning': pca_warning,
            'explained_variance_ratio': explained,
        }

    best_labels = None
    best_k = min_k
    best_score = -np.inf
    cluster_warning = None

    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore[reportMissingImports]
        from sklearn.metrics import silhouette_score  # type: ignore[reportMissingImports]

        # The reference method builds a Ward hierarchy up to 20 clusters, then
        # chooses an interpretable cluster count with silhouette analysis.
        for k in range(min_k, max_k + 1):
            if k <= 1 or k >= n_freq:
                continue
            labels = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(pca_feats)
            if len(np.unique(labels)) <= 1:
                continue
            score = float(silhouette_score(pca_feats, labels))
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k
    except Exception:
        best_labels = None
        cluster_warning = 'sklearn_clustering_unavailable_or_failed'

    if best_labels is None:
        return np.zeros(n_freq, dtype=int), 1, np.nan, {
            'backend': 'fallback_single_cluster',
            'warning': cluster_warning or 'silhouette_selection_failed',
            'pca_components_used': int(pca_used),
            'pca_elbow_components': int(pca_elbow),
            'pca_warning': pca_warning,
            'explained_variance_ratio': explained,
        }

    return np.asarray(best_labels, dtype=int), int(best_k), float(best_score), {
        'backend': 'ward_silhouette_after_pca_elbow',
        'warning': cluster_warning,
        'pca_components_used': int(pca_used),
        'pca_elbow_components': int(pca_elbow),
        'pca_warning': pca_warning,
        'explained_variance_ratio': explained,
    }


def _event_pc_refine_cluster_trace(trace, coeffs_cluster, fs, cfg):
    y = np.asarray(trace, dtype=float).copy()
    coeffs_out = np.asarray(coeffs_cluster).copy()
    info = {
        'ok': False,
        'reason': None,
        'peak_indices': np.array([], dtype=int),
        'peak_scores': np.array([], dtype=float),
        'peak_scores_z': np.array([], dtype=float),
        'peak_att_mask': np.array([], dtype=bool),
        'cutoff': float(cfg.get('event_refine_pc1_z_cutoff', -0.5)),
        'rejected_count': 0,
    }
    if y.size < 32:
        info['reason'] = 'trace_too_short'
        return coeffs_out, info

    sigma = _robust_mad(y)
    thr = float(cfg.get('event_refine_sigma', 2.0)) * sigma
    min_dist = max(1, int(round((2.0 / 1000.0) * float(fs))))
    peaks, _ = find_peaks(np.abs(y), height=thr, distance=min_dist)
    info['peak_indices'] = np.asarray(peaks, dtype=int)
    if peaks.size < 5:
        info['reason'] = 'not_enough_peaks'
        return coeffs_out, info

    half_w = max(2, int(round((float(cfg.get('event_refine_window_ms', 20.0)) / 1000.0) * float(fs) / 2.0)))
    waves = []
    valid_peaks = []
    for p in peaks:
        s = int(p - half_w)
        e = int(p + half_w)
        if s < 0 or e >= y.size:
            continue
        w = y[s:e].copy()
        w -= np.mean(w)
        waves.append(w)
        valid_peaks.append(int(p))

    info['peak_indices'] = np.asarray(valid_peaks, dtype=int)
    if len(waves) < 5:
        info['reason'] = 'not_enough_valid_waveforms'
        return coeffs_out, info

    W = np.vstack(waves)
    try:
        _, _, vt = np.linalg.svd(W, full_matrices=False)
        pc1 = vt[0]
        scores = W @ pc1
        if np.nanmean(scores[np.argsort(np.abs(scores))[-max(1, min(5, scores.size)):]]) < 0:
            scores = -scores
    except Exception:
        info['reason'] = 'svd_failed'
        return coeffs_out, info

    z = (scores - np.nanmedian(scores)) / (_robust_mad(scores) + 1e-9)
    cutoff = float(cfg.get('event_refine_pc1_z_cutoff', -0.5))
    atten = float(np.clip(cfg.get('event_refine_attenuation', 0.6), 0.0, 1.0))
    reject = np.asarray(z < cutoff, dtype=bool)

    for p, bad in zip(valid_peaks, reject):
        if not bad:
            continue
        s = max(0, int(p - half_w))
        e = min(y.size, int(p + half_w))
        coeffs_out[:, s:e] *= atten

    info.update({
        'ok': True,
        'reason': None,
        'peak_scores': np.asarray(scores, dtype=float),
        'peak_scores_z': np.asarray(z, dtype=float),
        'peak_att_mask': reject,
        'cutoff': float(cutoff),
        'rejected_count': int(np.sum(reject)),
    })
    return coeffs_out, info


def _calibrate_reconstruction(reference, reconstructed):
    ref = np.asarray(reference, dtype=float).ravel()
    rec = np.asarray(reconstructed, dtype=float).ravel()
    if ref.size != rec.size or ref.size == 0:
        return 1.0
    a = ref - np.nanmean(ref)
    b = rec - np.nanmean(rec)
    denom = float(np.nansum(b * b))
    if not np.isfinite(denom) or denom <= 1e-12:
        return 1.0
    gain = float(np.nansum(a * b) / denom)
    if not np.isfinite(gain) or gain <= 0:
        return 1.0
    return float(np.clip(gain, 0.05, 50.0))


def adaptive_wavelet_denoise(trace, fs, cfg=None):
    config = default_denoise_config()
    if cfg is not None:
        config.update(cfg)

    x = np.asarray(trace, dtype=float).ravel()
    out_meta = {
        'ok': False,
        'error': None,
        'wavelet': CMOR_WAVELET,
        'n_clusters': 1,
        'silhouette': np.nan,
        'cluster_backend': 'unknown',
        'cluster_warning': None,
        'cluster_pca_components_used': 0,
        'cluster_pca_elbow_components': 0,
        'cluster_pca_warning': None,
        'pca_explained_variance_ratio': np.array([], dtype=float),
        'freqs_hz': np.array([], dtype=float),
        'cluster_labels': np.array([], dtype=int),
        'coeff_abs_norm': np.array([], dtype=float),
        'attenuation_mask': np.ones_like(x),
        'cluster_stats': [],
        'event_refined_count': 0,
        'event_refine': {
            'ok': False,
            'reason': None,
            'peak_indices': np.array([], dtype=int),
            'peak_scores_z': np.array([], dtype=float),
            'peak_att_mask': np.array([], dtype=bool),
            'cutoff': float(config.get('event_refine_pc1_z_cutoff', -0.5)),
        },
    }

    if x.size < 32 or not np.isfinite(fs) or fs <= 0:
        out_meta['error'] = 'trace_too_short_or_invalid_fs'
        return x.copy(), out_meta

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        import pywt
    except Exception as e:
        out_meta['error'] = f'pywt_not_available: {e}'
        return x.copy(), out_meta

    try:
        f_min = float(max(0.5, config.get('f_min_hz', 3.0)))
        f_max = float(config.get('f_max_hz', 350.0))
        nyq = 0.5 * float(fs)
        f_max = min(f_max, nyq * 0.95)
        if f_max <= f_min:
            f_max = min(nyq * 0.95, f_min + 1.0)
        n_freqs = int(max(8, config.get('n_freqs', 40)))
        freqs = np.geomspace(f_min, f_max, n_freqs)

        wavelet = pywt.ContinuousWavelet(CMOR_WAVELET)
        central = float(pywt.central_frequency(wavelet))
        scales = central * float(fs) / np.maximum(freqs, 1e-9)
        coeffs, freqs_hz = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / float(fs))
        coeffs = np.asarray(coeffs, dtype=np.complex128)
        mag = np.abs(coeffs)

        med = np.nanmedian(mag, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(mag - med), axis=1, keepdims=True)
        scale = 1.4826 * np.maximum(mad, 1e-9)
        norm_mag = (mag - med) / scale
        norm_mag = np.nan_to_num(norm_mag, nan=0.0, posinf=0.0, neginf=0.0)

        labels, n_clusters, sil, cluster_diag = _cluster_frequency_profiles(norm_mag, config)

        coeffs_clean = np.zeros_like(coeffs)
        attenuation_mask = np.ones(x.shape, dtype=float)
        cluster_stats = []
        atten_min = float(np.clip(config.get('attenuation_min', 0.4), 0.0, 1.0))
        thr_sigma = float(max(0.1, config.get('threshold_sigma', 2.0)))
        refined_count = 0
        all_peak_idx = []
        all_peak_z = []
        all_peak_bad = []

        for cid in np.unique(labels):
            freq_idx = np.where(labels == cid)[0]
            if freq_idx.size == 0:
                continue

            coeffs_cluster = np.zeros_like(coeffs)
            cluster_coeffs = coeffs[freq_idx, :]
            cluster_env = np.mean(norm_mag[freq_idx, :], axis=0)
            f_cluster_max = float(np.max(freqs_hz[freq_idx]))
            win_sec = float(config.get('moving_cycles', 2.0)) / max(f_cluster_max, 0.5)
            win_n = int(max(5, round(win_sec * float(fs))))
            if win_n % 2 == 0:
                win_n += 1

            mov_mean = uniform_filter1d(cluster_env, size=win_n, mode='nearest')
            mov_mean_sq = uniform_filter1d(cluster_env * cluster_env, size=win_n, mode='nearest')
            mov_std = np.sqrt(np.maximum(mov_mean_sq - mov_mean * mov_mean, 1e-12))
            threshold = mov_mean + thr_sigma * mov_std
            keep = cluster_env >= threshold
            if bool(config.get('soft_threshold', False)):
                ratio = np.clip(cluster_env / np.maximum(threshold, 1e-9), 0.0, 1.0)
                mask = np.where(keep, 1.0, atten_min + (1.0 - atten_min) * ratio)
            else:
                mask = np.where(keep, 1.0, atten_min)

            coeffs_cluster[freq_idx, :] = cluster_coeffs * mask[None, :]
            cluster_trace_before_pc = _match_length(_inverse_cwt(coeffs_cluster, scales, sampling_period=1.0 / float(fs)), x.size)
            event_info = {
                'ok': False,
                'reason': 'disabled',
                'peak_indices': np.array([], dtype=int),
                'peak_scores': np.array([], dtype=float),
                'peak_scores_z': np.array([], dtype=float),
                'peak_att_mask': np.array([], dtype=bool),
                'cutoff': float(config.get('event_refine_pc1_z_cutoff', -0.5)),
                'rejected_count': 0,
            }
            if bool(config.get('event_refine_enabled', True)):
                coeffs_cluster, event_info = _event_pc_refine_cluster_trace(cluster_trace_before_pc, coeffs_cluster, fs, config)
                refined_count += int(event_info.get('rejected_count', 0))
                if np.asarray(event_info.get('peak_indices', [])).size > 0:
                    all_peak_idx.append(np.asarray(event_info.get('peak_indices', []), dtype=int))
                    all_peak_z.append(np.asarray(event_info.get('peak_scores_z', []), dtype=float))
                    all_peak_bad.append(np.asarray(event_info.get('peak_att_mask', []), dtype=bool))

            cluster_trace_after_pc = _match_length(_inverse_cwt(coeffs_cluster, scales, sampling_period=1.0 / float(fs)), x.size)
            coeffs_clean += coeffs_cluster
            attenuation_mask = np.minimum(attenuation_mask, mask)

            cluster_stats.append({
                'cluster_id': int(cid),
                'freq_min_hz': float(np.min(freqs_hz[freq_idx])),
                'freq_max_hz': float(np.max(freqs_hz[freq_idx])),
                'n_freqs': int(freq_idx.size),
                'envelope': cluster_env.astype(float),
                'threshold': threshold.astype(float),
                'mask': mask.astype(float),
                'attenuated_fraction': float(np.mean(mask < 0.99)),
                'trace': cluster_trace_after_pc.astype(float),
                'trace_before_event_pc': cluster_trace_before_pc.astype(float),
                'event_refine': event_info,
            })

        # Integrate per-cluster cleaned reconstructions as an in-band correction,
        # preserving out-of-band/raw baseline components without global rescaling.
        x_band = _match_length(_inverse_cwt(coeffs, scales, sampling_period=1.0 / float(fs)), x.size)
        y_band = _match_length(_inverse_cwt(coeffs_clean, scales, sampling_period=1.0 / float(fs)), x.size)
        reconstruction_gain = _calibrate_reconstruction(x, x_band)
        x_band = reconstruction_gain * x_band
        y_band = reconstruction_gain * y_band
        y = x + (y_band - x_band)
        y = np.asarray(y, dtype=float)

        if all_peak_idx:
            peak_idx = np.concatenate(all_peak_idx)
            peak_z = np.concatenate(all_peak_z) if all_peak_z else np.array([], dtype=float)
            peak_bad = np.concatenate(all_peak_bad) if all_peak_bad else np.array([], dtype=bool)
        else:
            peak_idx = np.array([], dtype=int)
            peak_z = np.array([], dtype=float)
            peak_bad = np.array([], dtype=bool)

        out_meta.update({
            'ok': True,
            'n_clusters': int(n_clusters),
            'silhouette': float(sil) if np.isfinite(sil) else np.nan,
            'cluster_backend': cluster_diag.get('backend', 'unknown'),
            'cluster_warning': cluster_diag.get('warning', None),
            'cluster_pca_components_used': int(cluster_diag.get('pca_components_used', 0)),
            'cluster_pca_elbow_components': int(cluster_diag.get('pca_elbow_components', 0)),
            'cluster_pca_warning': cluster_diag.get('pca_warning', None),
            'pca_explained_variance_ratio': np.asarray(cluster_diag.get('explained_variance_ratio', np.array([])), dtype=float),
            'freqs_hz': np.asarray(freqs_hz, dtype=float),
            'cluster_labels': np.asarray(labels, dtype=int),
            'coeff_abs_norm': np.asarray(norm_mag, dtype=float),
            'attenuation_mask': np.asarray(attenuation_mask, dtype=float),
            'cluster_stats': cluster_stats,
            'event_refined_count': int(refined_count),
            'event_refine': {
                'ok': bool(peak_idx.size > 0),
                'reason': None if peak_idx.size > 0 else 'no_cluster_events',
                'peak_indices': np.asarray(peak_idx, dtype=int),
                'peak_scores_z': np.asarray(peak_z, dtype=float),
                'peak_att_mask': np.asarray(peak_bad, dtype=bool),
                'cutoff': float(config.get('event_refine_pc1_z_cutoff', -0.5)),
            },
            'input_sigma': float(_robust_mad(x)),
            'output_sigma': float(_robust_mad(y)),
            'reconstruction_gain': float(reconstruction_gain),
        })
        return y, out_meta
    except Exception as e:
        out_meta['error'] = str(e)
        return x.copy(), out_meta
