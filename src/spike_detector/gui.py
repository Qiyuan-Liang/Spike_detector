"""
GUI for preprocessing and inspecting voltage imaging sessions.
Cross-platform (Mac/Windows) PyQt GUI that loads session folders containing
either analyzed NPZ files (created by your notebook) or raw Excel traces.

Features:
- Select master folder containing session subfolders
- List and load sessions (npz or xlsx)
- Visualize raw traces, baseline, CS/SS streams, and detected spikes
- Adjustable detection parameters (thresholds, filter bands, smoothing)
- Compute and show simple spike statistics (rates, SNR, FWHM summary)
- Export current figure as vector image (SVG or PDF)

Dependencies:
- PyQt5 (or PySide6 as fallback)
- numpy, scipy, pandas, matplotlib

Install with pip if needed:
pip install pyqt5 numpy scipy pandas matplotlib

Run:
python gui_preprocess.py

"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, fftconvolve
from scipy.ndimage import percentile_filter
from scipy.stats import median_abs_deviation
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# PyQt6-only target for packaging/release
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QColorDialog
from PyQt6.QtGui import QColor

from .core.detection import (
    TEMPLATE_TARGET_FS,
    _build_parallel_template_banks,
    _filter_peaks_by_reference,
    _resample_template_to_fs,
    apply_frame_processing,
    compute_event_snrs,
    get_interpolated_wave,
    get_wave_stats,
    process_cell_simple,
    process_cell_template_matching,
    _select_event_bank,
)
from .io.session import load_session_path
from .utils.export import save_figure_with_dialog
from .utils.stats import mean_std_count

# Orientation compatibility helper (works across PyQt6 / PySide6)
try:
    # PyQt6/PySide6: prefer Qt.Orientation.Horizontal
    ORIENT_HORIZONTAL = Qt.Orientation.Horizontal
except Exception:
    try:
        ORIENT_HORIZONTAL = QtCore.Qt.Orientation.Horizontal
    except Exception:
        try:
            ORIENT_HORIZONTAL = QtCore.Qt.Horizontal
        except Exception:
            # fallback integer (least preferred)
            ORIENT_HORIZONTAL = 1

# Alignment compatibility (AlignTop)
try:
    ALIGN_TOP = Qt.AlignTop
except Exception:
    try:
        ALIGN_TOP = QtCore.Qt.AlignTop
    except Exception:
        try:
            ALIGN_TOP = QtCore.Qt.AlignmentFlag.AlignTop
        except Exception:
            # fallback integer
            ALIGN_TOP = 1


def _get_screen_scale():
    """Return a scale factor based on the primary screen devicePixelRatio or logical DPI.
    Falls back to 1.0 when information is unavailable. This helps adapt sizes across
    different display resolutions and Windows scaling settings.
    """
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return 1.0
        screen = app.primaryScreen()
        if screen is None:
            return 1.0

        try:
            logical_dpi = float(screen.logicalDotsPerInch())
        except Exception:
            logical_dpi = 96.0
        try:
            dpr = float(screen.devicePixelRatio())
        except Exception:
            dpr = 1.0

        if sys.platform == 'darwin':
            # Keep macOS behavior unchanged (already tuned visually).
            return 1.0

        # Windows/Linux: account for OS display scaling from logical DPI.
        dpi_scale = logical_dpi / 96.0 if logical_dpi > 0 else 1.0
        scale = max(dpi_scale, dpr, 1.0)
        return float(max(0.9, min(2.0, scale)))
    except Exception:
        return 1.0

# QSizePolicy compatibility: resolve Expanding/Fixed across bindings
try:
    SIZEPOLICY_EXPANDING = QtWidgets.QSizePolicy.Expanding
    SIZEPOLICY_FIXED = QtWidgets.QSizePolicy.Fixed
except Exception:
    try:
        SIZEPOLICY_EXPANDING = QtWidgets.QSizePolicy.Policy.Expanding
        SIZEPOLICY_FIXED = QtWidgets.QSizePolicy.Policy.Fixed
    except Exception:
        try:
            SIZEPOLICY_EXPANDING = QtWidgets.QSizePolicy.Policy.Expanding.value
            SIZEPOLICY_FIXED = QtWidgets.QSizePolicy.Policy.Fixed.value
        except Exception:
            SIZEPOLICY_EXPANDING = 1
            SIZEPOLICY_FIXED = 0


def _scaled_size(px_w, px_h):
    """Scale a (width,height) pair by the detected screen scale factor and return ints."""
    try:
        s = _get_screen_scale()
        return max(1, int(round(float(px_w) * s))), max(1, int(round(float(px_h) * s)))
    except Exception:
        return px_w, px_h


def _make_figure(width, height, dpi=100):
    """Create a matplotlib Figure with reduced font sizes (approx. 4x smaller)
    to avoid overly large text on high-DPI displays. Uses a temporary rc_context
    so global rcParams are not permanently changed.
    """
    # Create a standard Figure and rely on native Qt6 font/DPI handling.
    return Figure(figsize=(width, height), dpi=dpi)


def _get_linewidth(base=1.0):
    """Return an adaptive line width scaled with screen DPI.
    Ensures lines are thicker on high-DPI displays and not too thin on low-DPI.
    """
    try:
        s = _get_screen_scale()
        lw = float(base) * (1.0 + 0.20 * max(0.0, s - 1.0))
        return float(max(0.5, lw))
    except Exception:
        return float(1.0)


def _scale_font(base=10):
    """Scale font size mildly on high-DPI displays while keeping macOS unchanged."""
    try:
        s = _get_screen_scale()
        return int(round(float(base) * (1.0 + 0.15 * max(0.0, s - 1.0))))
    except Exception:
        return base


def _set_frame_hline(frame):
    """Set a QFrame to horizontal line shape in a binding-compatible way."""
    try:
        frame.setFrameShape(QtWidgets.QFrame.HLine)
        return
    except Exception:
        pass
    try:
        frame.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        return
    except Exception:
        pass
    try:
        # PySide/PyQt variants where enum has .value
        frame.setFrameShape(QtWidgets.QFrame.HLine.value)
        return
    except Exception:
        pass
    try:
        # fallback: set a thin line by adjusting frame style
        frame.setLineWidth(1)
    except Exception:
        pass


# -------------------- Core processing helpers (adapted from notebook) --------------------

def estimate_noise_mad(trace):
    # Robust MAD-based noise estimator with fallbacks.
    try:
        mad = median_abs_deviation(trace, scale='normal')
        if np.isfinite(mad) and mad > 0:
            return mad
    except Exception:
        mad = None
    # fallback to standard deviation scaled to MAD equivalent if possible
    try:
        std = float(np.nanstd(trace))
        if np.isfinite(std) and std > 0:
            # For a normal distribution, MAD ~ std * 0.6745
            return std * 0.6745
    except Exception:
        pass
    # final tiny fallback to avoid zero thresholds
    return 1e-6


def detrend_trace(trace, fs, window_sec=0.05, percentile=20):
    window_samples = int(window_sec * fs)
    if window_samples < 5:
        window_samples = 5
    baseline = percentile_filter(trace, percentile, size=window_samples)
    return trace - baseline, baseline


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    # no filtering requested
    if lowcut is None and highcut is None:
        return None
    # normalize and guard frequencies to (0,1)
    def _norm(val):
        return None if val is None else float(val) / float(nyq)

    low_n = _norm(lowcut)
    high_n = _norm(highcut)
    # clamp to valid open interval (0,1)
    eps = 1e-6
    if low_n is not None and low_n <= 0.0:
        low_n = eps
    if high_n is not None and high_n >= 1.0:
        high_n = 1.0 - 1e-3
    # If both normalized present but inverted or too narrow, skip filtering
    if low_n is not None and high_n is not None and low_n >= high_n:
        return None
    try:
        if low_n is None:
            return butter(order, high_n, btype='low', output='sos')
        if high_n is None:
            return butter(order, low_n, btype='high', output='sos')
        return butter(order, [low_n, high_n], btype='band', output='sos')
    except ValueError:
        # fallback: do not filter if parameters invalid
        return None


def apply_filter(trace, fs, low=None, high=None, order=3):
    sos = butter_bandpass(low, high, fs, order=order)
    if sos is None:
        return trace
    return sosfiltfilt(sos, trace)


def apply_frame_processing(trace, frames=0, mode='Rolling average'):
    """Apply frame-based processing while preserving original signal length."""
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


TEMPLATE_TARGET_FS = 5000.0


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
            mask = (labels == ci)
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
        mask = (labels == cid)
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
    # allow upstream code to provide averaged + baseline-corrected (detrended) signal
    working = raw_trace * -1 if negative_going else raw_trace
    if use_preprocessed and pre_detrended is not None and pre_baseline is not None:
        detrended = pre_detrended
        baseline = pre_baseline
    else:
        detrended, baseline = detrend_trace(working, fs, window_sec=0.05, percentile=20)
    # If baseline is effectively disabled (all zeros), remove very-low-frequency
    # components for detection only so DC/drift doesn't dominate filtered traces.
    detr_for_detection = detrended
    try:
        if isinstance(baseline, np.ndarray) and np.allclose(baseline, 0):
            # apply a mild high-pass at 1 Hz to remove DC/drift for detection
            try:
                detr_for_detection = apply_filter(detrended, fs, low=1.0, high=None, order=3)
            except Exception:
                detr_for_detection = detrended
    except Exception:
        detr_for_detection = detrended
    global_sigma = estimate_noise_mad(detrended)

    # CS stream (lowpass)
    # Use detr_for_detection (possibly high-passed) for computing detection traces
    cs_trace = apply_filter(detr_for_detection, fs, low=None, high=cs_high_cut, order=cs_order)
    sigma_cs = estimate_noise_mad(cs_trace)
    cs_dist = int((cs_min_dist_ms / 1000.0) * fs)
    if cs_dist < 1:
        cs_dist = 1
    cs_candidates, _ = find_peaks(cs_trace, height=cs_thresh_sigma * sigma_cs, distance=cs_dist)
    # remove peaks that fall within the initial blank period (if any)
    if initial_blank_ms is not None and initial_blank_ms > 0:
        init_blank_samples = int((initial_blank_ms / 1000.0) * fs)
        cs_peaks = cs_candidates[cs_candidates >= init_blank_samples]
    else:
        cs_peaks = cs_candidates

    # SS stream (bandpass)
    ss_trace = apply_filter(detr_for_detection, fs, low=ss_low_cut, high=ss_high_cut, order=ss_order)
    ss_trace_clean = ss_trace.copy()
    blank_samples = int((ss_blank_ms / 1000.0) * fs)
    for cs_idx in cs_peaks:
        # blank centered on CS
        start = max(0, cs_idx - blank_samples // 2)
        end = min(len(raw_trace), start + blank_samples)
        ss_trace_clean[start:end] = 0
    # Robust sigma for SS: prefer non-zero samples after blanking, else use estimator on full signal
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
    # No local SNR filtering: accept SS candidates directly, but exclude initial blank
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
        # fallback to simple interpolation
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
    """Return a deterministic subset of peak indices for stats/SNR/FWHM.

    If number of peaks exceeds `max_per_cell`, select evenly spaced events
    across the full peak list to keep coverage while ensuring repeatability.
    """
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
    """Compute SNRs for events in a single result dict `res`.

    Returns a list of SNRs computed as max(|wave - baseline|)/sigma where baseline
    is the mean of the first 5 samples of the window. `spike_type` is 'CS' or 'SS'.
    This helper ensures consistent SNR calculations across summaries and viewers.
    """
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
            s = int(p - half_win); e = int(p + half_win)
            if s < 0 or e >= len(trace):
                continue
            wave = trace[s:e]
            if len(wave) != (2 * half_win):
                continue
            # local baseline and amplitude
            baseline = np.mean(wave[:5]) if len(wave) > 5 else 0.0
            amp = float(np.max(np.abs(wave - baseline)))
            if sigma == 0:
                continue
            snr = amp / float(sigma)
            snr_list.append(snr)
    except Exception:
        pass
    return snr_list


# Rebind processing helpers from extracted core module (keeps GUI code unchanged).
from .core.detection import (
    TEMPLATE_TARGET_FS,
    _build_parallel_template_banks,
    _filter_peaks_by_reference,
    _resample_template_to_fs,
    apply_frame_processing,
    compute_event_snrs,
    get_interpolated_wave,
    get_wave_stats,
    process_cell_simple,
    process_cell_template_matching,
    _select_event_bank,
)


# -------------------- GUI Application --------------------
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        # Create figure with reduced fonts for high-DPI safety
        fig = _make_figure(width, height, dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        try:
            self.setSizePolicy(QtWidgets.QSizePolicy(SIZEPOLICY_EXPANDING, SIZEPOLICY_EXPANDING))
            self.updateGeometry()
        except Exception:
            pass
        fig.tight_layout()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spike detector V3.3')
        self.resize(*_scaled_size(1150, 700))

        # State
        self.master_folder = ''
        self.sessions = []  # list of session paths
        self.session_names = []
        self.loaded_sessions = {}  # session_name: data dict
        self.selected_session = None
        self.selected_cell = 0

        # Baseline correction params
        self.baseline_params = {
            'method': 'Median',  # 'Percentile', 'Median', 'Savitzky-Golay'
            'window_ms': 40.0,
            'percentile': 20.0,
            'sgolay_polyorder': 3,
        }

        # Color scheme
        self.colors = {
            'raw': '#333333',
            'baseline': '#FFC20A',
        }

        # Detection params (defaults)
        self.params = {
            'NEGATIVE_GOING': True,
            'DETECTION_METHOD': 'Threshold',
            'CS_HIGH_CUT_HZ': 150.0,
            'CS_THRESHOLD_SIGMA': 6.0,
            'CS_MIN_DIST_MS': 25.0,
            'SS_LOW_CUT_HZ': 50.0,
            'SS_HIGH_CUT_HZ': 700.0,
            'SS_THRESHOLD_SIGMA': 2.0,
            'SS_MIN_DIST_MS': 2.0,
            'SS_BLANK_MS': 8.0,
            'INITIAL_BLANK_MS': 150.0,
            'CS_FILTER_ORDER': 4,
            'SS_FILTER_ORDER': 4,
            'FRAME_PROCESSING_MODE': 'Rolling average',
            'TEMPLATE_CS_WINDOW_MS': 30.0,
            'TEMPLATE_SS_WINDOW_MS': 8.0,
            'TEMPLATE_MATCH_METHOD': 'LLR Probability Vector',
            'TEMPLATE_PARALLEL': False,
            'TEMPLATE_CS_SIGMA': 6.0,
            'TEMPLATE_SS_SIGMA': 4.0,
            }

        # Template banks (each can be loaded from one file or a folder)
        self.template_store = {
            'cs_templates': [],
            'ss_templates': [],
            'fs_cs': [],
            'fs_ss': [],
            'cs_sources': [],
            'ss_sources': [],
        }

        # Interactive selection state on main plot
        self.template_selection_active = False
        self.template_selection_type = 'CS'
        self.template_selection_window_ms = 30.0
        self.template_selected_segments = {'CS': [], 'SS': []}
        self.template_selected_intervals_ms = {'CS': [], 'SS': []}
        self.btn_select_templates = None
        self.btn_abort_templates = None

        # Currently loaded data (for single-session views)
        self.data = None

        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # --- Upper block: Controls ---
        upper_layout = QtWidgets.QHBoxLayout()

        # Folder/session controls (compact folded list using QComboBox)
        vbox_left = QtWidgets.QVBoxLayout()
        vbox_left.setSpacing(2)
        btn_folder = QtWidgets.QPushButton('Select Folder')
        btn_folder.clicked.connect(self.select_folder)
        # place folder button on its own line
        h_folder = QtWidgets.QHBoxLayout()
        h_folder.addWidget(btn_folder)
        vbox_left.addLayout(h_folder)
        # session selector on its own line (between folder and cell selection)
        h_session = QtWidgets.QHBoxLayout()
        h_session.addWidget(QtWidgets.QLabel('Session:'))
        self.list_sessions = QtWidgets.QComboBox()
        # connect later; use currentIndexChanged without argument handling
        self.list_sessions.currentIndexChanged.connect(self.on_session_select)
        h_session.addWidget(self.list_sessions)
        vbox_left.addLayout(h_session)
        # cell selector inline
        h_cell = QtWidgets.QHBoxLayout()
        h_cell.addWidget(QtWidgets.QLabel('Cell:'))
        self.combo_cell = QtWidgets.QComboBox()
        self.combo_cell.currentIndexChanged.connect(self.on_cell_change)
        h_cell.addWidget(self.combo_cell)
        vbox_left.addLayout(h_cell)
        # keep compact, add to upper layout later
        # upper_layout.addLayout(vbox_left)

        # Baseline correction controls
        vbox_mid = QtWidgets.QVBoxLayout()
        vbox_mid.setSpacing(2)

        self.combo_baseline_method = QtWidgets.QComboBox()
        self.combo_baseline_method.addItems(['Disable', 'Percentile', 'Median', 'Savitzky-Golay'])
        self.combo_baseline_method.setCurrentText(self.baseline_params['method'])
        self.combo_baseline_method.currentTextChanged.connect(self.on_baseline_param_change)
        h_method = QtWidgets.QHBoxLayout()
        h_method.addWidget(QtWidgets.QLabel('Baseline correction method:'))
        h_method.addWidget(self.combo_baseline_method)
        vbox_mid.addLayout(h_method)
        self.spin_baseline_window = QtWidgets.QDoubleSpinBox(); self.spin_baseline_window.setRange(1.0, 1000.0); self.spin_baseline_window.setValue(self.baseline_params['window_ms'])
        self.spin_baseline_window.setSuffix(' ms')
        self.spin_baseline_window.valueChanged.connect(self.on_baseline_param_change)
        h_window = QtWidgets.QHBoxLayout()
        h_window.addWidget(QtWidgets.QLabel('Baseline correction window:'))
        h_window.addWidget(self.spin_baseline_window)
        vbox_mid.addLayout(h_window)
        self.spin_baseline_percentile = QtWidgets.QDoubleSpinBox(); self.spin_baseline_percentile.setRange(0.0, 100.0); self.spin_baseline_percentile.setValue(self.baseline_params['percentile'])
        self.spin_baseline_percentile.setSuffix(' %')
        self.spin_baseline_percentile.valueChanged.connect(self.on_baseline_param_change)
        h_percentile = QtWidgets.QHBoxLayout()
        h_percentile.addWidget(QtWidgets.QLabel('Percentile:'))
        h_percentile.addWidget(self.spin_baseline_percentile)
        vbox_mid.addLayout(h_percentile)
        self.spin_sgolay_polyorder = QtWidgets.QSpinBox(); self.spin_sgolay_polyorder.setRange(1, 7); self.spin_sgolay_polyorder.setValue(self.baseline_params['sgolay_polyorder'])
        self.spin_sgolay_polyorder.valueChanged.connect(self.on_baseline_param_change)
        h_sgolay = QtWidgets.QHBoxLayout()
        h_sgolay.addWidget(QtWidgets.QLabel('SGolay Polyorder:'))
        h_sgolay.addWidget(self.spin_sgolay_polyorder)
        vbox_mid.addLayout(h_sgolay)
        # Option to show baseline-corrected result (moved to full-width control row)
        self.chk_show_baseline = QtWidgets.QCheckBox('Baseline correction')
        self.chk_show_baseline.setChecked(False)
        self.chk_show_baseline.stateChanged.connect(self.update_plot)
        # Note: baseline / filter visualization checkboxes are placed
        # in the full-width control row below the separator (see controls_h)
        self.combo_frame_processing = QtWidgets.QComboBox()
        self.combo_frame_processing.addItems(['Rolling average', 'Downsampling frames'])
        self.combo_frame_processing.setCurrentText(str(self.params.get('FRAME_PROCESSING_MODE', 'Rolling average')))
        self.combo_frame_processing.currentTextChanged.connect(self.on_frame_processing_mode_changed)
        h_frame_mode = QtWidgets.QHBoxLayout()
        h_frame_mode.addWidget(QtWidgets.QLabel('Frame processing:'))
        h_frame_mode.addWidget(self.combo_frame_processing)
        vbox_mid.addLayout(h_frame_mode)
        # Averaging (frames) applied to visualization and detection
        self.spin_avg_frames = QtWidgets.QSpinBox(); self.spin_avg_frames.setRange(0, 500); self.spin_avg_frames.setValue(0)
        self.spin_avg_frames.valueChanged.connect(self.update_plot)
        h_avg = QtWidgets.QHBoxLayout()
        h_avg.addWidget(QtWidgets.QLabel('Averaging frames (0 = off):'))
        h_avg.addWidget(self.spin_avg_frames)
        vbox_mid.addLayout(h_avg)
        # put baseline controls into middle column
        # upper_layout.addLayout(vbox_mid)
        # Raw trace visualization controls (grouped)
        vbox_right = QtWidgets.QVBoxLayout()
        vbox_right.setSpacing(2)
        # Title removed: separator provides sufficient visual grouping
        # Time slider + window controls replace start/end numeric boxes
        self.slider_time = QtWidgets.QSlider()
        self.slider_time.setOrientation(ORIENT_HORIZONTAL)
        self.slider_time.setMinimum(0)
        self.slider_time.setMaximum(1000)
        self.slider_time.setValue(0)
        self.slider_time.valueChanged.connect(self.on_slider_time_changed)
        self.spin_window = QtWidgets.QDoubleSpinBox(); self.spin_window.setRange(1.0, 1e6); self.spin_window.setValue(500.0)
        self.spin_window.setSuffix(' ms')
        self.spin_window.valueChanged.connect(self.on_window_changed)
        # Note: time/window/Y-range controls are placed in a dedicated full-width
        # control row below the upper settings (see main layout), not inside
        # the right-side column. Keep widgets created here but do not add them
        # to this column to avoid duplicates.
        self.spin_zoom = QtWidgets.QDoubleSpinBox(); self.spin_zoom.setRange(0.0, 1e6); self.spin_zoom.setValue(0.0)
        self.spin_zoom.valueChanged.connect(self.update_plot)
        # Visualization checkboxes (created here so they exist when placed in control row)
        self.chk_show_cs = QtWidgets.QCheckBox('CS filter')
        self.chk_show_cs.setChecked(False)
        self.chk_show_cs.toggled.connect(self._on_chk_cs_toggled)
        self.chk_show_ss = QtWidgets.QCheckBox('SS filter')
        self.chk_show_ss.setChecked(False)
        self.chk_show_ss.toggled.connect(self._on_chk_ss_toggled)
        # Detection controls (CS/SS) added into main window (third column)
        vbox_right.addWidget(QtWidgets.QLabel('CS / SS Detection Settings'))
        self.spin_cs_high = QtWidgets.QDoubleSpinBox(); self.spin_cs_high.setRange(0.0, 5000.0); self.spin_cs_high.setValue(self.params.get('CS_HIGH_CUT_HZ',150.0))
        self.spin_cs_high.valueChanged.connect(lambda v: (self.params.update({'CS_HIGH_CUT_HZ': float(v)}), self.update_plot()))
        vbox_right.addWidget(QtWidgets.QLabel('CS high cut (Hz):'))
        vbox_right.addWidget(self.spin_cs_high)
        self.spin_cs_thresh = QtWidgets.QDoubleSpinBox(); self.spin_cs_thresh.setRange(0.01,50.0); self.spin_cs_thresh.setDecimals(3); self.spin_cs_thresh.setSingleStep(0.05); self.spin_cs_thresh.setValue(self.params.get('CS_THRESHOLD_SIGMA',6.0))
        self.spin_cs_thresh.valueChanged.connect(lambda v: (self.params.update({'CS_THRESHOLD_SIGMA': float(v)}), self.update_plot()))
        vbox_right.addWidget(QtWidgets.QLabel('CS threshold (σ):'))
        vbox_right.addWidget(self.spin_cs_thresh)
        # CS filter order (default 4)
        self.spin_cs_order = QtWidgets.QSpinBox(); self.spin_cs_order.setRange(1, 8); self.spin_cs_order.setValue(self.params.get('CS_FILTER_ORDER', 4))
        self.spin_cs_order.valueChanged.connect(lambda v: (self.params.update({'CS_FILTER_ORDER': int(v)}), self.update_plot()))
        h_cs_order_init = QtWidgets.QHBoxLayout()
        h_cs_order_init.addWidget(QtWidgets.QLabel('CS filter order:'))
        h_cs_order_init.addWidget(self.spin_cs_order)
        vbox_right.addLayout(h_cs_order_init)
        # CS min dist moved to Advanced Settings (SettingsDialog)

        self.spin_ss_low = QtWidgets.QDoubleSpinBox(); self.spin_ss_low.setRange(0.0,5000.0); self.spin_ss_low.setValue(self.params.get('SS_LOW_CUT_HZ',50.0))
        self.spin_ss_low.valueChanged.connect(lambda v: (self.params.update({'SS_LOW_CUT_HZ': float(v)}), self.update_plot()))
        vbox_right.addWidget(QtWidgets.QLabel('SS low cut (Hz):'))
        vbox_right.addWidget(self.spin_ss_low)
        self.spin_ss_high = QtWidgets.QDoubleSpinBox(); self.spin_ss_high.setRange(0.0,5000.0); self.spin_ss_high.setValue(self.params.get('SS_HIGH_CUT_HZ',700.0))
        self.spin_ss_high.valueChanged.connect(lambda v: (self.params.update({'SS_HIGH_CUT_HZ': float(v)}), self.update_plot()))
        vbox_right.addWidget(QtWidgets.QLabel('SS high cut (Hz):'))
        vbox_right.addWidget(self.spin_ss_high)
        self.spin_ss_thresh = QtWidgets.QDoubleSpinBox(); self.spin_ss_thresh.setRange(0.01,50.0); self.spin_ss_thresh.setDecimals(3); self.spin_ss_thresh.setSingleStep(0.05); self.spin_ss_thresh.setValue(self.params.get('SS_THRESHOLD_SIGMA',2.0))
        self.spin_ss_thresh.valueChanged.connect(lambda v: (self.params.update({'SS_THRESHOLD_SIGMA': float(v)}), self.update_plot()))
        vbox_right.addWidget(QtWidgets.QLabel('SS threshold (σ):'))
        vbox_right.addWidget(self.spin_ss_thresh)
        # SS filter order (applies to both HP and LP in SS bandpass)
        self.spin_ss_order = QtWidgets.QSpinBox(); self.spin_ss_order.setRange(1, 8); self.spin_ss_order.setValue(self.params.get('SS_FILTER_ORDER', 4))
        self.spin_ss_order.valueChanged.connect(lambda v: (self.params.update({'SS_FILTER_ORDER': int(v)}), self.update_plot()))
        h_ss_order_init = QtWidgets.QHBoxLayout()
        h_ss_order_init.addWidget(QtWidgets.QLabel('SS filter order:'))
        h_ss_order_init.addWidget(self.spin_ss_order)
        vbox_right.addLayout(h_ss_order_init)
        self.spin_ss_mind = QtWidgets.QDoubleSpinBox(); self.spin_ss_mind.setRange(0.0,1000.0); self.spin_ss_mind.setValue(self.params.get('SS_MIN_DIST_MS',2.0))
        self.spin_ss_mind.valueChanged.connect(lambda v: (self.params.update({'SS_MIN_DIST_MS': float(v)}), self.update_plot()))
        self.spin_ss_blank = QtWidgets.QDoubleSpinBox(); self.spin_ss_blank.setRange(0.0,1000.0); self.spin_ss_blank.setValue(self.params.get('SS_BLANK_MS',8.0))
        self.spin_ss_blank.valueChanged.connect(lambda v: (self.params.update({'SS_BLANK_MS': float(v)}), self.update_plot()))
        # Trace detection button
        btn_detect = QtWidgets.QPushButton('Spike Detection')
        btn_detect.clicked.connect(self.open_detection_settings_dialog)
        btn_two_step = QtWidgets.QPushButton('Two-step Detection')
        btn_two_step.clicked.connect(self.open_two_step_detection)
        # create Spike Stats button here so we can place them side-by-side later
        btn_stats = QtWidgets.QPushButton('Spike Statistics')
        btn_stats.clicked.connect(self.open_stats_viewer)
        # add detect button to the right-side vbox for initial layout; it will be
        # reparented into the column layout when arranged below
        vbox_right.addWidget(btn_detect)
        vbox_right.addWidget(QtWidgets.QLabel('Info:'))
        self.text_stats = QtWidgets.QTextEdit()
        self.text_stats.setReadOnly(True)
        self.text_stats.setMaximumHeight(160)
        vbox_right.addWidget(self.text_stats)
        # upper_layout.addLayout(vbox_right)

        # --- Now compose upper panel with 3 equal columns ---
        upper_panel = QtWidgets.QWidget()
        upper_h = QtWidgets.QHBoxLayout()
        upper_h.setSpacing(4)
        upper_h.setContentsMargins(2, 2, 2, 2)
        col1 = QtWidgets.QVBoxLayout(); col2 = QtWidgets.QVBoxLayout(); col3 = QtWidgets.QVBoxLayout()
        col1.setSpacing(2); col2.setSpacing(2); col3.setSpacing(2)
        col1.setContentsMargins(2,1,2,1); col2.setContentsMargins(2,1,2,1); col3.setContentsMargins(2,1,2,1)
        col1.addLayout(vbox_left)
        col1.addSpacing(2)
        col1.addLayout(vbox_mid)
        col1.addSpacing(3)
        raw_group = QtWidgets.QVBoxLayout()
        # increase internal spacing by ~30% for clearer separation
        raw_group.setSpacing(4)
        # explicit time controls row (center slider + window)
        h_time_local = QtWidgets.QHBoxLayout()
        lbl_center = QtWidgets.QLabel('Center:')
        h_time_local.addWidget(lbl_center)
        h_time_local.addWidget(self.slider_time)
        lbl_window = QtWidgets.QLabel('Window:')
        h_time_local.addWidget(lbl_window)
        h_time_local.addWidget(self.spin_window)
        # slightly more space between time controls (~30% more)
        h_time_local.setSpacing(3)
        # Note: per-user request, the center/window/Y-range controls are moved
        # into a single full-width control row below the upper panel. Do not
        # add them again here to avoid duplicate controls.
        col1.addLayout(raw_group)

        # Column 2: spike detection controls
        lbl_spike = QtWidgets.QLabel('Spike Detection')
        lbl_spike.setStyleSheet('font-weight: bold;')
        col2.addWidget(lbl_spike, 0, ALIGN_TOP)
        self.tabs_detection = QtWidgets.QTabWidget()

        # Threshold tab
        tab_threshold = QtWidgets.QWidget()
        lay_threshold = QtWidgets.QVBoxLayout()
        h_cs_high = QtWidgets.QHBoxLayout()
        h_cs_high.addWidget(QtWidgets.QLabel('CS high cut (Hz):'))
        h_cs_high.addWidget(self.spin_cs_high)
        lay_threshold.addLayout(h_cs_high)
        h_cs_order = QtWidgets.QHBoxLayout()
        h_cs_order.addWidget(QtWidgets.QLabel('CS filter order:'))
        h_cs_order.addWidget(self.spin_cs_order)
        lay_threshold.addLayout(h_cs_order)
        h_cs_thresh = QtWidgets.QHBoxLayout()
        h_cs_thresh.addWidget(QtWidgets.QLabel('CS threshold (σ):'))
        h_cs_thresh.addWidget(self.spin_cs_thresh)
        lay_threshold.addLayout(h_cs_thresh)
        lay_threshold.addSpacing(6)
        h_ss_low = QtWidgets.QHBoxLayout()
        h_ss_low.addWidget(QtWidgets.QLabel('SS low cut (Hz):'))
        h_ss_low.addWidget(self.spin_ss_low)
        lay_threshold.addLayout(h_ss_low)
        h_ss_high = QtWidgets.QHBoxLayout()
        h_ss_high.addWidget(QtWidgets.QLabel('SS high cut (Hz):'))
        h_ss_high.addWidget(self.spin_ss_high)
        lay_threshold.addLayout(h_ss_high)
        h_ss_order = QtWidgets.QHBoxLayout()
        h_ss_order.addWidget(QtWidgets.QLabel('SS filter order:'))
        h_ss_order.addWidget(self.spin_ss_order)
        lay_threshold.addLayout(h_ss_order)
        h_ss_thresh = QtWidgets.QHBoxLayout()
        h_ss_thresh.addWidget(QtWidgets.QLabel('SS threshold (σ):'))
        h_ss_thresh.addWidget(self.spin_ss_thresh)
        lay_threshold.addLayout(h_ss_thresh)
        lay_threshold.addStretch(1)
        tab_threshold.setLayout(lay_threshold)

        # Template matching tab
        tab_template = QtWidgets.QWidget()
        lay_template = QtWidgets.QVBoxLayout()
        btn_load_cs = QtWidgets.QPushButton('Load CS templates')
        btn_load_ss = QtWidgets.QPushButton('Load SS templates')
        btn_view_tpl = QtWidgets.QPushButton('View')
        btn_load_cs.clicked.connect(lambda: self.load_templates_for_type('CS'))
        btn_load_ss.clicked.connect(lambda: self.load_templates_for_type('SS'))
        btn_view_tpl.clicked.connect(self.open_template_viewer)
        h_tpl_load = QtWidgets.QHBoxLayout()
        h_tpl_load.addWidget(btn_load_cs)
        h_tpl_load.addWidget(btn_load_ss)
        lay_template.addLayout(h_tpl_load)
        h_tpl_view = QtWidgets.QHBoxLayout()
        h_tpl_view.addWidget(btn_view_tpl)
        self.chk_template_parallel = QtWidgets.QCheckBox('parallel')
        self.chk_template_parallel.setChecked(bool(self.params.get('TEMPLATE_PARALLEL', False)))
        self.chk_template_parallel.stateChanged.connect(self.on_template_controls_changed)
        h_tpl_view.addWidget(self.chk_template_parallel)
        h_tpl_view.addStretch(1)
        lay_template.addLayout(h_tpl_view)
        self.lbl_template_status = QtWidgets.QLabel('Templates: CS [0] from 0 source(s), SS [0] from 0 source(s)')
        self.lbl_template_status.setWordWrap(True)
        lay_template.addWidget(self.lbl_template_status)

        h_tmode = QtWidgets.QHBoxLayout()
        h_tmode.addWidget(QtWidgets.QLabel('Template method:'))
        self.combo_template_method = QtWidgets.QComboBox()
        self.combo_template_method.addItems(['LLR Probability Vector'])
        self.combo_template_method.setCurrentText(str(self.params.get('TEMPLATE_MATCH_METHOD', 'LLR Probability Vector')))
        self.combo_template_method.currentTextChanged.connect(self.on_template_controls_changed)
        h_tmode.addWidget(self.combo_template_method)
        lay_template.addLayout(h_tmode)

        h_cs_sigma = QtWidgets.QHBoxLayout()
        h_cs_sigma.addWidget(QtWidgets.QLabel('CS threshold (σ × MAD):'))
        self.spin_template_cs_sigma = QtWidgets.QDoubleSpinBox(); self.spin_template_cs_sigma.setRange(0.01, 20.0); self.spin_template_cs_sigma.setDecimals(3); self.spin_template_cs_sigma.setSingleStep(0.05); self.spin_template_cs_sigma.setValue(float(self.params.get('TEMPLATE_CS_SIGMA', 6.0)))
        self.spin_template_cs_sigma.valueChanged.connect(self.on_template_controls_changed)
        h_cs_sigma.addWidget(self.spin_template_cs_sigma)
        lay_template.addLayout(h_cs_sigma)

        h_ss_sigma = QtWidgets.QHBoxLayout()
        h_ss_sigma.addWidget(QtWidgets.QLabel('SS threshold (σ × MAD):'))
        self.spin_template_ss_sigma = QtWidgets.QDoubleSpinBox(); self.spin_template_ss_sigma.setRange(0.01, 20.0); self.spin_template_ss_sigma.setDecimals(3); self.spin_template_ss_sigma.setSingleStep(0.05); self.spin_template_ss_sigma.setValue(float(self.params.get('TEMPLATE_SS_SIGMA', 4.0)))
        self.spin_template_ss_sigma.valueChanged.connect(self.on_template_controls_changed)
        h_ss_sigma.addWidget(self.spin_template_ss_sigma)
        lay_template.addLayout(h_ss_sigma)

        lay_template.addStretch(1)
        tab_template.setLayout(lay_template)

        self.tabs_detection.addTab(tab_threshold, 'Threshold')
        self.tabs_detection.addTab(tab_template, 'Template Matching')
        self.tabs_detection.currentChanged.connect(self.on_detection_tab_changed)
        try:
            default_idx = 1 if self.params.get('DETECTION_METHOD', 'Threshold') == 'Template Matching' else 0
            self.tabs_detection.setCurrentIndex(default_idx)
            self.on_detection_tab_changed(default_idx)
        except Exception:
            pass
        col2.addWidget(self.tabs_detection)

        # place Spike Detection and Spike Stats side-by-side with equal stretch
        h_detect = QtWidgets.QVBoxLayout()
        try:
            btn_detect.setSizePolicy(QtWidgets.QSizePolicy(SIZEPOLICY_EXPANDING, SIZEPOLICY_FIXED))
        except Exception:
            try:
                btn_detect.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            except Exception:
                pass
        try:
            btn_stats.setSizePolicy(QtWidgets.QSizePolicy(SIZEPOLICY_EXPANDING, SIZEPOLICY_FIXED))
        except Exception:
            try:
                btn_stats.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            except Exception:
                pass
        h_detect.addWidget(btn_detect)
        h_detect.addSpacing(6)
        h_twostep_stats = QtWidgets.QHBoxLayout()
        h_twostep_stats.addWidget(btn_two_step)
        h_twostep_stats.addWidget(btn_stats)
        h_detect.addLayout(h_twostep_stats)

        # Column 3: advanced settings and info panel
        col3.addLayout(h_detect)
        col3.addSpacing(8)
        # Color scheme is accessible from Advanced Settings (no separate main button)
        # settings dialog button
        btn_settings = QtWidgets.QPushButton('Advanced Settings')
        btn_settings.clicked.connect(self.open_settings)
        col3.addWidget(btn_settings)
        # Spike statistics viewer button (moved next to Spike Detection)
        # Reset application state
        btn_reset = QtWidgets.QPushButton('Reset App')
        btn_reset.clicked.connect(self.reset_app)
        col3.addWidget(btn_reset)
        col3.addWidget(QtWidgets.QLabel('Info:'))
        col3.addWidget(self.text_stats)

        # place columns into a layout wrapper to ensure equal stretch
        upper_h.addLayout(col1, 1); upper_h.addLayout(col2, 1); upper_h.addLayout(col3, 1)
        upper_panel.setLayout(upper_h)
        # (do not cap upper panel height) place wrapper
        upper_layout_wrapper = upper_panel
        upper_layout.addWidget(upper_layout_wrapper)
        # --- Lower block: Visualization ---
        # make initial plot slightly shorter vertically (~20% shorter)
        self.canvas = PlotCanvas(self, width=10, height=3.2, dpi=100)

        # Main vertical layout with three stacked rows:
        # 1) upper settings (columns)
        # 2) narrow full-width control row (time slider, window, Y-range, save)
        # 3) large canvas for plotting
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(4,4,4,4)
        # Add upper settings panel (keeps previous 3-column layout)
        main_layout.addLayout(upper_layout, 0)

        # Create narrow full-width control row
        # add a thin separator line between the top three-column panel and this control row
        sep = QtWidgets.QFrame()
        _set_frame_hline(sep)
        try:
            sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        except Exception:
            try:
                sep.setFrameShadow(QtWidgets.QFrame.FrameShadow.Sunken)
            except Exception:
                pass
        sep.setFixedHeight(2)
        main_layout.addWidget(sep)

        controls_widget = QtWidgets.QWidget()
        controls_h = QtWidgets.QHBoxLayout()
        controls_h.setContentsMargins(2,2,2,2)
        controls_h.addWidget(QtWidgets.QLabel('Center:'))
        controls_h.addWidget(self.slider_time)
        controls_h.addWidget(QtWidgets.QLabel('Window:'))
        controls_h.addWidget(self.spin_window)
        # (Baseline checkbox is placed in the middle panel; do not add here)
        controls_h.addWidget(QtWidgets.QLabel('Y-Range:'))
        controls_h.addWidget(self.spin_zoom)
        # Visualization toggles (Baseline, CS filter, SS filter)
        # Baseline checkbox (moved here from middle panel)
        controls_h.addSpacing(6)
        controls_h.addWidget(self.chk_show_baseline)
        # CS/SS filter checkboxes (mutually exclusive)
        controls_h.addWidget(self.chk_show_cs)
        controls_h.addWidget(self.chk_show_ss)
        self.btn_select_templates = QtWidgets.QPushButton('Select templates')
        self.btn_select_templates.clicked.connect(self.toggle_template_selection_or_save)
        controls_h.addWidget(self.btn_select_templates)
        self.btn_abort_templates = QtWidgets.QPushButton('Abort')
        self.btn_abort_templates.clicked.connect(self.abort_template_selection)
        self.btn_abort_templates.setEnabled(False)
        controls_h.addWidget(self.btn_abort_templates)
        # Save Figure button (SVG/PNG/JPG)
        btn_save_fig = QtWidgets.QPushButton('Save Figure')
        btn_save_fig.clicked.connect(self.export_figure)
        controls_h.addStretch(1)
        controls_h.addWidget(btn_save_fig)
        controls_widget.setLayout(controls_h)
        # keep the controls row narrow
        controls_widget.setFixedHeight(52)
        main_layout.addWidget(controls_widget, 0)

        # Add canvas as the large plotting area
        main_layout.addWidget(self.canvas, 1)
        central.setLayout(main_layout)

        try:
            self.canvas.mpl_connect('button_press_event', self.on_main_plot_click)
        except Exception:
            pass

        # Hide irrelevant controls for baseline method
        self.on_baseline_param_change()
        self.on_template_controls_changed()
        self.update_template_status_label()

    def on_detection_tab_changed(self, idx):
        try:
            txt = self.tabs_detection.tabText(int(idx))
        except Exception:
            txt = 'Threshold'
        self.params['DETECTION_METHOD'] = 'Template Matching' if 'Template' in str(txt) else 'Threshold'

    def on_template_controls_changed(self):
        try:
            self.params['TEMPLATE_MATCH_METHOD'] = str(self.combo_template_method.currentText())
            self.params['TEMPLATE_PARALLEL'] = bool(self.chk_template_parallel.isChecked()) if hasattr(self, 'chk_template_parallel') else False
            self.params['TEMPLATE_CS_SIGMA'] = float(self.spin_template_cs_sigma.value())
            self.params['TEMPLATE_SS_SIGMA'] = float(self.spin_template_ss_sigma.value())
        except Exception:
            pass

    def open_two_step_detection(self):
        try:
            print('Running two-step detection (Template Matching + Threshold verification)...')
            self.on_template_controls_changed()
            sessions_done, total_cells = self.run_detection_all(two_step=True, force_detection_method='Template Matching')
            try:
                sessions_with_results = 0
                for sname, data in self.loaded_sessions.items():
                    if isinstance(data, dict) and 'results' in data and any([r is not None for r in data.get('results', [])]):
                        sessions_with_results += 1
                if sessions_with_results > 0:
                    self.text_stats.append(f'Two-step detection finished: {sessions_with_results} sessions, results available for viewing')
                    if self.selected_session in self.loaded_sessions:
                        d = self.loaded_sessions.get(self.selected_session)
                        if d and isinstance(d, dict) and 'results' in d and any([r is not None for r in d.get('results', [])]):
                            self.open_detection_viewer()
                else:
                    self.text_stats.append(f'Two-step detection finished: {sessions_done} sessions, {total_cells} cells processed (no valid results)')
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, 'Two-step Detection Error', str(e))
            try:
                self.text_stats.append(f'Two-step detection error: {e}')
            except Exception:
                pass

    def update_template_status_label(self):
        if getattr(self, 'lbl_template_status', None) is None:
            return
        cs_n = len(self.template_store.get('cs_templates', []))
        ss_n = len(self.template_store.get('ss_templates', []))
        cs_src = len(self.template_store.get('cs_sources', []))
        ss_src = len(self.template_store.get('ss_sources', []))
        info = f"CS [{cs_n}] from {cs_src} source(s), SS [{ss_n}] from {ss_src} source(s)"
        self.lbl_template_status.setText(info)

    def _read_template_npz(self, npz_path):
        out = {'cs_list': [], 'ss_list': [], 'fs_cs_list': [], 'fs_ss_list': []}
        try:
            d = np.load(npz_path, allow_pickle=True)
        except Exception:
            return out

        try:
            fs_default = float(d['fs']) if 'fs' in d else None
        except Exception:
            fs_default = None

        def _append_template(key_arr, key_fs, out_arr, out_fs):
            try:
                raw_arr = d[key_arr]
            except Exception:
                return
            local_fs = None
            try:
                local_fs = float(d[key_fs]) if key_fs in d else fs_default
            except Exception:
                local_fs = fs_default
            try:
                arr = np.asarray(raw_arr)
                if arr.dtype == object:
                    rows = [np.asarray(r, dtype=float).ravel() for r in list(arr)]
                elif arr.ndim == 1:
                    rows = [np.asarray(arr, dtype=float).ravel()]
                else:
                    rows = [np.asarray(r, dtype=float).ravel() for r in arr]
            except Exception:
                rows = []
            for rr in rows:
                if rr.size > 3 and np.all(np.isfinite(rr)):
                    out_arr.append(rr)
                    out_fs.append(local_fs)

        if 'template_cs' in d:
            _append_template('template_cs', 'fs_cs', out['cs_list'], out['fs_cs_list'])
        if 'cs_waveforms' in d:
            _append_template('cs_waveforms', 'fs_cs', out['cs_list'], out['fs_cs_list'])
        if 'template_ss' in d:
            _append_template('template_ss', 'fs_ss', out['ss_list'], out['fs_ss_list'])
        if 'ss_waveforms' in d:
            _append_template('ss_waveforms', 'fs_ss', out['ss_list'], out['fs_ss_list'])
        return out

    def _merge_templates_for_type(self, spike_type, info, source_path):
        if spike_type == 'CS':
            self.template_store['cs_templates'].extend(info.get('cs_list', []))
            self.template_store['fs_cs'].extend(info.get('fs_cs_list', []))
            if source_path not in self.template_store['cs_sources']:
                self.template_store['cs_sources'].append(source_path)
        else:
            self.template_store['ss_templates'].extend(info.get('ss_list', []))
            self.template_store['fs_ss'].extend(info.get('fs_ss_list', []))
            if source_path not in self.template_store['ss_sources']:
                self.template_store['ss_sources'].append(source_path)

    def _pick_file_or_folder(self, caption):
        fpaths, _ = QFileDialog.getOpenFileNames(self, caption + ' (single/multiple files)', os.path.expanduser('~'), 'NPZ Files (*.npz)')
        if fpaths and len(fpaths) > 0:
            return list(fpaths)
        return []

    def load_templates_for_type(self, spike_type='CS'):
        p = self._pick_file_or_folder(f'Load {spike_type} templates')
        if not p:
            return
        if spike_type == 'CS':
            self.template_store['cs_templates'] = []
            self.template_store['fs_cs'] = []
            self.template_store['cs_sources'] = []
        else:
            self.template_store['ss_templates'] = []
            self.template_store['fs_ss'] = []
            self.template_store['ss_sources'] = []

        if isinstance(p, (list, tuple)):
            files = [fp for fp in p if isinstance(fp, str) and fp.lower().endswith('.npz') and os.path.isfile(fp)]
        elif isinstance(p, str) and os.path.isfile(p):
            files = [p]
        elif isinstance(p, str) and os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.npz')))
        else:
            files = []
        if len(files) == 0:
            QMessageBox.warning(self, 'No templates', 'No NPZ template files selected/found.')
            return
        for fp in files:
            info = self._read_template_npz(fp)
            self._merge_templates_for_type(spike_type, info, fp)
        self.update_template_status_label()

    def _get_current_window_mask(self, t):
        t_min, t_max = t[0], t[-1]
        center_pct = float(self.slider_time.value()) / 1000.0 if hasattr(self, 'slider_time') else 0.0
        total_ms = t_max - t_min
        center_time = t_min + center_pct * total_ms
        window_ms = float(self.spin_window.value()) if hasattr(self, 'spin_window') else 500.0
        start_time = center_time - window_ms / 2.0
        end_time = center_time + window_ms / 2.0
        if start_time < t_min:
            start_time = t_min
            end_time = min(t_min + window_ms, t_max)
        if end_time > t_max:
            end_time = t_max
            start_time = max(t_min, t_max - window_ms)
        return (t >= start_time) & (t <= end_time)

    def _get_current_display_signal_full(self):
        if not self.selected_session or self.selected_session not in self.loaded_sessions:
            return None, None, None
        data = self.loaded_sessions[self.selected_session]
        idx = max(0, self.selected_cell)
        raw = data['raw_data'][:, idx]
        t = data['time_ms']
        fs = float(data['fs'])
        frames = int(self.spin_avg_frames.value()) if hasattr(self, 'spin_avg_frames') else 0
        raw_proc = apply_frame_processing(raw, frames=frames, mode=self._get_frame_processing_mode())
        baseline = self.compute_baseline(raw_proc, fs)
        method = self.baseline_params.get('method', '')
        use_baseline_vis = (method != 'Disable' and getattr(self, 'chk_show_baseline', None) and self.chk_show_baseline.isChecked())
        vis_source = (raw_proc - baseline) if use_baseline_vis else raw_proc
        if getattr(self, 'chk_show_cs', None) and self.chk_show_cs.isChecked():
            vis_signal = apply_filter(vis_source, fs, low=None, high=float(self.spin_cs_high.value()), order=int(self.spin_cs_order.value()))
        elif getattr(self, 'chk_show_ss', None) and self.chk_show_ss.isChecked():
            vis_signal = apply_filter(vis_source, fs, low=float(self.spin_ss_low.value()), high=float(self.spin_ss_high.value()), order=int(self.spin_ss_order.value()))
        else:
            vis_signal = vis_source
        return np.asarray(vis_signal, dtype=float), np.asarray(t), float(fs)

    def _get_current_template_signal_full(self):
        if not self.selected_session or self.selected_session not in self.loaded_sessions:
            return None, None, None
        data = self.loaded_sessions[self.selected_session]
        idx = max(0, self.selected_cell)
        raw = np.asarray(data['raw_data'][:, idx], dtype=float)
        t = np.asarray(data['time_ms'], dtype=float)
        fs = float(data['fs'])
        frames = int(self.spin_avg_frames.value()) if hasattr(self, 'spin_avg_frames') else 0
        raw_proc = apply_frame_processing(raw, frames=frames, mode=self._get_frame_processing_mode())
        try:
            baseline = self.compute_baseline(raw_proc, fs)
        except Exception:
            baseline = np.zeros_like(raw_proc)
        sig = raw_proc - baseline
        if bool(self.params.get('NEGATIVE_GOING', True)):
            sig = -sig
        return np.asarray(sig, dtype=float), t, fs

    def _reset_template_selection_button_text(self):
        if self.btn_select_templates is None:
            return
        self.btn_select_templates.setText('Save templates' if self.template_selection_active else 'Select templates')
        if self.btn_abort_templates is not None:
            self.btn_abort_templates.setEnabled(bool(self.template_selection_active))

    def abort_template_selection(self):
        self.template_selection_active = False
        self.template_selected_segments['CS'] = []
        self.template_selected_segments['SS'] = []
        self.template_selected_intervals_ms['CS'] = []
        self.template_selected_intervals_ms['SS'] = []
        self._reset_template_selection_button_text()
        self.update_plot()

    def toggle_template_selection_or_save(self):
        if self.template_selection_active:
            self.save_selected_templates()
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Select templates')
        form = QtWidgets.QFormLayout(dlg)
        combo_type = QtWidgets.QComboBox(dlg)
        combo_type.addItems(['CS', 'SS'])
        spin_cs_w = QtWidgets.QDoubleSpinBox(dlg); spin_cs_w.setRange(1.0, 200.0); spin_cs_w.setDecimals(1); spin_cs_w.setSuffix(' ms')
        spin_ss_w = QtWidgets.QDoubleSpinBox(dlg); spin_ss_w.setRange(1.0, 100.0); spin_ss_w.setDecimals(1); spin_ss_w.setSuffix(' ms')
        spin_cs_w.setValue(float(self.params.get('TEMPLATE_CS_WINDOW_MS', 30.0)))
        spin_ss_w.setValue(float(self.params.get('TEMPLATE_SS_WINDOW_MS', 8.0)))
        form.addRow('Template type:', combo_type)
        form.addRow('CS window:', spin_cs_w)
        form.addRow('SS window:', spin_ss_w)
        try:
            bb_buttons = QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        except Exception:
            bb_buttons = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        btn_box = QtWidgets.QDialogButtonBox(bb_buttons, parent=dlg)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        form.addRow(btn_box)
        try:
            accepted_code = QtWidgets.QDialog.DialogCode.Accepted
        except Exception:
            accepted_code = QtWidgets.QDialog.Accepted
        if dlg.exec() != accepted_code:
            return

        self.params['TEMPLATE_CS_WINDOW_MS'] = float(spin_cs_w.value())
        self.params['TEMPLATE_SS_WINDOW_MS'] = float(spin_ss_w.value())
        self.template_selection_type = str(combo_type.currentText()).upper()
        self.template_selection_window_ms = float(self.params['TEMPLATE_CS_WINDOW_MS'] if self.template_selection_type == 'CS' else self.params['TEMPLATE_SS_WINDOW_MS'])
        self.template_selection_active = True
        self.template_selected_segments[self.template_selection_type] = []
        self.template_selected_intervals_ms[self.template_selection_type] = []
        self._reset_template_selection_button_text()
        self.update_plot()

    def on_main_plot_click(self, event):
        if not self.template_selection_active:
            return
        if event is None or event.inaxes is None or event.xdata is None:
            return
        sig, t, fs = self._get_current_template_signal_full()
        if sig is None or t is None or fs is None or sig.size <= 3:
            return
        center_ms = float(event.xdata)
        half_ms = self.template_selection_window_ms / 2.0
        start_ms = center_ms - half_ms
        end_ms = center_ms + half_ms
        mask = (t >= start_ms) & (t <= end_ms)
        seg = np.asarray(sig[mask], dtype=float)
        if seg.size <= 3:
            return
        self.template_selected_segments[self.template_selection_type].append(seg)
        self.template_selected_intervals_ms[self.template_selection_type].append((start_ms, end_ms))
        self.update_plot()

    def save_selected_templates(self):
        st = self.template_selection_type
        segs = self.template_selected_segments.get(st, [])
        if len(segs) == 0:
            QMessageBox.warning(self, 'No templates', 'No selected periods yet. Click on the trace to select template windows.')
            return
        _, _, fs = self._get_current_template_signal_full()
        if fs is None:
            QMessageBox.warning(self, 'No data', 'No valid trace available for saving templates.')
            return
        filename, _ = QFileDialog.getSaveFileName(self, 'Save templates NPZ', os.path.expanduser('~/templates_selected.npz'), 'NPZ Files (*.npz)')
        if not filename:
            return
        if not filename.lower().endswith('.npz'):
            filename += '.npz'

        old = {}
        if os.path.exists(filename):
            try:
                old_npz = np.load(filename, allow_pickle=True)
                old = {k: old_npz[k] for k in old_npz.files}
            except Exception:
                old = {}

        payload = dict(old)
        target_fs = float(TEMPLATE_TARGET_FS)
        upsampled = [_resample_template_to_fs(np.asarray(s, dtype=float).ravel(), fs, target_fs) for s in segs]
        payload['fs'] = float(target_fs)
        seg_arr = np.array(upsampled, dtype=object)
        if st == 'CS':
            payload['cs_waveforms'] = seg_arr
            try:
                lens = [len(np.asarray(s).ravel()) for s in upsampled]
                min_len = int(np.min(lens))
                stack = np.vstack([np.asarray(s).ravel()[:min_len] for s in upsampled])
                payload['template_cs'] = np.mean(stack, axis=0)
                payload['fs_cs'] = float(target_fs)
            except Exception:
                pass
        else:
            payload['ss_waveforms'] = seg_arr
            try:
                lens = [len(np.asarray(s).ravel()) for s in upsampled]
                min_len = int(np.min(lens))
                stack = np.vstack([np.asarray(s).ravel()[:min_len] for s in upsampled])
                payload['template_ss'] = np.mean(stack, axis=0)
                payload['fs_ss'] = float(target_fs)
            except Exception:
                pass

        np.savez_compressed(filename, **payload)
        QMessageBox.information(self, 'Saved', f'{len(segs)} {st} template segment(s) saved to {filename}')

        # reload just-saved file into matching bank for this type
        info = self._read_template_npz(filename)
        if st == 'CS':
            self.template_store['cs_templates'] = []
            self.template_store['fs_cs'] = []
            self.template_store['cs_sources'] = []
        else:
            self.template_store['ss_templates'] = []
            self.template_store['fs_ss'] = []
            self.template_store['ss_sources'] = []
        self._merge_templates_for_type(st, info, filename)
        self.update_template_status_label()

        self.template_selection_active = False
        self.template_selected_segments[st] = []
        self.template_selected_intervals_ms[st] = []
        self._reset_template_selection_button_text()
        self.update_plot()

    def open_color_scheme_dialog(self):
        dlg = ColorSchemeDialog(self)
        if dlg.exec():
            scheme = dlg.get_scheme()
            self.colors.update(scheme)
            self.update_plot()

    def open_detection_settings_dialog(self):
        # Use main-window controls for detection (no popup). Run detection directly.
        try:
            print('Running detection on all sessions (main-window settings)...')
            try:
                if hasattr(self, 'tabs_detection') and self.tabs_detection is not None:
                    self.on_detection_tab_changed(self.tabs_detection.currentIndex())
            except Exception:
                pass
            # sync UI controls into params to ensure latest values are used
            try:
                # Read inline detection controls that remain in main UI
                self.params['CS_HIGH_CUT_HZ'] = float(self.spin_cs_high.value())
                self.params['CS_THRESHOLD_SIGMA'] = float(self.spin_cs_thresh.value())
                self.params['SS_LOW_CUT_HZ'] = float(self.spin_ss_low.value())
                self.params['SS_HIGH_CUT_HZ'] = float(self.spin_ss_high.value())
                self.params['SS_THRESHOLD_SIGMA'] = float(self.spin_ss_thresh.value())
                self.on_template_controls_changed()
            except Exception:
                pass
            sessions_done, total_cells = self.run_detection_all()
            # open detection viewer for current session when done only if we have results
            try:
                # Determine which sessions actually produced results
                sessions_with_results = 0
                cells_processed = 0
                for sname, data in self.loaded_sessions.items():
                    if isinstance(data, dict) and 'results' in data and any([r is not None for r in data.get('results', [])]):
                        sessions_with_results += 1
                        cells_processed += len(data.get('results', []))
                if sessions_with_results > 0:
                    self.text_stats.append(f'Detection finished: {sessions_with_results} sessions, results available for viewing')
                    # open viewer for selected session if it has results
                    if self.selected_session in self.loaded_sessions:
                        d = self.loaded_sessions.get(self.selected_session)
                        if d and isinstance(d, dict) and 'results' in d and any([r is not None for r in d.get('results', [])]):
                            self.open_detection_viewer()
                        else:
                            self.text_stats.append('Selected session has no valid results to view.')
                else:
                    self.text_stats.append(f'Detection finished: {sessions_done} sessions, {total_cells} cells processed (no valid results)')
            except Exception as e:
                print('Failed to open Detection Viewer:', e)
                try:
                    self.text_stats.append(f'Failed to open Detection Viewer: {e}')
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.critical(self, 'Detection Error', str(e))
            try:
                self.text_stats.append(f'Detection error: {e}')
            except Exception:
                pass
        
    
    def on_baseline_param_change(self):
        method = self.combo_baseline_method.currentText()
        self.baseline_params['method'] = method
        self.baseline_params['window_ms'] = float(self.spin_baseline_window.value())
        self.baseline_params['percentile'] = float(self.spin_baseline_percentile.value())
        self.baseline_params['sgolay_polyorder'] = int(self.spin_sgolay_polyorder.value())
        # Show/hide controls
        self.spin_baseline_percentile.setEnabled(method == 'Percentile')
        self.spin_sgolay_polyorder.setEnabled(method == 'Savitzky-Golay')
        # If baseline is disabled, disable the 'Show baseline-corrected' option
        try:
            if getattr(self, 'chk_show_baseline', None) is not None:
                if method == 'Disable':
                    try:
                        self.chk_show_baseline.setChecked(False)
                    except Exception:
                        pass
                    try:
                        self.chk_show_baseline.setEnabled(False)
                    except Exception:
                        pass
                else:
                    try:
                        self.chk_show_baseline.setEnabled(True)
                    except Exception:
                        pass
        except Exception:
            pass
        self.update_plot()

    def on_frame_processing_mode_changed(self):
        try:
            self.params['FRAME_PROCESSING_MODE'] = str(self.combo_frame_processing.currentText())
        except Exception:
            pass
        self.update_plot()

    def _get_frame_processing_mode(self):
        try:
            return str(self.combo_frame_processing.currentText())
        except Exception:
            return str(self.params.get('FRAME_PROCESSING_MODE', 'Rolling average'))

    def _on_chk_cs_toggled(self, checked):
        try:
            if checked and getattr(self, 'chk_show_ss', None) is not None:
                # enforce mutual exclusion
                self.chk_show_ss.setChecked(False)
        except Exception:
            pass
        try:
            self.update_plot()
        except Exception:
            pass

    def _on_chk_ss_toggled(self, checked):
        try:
            if checked and getattr(self, 'chk_show_cs', None) is not None:
                # enforce mutual exclusion
                self.chk_show_cs.setChecked(False)
        except Exception:
            pass
        try:
            self.update_plot()
        except Exception:
            pass


    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            # dialog updates parent.params and parent.colors directly
            self.update_plot()

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Master Folder', os.path.expanduser('~'))
        if folder:
            self.master_folder = folder
            self.statusBar().showMessage(f'Master folder: {folder}')
            self.refresh_sessions()

    def _count_sessions_and_cells(self, candidates):
        """Return (n_sessions, total_cells, cells_per_session list)
        candidates: list of file or directory paths"""
        total_cells = 0
        cells_per = []
        for s in candidates:
            try:
                if os.path.isfile(s):
                    if s.lower().endswith('.npz'):
                        npz = np.load(s, allow_pickle=True)
                        n = len(npz['cell_names']) if 'cell_names' in npz else 0
                    elif s.lower().endswith('.xlsx') or s.lower().endswith('.csv'):
                        # read header only
                        if s.lower().endswith('.xlsx'):
                            df = pd.read_excel(s, nrows=0, engine='openpyxl')
                        else:
                            df = pd.read_csv(s, nrows=0)
                        n = max(0, df.shape[1] - 1)
                    else:
                        n = 0
                else:
                    # directory: look for analyzed npz or xlsx inside
                    npz_list = glob.glob(os.path.join(s, '*_analyzed.npz'))
                    xls_list = glob.glob(os.path.join(s, '*.xlsx'))
                    csv_list = glob.glob(os.path.join(s, '*.csv'))
                    if npz_list:
                        npz = np.load(npz_list[0], allow_pickle=True)
                        n = len(npz['cell_names']) if 'cell_names' in npz else 0
                    elif xls_list:
                        df = pd.read_excel(xls_list[0], nrows=0, engine='openpyxl')
                        n = max(0, df.shape[1] - 1)
                    elif csv_list:
                        df = pd.read_csv(csv_list[0], nrows=0)
                        n = max(0, df.shape[1] - 1)
                    else:
                        n = 0
            except Exception:
                n = 0
            cells_per.append(n)
            total_cells += n
        return len(candidates), total_cells, cells_per
    def open_slider_viewer(self):
        if self.data is None:
            QMessageBox.warning(self, 'No data', 'Load a session first')
            return
        dlg = SliderViewerDialog(self.data, parent=self)
        dlg.exec()

    def run_detection_all(self, two_step=False, force_detection_method=None):
        if not self.sessions:
            QMessageBox.warning(self, 'No data', 'No sessions loaded. Select a folder first.')
            return 0, 0

        detection_method = force_detection_method if force_detection_method is not None else self.params.get('DETECTION_METHOD', 'Threshold')
        if detection_method == 'Template Matching':
            cs_n = len(self.template_store.get('cs_templates', []))
            ss_n = len(self.template_store.get('ss_templates', []))
            if cs_n == 0 and ss_n == 0:
                QMessageBox.warning(self, 'Templates missing', 'Template Matching is selected, but no CS/SS template is loaded.')
                return 0, 0

        # Ensure all sessions from the session list are loaded into memory
        for idx, session_path in enumerate(self.sessions):
            try:
                session_name = self.session_names[idx]
            except Exception:
                session_name = os.path.basename(session_path)
            if session_name not in self.loaded_sessions:
                d = self.load_session(session_path)
                if d is not None:
                    self.loaded_sessions[session_name] = d

        total_cells = 0
        sessions_done = 0

        # Aggregation containers for global statistics across all sessions/cells
        global_cs_rates = []
        global_ss_rates = []
        global_cs_snrs = []
        global_ss_snrs = []
        global_cs_fwhm = []
        global_ss_fwhm = []

        n_sessions = len(self.session_names)
        for sess_idx, sname in enumerate(self.session_names):
            data = self.loaded_sessions.get(sname, None)
            if data is None:
                continue
            try:
                # announce session start
                try:
                    if hasattr(self, 'text_stats') and self.text_stats is not None:
                        # single-line status: overwrite with current session
                        self.text_stats.setPlainText(f"Processing session {sess_idx+1}/{n_sessions}: {sname}")
                        QtWidgets.QApplication.processEvents()
                except Exception:
                    pass
                fs = float(data['fs'])
                n_cells = data['raw_data'].shape[1]
                data['results'] = [None] * n_cells
                data['spike_times_cs'] = [np.array([])] * n_cells
                data['spike_times_ss'] = [np.array([])] * n_cells
                for i in range(n_cells):
                    raw = data['raw_data'][:, i]
                    # apply averaging if requested in UI (applies to detection)
                    frames = int(self.spin_avg_frames.value()) if hasattr(self, 'spin_avg_frames') else 0
                    raw_proc = apply_frame_processing(raw, frames=frames, mode=self._get_frame_processing_mode())
                    # compute GUI baseline on processed (averaged) signal and provide detrended to detection
                    try:
                        baseline_gui = self.compute_baseline(raw_proc, fs)
                        detrended_gui = raw_proc - baseline_gui
                    except Exception:
                        baseline_gui = np.zeros_like(raw_proc)
                        detrended_gui = raw_proc
                    # ensure detrended provided to detection matches negative-going setting
                    pre_detr = detrended_gui * (-1.0 if self.params.get('NEGATIVE_GOING', True) else 1.0)
                    if detection_method == 'Template Matching':
                        res_tm = process_cell_template_matching(
                            raw_proc, fs,
                            template_cs_bank=self.template_store.get('cs_templates', []),
                            template_ss_bank=self.template_store.get('ss_templates', []),
                            template_cs_fs_bank=self.template_store.get('fs_cs', []),
                            template_ss_fs_bank=self.template_store.get('fs_ss', []),
                            negative_going=self.params.get('NEGATIVE_GOING', True),
                            cs_high_cut=self.params.get('CS_HIGH_CUT_HZ', 150.0),
                            cs_thresh_sigma=self.params.get('TEMPLATE_CS_SIGMA', 6.0),
                            cs_min_dist_ms=self.params.get('CS_MIN_DIST_MS', 25.0),
                            ss_low_cut=self.params.get('SS_LOW_CUT_HZ', 50.0),
                            ss_high_cut=self.params.get('SS_HIGH_CUT_HZ', 700.0),
                            ss_thresh_sigma=self.params.get('TEMPLATE_SS_SIGMA', 4.0),
                            ss_min_dist_ms=self.params.get('SS_MIN_DIST_MS', 2.0),
                            ss_blank_ms=self.params.get('SS_BLANK_MS', 8.0),
                            template_match_method=self.params.get('TEMPLATE_MATCH_METHOD', 'LLR Probability Vector'),
                            parallel_match=bool(self.params.get('TEMPLATE_PARALLEL', False)),
                            initial_blank_ms=self.params.get('INITIAL_BLANK_MS', 150.0),
                            use_preprocessed=True, pre_detrended=pre_detr, pre_baseline=baseline_gui,
                            cs_order=int(self.params.get('CS_FILTER_ORDER', 4)),
                            ss_order=int(self.params.get('SS_FILTER_ORDER', 4))
                        )
                        if bool(two_step):
                            res_simple = process_cell_simple(raw_proc, fs,
                                                             negative_going=self.params.get('NEGATIVE_GOING', True),
                                                             cs_high_cut=self.params.get('CS_HIGH_CUT_HZ', 150.0),
                                                             cs_thresh_sigma=self.params.get('CS_THRESHOLD_SIGMA', 6.0),
                                                             cs_min_dist_ms=self.params.get('CS_MIN_DIST_MS', 25.0),
                                                             ss_low_cut=self.params.get('SS_LOW_CUT_HZ', 50.0),
                                                             ss_high_cut=self.params.get('SS_HIGH_CUT_HZ', 700.0),
                                                             ss_thresh_sigma=self.params.get('SS_THRESHOLD_SIGMA', 2.0),
                                                             ss_min_dist_ms=self.params.get('SS_MIN_DIST_MS', 2.0),
                                                             ss_blank_ms=self.params.get('SS_BLANK_MS', 8.0),
                                                             initial_blank_ms=self.params.get('INITIAL_BLANK_MS', 150.0),
                                                             use_preprocessed=True, pre_detrended=pre_detr, pre_baseline=baseline_gui,
                                                             cs_order=int(self.params.get('CS_FILTER_ORDER', 4)),
                                                             ss_order=int(self.params.get('SS_FILTER_ORDER', 4)))
                            tol_cs = max(1, int(round((2.0 / 1000.0) * fs)))
                            tol_ss = max(1, int(round((1.0 / 1000.0) * fs)))
                            cs_keep = _filter_peaks_by_reference(res_tm.get('cs_peaks', []), res_simple.get('cs_peaks', []), tol_cs)
                            ss_keep = _filter_peaks_by_reference(res_tm.get('ss_peaks', []), res_simple.get('ss_peaks', []), tol_ss)
                            res = dict(res_tm)
                            res['cs_peaks'] = cs_keep
                            res['ss_peaks'] = ss_keep
                            res['cs_simple_trace'] = np.asarray(res_simple.get('cs_trace', np.zeros_like(raw_proc)), dtype=float)
                            res['ss_simple_trace'] = np.asarray(res_simple.get('ss_trace', np.zeros_like(raw_proc)), dtype=float)
                            res['cs_simple_sigma'] = float(res_simple.get('sigma_cs', np.nan))
                            res['ss_simple_sigma'] = float(res_simple.get('sigma_ss', np.nan))
                            res['cs_simple_threshold_used'] = float(self.params.get('CS_THRESHOLD_SIGMA', 6.0) * res.get('cs_simple_sigma', np.nan)) if np.isfinite(res.get('cs_simple_sigma', np.nan)) else np.nan
                            res['ss_simple_threshold_used'] = float(self.params.get('SS_THRESHOLD_SIGMA', 2.0) * res.get('ss_simple_sigma', np.nan)) if np.isfinite(res.get('ss_simple_sigma', np.nan)) else np.nan
                            res['two_step_enabled'] = True
                            res['det_method'] = f"{res_tm.get('det_method', 'Template Matching')} + Two-step"
                        else:
                            res = res_tm
                            res['two_step_enabled'] = False
                    else:
                        res = process_cell_simple(raw_proc, fs,
                                                  negative_going=self.params.get('NEGATIVE_GOING', True),
                                                  cs_high_cut=self.params.get('CS_HIGH_CUT_HZ', 150.0),
                                                  cs_thresh_sigma=self.params.get('CS_THRESHOLD_SIGMA', 6.0),
                                                  cs_min_dist_ms=self.params.get('CS_MIN_DIST_MS', 25.0),
                                                  ss_low_cut=self.params.get('SS_LOW_CUT_HZ', 50.0),
                                                  ss_high_cut=self.params.get('SS_HIGH_CUT_HZ', 700.0),
                                                  ss_thresh_sigma=self.params.get('SS_THRESHOLD_SIGMA', 2.0),
                                                  ss_min_dist_ms=self.params.get('SS_MIN_DIST_MS', 2.0),
                                                  ss_blank_ms=self.params.get('SS_BLANK_MS', 8.0),
                                                  initial_blank_ms=self.params.get('INITIAL_BLANK_MS', 150.0),
                                                  use_preprocessed=True, pre_detrended=pre_detr, pre_baseline=baseline_gui,
                                                  cs_order=int(self.params.get('CS_FILTER_ORDER', 4)),
                                                  ss_order=int(self.params.get('SS_FILTER_ORDER', 4)))
                    data['results'][i] = res
                    data['spike_times_cs'][i] = (res['cs_peaks'] / fs) * 1000.0
                    data['spike_times_ss'][i] = (res['ss_peaks'] / fs) * 1000.0
                total_cells += n_cells
                sessions_done += 1
                # After finishing a session, save results to an npz file adjacent to source
                try:
                    session_src = data.get('session_path', None)
                    save_dir = None
                    base = sname
                    if session_src and os.path.isfile(session_src):
                        base = os.path.splitext(os.path.basename(session_src))[0]
                        if session_src.lower().endswith('.xlsx'):
                            # Notebook creates a folder named after the file (base) and
                            # writes <base>_analyzed.npz inside it — match that behavior.
                            save_dir = os.path.join(os.path.dirname(session_src), base)
                        else:
                            save_dir = os.path.dirname(session_src)
                    elif session_src and os.path.isdir(session_src):
                        save_dir = session_src
                    else:
                        # fallback: use master_folder or current working dir
                        save_dir = getattr(self, 'master_folder', os.getcwd()) or os.getcwd()
                    try:
                        os.makedirs(save_dir, exist_ok=True)
                    except Exception:
                        pass
                    npz_name = os.path.join(save_dir, base + '_analyzed.npz')
                    # Prepare arrays for saving to match notebook structure
                    try:
                        time_ms = np.array(data.get('time_ms', []))
                        raw_data = np.array(data.get('raw_data', []))
                        # Build cs_traces and ss_traces column-stacked if results exist
                        cs_traces = np.array([])
                        ss_traces = np.array([])
                        raw_sigmas = np.array([])
                        try:
                            res_list = data.get('results', [])
                            if res_list and len(time_ms) > 0:
                                n_samples = len(time_ms)
                                cs_traces = np.column_stack([r.get('cs_trace', np.zeros(n_samples)) if r is not None else np.zeros(n_samples) for r in res_list])
                                ss_traces = np.column_stack([r.get('ss_trace', np.zeros(n_samples)) if r is not None else np.zeros(n_samples) for r in res_list])
                                raw_sigmas = np.array([r.get('raw_sigma', np.nan) if r is not None else np.nan for r in res_list])
                            else:
                                cs_traces = np.array([])
                                ss_traces = np.array([])
                                raw_sigmas = np.array([])
                        except Exception:
                            cs_traces = np.array([])
                            ss_traces = np.array([])
                            raw_sigmas = np.array([])

                        np.savez_compressed(npz_name,
                                            time_ms=time_ms,
                                            raw_data=raw_data,
                                            cs_traces=cs_traces,
                                            ss_traces=ss_traces,
                                            spike_times_cs=np.array(data.get('spike_times_cs', []), dtype=object),
                                            spike_times_ss=np.array(data.get('spike_times_ss', []), dtype=object),
                                            cell_names=np.array(data.get('cell_names', [])),
                                            fs=float(data.get('fs', 1000.0)),
                                            raw_sigmas=raw_sigmas)
                        # update single-line status to indicate session finished
                        try:
                            if hasattr(self, 'text_stats') and self.text_stats is not None:
                                self.text_stats.setPlainText(f"Finished session {sess_idx+1}/{n_sessions}: {sname}")
                                QtWidgets.QApplication.processEvents()
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"Failed to save detection results for {sname}: {e}")
                except Exception:
                    pass
            except Exception as e:
                # continue on error with other sessions
                print(f"Detection failed for session {sname}: {e}")
                continue
        # After all sessions processed: compute overall averaged statistics across sessions/cells
        try:
            window_ms = 100
            half_win = None
            for sname, data in self.loaded_sessions.items():
                try:
                    res_list = data.get('results', [])
                    if not res_list:
                        continue
                    fs = float(data.get('fs', 1000.0))
                    tvec = np.array(data.get('time_ms', []))
                    duration_s = (tvec[-1] - tvec[0]) / 1000.0 if tvec.size>1 else np.nan
                    if half_win is None:
                        half_win = int((window_ms/2.0) * fs / 1000.0)
                    for res in res_list:
                        if res is None:
                            continue
                        try:
                            if duration_s and duration_s > 0:
                                global_cs_rates.append(len(res.get('cs_peaks', [])) / duration_s)
                                global_ss_rates.append(len(res.get('ss_peaks', [])) / duration_s)
                        except Exception:
                            pass
                        # CS events
                        try:
                            cs_peaks = np.array(res.get('cs_peaks', []), dtype=int)
                            raw_cs = res.get('cs_trace', None)
                            raw_all = res.get('detrended', None)
                            sigma_cs = res.get('sigma_cs', np.nan)
                            sigma_raw = res.get('raw_sigma', np.nan)
                            if raw_cs is not None and cs_peaks.size>0:
                                cs_event_snrs = compute_event_snrs(res, 'CS', fs, window_ms=window_ms, max_per_cell=None)
                                global_cs_snrs.extend(cs_event_snrs)
                                chosen = _select_event_bank(cs_peaks, max_per_cell=None)
                                for p in chosen:
                                    s = int(p - half_win); e = int(p + half_win)
                                    if s < 0 or e >= len(raw_cs):
                                        continue
                                    wave = raw_cs[s:e]
                                    if len(wave) != (2*half_win):
                                        continue
                                    _, interp = get_interpolated_wave(wave, fs)
                                    t_d = np.linspace(-window_ms/2.0, window_ms/2.0, len(interp))
                                    _, fwhm = get_wave_stats(interp, t_d)
                                    if not np.isnan(fwhm):
                                        global_cs_fwhm.append(fwhm)
                        except Exception:
                            pass
                        # SS events
                        try:
                            ss_peaks = np.array(res.get('ss_peaks', []), dtype=int)
                            ss_trace = res.get('ss_trace', None)
                            ss_source = ss_trace if ss_trace is not None else raw_all
                            if ss_source is not None and ss_peaks.size>0:
                                ss_event_snrs = compute_event_snrs(res, 'SS', fs, window_ms=window_ms, max_per_cell=None)
                                global_ss_snrs.extend(ss_event_snrs)
                                chosen = _select_event_bank(ss_peaks, max_per_cell=None)
                                for p in chosen:
                                    s = int(p - half_win); e = int(p + half_win)
                                    if s < 0 or e >= len(ss_source):
                                        continue
                                    wave = ss_source[s:e]
                                    if len(wave) != (2*half_win):
                                        continue
                                    _, interp = get_interpolated_wave(wave, fs)
                                    t_d = np.linspace(-window_ms/2.0, window_ms/2.0, len(interp))
                                    _, fwhm = get_wave_stats(interp, t_d)
                                    if not np.isnan(fwhm):
                                        global_ss_fwhm.append(fwhm)
                        except Exception:
                            pass
                except Exception:
                    continue
            cs_r_mean, cs_r_std, cs_r_n = mean_std_count(global_cs_rates)
            ss_r_mean, ss_r_std, ss_r_n = mean_std_count(global_ss_rates)
            cs_snr_mean, cs_snr_std, cs_snr_n = mean_std_count(global_cs_snrs)
            ss_snr_mean, ss_snr_std, ss_snr_n = mean_std_count(global_ss_snrs)
            cs_fwhm_mean, cs_fwhm_std, cs_fwhm_n = mean_std_count(global_cs_fwhm)
            ss_fwhm_mean, ss_fwhm_std, ss_fwhm_n = mean_std_count(global_ss_fwhm)

            # Replace info panel content with concise overall summary
            try:
                if hasattr(self, 'text_stats') and self.text_stats is not None:
                    summary = []
                    summary.append(f'Detection finished: {sessions_done} sessions, {total_cells} cells')
                    summary.append(f"Method: {detection_method}")
                    if detection_method == 'Template Matching':
                        summary.append(
                            f"Template mode: {self.params.get('TEMPLATE_MATCH_METHOD', 'LLR Probability Vector')} | "
                            f"CS templates: {len(self.template_store.get('cs_templates', []))}, "
                            f"SS templates: {len(self.template_store.get('ss_templates', []))}"
                        )
                        summary.append(f"Parallel matching: {'ON' if bool(self.params.get('TEMPLATE_PARALLEL', False)) else 'OFF'}")
                        summary.append(f"Two-step detection: {'ON' if bool(two_step) else 'OFF'}")
                        summary.append(
                            f"Template thresholds (Sigma x MAD): CS sigma={self.params.get('TEMPLATE_CS_SIGMA', 6.0):.2f}, "
                            f"SS sigma={self.params.get('TEMPLATE_SS_SIGMA', 4.0):.2f}"
                        )
                    summary.append('Overall CS:')
                    summary.append(f'  Rate: {cs_r_mean:.2f}±{cs_r_std:.2f} Hz (cells: {cs_r_n}) | SNR: {cs_snr_mean:.2f}±{cs_snr_std:.2f} (events: {cs_snr_n}) | FWHM: {cs_fwhm_mean:.2f}±{cs_fwhm_std:.2f} ms (events: {cs_fwhm_n})')
                    summary.append('Overall SS:')
                    summary.append(f'  Rate: {ss_r_mean:.2f}±{ss_r_std:.2f} Hz (cells: {ss_r_n}) | SNR: {ss_snr_mean:.2f}±{ss_snr_std:.2f} (events: {ss_snr_n}) | FWHM: {ss_fwhm_mean:.2f}±{ss_fwhm_std:.2f} ms (events: {ss_fwhm_n})')
                    self.text_stats.setPlainText('\n'.join(summary))
            except Exception:
                pass
        except Exception:
            pass

        return sessions_done, total_cells

    def open_detection_viewer(self):
        # prefer using currently selected session data from loaded_sessions
        data = None
        if self.selected_session and self.selected_session in self.loaded_sessions:
            data = self.loaded_sessions[self.selected_session]
        elif self.data is not None:
            data = self.data
        if data is None or 'results' not in data:
            QMessageBox.warning(self, 'No detection', 'Run detection first (Spike Detection).')
            return
        # Ensure results contain at least one non-empty entry for plotting
        try:
            results = data.get('results', [])
            if not any([r is not None for r in results]):
                QMessageBox.warning(self, 'No detection', 'Run detection first (Spike Detection). No valid results found.')
                return
        except Exception:
            QMessageBox.warning(self, 'No detection', 'Run detection first (Spike Detection).')
            return
        dlg = DetectionViewerDialog(data, parent=self)
        dlg.exec()

    def open_stats_viewer(self):
        if self.data is None or 'results' not in self.data:
            QMessageBox.warning(self, 'No detection', 'Run detection first (Run Detection (all cells)).')
            return
        dlg = StatsViewerDialog(self.data, parent=self)
        dlg.exec()

    def open_template_viewer(self):
        dlg = TemplateViewerDialog(self.template_store, parent=self)
        dlg.exec()

    def reset_app(self):
        try:
            buttons = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            reply = QMessageBox.question(self, 'Reset App', 'Reset application state and clear loaded sessions?', buttons)
            yes_btn = QMessageBox.StandardButton.Yes
        except Exception:
            # Fallback for bindings where StandardButton namespace differs
            buttons = QMessageBox.Yes | QMessageBox.No if hasattr(QMessageBox, 'Yes') else 1 | 0
            reply = QMessageBox.question(self, 'Reset App', 'Reset application state and clear loaded sessions?', buttons)
            yes_btn = QMessageBox.Yes if hasattr(QMessageBox, 'Yes') else 1
        if reply != yes_btn:
            return
        # Clear loaded data and UI
        try:
            self.loaded_sessions = {}
            self.sessions = []
            self.session_names = []
            self.list_sessions.clear()
            self.combo_cell.clear()
            self.data = None
            self.selected_session = None
            self.selected_cell = 0
            # reset params to defaults
            self.params = {
                'NEGATIVE_GOING': True,
                'DETECTION_METHOD': 'Threshold',
                'CS_HIGH_CUT_HZ': 150.0,
                'CS_THRESHOLD_SIGMA': 6.0,
                'CS_MIN_DIST_MS': 25.0,
                'SS_LOW_CUT_HZ': 50.0,
                'SS_HIGH_CUT_HZ': 700.0,
                'SS_THRESHOLD_SIGMA': 2.0,
                'SS_MIN_DIST_MS': 2.0,
                'SS_BLANK_MS': 8.0,
                'INITIAL_BLANK_MS': 150.0,
                'CS_FILTER_ORDER': 4,
                'SS_FILTER_ORDER': 4,
                'FRAME_PROCESSING_MODE': 'Rolling average',
                'TEMPLATE_CS_WINDOW_MS': 30.0,
                'TEMPLATE_SS_WINDOW_MS': 8.0,
                'TEMPLATE_MATCH_METHOD': 'LLR Probability Vector',
                'TEMPLATE_PARALLEL': False,
                'TEMPLATE_CS_SIGMA': 6.0,
                'TEMPLATE_SS_SIGMA': 4.0,
            }
            self.template_store = {
                'cs_templates': [],
                'ss_templates': [],
                'fs_cs': [],
                'fs_ss': [],
                'cs_sources': [],
                'ss_sources': [],
            }
            # reset baseline params and UI controls
            self.baseline_params = {
                'method': 'Median',
                'window_ms': 40.0,
                'percentile': 20.0,
                'sgolay_polyorder': 3,
            }
            self.combo_baseline_method.setCurrentText(self.baseline_params['method'])
            self.spin_baseline_window.setValue(self.baseline_params['window_ms'])
            self.spin_baseline_percentile.setValue(self.baseline_params['percentile'])
            self.spin_sgolay_polyorder.setValue(self.baseline_params['sgolay_polyorder'])
            # reset detection UI controls
            try:
                self.spin_cs_high.setValue(self.params['CS_HIGH_CUT_HZ'])
                self.spin_cs_thresh.setValue(self.params['CS_THRESHOLD_SIGMA'])
                # CS min dist moved to Advanced Settings — keep parent.params in sync
                # ensure main UI does not refer to removed widget
                self.spin_ss_low.setValue(self.params['SS_LOW_CUT_HZ'])
                self.spin_ss_high.setValue(self.params['SS_HIGH_CUT_HZ'])
                self.spin_ss_thresh.setValue(self.params['SS_THRESHOLD_SIGMA'])
                self.spin_ss_mind.setValue(self.params['SS_MIN_DIST_MS'])
                self.spin_ss_blank.setValue(self.params['SS_BLANK_MS'])
                self.combo_frame_processing.setCurrentText(self.params['FRAME_PROCESSING_MODE'])
                self.combo_template_method.setCurrentText(self.params['TEMPLATE_MATCH_METHOD'])
                if hasattr(self, 'chk_template_parallel') and self.chk_template_parallel is not None:
                    self.chk_template_parallel.setChecked(bool(self.params.get('TEMPLATE_PARALLEL', False)))
                self.spin_template_cs_sigma.setValue(self.params['TEMPLATE_CS_SIGMA'])
                self.spin_template_ss_sigma.setValue(self.params['TEMPLATE_SS_SIGMA'])
            except Exception:
                pass
            try:
                if hasattr(self, 'tabs_detection') and self.tabs_detection is not None:
                    self.tabs_detection.setCurrentIndex(0)
                self.update_template_status_label()
            except Exception:
                pass
            # clear canvas and stats
            try:
                self.canvas.figure.clf()
                self.canvas.ax = self.canvas.figure.add_subplot(111)
                self.canvas.draw()
            except Exception:
                pass
            try:
                self.text_stats.clear()
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, 'Reset Error', f'Failed to reset application: {e}')

    def refresh_sessions(self):
        if not self.master_folder:
            QMessageBox.warning(self, 'No folder', 'Please select a master folder first.')
            return
        candidates = []
        names = []
        # 1) Top-level files
        top_xls = sorted(glob.glob(os.path.join(self.master_folder, '*.xlsx')))
        top_csv = sorted(glob.glob(os.path.join(self.master_folder, '*.csv')))
        for f in top_xls + top_csv:
            candidates.append(f)
            names.append(os.path.basename(f))
        # 2) Subfolders
        for d in sorted(glob.glob(os.path.join(self.master_folder, '*'))):
            if os.path.isdir(d):
                xls_list = glob.glob(os.path.join(d, '*.xlsx'))
                csv_list = glob.glob(os.path.join(d, '*.csv'))
                if xls_list or csv_list:
                    candidates.append(d)
                    names.append(os.path.basename(d))
        self.sessions = candidates
        self.session_names = names
        # populate compact combo box
        self.list_sessions.clear()
        for n in self.session_names:
            self.list_sessions.addItem(n)
        self.loaded_sessions = {}
        if self.sessions:
            # set current index and explicitly call handler to load session
            self.list_sessions.setCurrentIndex(0)
            try:
                self.on_session_select(0)
            except Exception:
                pass

    def on_session_select(self, idx=None):
        # idx may be passed from QComboBox signal or we read currentIndex
        try:
            if idx is None:
                idx = int(self.list_sessions.currentIndex())
        except Exception:
            return
        if idx < 0 or idx >= len(self.sessions):
            return
        session_path = self.sessions[idx]
        session_name = self.session_names[idx]
        self.selected_session = session_name
        if session_name not in self.loaded_sessions:
            data = self.load_session(session_path)
            if data is not None:
                self.loaded_sessions[session_name] = data
        # set current data reference for single-session visualizations
        if self.selected_session in self.loaded_sessions:
            self.data = self.loaded_sessions[self.selected_session]
        self.update_cell_selector()
        self.update_plot()

    def load_session(self, session_path):
        try:
            return load_session_path(session_path, default_fs=float(self.params.get('FS', 1000.0)))
        except FileNotFoundError as e:
            QMessageBox.warning(self, 'No data', str(e))
            return None
        except ValueError as e:
            QMessageBox.warning(self, 'Unsupported file', str(e))
            return None
        except Exception as e:
            QMessageBox.warning(self, 'Load error', f'Failed to load session: {e}')
            return None

    def update_cell_selector(self):
        self.combo_cell.clear()
        if self.selected_session and self.selected_session in self.loaded_sessions:
            data = self.loaded_sessions[self.selected_session]
            for name in data['cell_names']:
                self.combo_cell.addItem(name)
            self.combo_cell.setCurrentIndex(0)
            self.selected_cell = 0

    def on_cell_change(self):
        self.selected_cell = self.combo_cell.currentIndex()
        self.update_plot()

    def update_info_text(self):
        if not self.selected_session or self.selected_session not in self.loaded_sessions:
            return
        data = self.loaded_sessions[self.selected_session]
        idx = max(0, self.selected_cell)
        t = data['time_ms']
        duration_s = (t[-1] - t[0]) / 1000.0
        n_samples = len(t)
        fs = float(data['fs'])
        txt = f"Session: {self.selected_session}\n"
        txt += f"Cell: {data['cell_names'][idx]}\n"
        txt += f"Duration: {duration_s:.2f} s\n"
        txt += f"Samples: {n_samples}\n"
        txt += f"Fs: {fs:.1f} Hz"
        self.text_stats.setPlainText(txt)

    def on_slider_time_changed(self):
        self.slider_position = self.slider_time.value() / 1000.0
        self.update_plot()

    def on_window_changed(self):
        # Update window size from control and refresh plot
        try:
            self.window_ms = float(self.spin_window.value())
        except Exception:
            self.window_ms = float(self.window_ms)
        self.update_plot()

    def update_plot(self):
        self.canvas.figure.clf()
        ax = self.canvas.figure.add_subplot(111)
        # ensure top and right spines are hidden even when no data is plotted
        try:
            ax.spines['top'].set_visible(False)
        except Exception:
            pass
        try:
            ax.spines['right'].set_visible(False)
        except Exception:
            pass
        if not self.selected_session or self.selected_session not in self.loaded_sessions:
            ax.text(0.5, 0.5, 'No data loaded', ha='center')
            self.canvas.draw()
            return
        data = self.loaded_sessions[self.selected_session]
        idx = max(0, self.selected_cell)
        raw = data['raw_data'][:, idx]
        t = data['time_ms']
        fs = float(data['fs'])
        # Apply averaging (smoothing) if requested, then compute baseline on processed signal
        frames = int(self.spin_avg_frames.value()) if hasattr(self, 'spin_avg_frames') else 0
        raw_proc = apply_frame_processing(raw, frames=frames, mode=self._get_frame_processing_mode())
        baseline = self.compute_baseline(raw_proc, fs)
        # Sliding window using center slider + window size
        t_min, t_max = t[0], t[-1]
        center_pct = float(self.slider_time.value()) / 1000.0 if hasattr(self, 'slider_time') else 0.0
        total_ms = t_max - t_min
        center_time = t_min + center_pct * total_ms
        window_ms = float(self.spin_window.value()) if hasattr(self, 'spin_window') else (500.0)
        start_time = center_time - window_ms / 2.0
        end_time = center_time + window_ms / 2.0
        # Clamp to valid range
        if start_time < t_min:
            start_time = t_min
            end_time = min(t_min + window_ms, t_max)
        if end_time > t_max:
            end_time = t_max
            start_time = max(t_min, t_max - window_ms)
        mask = (t >= start_time) & (t <= end_time)
        # Plot: either raw + baseline OR baseline-corrected (replace raw)
        # If baseline method is 'Disable', do not apply or show baseline correction
        # Plot selection: raw / baseline-corrected / CS-filter / SS-filter (CS/SS mutually exclusive)
        method = self.baseline_params.get('method', '')
        use_baseline_vis = (method != 'Disable' and getattr(self, 'chk_show_baseline', None) and self.chk_show_baseline.isChecked())
        try:
            vis_source = (raw_proc - baseline) if use_baseline_vis else raw_proc
        except Exception:
            vis_source = raw_proc

        # If any of the visualization checkboxes are active, force black plotting
        # and suppress figure legend for clarity per user request.
        try:
            any_vis_override = False
            if (getattr(self, 'chk_show_baseline', None) and self.chk_show_baseline.isChecked()) or \
               (getattr(self, 'chk_show_cs', None) and self.chk_show_cs.isChecked()) or \
               (getattr(self, 'chk_show_ss', None) and self.chk_show_ss.isChecked()):
                any_vis_override = True
            # Determine the main display color when override is active (follow raw color scheme)
            if any_vis_override:
                try:
                    color_main = self.colors.get('raw', '#333333')
                except Exception:
                    color_main = '#333333'
            else:
                color_main = None
        except Exception:
            any_vis_override = False
            color_main = None

        # If CS filter visualization is requested
        if getattr(self, 'chk_show_cs', None) and self.chk_show_cs.isChecked():
            try:
                cs_vis = apply_filter(vis_source, fs, low=None, high=float(self.spin_cs_high.value()), order=int(self.spin_cs_order.value()))
                vis_for_ylim = cs_vis
                plot_color = color_main if color_main is not None else self.colors.get('baseline', '#FFC20A')
                ax.plot(t[mask], cs_vis[mask], color=plot_color, lw=_get_linewidth(1.2), label='CS filter')
            except Exception:
                vis_for_ylim = vis_source
                plot_color = color_main if color_main is not None else self.colors.get('raw', '#333333')
                ax.plot(t[mask], vis_source[mask], color=plot_color, lw=_get_linewidth(1), label='Raw')

        # If SS filter visualization is requested
        elif getattr(self, 'chk_show_ss', None) and self.chk_show_ss.isChecked():
            try:
                ss_vis = apply_filter(vis_source, fs, low=float(self.spin_ss_low.value()), high=float(self.spin_ss_high.value()), order=int(self.spin_ss_order.value()))
                vis_for_ylim = ss_vis
                plot_color = color_main if color_main is not None else '#2ca02c'
                ax.plot(t[mask], ss_vis[mask], color=plot_color, lw=_get_linewidth(1.2), label='SS filter')
            except Exception:
                vis_for_ylim = vis_source
                plot_color = color_main if color_main is not None else self.colors.get('raw', '#333333')
                ax.plot(t[mask], vis_source[mask], color=plot_color, lw=_get_linewidth(1), label='Raw')

        else:
            # Default: plot raw or baseline-corrected depending on checkbox
            vis_for_ylim = vis_source
            lbl = 'Baseline-corrected' if use_baseline_vis else 'Raw'
            plot_color = color_main if color_main is not None else self.colors.get('raw', '#333333')
            ax.plot(t[mask], vis_source[mask], color=plot_color, alpha=0.9, lw=_get_linewidth(1), label=lbl)
            # Optionally overlay baseline (when baseline method active and available)
            try:
                # Hide baseline overlay when simplified/black view is active
                if not any_vis_override and method != 'Disable' and baseline is not None:
                    ax.plot(t[mask], baseline[mask], color=self.colors.get('baseline', '#FFC20A'), ls='--', lw=_get_linewidth(2), label='Baseline')
            except Exception:
                pass

        # Draw shaded selected template intervals (current type)
        try:
            for tp, col in [('CS', '#009E73'), ('SS', '#D55E00')]:
                intervals = self.template_selected_intervals_ms.get(tp, []) if hasattr(self, 'template_selected_intervals_ms') else []
                for s_ms, e_ms in intervals:
                    if e_ms < start_time or s_ms > end_time:
                        continue
                    ax.axvspan(max(s_ms, start_time), min(e_ms, end_time), color=col, alpha=0.18, lw=0)
        except Exception:
            pass
        ax.set_title(f"{data['cell_names'][idx]} Raw Trace", fontweight='bold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Signal')
        # tidy legend (no frame) unless user requested simplified black view
        try:
            if not any_vis_override:
                ax.legend(frameon=False)
        except Exception:
            pass
        try:
            ax.spines['top'].set_visible(False)
        except Exception:
            pass
        try:
            ax.spines['right'].set_visible(False)
        except Exception:
            pass
        # Y-Range semantics: if spin_zoom (Y-Range) > 0, center plot on trace and show that full range.
        # If 0 => auto-scale with padding.
        y_range = float(self.spin_zoom.value()) if hasattr(self, 'spin_zoom') else 0.0
        if np.any(mask):
            v = vis_for_ylim[mask]
            vmin = np.min(v); vmax = np.max(v)
            span = vmax - vmin
            local_std = np.std(v) if len(v) > 0 else 1.0
            if y_range <= 0.0:
                pad = max(span * 0.05, local_std * 0.5, 1e-9)
                ax.set_ylim(vmin - pad, vmax + pad)
            else:
                center = 0.5 * (vmax + vmin)
                half_range = float(y_range) / 2.0
                ax.set_ylim(center - half_range, center + half_range)
            ax.set_xlim(start_time, end_time)
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.update_info_text()
    def compute_baseline(self, trace, fs):
        method = self.baseline_params['method']
        window_ms = self.baseline_params['window_ms']
        window_samples = int(window_ms * fs / 1000.0)
        # safety caps to avoid extreme window sizes that kill performance
        max_samples_cap = max(5, min(len(trace)//2, 5000))
        if window_samples < 5:
            window_samples = 5
        if window_samples > max_samples_cap:
            window_samples = max_samples_cap
        if method == 'Percentile':
            percentile = self.baseline_params['percentile']
            baseline = percentile_filter(trace, percentile, size=window_samples)
        elif method == 'Median':
            from scipy.ndimage import median_filter
            baseline = median_filter(trace, size=window_samples)
        elif method == 'Savitzky-Golay':
            from scipy.signal import savgol_filter
            polyorder = self.baseline_params['sgolay_polyorder']
            if window_samples % 2 == 0:
                window_samples += 1
            if window_samples <= polyorder:
                window_samples = polyorder + 2 + (polyorder % 2)
            baseline = savgol_filter(trace, window_length=window_samples, polyorder=polyorder)
        else:
            baseline = trace * 0
        return baseline

    def export_figure(self):
        if self.data is None:
            QMessageBox.warning(self, 'No data', 'Load a session and plot something first.')
            return
        try:
            filename = save_figure_with_dialog(
                lambda title, default, flt: QFileDialog.getSaveFileName(self, title, default, flt),
                self.canvas.figure,
                default_name='figure.svg',
            )
            if filename:
                QMessageBox.information(self, 'Saved', f'Figure saved as {filename}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save figure: {e}')
class DetectionSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Detection Settings')
        self.resize(*_scaled_size(400, 400))
        self.parent_win = parent
        layout = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        p = parent.params if parent is not None else {}
        # Filtering settings
        self.spin_cs_high = QtWidgets.QDoubleSpinBox(); self.spin_cs_high.setRange(0.0, 5000.0); self.spin_cs_high.setValue(p.get('CS_HIGH_CUT_HZ',150.0))
        self.spin_cs_thresh = QtWidgets.QDoubleSpinBox(); self.spin_cs_thresh.setRange(0.1, 50.0); self.spin_cs_thresh.setValue(p.get('CS_THRESHOLD_SIGMA',6.0))
        self.spin_ss_low = QtWidgets.QDoubleSpinBox(); self.spin_ss_low.setRange(0.0,5000.0); self.spin_ss_low.setValue(p.get('SS_LOW_CUT_HZ',50.0))
        self.spin_ss_high = QtWidgets.QDoubleSpinBox(); self.spin_ss_high.setRange(0.0,5000.0); self.spin_ss_high.setValue(p.get('SS_HIGH_CUT_HZ',700.0))
        self.spin_ss_thresh = QtWidgets.QDoubleSpinBox(); self.spin_ss_thresh.setRange(0.1,50.0); self.spin_ss_thresh.setValue(p.get('SS_THRESHOLD_SIGMA',2.0))
        form.addRow('CS high cut (Hz):', self.spin_cs_high)
        form.addRow('CS threshold (sigma):', self.spin_cs_thresh)
        # CS min dist is managed in Advanced Settings (SettingsDialog)
        form.addRow('SS low cut (Hz):', self.spin_ss_low)
        form.addRow('SS high cut (Hz):', self.spin_ss_high)
        form.addRow('SS threshold (sigma):', self.spin_ss_thresh)
        # SS min dist / SS blank moved to Advanced Settings (use parent.params)
        layout.addLayout(form)
        btns = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton('OK'); cancel = QtWidgets.QPushButton('Cancel')
        ok.clicked.connect(self.accept); cancel.clicked.connect(self.reject)
        btns.addStretch(1); btns.addWidget(ok); btns.addWidget(cancel)
        layout.addLayout(btns)
        self.setLayout(layout)
    def get_params(self):
        return {
            'CS_HIGH_CUT_HZ': float(self.spin_cs_high.value()),
            'CS_THRESHOLD_SIGMA': float(self.spin_cs_thresh.value()),
            'CS_MIN_DIST_MS': float(self.parent_win.params.get('CS_MIN_DIST_MS', 25.0)) if hasattr(self, 'parent_win') and self.parent_win is not None else 25.0,
            'SS_LOW_CUT_HZ': float(self.spin_ss_low.value()),
            'SS_HIGH_CUT_HZ': float(self.spin_ss_high.value()),
            'SS_THRESHOLD_SIGMA': float(self.spin_ss_thresh.value()),
            'SS_MIN_DIST_MS': float(self.parent_win.params.get('SS_MIN_DIST_MS', 2.0)) if hasattr(self, 'parent_win') and self.parent_win is not None else 2.0,
            'SS_BLANK_MS': float(self.parent_win.params.get('SS_BLANK_MS', 8.0)) if hasattr(self, 'parent_win') and self.parent_win is not None else 8.0,
        }
    # (Moved compute_baseline and export_figure into MainWindow)


class SliderViewerDialog(QtWidgets.QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Slider Viewer')
        # Base scaled size, then shrink further on low-resolution screens (e.g., 1080p)
        w, h = _scaled_size(1020, 700)
        try:
            app = QtWidgets.QApplication.instance()
            screen = app.primaryScreen() if app is not None else None
            scr_h = None
            if screen is not None:
                try:
                    scr_h = int(screen.size().height())
                except Exception:
                    try:
                        scr_h = int(screen.geometry().height())
                    except Exception:
                        scr_h = None
        except Exception:
            scr_h = None
        # If low-resolution screen, enforce a 30% smaller logical size
        s = _get_screen_scale()
        if scr_h is not None and scr_h <= 1080:
            # compute target size based on logical pixels (reduce influence of DPI)
            w = int(1000 * 0.7 / max(1.0, s))
            h = int(700 * 0.7 / max(1.0, s))
        else:
            self.resize(w, h)
            # when not low-res, already resized above; return early
            # (keep existing resize behavior)
            pass
        # apply final resize for low-res branch
        try:
            if scr_h is not None and scr_h <= 1080:
                self.resize(w, h)
        except Exception:
            self.resize(w, h)
        self.data = data
        self.fig = _make_figure(8, 6)
        self.canvas = FigureCanvas(self.fig)
        self.ax_raw = self.fig.add_subplot(311)
        self.ax_cs = self.fig.add_subplot(312, sharex=self.ax_raw)
        self.ax_ss = self.fig.add_subplot(313, sharex=self.ax_raw)

        # Controls
        self.slider_time = QtWidgets.QSlider()
        self.slider_time.setOrientation(ORIENT_HORIZONTAL)
        self.slider_time.setMinimum(0)
        self.slider_time.setMaximum(1000)
        self.slider_time.setValue(200)
        self.slider_time.valueChanged.connect(self.update_plot)

        self.spin_zoom = QtWidgets.QDoubleSpinBox(); self.spin_zoom.setRange(0.0, 1e6); self.spin_zoom.setValue(0.0)
        self.spin_zoom.valueChanged.connect(self.update_plot)

        # Controls row (move on top of plot)
        layout = QtWidgets.QVBoxLayout()
        hl = QtWidgets.QHBoxLayout(); hl.addWidget(QtWidgets.QLabel('Time:')); hl.addWidget(self.slider_time)
        # start time label
        self.lbl_start_time = QtWidgets.QLabel('Start: 0 ms')
        hl.addWidget(self.lbl_start_time)
        hl.addWidget(QtWidgets.QLabel('Y-Range (0=auto):')); hl.addWidget(self.spin_zoom)
        # Export button
        btn_export = QtWidgets.QPushButton('Save Figure')
        btn_export.clicked.connect(self.save_figure)
        hl.addWidget(btn_export)
        layout.addLayout(hl)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # default cell
        self.cell_idx = 0
        self.update_plot()

    def update_plot(self):
        t = self.data['time_ms']
        raw = self.data['raw_data'][:, self.cell_idx]
        fs = float(self.data['fs'])
        center_pct = self.slider_time.value() / 1000.0
        n = len(t)
        win_ms = 500.0
        total_ms = t[-1] - t[0]
        center_time = t[0] + center_pct * total_ms
        start_time = center_time - win_ms/2.0
        mask = (t >= start_time) & (t <= center_time + win_ms/2.0)
        # update start time label
        try:
            self.lbl_start_time.setText(f'Start: {int(start_time)} ms')
        except Exception:
            pass
        zoom = float(self.spin_zoom.value())

        self.fig.suptitle(f"Cell: {self.data['cell_names'][self.cell_idx]}")
        parent = self.parent()
        colors = parent.colors if parent is not None else {'raw':'#333333','drift':'#FFC20A','cs':'#009E73','ss':'#D55E00','cs_th':'#56B4E9','ss_th':'#CC79A7'}
        params = parent.params if parent is not None else {}
        self.ax_raw.clear(); self.ax_cs.clear(); self.ax_ss.clear()
        # apply frame-based averaging for visualization if enabled
        frames = int(parent.spin_avg_frames.value()) if (parent is not None and hasattr(parent, 'spin_avg_frames')) else 0
        mode = parent._get_frame_processing_mode() if (parent is not None and hasattr(parent, '_get_frame_processing_mode')) else 'Rolling average'
        raw_viz = apply_frame_processing(raw, frames=frames, mode=mode)

        # compute baseline using GUI baseline settings (after averaging)
        baseline_gui = parent.compute_baseline(raw_viz, fs) if (parent is not None and hasattr(parent, 'compute_baseline')) else np.zeros_like(raw_viz)
        detrended_gui = raw_viz - baseline_gui
        pre_detr_viz = detrended_gui * (-1.0 if params.get('NEGATIVE_GOING', True) else 1.0)
        det = process_cell_simple(raw_viz, fs,
                  negative_going=params.get('NEGATIVE_GOING', True),
                  cs_high_cut=params.get('CS_HIGH_CUT_HZ', 150.0),
                  cs_thresh_sigma=params.get('CS_THRESHOLD_SIGMA', 6.0),
                  cs_min_dist_ms=params.get('CS_MIN_DIST_MS', 25.0),
                  ss_low_cut=params.get('SS_LOW_CUT_HZ', 50.0),
                  ss_high_cut=params.get('SS_HIGH_CUT_HZ', 700.0),
                  ss_thresh_sigma=params.get('SS_THRESHOLD_SIGMA', 2.0),
                  ss_min_dist_ms=params.get('SS_MIN_DIST_MS', 2.0),
                  ss_blank_ms=params.get('SS_BLANK_MS', 8.0),
                  use_preprocessed=True, pre_detrended=pre_detr_viz, pre_baseline=baseline_gui)

        # baseline_display should be the baseline estimated for the raw (after averaging).
        # Use the GUI baseline (computed above) or the one returned by detection if present.
        baseline_display = det.get('baseline', baseline_gui) if det.get('baseline', None) is not None else baseline_gui
        # show corrected-only or raw+baseline depending on checkbox
        if getattr(parent, 'chk_show_baseline', None) and parent.chk_show_baseline.isChecked():
            corrected = raw_viz - det['baseline']
            vis_signal = corrected
            self.ax_raw.plot(t[mask], vis_signal[mask], color=colors.get('raw', '#333333'), lw=_get_linewidth(1))
        else:
            vis_signal = raw_viz
            self.ax_raw.plot(t[mask], raw_viz[mask], color=colors.get('raw', '#333333'), lw=_get_linewidth(1))
            self.ax_raw.plot(t[mask], baseline_display[mask], color=colors.get('baseline', '#FFC20A'), ls='--', lw=_get_linewidth(1.6))

        self.ax_cs.plot(t[mask], det['cs_trace'][mask], color=colors.get('cs_trace', '#009E73'))
        self.ax_cs.axhline(params.get('CS_THRESHOLD_SIGMA', 6.0) * det['sigma_cs'], color=colors.get('cs_thresh', '#56B4E9'), ls='--')
        self.ax_ss.plot(t[mask], det['ss_trace'][mask], color=colors.get('ss_trace', '#D55E00'))
        self.ax_ss.axhline(params.get('SS_THRESHOLD_SIGMA', 2.0) * det['sigma_ss'], color=colors.get('ss_thresh', '#CC79A7'), ls='--')

        # overlay spikes
        cs_times = (det['cs_peaks'] / fs) * 1000.0
        ss_times = (det['ss_peaks'] / fs) * 1000.0
        self.ax_raw.plot(cs_times, [np.max(vis_signal)] * len(cs_times), 'x', color=colors.get('cs_trace', '#009E73'))
        self.ax_raw.plot(ss_times, [np.min(vis_signal)] * len(ss_times), '|', color=colors.get('ss_trace', '#D55E00'))

        self.ax_raw.set_xlim(center_time - win_ms/2.0, center_time + win_ms/2.0)
        if np.any(mask):
            v_lim = vis_signal[mask]
            self.ax_raw.set_ylim(np.min(v_lim) - 0.1*np.ptp(v_lim), np.max(v_lim) + 0.1*np.ptp(v_lim))

        # Apply Y-Range uniformly to raw/CS/SS axes (same semantics as main & detection viewers)
        y_range = float(self.spin_zoom.value()) if hasattr(self, 'spin_zoom') else 0.0
        if np.any(mask):
            vmin = np.min(vis_signal[mask]); vmax = np.max(vis_signal[mask])
            span = vmax - vmin
            local_std = np.std(vis_signal[mask]) if len(vis_signal[mask])>0 else 1.0
            if y_range <= 0.0:
                pad = max(span * 0.05, local_std * 0.5, 1e-9)
                ylim = (vmin - pad, vmax + pad)
            else:
                center = 0.5 * (vmax + vmin)
                half_range = float(y_range) / 2.0
                ylim = (center - half_range, center + half_range)
        else:
            ylim = (-1.0, 1.0)
        for a in [self.ax_raw, self.ax_cs, self.ax_ss]:
            a.set_ylim(ylim)

        self.canvas.draw()

    def save_figure(self):
        try:
            filename = save_figure_with_dialog(
                lambda title, default, flt: QFileDialog.getSaveFileName(self, title, default, flt),
                self.canvas.figure,
                default_name='figure.svg',
            )
            if filename:
                QMessageBox.information(self, 'Saved', f'Figure saved as {filename}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save figure: {e}')


class DetectionViewerDialog(QtWidgets.QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Detection Viewer')
        self.resize(*_scaled_size(1000, 700))
        self.data = data
        self.fig = _make_figure(8, 6)
        self.canvas = FigureCanvas(self.fig)
        self._viewer_mode = None
        self._build_axes(two_step_mode=False)

        # Controls: time slider, window, zoom
        # Session and cell selectors (allow switching between loaded sessions/cells)
        self.combo_session = QtWidgets.QComboBox()
        parent = self.parent()
        session_names = []
        try:
            if parent is not None and hasattr(parent, 'session_names') and parent.session_names:
                session_names = list(parent.session_names)
            elif parent is not None and hasattr(parent, 'loaded_sessions'):
                session_names = list(parent.loaded_sessions.keys())
        except Exception:
            session_names = []
        self.combo_session.addItems(session_names)
        # try to set current session name matching provided data
        try:
            cur_name = None
            if parent is not None and parent.selected_session:
                cur_name = parent.selected_session
            elif isinstance(self.data, dict) and 'session_name' in self.data:
                cur_name = self.data.get('session_name')
            if cur_name is not None and cur_name in session_names:
                self.combo_session.setCurrentText(cur_name)
        except Exception:
            pass
        self.combo_session.currentTextChanged.connect(self._on_session_changed)

        self.combo_cell = QtWidgets.QComboBox()
        # populate cells based on initial data
        try:
            n_cells = int(self.data['raw_data'].shape[1]) if (self.data is not None and 'raw_data' in self.data) else 0
            cell_names = self.data.get('cell_names', [f'Cell {i}' for i in range(n_cells)]) if isinstance(self.data, dict) else []
            self.combo_cell.addItems(cell_names)
        except Exception:
            pass
        self.combo_cell.currentIndexChanged.connect(self._on_cell_changed)

        self.slider_time = QtWidgets.QSlider()
        self.slider_time.setOrientation(ORIENT_HORIZONTAL)
        self.slider_time.setMinimum(0)
        self.slider_time.setMaximum(1000)
        self.slider_time.setValue(200)
        self.slider_time.valueChanged.connect(self.plot_detection)

        self.spin_window = QtWidgets.QDoubleSpinBox(); self.spin_window.setRange(10.0, 10000.0); self.spin_window.setValue(500.0)
        self.spin_window.valueChanged.connect(self.plot_detection)
        self.spin_zoom = QtWidgets.QDoubleSpinBox(); self.spin_zoom.setRange(0.0, 1e6); self.spin_zoom.setValue(0.0)
        self.spin_zoom.valueChanged.connect(self.plot_detection)

        # Two-row control panel to avoid overly wide layout
        ctrl_top = QtWidgets.QHBoxLayout()
        ctrl_top.addWidget(QtWidgets.QLabel('Session:'))
        ctrl_top.addWidget(self.combo_session)
        ctrl_top.addWidget(QtWidgets.QLabel('Cell:'))
        ctrl_top.addWidget(self.combo_cell)
        ctrl_top.addWidget(QtWidgets.QLabel('Time'))
        ctrl_top.addWidget(self.slider_time)

        ctrl_bottom = QtWidgets.QHBoxLayout()
        # keep start time label on the bottom row to show current window start
        self.lbl_start_time = QtWidgets.QLabel('Start: 0 ms')
        ctrl_bottom.addWidget(self.lbl_start_time)
        ctrl_bottom.addWidget(QtWidgets.QLabel('Window (ms):'))
        ctrl_bottom.addWidget(self.spin_window)
        ctrl_bottom.addSpacing(8)
        ctrl_bottom.addWidget(QtWidgets.QLabel('Scale (Y-Range):'))
        ctrl_bottom.addWidget(self.spin_zoom)
        btn_extract_tpl = QtWidgets.QPushButton('Export as templates')
        btn_extract_tpl.clicked.connect(self.export_detected_templates)
        ctrl_bottom.addWidget(btn_extract_tpl)
        btn_export = QtWidgets.QPushButton('Save Figure')
        btn_export.clicked.connect(self.save_figure)
        ctrl_bottom.addStretch(1)
        ctrl_bottom.addWidget(btn_export)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(ctrl_top)
        layout.addLayout(ctrl_bottom)
        layout.addWidget(self.canvas)
        # Make info (ctrl_bottom) taller by increasing its stretch relative to canvas.
        # Stretch indices: 0=ctrl_top,1=ctrl_bottom,2=canvas
        layout.setStretch(0, 0)
        # restore detection viewer info panel stretch to previous value
        layout.setStretch(1, 3)
        layout.setStretch(2, 4)
        self.setLayout(layout)

        # default to first cell
        self.cell_idx = 0
        self.plot_detection()

    def _build_axes(self, two_step_mode=False):
        self.fig.clf()
        if two_step_mode:
            self.resize(*_scaled_size(1000, 840))
            gs = self.fig.add_gridspec(6, 1, height_ratios=[1.05, 1.0, 1.0, 1.0, 1.0, 1.2], hspace=0.06)
            self.ax_raw_corr = self.fig.add_subplot(gs[0, 0])
            self.ax_cs_score = self.fig.add_subplot(gs[1, 0], sharex=self.ax_raw_corr)
            self.ax_ss_score = self.fig.add_subplot(gs[2, 0], sharex=self.ax_raw_corr)
            self.ax_cs_simple = self.fig.add_subplot(gs[3, 0], sharex=self.ax_raw_corr)
            self.ax_ss_simple = self.fig.add_subplot(gs[4, 0], sharex=self.ax_raw_corr)
            self.ax_raw_spikes = self.fig.add_subplot(gs[5, 0], sharex=self.ax_raw_corr)
            self._viewer_mode = 'two_step'
        else:
            self.resize(*_scaled_size(1000, 700))
            gs = self.fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.2], hspace=0.05)
            self.ax_raw_top = self.fig.add_subplot(gs[0, 0])
            self.ax_cs = self.fig.add_subplot(gs[1, 0], sharex=self.ax_raw_top)
            self.ax_ss = self.fig.add_subplot(gs[2, 0], sharex=self.ax_raw_top)
            self.ax_raw_bot = self.fig.add_subplot(gs[3, 0], sharex=self.ax_raw_top)
            self._viewer_mode = 'standard'

    def _on_session_changed(self, text):
        # Switch to a different session from the parent loaded sessions
        parent = self.parent()
        if parent is None:
            return
        try:
            data = parent.loaded_sessions.get(text, None)
            if data is None:
                return
            self.data = data
            # repopulate cell list
            try:
                n_cells = int(self.data['raw_data'].shape[1])
                cell_names = self.data.get('cell_names', [f'Cell {i}' for i in range(n_cells)])
                self.combo_cell.clear()
                self.combo_cell.addItems(cell_names)
                self.cell_idx = 0
            except Exception:
                pass
            self.plot_detection()
        except Exception:
            return

    def _on_cell_changed(self, idx):
        try:
            self.cell_idx = int(idx)
        except Exception:
            self.cell_idx = 0
        self.plot_detection()

    def _draw_classic_scale_bar(self, axis, visible_ms, ylim, sigma, anchor_top_y_data=None, shift_frac=0.10):
        def _nice_round_time(ms_val):
            if ms_val <= 0:
                return 1.0
            exp = int(np.floor(np.log10(ms_val)))
            candidates = [1, 2, 2.5, 5, 10]
            best = None
            best_diff = None
            for m in candidates:
                cand = float(m) * (10 ** exp)
                diff = abs(cand - ms_val)
                if best is None or diff < best_diff:
                    best = cand
                    best_diff = diff
            next_cand = 10 ** (exp + 1)
            if abs(next_cand - ms_val) < best_diff:
                best = float(next_cand)
            return float(best)

        try:
            visible_ms = float(visible_ms)
        except Exception:
            visible_ms = 1.0
        target_ms = max(1.0, visible_ms / 20.0)
        nice_ms = _nice_round_time(target_ms)
        horiz_frac = float(nice_ms) / max(1e-9, visible_ms)
        horiz_frac = float(max(0.02, min(0.45, horiz_frac)))

        rx = 1.0 - 0.005
        lx = rx - horiz_frac

        span = (ylim[1] - ylim[0]) if isinstance(ylim, tuple) else max(1e-9, float(ylim))
        vertical_multiplier = 5.0
        vertical_frac = float(max(0.02, min(0.9, horiz_frac * vertical_multiplier)))

        if anchor_top_y_data is None:
            anchor_top_y_data = ylim[0] + 0.08 * span
        ty_axes = (float(anchor_top_y_data) - ylim[0]) / max(1e-9, span)
        ty_axes = float(max(ty_axes - 0.01 - shift_frac, -0.5))
        ry_axes = ty_axes - vertical_frac
        if ry_axes < -0.5:
            ry_axes = -0.5
            ty_axes = ry_axes + vertical_frac

        axis.plot([rx, rx], [ry_axes, ty_axes], transform=axis.transAxes, color='k', lw=_get_linewidth(2), clip_on=False)
        axis.plot([lx, rx], [ry_axes, ry_axes], transform=axis.transAxes, color='k', lw=_get_linewidth(2), clip_on=False)

        win_label = f"{int(nice_ms)} ms"
        sigma_safe = sigma if (sigma is not None and np.isfinite(sigma) and sigma != 0) else 1.0
        actual_z = (vertical_frac * span) / sigma_safe
        amp_z_label = float(round(actual_z * 2.0) / 2.0)
        if amp_z_label <= 0:
            amp_z_label = max(0.5, actual_z)

        axis.text((lx + rx) / 2.0, ry_axes - 0.02, win_label, transform=axis.transAxes,
                  va='top', ha='center', clip_on=False, fontsize=_scale_font(9))
        axis.text(rx + 0.01, (ry_axes + ty_axes) / 2.0, f"{amp_z_label:.1f} σ", transform=axis.transAxes,
                  va='center', ha='left', rotation='vertical', clip_on=False, fontsize=_scale_font(9))

    def plot_detection(self):
        t = self.data['time_ms']
        raw = self.data['raw_data'][:, self.cell_idx]
        fs = float(self.data['fs'])
        res = self.data['results'][self.cell_idx]

        parent = self.parent()
        colors = parent.colors if parent is not None else {'raw':'#333333','baseline':'#FFC20A','cs_trace':'#009E73','ss_trace':'#D55E00','cs_thresh':'#56B4E9','ss_thresh':'#CC79A7'}
        params = parent.params if parent is not None else {}

        # compute window
        center_pct = self.slider_time.value() / 1000.0
        total_ms = t[-1] - t[0]
        center_time = t[0] + center_pct * total_ms
        win_ms = float(self.spin_window.value())
        window_start = center_time - win_ms/2.0
        window_end = center_time + win_ms/2.0
        mask = (t >= window_start) & (t <= window_end)
        try:
            self.lbl_start_time.setText(f'Start: {int(window_start)} ms')
        except Exception:
            pass
        two_step_mode = bool(res.get('two_step_enabled', False))
        desired_mode = 'two_step' if two_step_mode else 'standard'
        if self._viewer_mode != desired_mode:
            self._build_axes(two_step_mode)

        # apply frame averaging if requested
        frames = int(parent.spin_avg_frames.value()) if (parent is not None and hasattr(parent, 'spin_avg_frames')) else 0
        mode = parent._get_frame_processing_mode() if (parent is not None and hasattr(parent, '_get_frame_processing_mode')) else 'Rolling average'
        raw_proc = apply_frame_processing(raw, frames=frames, mode=mode)

        baseline_gui = parent.compute_baseline(raw_proc, fs) if (parent is not None and hasattr(parent, 'compute_baseline')) else np.zeros_like(raw_proc)

        if not two_step_mode:
            for a in [self.ax_raw_top, self.ax_cs, self.ax_ss, self.ax_raw_bot]:
                a.clear()

            baseline_display = res.get('baseline', baseline_gui)
            vis_top = raw_proc
            self.ax_raw_top.plot(t[mask], vis_top[mask], color=colors.get('raw', '#333333'), lw=_get_linewidth(1))
            self.ax_raw_top.plot(t[mask], baseline_display[mask], color=colors.get('baseline', '#FFC20A'), ls='--', lw=_get_linewidth(1.6))
            self.ax_raw_bot.plot(t[mask], raw_proc[mask], color=colors.get('raw', '#333333'), lw=_get_linewidth(1))

            det_method = str(res.get('det_method', 'Threshold'))
            is_template_mode = ('Template' in det_method)
            cs_plot_trace = res.get('cs_similarity_trace', res.get('cs_trace', np.zeros_like(raw_proc))) if is_template_mode else res.get('cs_trace', np.zeros_like(raw_proc))
            ss_plot_trace = res.get('ss_similarity_trace', res.get('ss_trace', np.zeros_like(raw_proc))) if is_template_mode else res.get('ss_trace', np.zeros_like(raw_proc))
            self.ax_cs.plot(t[mask], cs_plot_trace[mask], color=colors.get('cs_trace', '#009E73'))
            self.ax_ss.plot(t[mask], ss_plot_trace[mask], color=colors.get('ss_trace', '#D55E00'))

            if is_template_mode:
                cs_thr_line = float(res.get('cs_threshold_used', np.nan))
                ss_thr_line = float(res.get('ss_threshold_used', np.nan))
                if np.isfinite(cs_thr_line):
                    self.ax_cs.axhline(cs_thr_line, color=colors.get('cs_thresh', '#56B4E9'), ls='--')
                if np.isfinite(ss_thr_line):
                    self.ax_ss.axhline(ss_thr_line, color=colors.get('ss_thresh', '#CC79A7'), ls='--')
            else:
                self.ax_cs.axhline(params.get('CS_THRESHOLD_SIGMA', 6.0) * res.get('sigma_cs', 0.0), color=colors.get('cs_thresh', '#56B4E9'), ls='--')
                self.ax_ss.axhline(params.get('SS_THRESHOLD_SIGMA', 2.0) * res.get('sigma_ss', 0.0), color=colors.get('ss_thresh', '#CC79A7'), ls='--')

            cs_idx = np.asarray(res.get('cs_peaks', []), dtype=int)
            ss_idx = np.asarray(res.get('ss_peaks', []), dtype=int)
            valid_cs = cs_idx[(t[cs_idx] >= window_start) & (t[cs_idx] <= window_end)] if len(cs_idx)>0 else np.array([])
            valid_ss = ss_idx[(t[ss_idx] >= window_start) & (t[ss_idx] <= window_end)] if len(ss_idx)>0 else np.array([])

            for a in [self.ax_raw_top, self.ax_cs, self.ax_ss, self.ax_raw_bot]:
                a.set_xlim(window_start, window_end)

            y_range = float(self.spin_zoom.value()) if hasattr(self, 'spin_zoom') else 0.0
            if np.any(mask):
                center = 0.5 * (np.max(vis_top[mask]) + np.min(vis_top[mask]))
                vmin = np.min(vis_top[mask]); vmax = np.max(vis_top[mask])
                span = vmax - vmin
                local_std = np.std(vis_top[mask]) if len(vis_top[mask])>0 else 1.0
                if y_range <= 0.0:
                    pad = max(span * 0.05, local_std * 0.5, 1e-9)
                    ylim = (vmin - pad, vmax + pad)
                else:
                    half_range = float(y_range) / 2.0
                    ylim = (center - half_range, center + half_range)
            else:
                ylim = (-1.0, 1.0)
            for a in [self.ax_raw_top, self.ax_raw_bot]:
                a.set_ylim(ylim)

            raw_span_original = max(1e-9, span if 'span' in locals() else 1.0)
            factor = float(y_range) / raw_span_original if y_range > 0.0 else 1.0
            cs_masked = cs_plot_trace[mask] if np.any(mask) else np.array([])
            ss_masked = ss_plot_trace[mask] if np.any(mask) else np.array([])
            cs_std = np.std(cs_masked) if cs_masked.size > 0 else 1.0
            ss_std = np.std(ss_masked) if ss_masked.size > 0 else 1.0
            if is_template_mode:
                cs_thresh = abs(float(res.get('cs_threshold_used', np.nan))) if np.isfinite(res.get('cs_threshold_used', np.nan)) else cs_std
                ss_thresh = abs(float(res.get('ss_threshold_used', np.nan))) if np.isfinite(res.get('ss_threshold_used', np.nan)) else ss_std
                cs_half = max(cs_thresh * 1.25, cs_std * 4.0)
                ss_half = max(ss_thresh * 1.25, ss_std * 5.0)
            else:
                cs_thresh = abs(params.get('CS_THRESHOLD_SIGMA', 6.0) * res.get('sigma_cs', 1.0))
                ss_thresh = abs(params.get('SS_THRESHOLD_SIGMA', 2.0) * res.get('sigma_ss', 1.0))
                cs_half = max(cs_thresh * 1.5, cs_std * 3.0) * factor
                ss_half = max(ss_thresh * 1.5, ss_std * 3.0) * factor
            if cs_masked.size > 0:
                cs_half = max(cs_half, np.max(np.abs(cs_masked)) * 1.2)
            if ss_masked.size > 0:
                ss_half = max(ss_half, np.max(np.abs(ss_masked)) * (1.35 if is_template_mode else 1.2))
            self.ax_cs.set_ylim(-max(cs_half, 1e-6), max(cs_half, 1e-6))
            self.ax_ss.set_ylim(-max(ss_half, 1e-6), max(ss_half, 1e-6))

            shift_frac = 0.10
            span_raw = (ylim[1] - ylim[0])
            ss_mark_center_desired = ylim[0] + 0.10 * span_raw - shift_frac * span_raw
            ss_mark_center = max(ylim[0] + 0.01 * span_raw, ss_mark_center_desired)
            ss_half_h = 0.04 * span_raw
            for p in valid_ss:
                if 0 <= p < len(t):
                    self.ax_raw_bot.plot([t[p], t[p]], [ss_mark_center - ss_half_h, ss_mark_center + ss_half_h], color=colors.get('ss_trace', '#D55E00'), lw=_get_linewidth(1.8))

            half_win = int((100.0/2.0) * fs / 1000.0)
            for p in valid_cs:
                if p < 0 or p >= len(res.get('cs_trace', [])):
                    continue
                s = max(0, int(p - half_win)); e = min(len(res.get('cs_trace', [])), int(p + half_win) + 1)
                wave = np.asarray(res.get('cs_trace', np.array([])))[s:e]
                if len(wave) <= 5:
                    continue
                x_new, interp = get_interpolated_wave(wave, fs)
                center_offset = (p - s)
                time_axis_ms = (x_new - center_offset) * 1000.0 / fs
                _, fwhm = get_wave_stats(interp, time_axis_ms)
                if np.isnan(fwhm) or fwhm <= 0:
                    continue
                x0 = t[p] - fwhm / 2.0
                x1 = t[p] + fwhm / 2.0
                y = max(ss_mark_center - ss_half_h, ylim[0] + 0.005 * span_raw)
                self.ax_raw_bot.plot([x0, x1], [y, y], color=colors.get('cs_trace', '#009E73'), lw=_get_linewidth(3))

            for a in [self.ax_raw_top, self.ax_cs, self.ax_ss, self.ax_raw_bot]:
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(False)

            try:
                vis_ms = float(window_end - window_start)
            except Exception:
                vis_ms = float(total_ms) if 'total_ms' in locals() else 1.0
            self._draw_classic_scale_bar(
                self.ax_raw_bot,
                vis_ms,
                ylim,
                res.get('raw_sigma', 1.0),
                anchor_top_y_data=(ss_mark_center - ss_half_h),
                shift_frac=shift_frac,
            )

            self.canvas.draw()
            return

        # two-step mode (6 traces)
        axes = [self.ax_raw_corr, self.ax_cs_score, self.ax_ss_score, self.ax_cs_simple, self.ax_ss_simple, self.ax_raw_spikes]
        for a in axes:
            a.clear()

        # Keep same style as classic 4-plot mode:
        # top raw + baseline, middle traces, bottom raw + final spikes
        baseline_display = res.get('baseline', baseline_gui)
        vis_top = raw_proc
        self.ax_raw_corr.plot(t[mask], vis_top[mask], color=colors.get('raw', '#333333'), lw=_get_linewidth(1))
        self.ax_raw_corr.plot(t[mask], baseline_display[mask], color=colors.get('baseline', '#FFC20A'), ls='--', lw=_get_linewidth(1.6))

        cs_score = np.asarray(res.get('cs_similarity_trace', np.zeros_like(raw_proc)), dtype=float)
        ss_score = np.asarray(res.get('ss_similarity_trace', np.zeros_like(raw_proc)), dtype=float)
        self.ax_cs_score.plot(t[mask], cs_score[mask], color=colors.get('cs_trace', '#009E73'))
        self.ax_ss_score.plot(t[mask], ss_score[mask], color=colors.get('ss_trace', '#D55E00'))
        cs_thr_line = float(res.get('cs_threshold_used', np.nan))
        ss_thr_line = float(res.get('ss_threshold_used', np.nan))
        if np.isfinite(cs_thr_line):
            self.ax_cs_score.axhline(cs_thr_line, color=colors.get('cs_thresh', '#56B4E9'), ls='--')
        if np.isfinite(ss_thr_line):
            self.ax_ss_score.axhline(ss_thr_line, color=colors.get('ss_thresh', '#CC79A7'), ls='--')

        cs_simple = np.asarray(res.get('cs_simple_trace', np.zeros_like(raw_proc)), dtype=float)
        ss_simple = np.asarray(res.get('ss_simple_trace', np.zeros_like(raw_proc)), dtype=float)
        self.ax_cs_simple.plot(t[mask], cs_simple[mask], color=colors.get('cs_trace', '#009E73'))
        self.ax_ss_simple.plot(t[mask], ss_simple[mask], color=colors.get('ss_trace', '#D55E00'))
        cs_simple_thr = float(res.get('cs_simple_threshold_used', np.nan))
        ss_simple_thr = float(res.get('ss_simple_threshold_used', np.nan))
        if np.isfinite(cs_simple_thr):
            self.ax_cs_simple.axhline(cs_simple_thr, color=colors.get('cs_thresh', '#56B4E9'), ls='--')
        if np.isfinite(ss_simple_thr):
            self.ax_ss_simple.axhline(ss_simple_thr, color=colors.get('ss_thresh', '#CC79A7'), ls='--')

        self.ax_raw_spikes.plot(t[mask], raw_proc[mask], color=colors.get('raw', '#333333'), lw=_get_linewidth(1))
        cs_idx = np.asarray(res.get('cs_peaks', []), dtype=int)
        ss_idx = np.asarray(res.get('ss_peaks', []), dtype=int)
        valid_cs = cs_idx[(t[cs_idx] >= window_start) & (t[cs_idx] <= window_end)] if cs_idx.size > 0 else np.array([])
        valid_ss = ss_idx[(t[ss_idx] >= window_start) & (t[ss_idx] <= window_end)] if ss_idx.size > 0 else np.array([])

        for a in axes:
            a.set_xlim(window_start, window_end)

        # --- scaling: match classic 4-plot behavior ---
        y_range = float(self.spin_zoom.value()) if hasattr(self, 'spin_zoom') else 0.0
        if np.any(mask):
            vmin = np.min(vis_top[mask]); vmax = np.max(vis_top[mask])
            span = vmax - vmin
            local_std = np.std(vis_top[mask]) if len(vis_top[mask]) > 0 else 1.0
            if y_range <= 0.0:
                pad = max(span * 0.05, local_std * 0.5, 1e-9)
                ylim_raw = (vmin - pad, vmax + pad)
            else:
                center = 0.5 * (vmax + vmin)
                half_range = float(y_range) / 2.0
                ylim_raw = (center - half_range, center + half_range)
        else:
            ylim_raw = (-1.0, 1.0)
            span = 2.0

        self.ax_raw_corr.set_ylim(ylim_raw)
        self.ax_raw_spikes.set_ylim(ylim_raw)

        # short CS/SS markers in bottom raw panel (same style as standard mode)
        span_raw = (ylim_raw[1] - ylim_raw[0])
        shift_frac = 0.10
        ss_mark_center_desired = ylim_raw[0] + 0.10 * span_raw - shift_frac * span_raw
        ss_mark_center = max(ylim_raw[0] + 0.01 * span_raw, ss_mark_center_desired)
        ss_half_h = 0.04 * span_raw
        for p in valid_ss:
            if 0 <= p < len(t):
                self.ax_raw_spikes.plot([t[p], t[p]], [ss_mark_center - ss_half_h, ss_mark_center + ss_half_h], color=colors.get('ss_trace', '#D55E00'), lw=_get_linewidth(1.8))

        half_win = int((100.0/2.0) * fs / 1000.0)
        cs_trace_for_fwhm = np.asarray(res.get('cs_trace', np.array([])), dtype=float)
        for p in valid_cs:
            if p < 0 or p >= len(cs_trace_for_fwhm):
                continue
            s = max(0, int(p - half_win)); e = min(len(cs_trace_for_fwhm), int(p + half_win) + 1)
            wave = cs_trace_for_fwhm[s:e]
            if len(wave) <= 5:
                continue
            x_new, interp = get_interpolated_wave(wave, fs)
            center_offset = (p - s)
            time_axis_ms = (x_new - center_offset) * 1000.0 / fs
            _, fwhm = get_wave_stats(interp, time_axis_ms)
            if np.isnan(fwhm) or fwhm <= 0:
                continue
            x0 = t[p] - fwhm / 2.0
            x1 = t[p] + fwhm / 2.0
            y = max(ss_mark_center - ss_half_h, ylim_raw[0] + 0.005 * span_raw)
            self.ax_raw_spikes.plot([x0, x1], [y, y], color=colors.get('cs_trace', '#009E73'), lw=_get_linewidth(3))

        raw_span_original = max(1e-9, span)
        factor = float(y_range) / raw_span_original if y_range > 0.0 else 1.0

        # template score traces use template-style scaling
        cs_masked = cs_score[mask] if np.any(mask) else np.array([])
        ss_masked = ss_score[mask] if np.any(mask) else np.array([])
        cs_std = np.std(cs_masked) if cs_masked.size > 0 else 1.0
        ss_std = np.std(ss_masked) if ss_masked.size > 0 else 1.0
        cs_thresh = abs(float(cs_thr_line)) if np.isfinite(cs_thr_line) else cs_std
        ss_thresh = abs(float(ss_thr_line)) if np.isfinite(ss_thr_line) else ss_std
        cs_half = max(cs_thresh * 1.25, cs_std * 4.0)
        ss_half = max(ss_thresh * 1.25, ss_std * 5.0)
        if cs_masked.size > 0:
            cs_half = max(cs_half, np.max(np.abs(cs_masked)) * 1.2)
        if ss_masked.size > 0:
            ss_half = max(ss_half, np.max(np.abs(ss_masked)) * 1.35)
        self.ax_cs_score.set_ylim(-max(cs_half, 1e-6), max(cs_half, 1e-6))
        self.ax_ss_score.set_ylim(-max(ss_half, 1e-6), max(ss_half, 1e-6))

        # simple threshold traces use threshold-style scaling
        cs_simple_masked = cs_simple[mask] if np.any(mask) else np.array([])
        ss_simple_masked = ss_simple[mask] if np.any(mask) else np.array([])
        cs_simple_std = np.std(cs_simple_masked) if cs_simple_masked.size > 0 else 1.0
        ss_simple_std = np.std(ss_simple_masked) if ss_simple_masked.size > 0 else 1.0
        cs_simple_thr_abs = abs(float(cs_simple_thr)) if np.isfinite(cs_simple_thr) else cs_simple_std
        ss_simple_thr_abs = abs(float(ss_simple_thr)) if np.isfinite(ss_simple_thr) else ss_simple_std
        cs_simple_half = max(cs_simple_thr_abs * 1.5, cs_simple_std * 3.0) * factor
        ss_simple_half = max(ss_simple_thr_abs * 1.5, ss_simple_std * 3.0) * factor
        if cs_simple_masked.size > 0:
            cs_simple_half = max(cs_simple_half, np.max(np.abs(cs_simple_masked)) * 1.2)
        if ss_simple_masked.size > 0:
            ss_simple_half = max(ss_simple_half, np.max(np.abs(ss_simple_masked)) * 1.2)
        self.ax_cs_simple.set_ylim(-max(cs_simple_half, 1e-6), max(cs_simple_half, 1e-6))
        self.ax_ss_simple.set_ylim(-max(ss_simple_half, 1e-6), max(ss_simple_half, 1e-6))

        # same clean frame/axis style as classic mode
        for a in axes:
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)

        # left indicators only
        labels = ['Raw', 'CS score', 'SS score', 'CS filt', 'SS filt', 'Raw+spk']
        for ax, txt in zip(axes, labels):
            ax.text(-0.03, 0.5, txt, transform=ax.transAxes, ha='right', va='center', fontsize=_scale_font(8), clip_on=False)

        try:
            vis_ms = float(window_end - window_start)
        except Exception:
            vis_ms = float(total_ms) if 'total_ms' in locals() else 1.0
        self._draw_classic_scale_bar(
            self.ax_raw_spikes,
            vis_ms,
            ylim_raw,
            res.get('raw_sigma', 1.0),
            anchor_top_y_data=(ss_mark_center - ss_half_h),
            shift_frac=shift_frac,
        )

        self.canvas.draw()

    def _extract_waveforms(self, trace, peaks, half_win):
        waves = []
        if trace is None or peaks is None:
            return np.array([])
        arr = np.asarray(trace, dtype=float).ravel()
        pks = np.asarray(peaks, dtype=int).ravel()
        for p in pks:
            s = int(p - half_win)
            e = int(p + half_win)
            if s < 0 or e > arr.size:
                continue
            w = arr[s:e]
            if w.size == 2 * half_win:
                waves.append(w)
        if len(waves) == 0:
            return np.array([])
        return np.asarray(waves, dtype=float)

    def _preprocess_for_template_export(self, raw, fs):
        parent = self.parent()
        sig = np.asarray(raw, dtype=float)
        frames = int(parent.spin_avg_frames.value()) if (parent is not None and hasattr(parent, 'spin_avg_frames')) else 0
        mode = parent._get_frame_processing_mode() if (parent is not None and hasattr(parent, '_get_frame_processing_mode')) else 'Rolling average'
        sig = apply_frame_processing(sig, frames=frames, mode=mode)
        try:
            baseline = parent.compute_baseline(sig, fs) if (parent is not None and hasattr(parent, 'compute_baseline')) else np.zeros_like(sig)
        except Exception:
            baseline = np.zeros_like(sig)
        sig = sig - baseline
        try:
            if parent is not None and bool(parent.params.get('NEGATIVE_GOING', True)):
                sig = -sig
        except Exception:
            pass
        return np.asarray(sig, dtype=float)

    def export_detected_templates(self):
        try:
            parent = self.parent()
            params = parent.params if parent is not None else {}

            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle('Export as templates')
            form = QtWidgets.QFormLayout(dlg)
            combo_scope = QtWidgets.QComboBox(dlg)
            combo_scope.addItems(['Current Cell', 'All Cells'])
            combo_scope.setCurrentText('All Cells')
            chk_cs = QtWidgets.QCheckBox('Export CS templates', dlg)
            chk_ss = QtWidgets.QCheckBox('Export SS templates', dlg)
            chk_cs.setChecked(True)
            chk_ss.setChecked(True)
            form.addRow('Extract from:', combo_scope)
            form.addRow(chk_cs)
            form.addRow(chk_ss)
            spin_cs_w = QtWidgets.QDoubleSpinBox(dlg)
            spin_cs_w.setRange(1.0, 200.0)
            spin_cs_w.setDecimals(1)
            spin_cs_w.setSuffix(' ms')
            spin_cs_w.setValue(float(params.get('TEMPLATE_CS_WINDOW_MS', 30.0)))
            spin_ss_w = QtWidgets.QDoubleSpinBox(dlg)
            spin_ss_w.setRange(1.0, 100.0)
            spin_ss_w.setDecimals(1)
            spin_ss_w.setSuffix(' ms')
            spin_ss_w.setValue(float(params.get('TEMPLATE_SS_WINDOW_MS', 8.0)))
            form.addRow('CS window:', spin_cs_w)
            form.addRow('SS window:', spin_ss_w)
            try:
                bb_buttons = QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            except Exception:
                bb_buttons = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            btn_box = QtWidgets.QDialogButtonBox(bb_buttons, parent=dlg)
            btn_box.accepted.connect(dlg.accept)
            btn_box.rejected.connect(dlg.reject)
            form.addRow(btn_box)
            try:
                accepted_code = QtWidgets.QDialog.DialogCode.Accepted
            except Exception:
                accepted_code = QtWidgets.QDialog.Accepted
            ok = (dlg.exec() == accepted_code)
            if not ok:
                return
            scope = str(combo_scope.currentText())
            do_cs = bool(chk_cs.isChecked())
            do_ss = bool(chk_ss.isChecked())
            if not do_cs and not do_ss:
                QMessageBox.warning(self, 'No selection', 'Please check CS and/or SS for export.')
                return

            fs = float(self.data.get('fs', 1000.0))
            cs_w_ms = float(spin_cs_w.value())
            ss_w_ms = float(spin_ss_w.value())
            try:
                params['TEMPLATE_CS_WINDOW_MS'] = cs_w_ms
                params['TEMPLATE_SS_WINDOW_MS'] = ss_w_ms
            except Exception:
                pass
            cs_half = max(2, int((cs_w_ms / 1000.0) * fs / 2.0))
            ss_half = max(2, int((ss_w_ms / 1000.0) * fs / 2.0))

            cell_indices = [self.cell_idx] if scope == 'Current Cell' else list(range(int(self.data['raw_data'].shape[1])))
            cs_all = []
            ss_all = []
            for ci in cell_indices:
                try:
                    res = self.data['results'][ci]
                except Exception:
                    continue
                if res is None:
                    continue
                raw_cell = self.data['raw_data'][:, ci]
                src = self._preprocess_for_template_export(raw_cell, fs)
                cs_src = src
                ss_src = src
                cs_w = self._extract_waveforms(cs_src, res.get('cs_peaks', None), cs_half)
                ss_w = self._extract_waveforms(ss_src, res.get('ss_peaks', None), ss_half)
                if cs_w.size > 0:
                    cs_all.append(cs_w)
                if ss_w.size > 0:
                    ss_all.append(ss_w)

            cs_stack = np.vstack(cs_all) if len(cs_all) > 0 else np.array([])
            ss_stack = np.vstack(ss_all) if len(ss_all) > 0 else np.array([])
            if (do_cs and cs_stack.size == 0) and (do_ss and ss_stack.size == 0):
                QMessageBox.warning(self, 'No spikes', 'No detected spikes found for template extraction in selected scope.')
                return

            saved_files = []
            if do_cs:
                if cs_stack.size == 0:
                    QMessageBox.warning(self, 'No CS spikes', 'No detected CS spikes found for selected scope.')
                else:
                    cs_name, _ = QFileDialog.getSaveFileName(self, 'Save CS Templates', os.path.expanduser('~/detected_cs_templates.npz'), 'NPZ Files (*.npz)')
                    if cs_name:
                        if not cs_name.lower().endswith('.npz'):
                            cs_name += '.npz'
                        fs_out = float(TEMPLATE_TARGET_FS)
                        cs_stack_rs = np.vstack([_resample_template_to_fs(w, fs, fs_out) for w in cs_stack])
                        np.savez_compressed(
                            cs_name,
                            fs=fs_out,
                            fs_cs=fs_out,
                            scope=scope,
                            cs_waveforms=cs_stack_rs,
                            template_cs=np.mean(cs_stack_rs, axis=0),
                        )
                        saved_files.append(cs_name)

            if do_ss:
                if ss_stack.size == 0:
                    QMessageBox.warning(self, 'No SS spikes', 'No detected SS spikes found for selected scope.')
                else:
                    ss_name, _ = QFileDialog.getSaveFileName(self, 'Save SS Templates', os.path.expanduser('~/detected_ss_templates.npz'), 'NPZ Files (*.npz)')
                    if ss_name:
                        if not ss_name.lower().endswith('.npz'):
                            ss_name += '.npz'
                        fs_out = float(TEMPLATE_TARGET_FS)
                        ss_stack_rs = np.vstack([_resample_template_to_fs(w, fs, fs_out) for w in ss_stack])
                        np.savez_compressed(
                            ss_name,
                            fs=fs_out,
                            fs_ss=fs_out,
                            scope=scope,
                            ss_waveforms=ss_stack_rs,
                            template_ss=np.mean(ss_stack_rs, axis=0),
                        )
                        saved_files.append(ss_name)

            if len(saved_files) > 0:
                QMessageBox.information(self, 'Saved', 'Templates saved:\n' + '\n'.join(saved_files))
        except Exception as e:
            QMessageBox.critical(self, 'Export Error', f'Failed to export templates: {e}')

    def save_figure(self):
        try:
            filename = save_figure_with_dialog(
                lambda title, default, flt: QFileDialog.getSaveFileName(self, title, default, flt),
                self.canvas.figure,
                default_name='detection_figure.svg',
            )
            if filename:
                QMessageBox.information(self, 'Saved', f'Figure saved as {filename}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save figure: {e}')


class TemplateViewerDialog(QtWidgets.QDialog):
    def __init__(self, template_store, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Template Waveforms')
        self.resize(*_scaled_size(900, 420))
        self.template_store = template_store if isinstance(template_store, dict) else {}
        self.fig = _make_figure(9, 4)
        self.canvas = FigureCanvas(self.fig)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.plot_templates()

    def _plot_bank(self, ax, bank_key, fs_key, color, title):
        bank = self.template_store.get(bank_key, [])
        fs_bank = self.template_store.get(fs_key, [])
        if bank is None or len(bank) == 0:
            ax.text(0.5, 0.5, 'No templates', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        traces = []
        for i, tpl in enumerate(bank):
            try:
                t = np.asarray(tpl, dtype=float).ravel()
                if t.size > 3 and np.all(np.isfinite(t)):
                    traces.append(t)
            except Exception:
                continue
        if len(traces) == 0:
            ax.text(0.5, 0.5, 'No valid templates', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        min_len = int(np.min([tr.size for tr in traces]))
        stack = np.vstack([tr[:min_len] for tr in traces])
        mean_tpl = np.mean(stack, axis=0)
        try:
            fs0 = float(fs_bank[0]) if fs_bank is not None and len(fs_bank) > 0 and fs_bank[0] is not None else TEMPLATE_TARGET_FS
        except Exception:
            fs0 = TEMPLATE_TARGET_FS
        if not np.isfinite(fs0) or fs0 <= 0:
            fs0 = TEMPLATE_TARGET_FS
        t_ms = np.arange(min_len, dtype=float) * 1000.0 / fs0
        for tr in stack:
            ax.plot(t_ms, tr, color=color, alpha=0.2, lw=0.8)
        ax.plot(t_ms, mean_tpl, color=color, lw=2.0)
        ax.set_title(f'{title} (n={stack.shape[0]})')
        ax.set_xlabel('ms')
        try:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        except Exception:
            pass

    def _plot_parallel_bank(self, ax, bank_key, fs_key, color, title):
        bank = self.template_store.get(bank_key, [])
        fs_bank = self.template_store.get(fs_key, [])
        groups = _build_parallel_template_banks(bank, fs_bank, TEMPLATE_TARGET_FS, force_peak_positive=False, max_use_types=3)
        if groups is None or len(groups) == 0:
            self._plot_bank(ax, bank_key, fs_key, color, title)
            return
        for gi, (tpl_bank, tpl_fs_bank) in enumerate(groups):
            if tpl_bank is None or len(tpl_bank) == 0:
                continue
            rows = [np.asarray(x, dtype=float).ravel() for x in tpl_bank if np.asarray(x, dtype=float).ravel().size > 3]
            if len(rows) == 0:
                continue
            min_len = int(np.min([r.size for r in rows]))
            stack = np.vstack([r[:min_len] for r in rows])
            mean_tpl = np.mean(stack, axis=0)
            try:
                fs0 = float(tpl_fs_bank[0]) if tpl_fs_bank is not None and len(tpl_fs_bank) > 0 else TEMPLATE_TARGET_FS
            except Exception:
                fs0 = TEMPLATE_TARGET_FS
            if not np.isfinite(fs0) or fs0 <= 0:
                fs0 = TEMPLATE_TARGET_FS
            t_ms = np.arange(min_len, dtype=float) * 1000.0 / fs0
            ax.plot(t_ms, mean_tpl, lw=2.0, alpha=0.95, label=f'Type {gi+1} (n={stack.shape[0]})', color=color)
        ax.set_title(f'{title} (parallel types)')
        ax.set_xlabel('ms')
        try:
            ax.legend(frameon=False, fontsize=_scale_font(8), loc='best')
        except Exception:
            pass
        try:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        except Exception:
            pass

    def plot_templates(self):
        self.fig.clf()
        parent = self.parent()
        cols = parent.colors if parent is not None and hasattr(parent, 'colors') else {}
        cs_color = cols.get('cs_trace', '#009E73')
        ss_color = cols.get('ss_trace', '#D55E00')
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax2 = self.fig.add_subplot(1, 2, 2)
        parallel_mode = bool(getattr(parent, 'params', {}).get('TEMPLATE_PARALLEL', False)) if parent is not None else False
        if parallel_mode:
            self._plot_parallel_bank(ax1, 'cs_templates', 'fs_cs', cs_color, 'CS Templates')
            self._plot_parallel_bank(ax2, 'ss_templates', 'fs_ss', ss_color, 'SS Templates')
        else:
            self._plot_bank(ax1, 'cs_templates', 'fs_cs', cs_color, 'CS Templates')
            self._plot_bank(ax2, 'ss_templates', 'fs_ss', ss_color, 'SS Templates')
        self.fig.tight_layout()
        self.canvas.draw()


class StatsViewerDialog(QtWidgets.QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Spike Statistics')
        # make stats dialog 20% wider (additional 20% -> total 1296px width)
        self.resize(*_scaled_size(1296, 600))
        self.data = data
        self.fig = _make_figure(10, 8)
        self.canvas = FigureCanvas(self.fig)
        # Top control row: session selector, cell selector, Save Figure
        top_h = QtWidgets.QHBoxLayout()
        parent_win = parent
        # session selector (All + session names)
        self.combo_session = QtWidgets.QComboBox()
        sess_names = []
        try:
            if parent_win is not None and hasattr(parent_win, 'session_names'):
                sess_names = list(parent_win.session_names)
            else:
                sess_names = list(parent_win.loaded_sessions.keys()) if parent_win is not None else []
        except Exception:
            sess_names = []
        self.combo_session.addItem('All')
        for n in sess_names:
            self.combo_session.addItem(n)
        self.combo_session.currentIndexChanged.connect(self._on_session_change)

        # cell selector (will be populated by session change)
        self.combo_cell = QtWidgets.QComboBox()
        self.combo_cell.addItem('All')
        self.combo_cell.currentIndexChanged.connect(self.compute_stats)

        top_h.addWidget(QtWidgets.QLabel('Session:'))
        top_h.addWidget(self.combo_session)
        top_h.addWidget(QtWidgets.QLabel('Cell:'))
        top_h.addWidget(self.combo_cell)
        top_h.addStretch(1)
        btn_save = QtWidgets.QPushButton('Save Figure')
        btn_save.clicked.connect(self.save_figure)
        top_h.addWidget(btn_save)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_h)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # initialize cell list based on current session selection
        self._on_session_change(0)
        # initial plot
        self.compute_stats()

    def _on_session_change(self, idx):
        # Populate the cell combobox based on selected session.
        self.combo_cell.blockSignals(True)
        try:
            self.combo_cell.clear()
            self.combo_cell.addItem('All')
            parent_win = self.parent()
            sel = 'All'
            try:
                sel = self.combo_session.currentText()
            except Exception:
                sel = 'All'
            if sel == 'All':
                # Collect union of cell names across sessions
                names = []
                try:
                    if parent_win is not None and hasattr(parent_win, 'session_names'):
                        for s in parent_win.session_names:
                            sdata = parent_win.loaded_sessions.get(s, None)
                            if sdata is None:
                                continue
                            for n in sdata.get('cell_names', []):
                                if n not in names:
                                    names.append(n)
                except Exception:
                    names = []
                for n in names:
                    self.combo_cell.addItem(n)
            else:
                try:
                    sdata = parent_win.loaded_sessions.get(sel, None) if parent_win is not None else None
                    if sdata is not None:
                        for n in sdata.get('cell_names', []):
                            self.combo_cell.addItem(n)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self.combo_cell.blockSignals(False)
        # Trigger recompute
        try:
            self.compute_stats()
        except Exception:
            pass

    def compute_stats(self):
        parent = self.parent()
        defaults = {'cs_trace': '#009E73', 'ss_trace': '#D55E00', 'raw': '#333333', 'baseline': '#FFC20A'}
        colors = dict(defaults)
        if parent is not None and isinstance(parent, object):
            try:
                colors.update(parent.colors)
            except Exception:
                pass
        # Build list of (res, session_data) according to user selection
        sel_session = None
        sel_cell = None
        try:
            sel_session = self.combo_session.currentText()
        except Exception:
            sel_session = 'All'
        try:
            sel_cell = self.combo_cell.currentText()
        except Exception:
            sel_cell = 'All'

        # gather items: list of tuples (res, session_data)
        items = []
        parent_win = self.parent()
        if sel_session == 'All':
            # iterate all sessions
            sessions_iter = []
            try:
                if parent_win is not None and hasattr(parent_win, 'session_names'):
                    sessions_iter = list(parent_win.session_names)
                else:
                    sessions_iter = list(parent_win.loaded_sessions.keys()) if parent_win is not None else []
            except Exception:
                sessions_iter = []
            for sname in sessions_iter:
                sdata = parent_win.loaded_sessions.get(sname, None) if parent_win is not None else None
                if sdata is None:
                    continue
                # If a specific cell name requested, try to find its index in this session
                if sel_cell != 'All':
                    try:
                        idx = sdata.get('cell_names', []).index(sel_cell)
                        res = sdata.get('results', [None])[idx]
                        items.append((res, sdata))
                    except Exception:
                        # cell not found in this session
                        continue
                else:
                    # add all cells for this session
                    for res in sdata.get('results', []):
                        items.append((res, sdata))
        else:
            # single session selected
            sdata = parent_win.loaded_sessions.get(sel_session, None) if parent_win is not None else None
            if sdata is not None:
                if sel_cell != 'All':
                    try:
                        idx = sdata.get('cell_names', []).index(sel_cell)
                        res = sdata.get('results', [None])[idx]
                        items.append((res, sdata))
                    except Exception:
                        pass
                else:
                    for res in sdata.get('results', []):
                        items.append((res, sdata))

        # Now compute stats across collected items
        window_ms = 100
        all_stats = {'CS': {'waves': [], 'snr': [], 'fwhm': []}, 'SS': {'waves': [], 'snr': [], 'fwhm': []}}
        rates = {'CS': [], 'SS': []}
        for res, sdata in items:
            if res is None or sdata is None:
                continue
            try:
                fs = float(sdata.get('fs', 1000.0))
                duration_s = (np.array(sdata.get('time_ms', [])).flatten()[-1] - np.array(sdata.get('time_ms', [])).flatten()[0]) / 1000.0 if len(sdata.get('time_ms', []))>1 else np.nan
            except Exception:
                fs = float(self.data.get('fs', 1000.0))
                duration_s = np.nan
            half_win = int((window_ms/2.0) * fs / 1000.0)
            for spike_type in ['CS', 'SS']:
                try:
                    peaks_key = 'cs_peaks' if spike_type == 'CS' else 'ss_peaks'
                    peaks = np.array(res.get(peaks_key, []), dtype=int)
                    if duration_s and duration_s > 0:
                        rates[spike_type].append(len(peaks) / duration_s)
                    # compute SNRs
                    event_snrs = compute_event_snrs(res, spike_type, fs, window_ms=window_ms, max_per_cell=None)
                    if len(event_snrs) > 0:
                        all_stats[spike_type]['snr'].extend(event_snrs)
                    # collect waveforms and fwhm
                    trace = res.get('cs_trace') if spike_type == 'CS' else (res.get('ss_trace') if res.get('ss_trace', None) is not None else res.get('detrended', None))
                    if trace is None or peaks.size == 0:
                        continue
                    chosen = _select_event_bank(peaks, max_per_cell=None)
                    for p in chosen:
                        s = int(p - half_win); e = int(p + half_win)
                        if s < 0 or e >= len(trace):
                            continue
                        wave = trace[s:e]
                        if len(wave) != (2 * half_win):
                            continue
                        if len(wave) > 5:
                            wave = wave - np.mean(wave[:5])
                        all_stats[spike_type]['waves'].append(wave)
                        _, interp_wave = get_interpolated_wave(wave, fs)
                        t_d = np.linspace(-window_ms/2.0, window_ms/2.0, len(interp_wave))
                        _, fwhm = get_wave_stats(interp_wave, t_d)
                        if not np.isnan(fwhm):
                            all_stats[spike_type]['fwhm'].append(fwhm)
                except Exception:
                    pass

        cs_r_mean, cs_r_std, cs_r_n = mean_std_count(rates['CS'])
        ss_r_mean, ss_r_std, ss_r_n = mean_std_count(rates['SS'])
        cs_snr_mean, cs_snr_std, cs_snr_n = mean_std_count(all_stats['CS']['snr'])
        ss_snr_mean, ss_snr_std, ss_snr_n = mean_std_count(all_stats['SS']['snr'])
        cs_fwhm_mean, cs_fwhm_std, cs_fwhm_n = mean_std_count(all_stats['CS']['fwhm'])
        ss_fwhm_mean, ss_fwhm_std, ss_fwhm_n = mean_std_count(all_stats['SS']['fwhm'])

        # Plot: 2 rows x 4 cols (last col for text summary)
        self.fig.clf()
        gs = self.fig.add_gridspec(2, 4, width_ratios=[2.0, 1.0, 1.0, 0.8], wspace=0.4, hspace=0.5)
        def _pick_color(d, keys, default):
            for k in keys:
                try:
                    if k in d and d.get(k) is not None:
                        return d.get(k)
                except Exception:
                    pass
            return default

        for row, spike_type in enumerate(['CS', 'SS']):
            stats = all_stats[spike_type]
            waves = stats['waves']
            snr = stats['snr']
            fwhm = stats['fwhm']
            # Try common color keys in parent/colors for robustness: prefer *_trace, then short name
            if spike_type == 'CS':
                color = _pick_color(colors, ['cs_trace', 'cs'], '#009E73')
            else:
                color = _pick_color(colors, ['ss_trace', 'ss'], '#D55E00')

            # per-spike-type time window: CS=100ms, SS=30ms
            w_ms = 100 if spike_type == 'CS' else 30

            ax1 = self.fig.add_subplot(gs[row, 0])
            if len(waves) > 0:
                max_bg = min(len(waves), 2000)
                # Use different background-alpha for CS vs SS to control trace visibility
                bg_alpha = 0.3 if spike_type == 'CS' else 0.003
                for w in waves[:max_bg]:
                    ax1.plot(np.linspace(-w_ms/2.0, w_ms/2.0, len(w)), w, color=color, alpha=bg_alpha, lw=_get_linewidth(0.5))
                ax1.plot(np.linspace(-w_ms/2.0, w_ms/2.0, len(waves[0])), np.mean(waves, axis=0), color=color, lw=_get_linewidth(3))
            ax1.set_title(f"{spike_type} Waveforms")
            ax1.set_xlabel('ms')
            try:
                ax1.spines['top'].set_visible(False)
            except Exception:
                pass
            try:
                ax1.spines['right'].set_visible(False)
            except Exception:
                pass

            ax2 = self.fig.add_subplot(gs[row, 1])
            if len(fwhm) > 0:
                ax2.hist(fwhm, bins=20, color=color, alpha=0.8)
                ax2.set_title('FWHM')
            try:
                ax2.spines['top'].set_visible(False)
            except Exception:
                pass
            try:
                ax2.spines['right'].set_visible(False)
            except Exception:
                pass
            if len(fwhm) == 0:
                ax2.text(0.5, 0.5, 'No data', ha='center')

            ax3 = self.fig.add_subplot(gs[row, 2])
            if len(snr) > 0:
                ax3.hist(snr, bins=20, color=color, alpha=0.8)
                ax3.set_title('SNR')
            try:
                ax3.spines['top'].set_visible(False)
            except Exception:
                pass
            try:
                ax3.spines['right'].set_visible(False)
            except Exception:
                pass
            if len(snr) == 0:
                ax3.text(0.5, 0.5, 'No data', ha='center')

            # right-side text summary
            ax_txt = self.fig.add_subplot(gs[row, 3])
            ax_txt.axis('off')
            txt_lines = []
            txt_lines.append(f"Rate: {cs_r_mean:.2f}±{cs_r_std:.2f} Hz" if spike_type == 'CS' else f"Rate: {ss_r_mean:.2f}±{ss_r_std:.2f} Hz")
            if spike_type == 'CS':
                txt_lines.append(f"FWHM: {cs_fwhm_mean:.2f}±{cs_fwhm_std:.2f} ms (n={cs_fwhm_n})")
                txt_lines.append(f"SNR: {cs_snr_mean:.2f}±{cs_snr_std:.2f} (n={cs_snr_n})")
            else:
                txt_lines.append(f"FWHM: {ss_fwhm_mean:.2f}±{ss_fwhm_std:.2f} ms (n={ss_fwhm_n})")
                txt_lines.append(f"SNR: {ss_snr_mean:.2f}±{ss_snr_std:.2f} (n={ss_snr_n})")
            ax_txt.text(0.02, 0.5, '\n'.join(txt_lines), va='center', ha='left', fontsize=10)

        self.canvas.draw()

    def save_figure(self):
        try:
            filename = save_figure_with_dialog(
                lambda title, default, flt: QFileDialog.getSaveFileName(self, title, default, flt),
                self.canvas.figure,
                default_name='stats_figure.svg',
            )
            if filename:
                QMessageBox.information(self, 'Saved', f'Figure saved as {filename}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save figure: {e}')


class SettingsDialog(QtWidgets.QDialog):
    """Advanced settings: keep only detection toggle; color scheme stays in main UI."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Advanced Settings')
        self.resize(*_scaled_size(460, 280))
        self.parent_win = parent

        layout = QtWidgets.QVBoxLayout()

        form = QtWidgets.QFormLayout()
        p = parent.params if parent is not None else {}

        # Negative going toggle (only detection-related control retained)
        self.chk_negative = QtWidgets.QCheckBox('Negative going (invert)')
        self.chk_negative.setChecked(p.get('NEGATIVE_GOING', True))
        form.addRow(self.chk_negative)
        # CS min dist and SS-specific params moved into Advanced Settings
        self.spin_cs_mind = QtWidgets.QDoubleSpinBox(); self.spin_cs_mind.setRange(0.0,1000.0); self.spin_cs_mind.setValue(p.get('CS_MIN_DIST_MS', 25.0))
        self.spin_ss_mind = QtWidgets.QDoubleSpinBox(); self.spin_ss_mind.setRange(0.0,1000.0); self.spin_ss_mind.setValue(p.get('SS_MIN_DIST_MS', 2.0))
        self.spin_ss_blank = QtWidgets.QDoubleSpinBox(); self.spin_ss_blank.setRange(0.0,1000.0); self.spin_ss_blank.setValue(p.get('SS_BLANK_MS', 8.0))
        self.spin_initial_blank = QtWidgets.QDoubleSpinBox(); self.spin_initial_blank.setRange(0.0,5000.0); self.spin_initial_blank.setValue(p.get('INITIAL_BLANK_MS', 150.0))
        self.spin_tpl_cs_window = QtWidgets.QDoubleSpinBox(); self.spin_tpl_cs_window.setRange(1.0,200.0); self.spin_tpl_cs_window.setDecimals(1); self.spin_tpl_cs_window.setValue(p.get('TEMPLATE_CS_WINDOW_MS', 30.0))
        self.spin_tpl_ss_window = QtWidgets.QDoubleSpinBox(); self.spin_tpl_ss_window.setRange(1.0,100.0); self.spin_tpl_ss_window.setDecimals(1); self.spin_tpl_ss_window.setValue(p.get('TEMPLATE_SS_WINDOW_MS', 8.0))
        form.addRow('CS min dist (ms):', self.spin_cs_mind)
        form.addRow('SS min dist (ms):', self.spin_ss_mind)
        form.addRow('SS blank (ms):', self.spin_ss_blank)
        form.addRow('Initial blank (ms):', self.spin_initial_blank)
        form.addRow('Template CS window (ms):', self.spin_tpl_cs_window)
        form.addRow('Template SS window (ms):', self.spin_tpl_ss_window)

        layout.addLayout(form)

        # Inline color pickers (embed color scheme settings here — no extra dialog)
        layout.addWidget(QtWidgets.QLabel('Colors:'))
        self.color_keys = ['raw','baseline','cs_trace','ss_trace','cs_thresh','ss_thresh']
        self.color_labels = {}
        grid = QtWidgets.QGridLayout()
        cols = parent.colors if parent is not None else {}
        # default (original) color scheme to fall back to when parent lacks keys
        default_colors = {
            'raw': '#333333',
            'baseline': '#FFC20A',
            'cs_trace': '#009E73',
            'ss_trace': '#D55E00',
            'cs_thresh': '#56B4E9',
            'ss_thresh': '#CC79A7'
        }
        # store defaults on the instance for use in other methods
        self.default_colors = dict(default_colors)
        color_display = ['Raw', 'Baseline', 'CS Trace', 'SS Trace', 'CS Threshold', 'SS Threshold']
        for i, key in enumerate(self.color_keys):
            grid.addWidget(QtWidgets.QLabel(color_display[i]+':'), i, 0)
            lbl = QtWidgets.QLabel('   ')
            lbl.setFixedHeight(20)
            curcol = cols.get(key, default_colors.get(key, '#FFFFFF'))
            lbl.setStyleSheet(f'background-color: {curcol};')
            lbl.setProperty('color_hex', curcol)
            btn = QtWidgets.QPushButton('Pick')
            # handler captures label and key
            def _make_handler(l=lbl, k=key):
                def _h():
                    cur = l.property('color_hex') or cols.get(k, '#ffffff')
                    col = QColorDialog.getColor(QColor(cur), self, f'Pick color for {k}')
                    if col.isValid():
                        hexc = col.name()
                        l.setStyleSheet(f'background-color: {hexc};')
                        l.setProperty('color_hex', hexc)
                return _h
            btn.clicked.connect(_make_handler())
            grid.addWidget(lbl, i, 1)
            grid.addWidget(btn, i, 2)
            self.color_labels[key] = lbl
        layout.addLayout(grid)

        # Buttons
        btns = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton('OK'); cancel = QtWidgets.QPushButton('Cancel')
        ok.clicked.connect(self.save_and_close); cancel.clicked.connect(self.reject)
        btns.addStretch(1); btns.addWidget(ok); btns.addWidget(cancel)
        layout.addLayout(btns)

        self.setLayout(layout)

    def save_and_close(self):
        # Apply negative-going toggle
        try:
            self.parent_win.params['NEGATIVE_GOING'] = bool(self.chk_negative.isChecked())
        except Exception:
            pass
        # Apply SS settings
        try:
            self.parent_win.params['CS_MIN_DIST_MS'] = float(self.spin_cs_mind.value())
            self.parent_win.params['SS_MIN_DIST_MS'] = float(self.spin_ss_mind.value())
            self.parent_win.params['SS_BLANK_MS'] = float(self.spin_ss_blank.value())
            self.parent_win.params['INITIAL_BLANK_MS'] = float(self.spin_initial_blank.value())
            self.parent_win.params['TEMPLATE_CS_WINDOW_MS'] = float(self.spin_tpl_cs_window.value())
            self.parent_win.params['TEMPLATE_SS_WINDOW_MS'] = float(self.spin_tpl_ss_window.value())
        except Exception:
            pass
        # Apply inline color choices to parent.colors
        try:
            cols = self.parent_win.colors
            for k, lbl in self.color_labels.items():
                hexc = lbl.property('color_hex') if lbl.property('color_hex') is not None else self.default_colors.get(k, cols.get(k))
                if hexc is None:
                    continue
                cols[k] = hexc
        except Exception:
            pass
        self.accept()


class ColorSchemeDialog(QtWidgets.QDialog):
    """Dialog for selecting color scheme for plots"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Color Scheme Settings')
        self.resize(*_scaled_size(400, 300))
        
        self.color_schemes = {
            'Default': {
                'raw': '#333333',
                'baseline': '#FFC20A',
                'cs_trace': '#009E73',
                'ss_trace': '#D55E00',
                'cs_thresh': '#56B4E9',
                'ss_thresh': '#CC79A7'
            },
            'Dark': {
                'raw': '#CCCCCC',
                'baseline': '#FFD700',
                'cs_trace': '#00FF41',
                'ss_trace': '#FF6B35',
                'cs_thresh': '#00D4FF',
                'ss_thresh': '#FF1493'
            },
            'Light': {
                'raw': '#000000',
                'baseline': '#FF8C00',
                'cs_trace': '#228B22',
                'ss_trace': '#DC143C',
                'cs_thresh': '#4169E1',
                'ss_thresh': '#8B008B'
            },
            'Colorblind-friendly': {
                'raw': '#000000',
                'baseline': '#FF7F00',
                'cs_trace': '#0173B2',
                'ss_trace': '#DE8F05',
                'cs_thresh': '#CC78BC',
                'ss_thresh': '#CA9161'
            }
        }
        
        layout = QtWidgets.QVBoxLayout()
        
        label = QtWidgets.QLabel('Select Color Scheme:')
        layout.addWidget(label)
        
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(self.color_schemes.keys())
        layout.addWidget(self.combo)
        
        # Preview and customize area
        preview_group = QtWidgets.QGroupBox('Color Preview / Customize')
        preview_layout = QtWidgets.QGridLayout()
        self.color_labels = {}
        self.color_buttons = {}
        color_names = ['raw', 'baseline', 'cs_trace', 'ss_trace', 'cs_thresh', 'ss_thresh']
        color_display = ['Raw', 'Baseline', 'CS Trace', 'SS Trace', 'CS Threshold', 'SS Threshold']

        for idx, (color_key, color_display_name) in enumerate(zip(color_names, color_display)):
            preview_layout.addWidget(QtWidgets.QLabel(color_display_name), idx, 0)
            self.color_labels[color_key] = QtWidgets.QLabel('   ')
            self.color_labels[color_key].setFixedHeight(22)
            preview_layout.addWidget(self.color_labels[color_key], idx, 1)
            btn = QtWidgets.QPushButton('Pick')
            btn.clicked.connect(self._make_pick_handler(color_key))
            preview_layout.addWidget(btn, idx, 2)
            self.color_buttons[color_key] = btn

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Buttons: OK / Cancel
        btns_h = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton('OK')
        btn_cancel = QtWidgets.QPushButton('Cancel')
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btns_h.addStretch(1); btns_h.addWidget(btn_ok); btns_h.addWidget(btn_cancel)
        layout.addLayout(btns_h)

        self.setLayout(layout)
        self.combo.currentTextChanged.connect(self.update_preview)
        self.update_preview()

    def _make_pick_handler(self, key):
        def handler():
            cur = self.color_labels[key].property('color_hex') or self.color_schemes[self.combo.currentText()].get(key, '#ffffff')
            col = QColorDialog.getColor(QColor(cur), self, f'Pick color for {key}')
            if col.isValid():
                hexc = col.name()
                self.color_labels[key].setStyleSheet(f'background-color: {hexc};')
                self.color_labels[key].setProperty('color_hex', hexc)
        return handler
    
    def update_preview(self):
        scheme = self.color_schemes[self.combo.currentText()]
        for color_key, label in self.color_labels.items():
            color = scheme.get(color_key, '#FFFFFF')
            label.setStyleSheet(f'background-color: {color};')
            label.setProperty('color_hex', color)
    
    def get_scheme(self):
        # return current edited scheme
        scheme = dict(self.color_schemes[self.combo.currentText()])
        for k, lbl in self.color_labels.items():
            hexc = lbl.property('color_hex') if lbl.property('color_hex') is not None else scheme.get(k)
            scheme[k] = hexc
        return scheme


def main():
    try:
        policy = Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(policy)
    except Exception:
        pass
    app = QtWidgets.QApplication(sys.argv)
    try:
        s = _get_screen_scale()
        if s > 1.0 and sys.platform != 'darwin':
            f = app.font()
            ps = float(f.pointSizeF()) if f.pointSizeF() > 0 else float(f.pointSize())
            if ps > 0:
                f.setPointSizeF(ps * (1.0 + 0.12 * max(0.0, s - 1.0)))
                app.setFont(f)
    except Exception:
        pass
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
