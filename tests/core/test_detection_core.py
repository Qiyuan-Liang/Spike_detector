import numpy as np

from spike_detector.core.detection import (
    TEMPLATE_TARGET_FS,
    apply_frame_processing,
    estimate_noise_mad,
    process_cell_simple,
    _resample_template_to_fs,
)


def test_noise_estimator_returns_positive_value():
    x = np.random.default_rng(42).normal(0, 1, 2000)
    sigma = estimate_noise_mad(x)
    assert np.isfinite(sigma)
    assert sigma > 0


def test_apply_frame_processing_keeps_length():
    x = np.linspace(0, 1, 1000)
    y_roll = apply_frame_processing(x, frames=5, mode='Rolling average')
    y_down = apply_frame_processing(x, frames=5, mode='Downsampling')
    assert len(y_roll) == len(x)
    assert len(y_down) == len(x)


def test_template_resample_changes_fs():
    tpl = np.sin(np.linspace(0, np.pi, 60))
    rs = _resample_template_to_fs(tpl, template_fs=1000.0, target_fs=TEMPLATE_TARGET_FS)
    assert rs.size > tpl.size
    assert np.isfinite(rs).all()


def test_process_cell_simple_output_shape_and_keys():
    fs = 5000.0
    t = np.arange(0, 2.0, 1.0 / fs)
    x = 0.05 * np.random.default_rng(0).normal(0, 1, t.size)
    # add simple positive spikes
    for s in [0.3, 0.7, 1.2, 1.6]:
        idx = int(s * fs)
        x[idx:idx + 3] += np.array([0.5, 1.0, 0.4])

    res = process_cell_simple(
        x,
        fs,
        negative_going=False,
        cs_high_cut=150.0,
        cs_thresh_sigma=3.0,
        ss_low_cut=50.0,
        ss_high_cut=700.0,
        ss_thresh_sigma=2.0,
    )

    for key in ['detrended', 'baseline', 'cs_trace', 'ss_trace', 'cs_peaks', 'ss_peaks', 'sigma_cs', 'sigma_ss', 'raw_sigma']:
        assert key in res

    assert len(res['detrended']) == len(x)
    assert len(res['cs_trace']) == len(x)
    assert len(res['ss_trace']) == len(x)
    assert np.asarray(res['cs_peaks']).ndim == 1
    assert np.asarray(res['ss_peaks']).ndim == 1
