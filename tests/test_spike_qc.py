"""Shared QC gates, CS/SS masking order and persistence of measured metrics."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/spike-detector-test-mpl')
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PyQt6 import QtWidgets
from spike_detector.gui import (MainWindow, SettingsDialog, DetectionViewerDialog,
                                SliderViewerDialog, DenoisingSettingsDialog, _detect_cell)
from spike_detector.utils.qc import run_with_qc, validate_qc, migrate_qc_settings
from spike_detector.utils.processing import compute_event_snrs
from spike_detector.utils.denoise import default_denoise_config


class SpikeQCTests(unittest.TestCase):
    def fixture(self):
        fs = 5000.
        t = np.arange(12000) / fs * 1000
        source = np.random.default_rng(12).normal(0, .1, len(t))
        trace = np.zeros(len(t))
        peaks = np.array([2000, 5000, 8000])
        for p, height, sigma in zip(peaks, [8, 2, 8], [.45, .45, 2.8]):
            wave = height * np.exp(-.5*((t-t[p])/sigma)**2)
            source += wave
            trace += wave
        def detect(cs_keep_indices=None):
            return dict(cs_peaks=np.array([], dtype=int), ss_peaks=peaks.copy(),
                cs_trace=trace, ss_trace=trace, detrended=source,
                cs_exclusion_mask=np.zeros(len(t), bool), ss_exclusion_mask=np.zeros(len(t), bool))
        return source, fs, detect

    def test_amplitude_min_and_both_width_limits_and_zero_disable(self):
        source, fs, detect = self.fixture()
        # Disable all limits first; all three proposals survive.
        limits = dict(CS_MIN_FWHM_MS=0, SS_MAX_FWHM_MS=0)
        all_events = run_with_qc(detect, source, fs, limits)
        np.testing.assert_array_equal(all_events['ss_peaks'], [2000, 5000, 8000])
        # One event fails amplitude, one is too broad, one passes.
        filtered = run_with_qc(detect, source, fs,
            dict(limits, SS_MIN_AMPLITUDE_SNR=40, SS_MIN_FWHM_MS=.5, SS_MAX_FWHM_MS=4.5))
        np.testing.assert_array_equal(filtered['ss_peaks'], [2000])
        self.assertEqual(filtered['spike_qc']['ss']['amplitude_rejected'], 1)
        self.assertEqual(filtered['spike_qc']['ss']['width_rejected'], 1)
        narrow = run_with_qc(detect, source, fs, dict(limits, SS_MIN_FWHM_MS=2))
        np.testing.assert_array_equal(narrow['ss_peaks'], [8000])
        # Saved SNR equals the QC decision metric, without re-estimating noise
        # from a different mask after rejected events disappear.
        original = compute_event_snrs(all_events, 'SS', fs)
        self.assertEqual(compute_event_snrs(filtered, 'SS', fs), original[:1])
        self.assertEqual(filtered['event_widths_ss'][0]['amplitude_snr'], original[0])

    def test_rejected_cs_does_not_keep_masking_ss(self):
        source, fs, ss_detect = self.fixture()
        calls = []
        def detect(cs_keep_indices=None):
            calls.append(cs_keep_indices)
            res = ss_detect()
            res['cs_peaks'] = np.array([8000]) if cs_keep_indices is None else np.asarray(cs_keep_indices)
            res['ss_peaks'] = np.array([2000]) if len(res['cs_peaks']) else np.array([2000, 8000])
            res['ss_exclusion_mask'][7900:8100] = bool(len(res['cs_peaks']))
            return res
        res = run_with_qc(detect, source, fs, dict(CS_MAX_FWHM_MS=5, SS_MAX_FWHM_MS=0))
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(res['cs_peaks']), 0)
        self.assertIn(8000, res['ss_peaks'])
        self.assertFalse(res['ss_exclusion_mask'][8000])

    def test_unknown_width_flagged_unknown_amplitude_explicit(self):
        source, fs, detect = self.fixture()
        source[:] = 0  # no estimable noise, hence no amplitude SNR
        res = run_with_qc(detect, source, fs, dict(SS_MIN_AMPLITUDE_SNR=1))
        self.assertEqual(len(res['ss_peaks']), 0)
        self.assertEqual(res['spike_qc']['ss']['amplitude_unavailable'], 3)
        def uncertain(**kwargs):
            result = detect(**kwargs)
            result['ss_trace'][:] = 1
            return result
        res = run_with_qc(uncertain, source, fs, {})
        self.assertEqual(len(res['ss_peaks']), 3)
        self.assertEqual(res['spike_qc']['ss']['width_uncertain'], 3)
        self.assertTrue(all(row['decision'] == 'uncertain' for row in res['event_widths_ss']))

    def test_invalid_limits_and_legacy_projection_not_reinterpreted_as_snr(self):
        for params in (dict(CS_MIN_FWHM_MS=10, CS_MAX_FWHM_MS=4),
                       dict(SS_MIN_AMPLITUDE_SNR=float('nan')), dict(SS_MIN_FWHM_MS=-1)):
            with self.assertRaises(ValueError):
                validate_qc(params)
        old = migrate_qc_settings(dict(TEMPLATE_MIN_RESPONSE_SIGMA=9,
            TEMPLATE_CS_MIN_FILTERED_PEAK_SIGMA=7, SS_MAX_FWHM_FILTER_ENABLED=False))
        self.assertEqual(old['SS_MAX_FWHM_MS'], 0)
        self.assertEqual(old['SS_MIN_AMPLITUDE_SNR'], 0)
        self.assertEqual(old['CS_MIN_AMPLITUDE_SNR'], 0)
        self.assertNotIn('TEMPLATE_MIN_RESPONSE_SIGMA', old)

    def test_actual_worker_gates_cs_and_ss_in_every_matching_mode(self):
        source, fs, _ = self.fixture()
        cs_bank = [np.exp(-.5*((np.arange(201)-100)/14)**2)]
        ss_bank = [np.exp(-.5*((np.arange(41)-20)/2.25)**2)]
        context = dict(baseline_params=dict(method='Disable', window_ms=30, percentile=20),
            frames=0, frame_mode='Rolling average', templates=dict(cs_templates=cs_bank,
                fs_cs=[fs], ss_templates=ss_bank, fs_ss=[fs]), denoise_cfg=default_denoise_config())
        for mode in ('Threshold', 'LLR Probability Vector', 'Burst-aware LLR', 'Normalized Similarity', 'Combo'):
            with self.subTest(mode=mode):
                params = dict(NEGATIVE_GOING=False, INITIAL_BLANK_MS=0,
                    DETECTION_METHOD='Threshold' if mode in ('Threshold','Combo') else 'Template Matching',
                    COMBO_DETECTION=mode == 'Combo', TEMPLATE_MATCH_METHOD=mode if mode not in ('Threshold','Combo') else 'LLR Probability Vector',
                    CS_MIN_AMPLITUDE_SNR=1000, SS_MIN_AMPLITUDE_SNR=1000,
                    CS_MIN_FWHM_MS=0, SS_MAX_FWHM_MS=0, CS_THRESHOLD_SIGMA=5, SS_THRESHOLD_SIGMA=5)
                res = _detect_cell(source, fs, params=params, **context)
                self.assertEqual(len(res['cs_peaks']), 0)
                self.assertEqual(len(res['ss_peaks']), 0)
                self.assertEqual(res['cs_min_filtered_peak_sigma_used'] if 'cs_min_filtered_peak_sigma_used' in res else 0, 0)
                self.assertEqual(res['spike_qc']['limits']['SS_MIN_AMPLITUDE_SNR'], 1000)


class SpikeQCGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_native_polarity_tabs_qc_visible_and_zero_help_only_in_tooltips(self):
        win = MainWindow()
        try:
            win.show(); self.app.processEvents()
            self.assertIsInstance(win.tabs_polarity, QtWidgets.QTabWidget)
            self.assertEqual(win.tabs_polarity.styleSheet(), '')
            self.assertFalse(win.tabs_polarity.tabBar().usesScrollButtons())
            self.assertGreaterEqual(win.tabs_polarity.width(), win.tabs_polarity.tabBar().sizeHint().width())
            for tab in (0, 1):
                win.tabs_detection.setCurrentIndex(tab); self.app.processEvents()
                self.assertTrue(win.group_spike_qc.isVisible())
            self.assertEqual(len(win.qc_inputs), 6)
            for spin in win.qc_inputs.values():
                self.assertIn('0 disables', spin.toolTip())
                self.assertEqual(spin.lineEdit().toolTip(), spin.toolTip())
            for label in win.findChildren(QtWidgets.QLabel):
                self.assertNotIn('0=off', label.text())
                self.assertNotIn('0 = off', label.text())
            advanced = SettingsDialog(win)
            self.assertFalse(hasattr(advanced, 'spin_cs_min_fwhm'))
            self.assertFalse(hasattr(advanced, 'spin_ss_max_fwhm'))
            advanced.close()
        finally:
            win.close()

    def test_legacy_load_width_disable_and_new_panel_roundtrip(self):
        win = MainWindow()
        try:
            with tempfile.TemporaryDirectory() as folder:
                cfg = Path(folder) / 'old.json'
                cfg.write_text(json.dumps(dict(app_version='3.6.41', params=dict(
                    SS_MAX_FWHM_FILTER_ENABLED=False, SS_MAX_FWHM_MS=8,
                    TEMPLATE_CS_MIN_FILTERED_PEAK_SIGMA=17, TEMPLATE_MIN_RESPONSE_SIGMA=8))))
                with patch('spike_detector.gui.QFileDialog.getOpenFileName', return_value=(str(cfg), '')), \
                     patch('spike_detector.gui.QMessageBox.information') as loaded:
                    win.load_all_settings()
                self.assertIn('not amplitude SNR', loaded.call_args.args[2])
                self.assertEqual(win.qc_inputs['SS_MAX_FWHM_MS'].value(), 0)
                self.assertEqual(win.qc_inputs['CS_MIN_AMPLITUDE_SNR'].value(), 0)
                self.assertNotIn('TEMPLATE_MIN_RESPONSE_SIGMA', win._current_settings_snapshot()['params'])
        finally:
            win.close()

    def test_viewer_display_and_denoise_visibility_persist_with_settings(self):
        win = MainWindow()
        data = dict(raw_data=np.zeros((300, 1)), time_ms=np.arange(300),
                    cell_names=['Cell'], fs=1000.0, results=[None])
        try:
            viewer = DetectionViewerDialog(data, win)
            viewer.spin_window.setValue(750)
            viewer.spin_zoom.setValue(12)
            viewer.slider_time.setValue(450)
            viewer.close()
            raw_viewer = SliderViewerDialog(data, win)
            raw_viewer.spin_zoom.setValue(7)
            raw_viewer.slider_time.setValue(600)
            raw_viewer.close()
            denoise = DenoisingSettingsDialog(win)
            denoise._preview_timer.stop()
            denoise.chk_show_advanced.setChecked(True)
            denoise.save_and_accept()
            denoise.close()
            snapshot = win._current_settings_snapshot()
            with tempfile.TemporaryDirectory() as folder:
                cfg = Path(folder) / 'settings.json'
                cfg.write_text(json.dumps(snapshot))
                win.params.update(DETECTION_VIEW_WINDOW_MS=500, DETECTION_VIEW_Y_RANGE=0,
                    DETECTION_VIEW_CENTER=0, SLIDER_VIEW_Y_RANGE=0,
                    SLIDER_VIEW_CENTER=0, DENOISE_SHOW_ADVANCED=False)
                with patch('spike_detector.gui.QFileDialog.getOpenFileName', return_value=(str(cfg), '')), \
                     patch('spike_detector.gui.QMessageBox.information'), \
                     patch('spike_detector.gui.QMessageBox.critical') as failed:
                    win.load_all_settings()
                failed.assert_not_called()
            viewer = DetectionViewerDialog(data, win)
            self.assertEqual(viewer.spin_window.value(), 750)
            self.assertEqual(viewer.spin_zoom.value(), 12)
            self.assertEqual(viewer.slider_time.value(), 450)
            viewer.close()
            raw_viewer = SliderViewerDialog(data, win)
            self.assertEqual(raw_viewer.spin_zoom.value(), 7)
            self.assertEqual(raw_viewer.slider_time.value(), 600)
            raw_viewer.close()
            denoise = DenoisingSettingsDialog(win)
            denoise._preview_timer.stop()
            self.assertTrue(denoise.chk_show_advanced.isChecked())
            self.assertTrue(all(not group.isHidden() for group in denoise._advanced_groups))
            denoise.close()
            self.assertIn('rolling', win.chk_local_baseline.toolTip())
            self.assertNotIn('subtract', win.spin_local_cs_ms.toolTip().lower())
        finally:
            win.close()
