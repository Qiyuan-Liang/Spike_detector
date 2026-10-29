"""Combo learns shapes from seeds without imposing seed agreement on final events."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/spike-detector-test-mpl')
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from PyQt6 import QtWidgets, QtCore
from spike_detector.gui import MainWindow, DetectionViewerDialog, _detect_cell, _polarity_reversed_control
from spike_detector.utils.denoise import default_denoise_config


class ComboTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def signal(self):
        fs = 5000.0
        t = np.arange(15000)
        raw = np.random.default_rng(28).normal(0, .1, len(t))
        for peak in (2000, 7000, 12000):
            raw += 3 * np.exp(-.5 * ((t-peak)/15)**2)
        for peak in (1500, 4500, 8000, 11000):
            raw += 1.2 * np.exp(-.5 * ((t-peak)/2)**2)
        raw += .3 * np.exp(-.5 * ((t-9000)/2)**2)
        params = dict(COMBO_DETECTION=True, DETECTION_METHOD='Threshold', NEGATIVE_GOING=False,
            CS_THRESHOLD_SIGMA=8, SS_THRESHOLD_SIGMA=6, TEMPLATE_SS_SIGMA=3,
            INITIAL_BLANK_MS=0, TEMPLATE_SS_LOWPASS_HZ=700)
        context = dict(params=params, baseline_params=dict(method='Disable', window_ms=30, percentile=20),
            frames=0, frame_mode='Rolling average', templates={}, denoise_cfg=default_denoise_config())
        return raw, fs, context

    def test_learns_both_types_and_recovers_event_below_seed_cutoff(self):
        raw, fs, context = self.signal()
        result = _detect_cell(raw, fs, **context)
        self.assertEqual(result['combo']['cs']['source'], 'auto')
        self.assertEqual(result['combo']['ss']['source'], 'auto')
        self.assertFalse(any(abs(p-9000) < 5 for p in result['combo']['ss']['seed_indices']))
        self.assertTrue(any(abs(p-9000) < 5 for p in result['ss_peaks']))
        self.assertFalse(np.any(np.abs(result['ss_peaks'] - 7000) < 40))
        self.assertEqual(context['templates'], {})
        json.dumps(result['combo'], allow_nan=False)
        with patch('spike_detector.gui.process_cell_combo', side_effect=AssertionError('control retrained')):
            control = _polarity_reversed_control(raw, fs, context, result)
        self.assertIn('ss_kept', control)

    def test_missing_seeds_and_loaded_fallback_are_explicit(self):
        raw, fs, context = self.signal()
        context['params']['CS_THRESHOLD_SIGMA'] = 10000
        context['params']['SS_THRESHOLD_SIGMA'] = 10000
        result = _detect_cell(raw, fs, **context)
        self.assertEqual(result['combo']['ss']['source'], 'unavailable')
        self.assertEqual(len(result['ss_peaks']), 0)
        wave = np.exp(-.5*((np.arange(41)-20)/2)**2)
        context['templates'] = dict(ss_templates=[wave], fs_ss=[fs])
        result = _detect_cell(raw, fs, **context)
        self.assertEqual(result['combo']['ss']['source'], 'loaded fallback')
        self.assertGreater(len(result['ss_peaks']), 0)
        np.testing.assert_array_equal(context['templates']['ss_templates'][0], wave)

    def test_combo_whitens_final_pass_and_control_reuses_noise_fit(self):
        raw, fs, context = self.signal()
        context['params']['TEMPLATE_WHITENING'] = True
        # Explicit amplitude QC replaces the former implicit CS height floor.
        context['params']['CS_MIN_AMPLITUDE_SNR'] = 3.0
        result = _detect_cell(raw, fs, **context)
        for kind in ('cs', 'ss'):
            self.assertEqual(result['combo'][kind]['source'], 'auto')
            self.assertTrue(result['template_whitening'][kind]['applied'])
            self.assertGreater(len(result[kind+'_peaks']), 0)
        with patch('spike_detector.utils.whitening.fit_whitening', side_effect=AssertionError('control refitted noise')):
            control = _polarity_reversed_control(raw, fs, context, result)
        self.assertIn('cs_kept', control)
        # Also exercise the GUI worker's ordinary loaded-template route.
        context['params'].update(COMBO_DETECTION=False, DETECTION_METHOD='Template Matching')
        context['templates'] = result['_combo_templates']
        loaded = _detect_cell(raw, fs, **context)
        np.testing.assert_array_equal(loaded['cs_peaks'], result['cs_peaks'])
        np.testing.assert_array_equal(loaded['ss_peaks'], result['ss_peaks'])

    def test_reset_turns_whitening_off_even_when_method_changes(self):
        win = MainWindow()
        try:
            win.spin_template_cs_sigma.setValue(17)
            win.spin_template_ss_sigma.setValue(8)
            win.spin_ss_thresh.setValue(4.2)
            win.chk_template_whitening.setChecked(True)
            win.combo_template_method.setCurrentIndex(1)
            self.assertTrue(win._current_settings_snapshot()['params']['TEMPLATE_WHITENING'])
            with patch('spike_detector.gui.QMessageBox.question', return_value=QtWidgets.QMessageBox.StandardButton.Yes):
                win.reset_app()
            self.assertFalse(win.chk_template_whitening.isChecked())
            self.assertFalse(win.params['TEMPLATE_WHITENING'])
            self.assertEqual(win.spin_template_cs_sigma.value(),30)
            self.assertEqual(win.spin_template_ss_sigma.value(),5)
            self.assertEqual(win.spin_ss_thresh.value(),3)
        finally:
            win.close()

    def test_all_template_methods_parallel_and_sparse_cs_mask(self):
        raw, fs, context = self.signal()
        context['params'].update(TEMPLATE_PARALLEL=True, TEMPLATE_PARALLEL_GROUPS=3,
                                 TEMPLATE_CS_SELECTED_GROUPS=[2], TEMPLATE_SS_SELECTED_GROUPS=[2])
        for method in ('LLR Probability Vector', 'Normalized Similarity', 'Burst-aware LLR'):
            with self.subTest(method=method):
                context['params']['TEMPLATE_MATCH_METHOD'] = method
                result = _detect_cell(raw, fs, **context)
                self.assertEqual(result['combo']['ss']['source'], 'auto')
                self.assertGreater(len(result['ss_peaks']), 0)
        t = np.arange(len(raw))
        for peak in (2000, 12000):
            raw -= 3 * np.exp(-.5*((t-peak)/15)**2)
        result = _detect_cell(raw, fs, **context)
        self.assertEqual(result['combo']['cs']['source'], 'unavailable')
        self.assertEqual(len(result['cs_peaks']), 0)
        self.assertTrue(result['ss_exclusion_mask'][7000])
        self.assertFalse(np.any(np.abs(result['ss_peaks']-7000) < 40))

    def test_checkbox_batch_banks_export_and_async_detection(self):
        win = MainWindow()
        try:
            self.assertIs(win.tabs_detection.cornerWidget(QtCore.Qt.Corner.TopRightCorner), win.chk_combo)
            self.assertEqual(win.spin_template_cs_sigma.value(),30)
            self.assertEqual(win.spin_template_ss_sigma.value(),5)
            self.assertEqual(win.spin_ss_thresh.value(),3)
            self.assertEqual(win.lbl_ss_threshold.text(),'Threshold (σ):')
            win.chk_combo.setChecked(True)
            self.assertEqual(win.lbl_ss_threshold.text(),'Seed (σ):')
            win.chk_template_whitening.setChecked(True)
            self.assertTrue(win._current_settings_snapshot()['params']['COMBO_DETECTION'])
            win.tabs_detection.setCurrentIndex(1)
            win.params['CS_MIN_FWHM_MS'] = 100
            win.spin_ss_thresh.setValue(5)
            with tempfile.TemporaryDirectory() as folder:
                paths = []
                for i in range(2):
                    root = Path(folder) / str(i)
                    root.mkdir()
                    t = np.arange(1500)
                    raw = 100 + np.random.default_rng(i).normal(0, .15, len(t))
                    for peak in (250, 510, 800, 1220):
                        raw -= 5 * np.exp(-.5*((t-peak)/(.8+i*.2))**2)
                    pd.DataFrame({'time_ms': t, 'cell': raw}).to_csv(root/'recording.csv', index=False)
                    paths.append(str(root))
                win.load_batch_folders(paths)
                with patch('spike_detector.gui.QMessageBox.warning') as warning:
                    self.assertTrue(win.start_detection_async())
                    loop = QtCore.QEventLoop()
                    win.detection_completed.connect(loop.quit)
                    timer = QtCore.QTimer()
                    timer.setSingleShot(True)
                    timer.timeout.connect(loop.quit)
                    timer.start(15000)
                    loop.exec()
                    timer.stop()
                    self.assertFalse(win._detection_running)
                    warning.assert_not_called()
                banks = []
                for data in win.loaded_sessions.values():
                    result = data['results'][0]
                    self.assertEqual(result['combo']['ss']['source'], 'auto')
                    self.assertEqual(result['combo']['cs']['source'], 'unavailable')
                    self.assertTrue(result['template_whitening']['requested'])
                    self.assertTrue(result['template_whitening']['ss']['applied'])
                    banks.append(result['_combo_templates']['ss_templates'])
                    with np.load(data['results_file'], allow_pickle=True) as saved:
                        settings = json.loads(str(saved['analysis_settings_json']))
                        self.assertTrue(settings['combo_detection'])
                        self.assertTrue(settings['params']['TEMPLATE_WHITENING'])
                        self.assertEqual(settings['event_snr_policy']['noise_estimation_interval_ms'], [-150., -5.])
                        self.assertTrue(settings['template_whitening_cells'][0]['ss']['applied'])
                        self.assertIn('templates', settings['combo_cells'][0])
                self.assertIsNot(banks[0], banks[1])
                self.assertEqual(win.template_store['ss_templates'], [])
                viewer = DetectionViewerDialog(win.data, win)
                self.assertIn('Combo:', viewer.lbl_candidate_info.text())
                self.assertIn('Whitening:', viewer.lbl_candidate_info.text())
                viewer.close()
        finally:
            win.close()


if __name__ == '__main__':
    unittest.main()
