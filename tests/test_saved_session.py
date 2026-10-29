"""Archived results restore their own settings and never rerun detection for viewing."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/spike-detector-test-mpl')
import numpy as np
import pandas as pd
from PyQt6 import QtWidgets
from spike_detector.gui import MainWindow, DetectionViewerDialog, StatsViewerDialog, SettingsDialog
from spike_detector.utils.saved_session import load_saved_detection, load_saved_detection_folder


class SavedSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        t = 100 + np.arange(1600, dtype=float)
        wave = 100 + np.random.default_rng(18).normal(0, .1, len(t))
        wave += 8*np.exp(-.5*((t-600)/7)**2) + 4*np.exp(-.5*((t-1100)/.8)**2)
        pd.DataFrame({'time_ms': t, 'cell': wave}).to_csv(self.root/'session.csv', index=False)
        self.win = MainWindow()
        self.win.tabs_polarity.setCurrentIndex(0)
        self.win.spin_avg_frames.setValue(0)
        self.win.qc_inputs['SS_MAX_FWHM_MS'].setValue(0)
        self.win.master_folder = str(self.root)
        self.win.refresh_sessions()
        self.win.chk_detection_save.setChecked(True)
        self.assertEqual(self.win.run_detection_all(), (1, 1))
        self.original = self.win.data
        self.archive = Path(self.original['results_file'])

    def tearDown(self):
        self.win.close()
        self.tmp.cleanup()

    def rewrite(self, changes=None, drop=()):
        with np.load(self.archive, allow_pickle=True) as z:
            fields = {key: z[key] for key in z.files if key not in drop}
        fields.update(changes or {})
        np.savez_compressed(self.archive, **fields)

    def test_exact_results_metrics_and_preprocessing_restore_without_detection(self):
        self.win.spin_avg_frames.setValue(5)
        self.win.spin_baseline_window.setValue(300)
        self.win.tabs_polarity.setCurrentIndex(1)
        self.win.selected_cell = 7  # previous recording had more cells
        with patch('spike_detector.gui._detect_cell', side_effect=AssertionError('must not detect')):
            self.assertEqual(self.win.load_saved_folder(self.root), (1, 1))
            loaded = self.win.data
            self.assertEqual(self.win.master_folder, str(self.root))
            self.assertEqual(self.win.selected_session, 'session.csv')
            self.assertEqual(self.win.spin_avg_frames.value(), 0)
            self.assertEqual(self.win.tabs_polarity.currentIndex(), 0)
            self.assertEqual(self.win.spin_baseline_window.value(), 30)
            for kind in ('cs', 'ss'):
                a, b = self.original['results'][0], loaded['results'][0]
                for key in (kind+'_peaks', kind+'_trace', kind+'_exclusion_mask', 'event_amplitude_snr_'+kind, 'event_minus_dff_percent_'+kind):
                    np.testing.assert_allclose(a[key], b[key], equal_nan=True)
            self.win.spin_avg_frames.setValue(7)
            self.win.spin_baseline_window.setValue(900)
            self.win.update_plot()
            line = self.win.canvas.figure.axes[0].lines[0]
            x, y = line.get_data()
            idx = np.searchsorted(loaded['time_ms'], x)
            np.testing.assert_allclose(y, loaded['processed_raw_data'][idx, 0])
            viewer = DetectionViewerDialog(loaded, parent=self.win)
            self.assertFalse(viewer.btn_polarity_control.isEnabled())
            viewer.close()
            stats = StatsViewerDialog(loaded, parent=self.win)
            self.assertTrue(stats.fig.axes)
            stats.close()
            np.testing.assert_allclose(self.win._get_waveform_source_trace_for_stats(loaded, 0),
                                       self.original['results'][0]['stats_source_trace'])

    def test_embedded_run_settings_win_and_switching_recordings_restores_each_run(self):
        with np.load(self.archive, allow_pickle=True) as z:
            fields = {key: z[key] for key in z.files}
        params = json.loads(str(fields['detection_params_json'].item()))
        base = json.loads(str(fields['baseline_params_json'].item()))
        params.update(NEGATIVE_GOING=True, CS_THRESHOLD_SIGMA=7.25, STATS_SS_ACG_BIN_MS=1.5)
        base['window_ms'] = 63.25
        fields['detection_params_json'] = json.dumps(params)
        fields['baseline_params_json'] = json.dumps(base)
        np.savez_compressed(self.archive.parent/'second_analyzed.npz', **fields)
        sidecar = self.archive.with_name(self.archive.stem+'_settings.json')
        cfg = json.loads(sidecar.read_text());cfg['params']['CS_THRESHOLD_SIGMA'] = 99
        sidecar.write_text(json.dumps(cfg))
        self.win.load_saved_folder(self.root)
        self.assertEqual(self.win.params['CS_THRESHOLD_SIGMA'], 7.25) # second sorts first
        self.assertEqual(self.win.baseline_params['window_ms'], 63.25)
        self.assertEqual(self.win.spin_cs_thresh.value(), 7.3)
        self.win.list_sessions.setCurrentText('session.csv')
        self.assertEqual(self.win.params['CS_THRESHOLD_SIGMA'], 6)
        self.assertFalse(self.win.params['NEGATIVE_GOING'])
        viewer = DetectionViewerDialog(self.win.data, parent=self.win)
        viewer.combo_session.setCurrentText('second_analyzed.npz')
        self.assertEqual(self.win.selected_session, 'second_analyzed.npz')
        self.assertEqual(self.win.params['CS_THRESHOLD_SIGMA'], 7.25)
        self.assertTrue(self.win.params['NEGATIVE_GOING'])
        viewer.close()

    def test_legacy_template_scores_not_invented_and_missing_templates_do_not_block(self):
        with np.load(self.archive, allow_pickle=True) as z:
            params=json.loads(str(z['detection_params_json'].item()))
            analysis=json.loads(str(z['analysis_settings_json'].item()))
            cfg=json.loads(str(z['gui_settings_json'].item()))
        params['DETECTION_METHOD']='Template Matching'
        analysis['detection_method']='Template Matching'
        cfg['template_sources']={'CS':['missing-bank.npz'],'SS':[]}
        self.rewrite({'detection_params_json':json.dumps(params), 'analysis_settings_json':json.dumps(analysis),
                      'gui_settings_json':json.dumps(cfg)}, drop=('detection_runtime_json',
                    'cs_similarity_traces','ss_similarity_traces','cs_threshold_traces','ss_threshold_traces'))
        self.win.load_saved_folder(self.root/'spike_detection')
        result=self.win.data['results'][0]
        self.assertTrue(result['saved_score_unavailable'])
        self.assertTrue(self.win._settings_restore_warnings)
        viewer=DetectionViewerDialog(self.win.data,parent=self.win)
        self.assertIn('scores unavailable',viewer.lbl_candidate_info.text())
        self.assertEqual(len(viewer.ax_cs.lines),1)
        viewer.close()

    def test_invalid_archive_keeps_current_session_and_partial_folder_skips_invalid(self):
        old=self.win.data
        bad=self.root/'bad';bad.mkdir()
        np.savez(bad/'wrong_analyzed.npz',raw_data=[1])
        with self.assertRaisesRegex(ValueError,'No compatible'):
            self.win.load_saved_folder(bad)
        self.assertIs(self.win.data,old)
        np.savez(self.archive.parent/'wrong_analyzed.npz',raw_data=[1])
        folder,loaded,issues=load_saved_detection_folder(self.root)
        self.assertEqual(len(loaded),1)
        self.assertEqual(len(issues),1)
        # The current input can disappear; a saved session is self-contained.
        (self.root/'session.csv').unlink()
        self.win.load_saved_folder(self.root)
        self.assertTrue(self.win.data['results'])

    def test_async_load_reenables_gui_and_opens_detection_viewer(self):
        with (patch('spike_detector.gui.QFileDialog.getExistingDirectory',return_value=str(self.root)),
              patch.object(self.win,'open_detection_viewer') as opened,
              patch('spike_detector.gui.QMessageBox.warning') as warning):
            self.win.btn_load_session.click()
            deadline=time.monotonic()+10
            while not opened.called and time.monotonic()<deadline:
                self.app.processEvents()
                time.sleep(.01)
            warning.assert_not_called()
            opened.assert_called_once()
            self.assertTrue(self.win.centralWidget().isEnabled())
            self.assertIsNone(self.win._saved_load_worker)

    def test_new_detection_clears_archival_bindings_and_uses_edited_controls(self):
        self.win.load_saved_folder(self.root)
        self.win.chk_detection_save.setChecked(False)
        self.win.spin_avg_frames.setValue(3)
        self.win.spin_cs_thresh.setValue(0)
        self.assertEqual(self.win.run_detection_all(),(1,1))
        self.assertNotIn('_saved_detection_params',self.win.data)
        self.assertFalse(self.win.data.get('loaded_detection_result'))
        self.assertEqual(len(self.win.data['results'][0]['cs_peaks']),0)
        self.assertTrue(self.win.data['temporary_result'])

    def test_template_and_combo_scores_masks_and_runtime_restore_exactly(self):
        for combo in (False, True):
            self.win.tabs_detection.setCurrentIndex(1)
            self.win.chk_combo.setChecked(combo)
            self.win.chk_template_whitening.setChecked(True)
            self.win.spin_template_lowpass.setValue(0)
            t=np.arange(31)-15
            self.win.template_store.update(cs_templates=[np.exp(-.5*(t/7)**2)],
                fs_cs=[1000], cs_sources=[],
                ss_templates=[np.exp(-.5*((np.arange(9)-4)/.8)**2)], fs_ss=[1000], ss_sources=[])
            with patch.object(self.win,'_confirm_result_replacement',return_value=True):
                self.assertEqual(self.win.run_detection_all(),(1,1))
            original=self.win.data['results'][0]
            saved=load_saved_detection(self.win.data['results_file'])
            restored=saved['results'][0]
            for key in ('cs_similarity_trace','ss_similarity_trace','cs_threshold_used','ss_threshold_used',
                        'cs_peaks','ss_peaks','cs_exclusion_mask','ss_exclusion_mask'):
                np.testing.assert_allclose(original[key],restored[key],equal_nan=True)
            self.assertEqual(original['template_whitening'],restored['template_whitening'])
            if combo:
                self.assertEqual(json.dumps(original['combo'],default=lambda v:v.tolist()),json.dumps(restored['combo']))
            self.assertFalse(restored['saved_score_unavailable'])
            self.win.load_saved_folder(self.root)
            viewer=DetectionViewerDialog(self.win.data,parent=self.win)
            self.assertNotIn('scores unavailable',viewer.lbl_candidate_info.text())
            viewer.close()

    def test_local_threshold_vectors_restore_and_legacy_missing_vectors_are_labelled(self):
        self.win.chk_local_baseline.setChecked(True)
        with patch.object(self.win,'_confirm_result_replacement',return_value=True):
            self.win.run_detection_all()
        original=self.win.data['results'][0]
        restored=load_saved_detection(self.archive)['results'][0]
        np.testing.assert_allclose(original['cs_threshold_trace'],restored['cs_threshold_trace'])
        self.assertFalse(restored['loaded_local_threshold_unavailable'])
        self.rewrite(drop=('cs_threshold_traces','ss_threshold_traces'))
        self.win.load_saved_folder(self.root)
        viewer=DetectionViewerDialog(self.win.data,parent=self.win)
        self.assertIn('local threshold vectors unavailable',viewer.lbl_candidate_info.text())
        self.assertEqual(len(viewer.ax_cs.lines),1)
        viewer.close()

    def test_controls_precision_reset_location_and_spacing(self):
        for spin in (self.win.spin_cs_thresh,self.win.spin_ss_thresh,self.win.spin_template_cs_sigma,
                     self.win.spin_window,self.win.spin_baseline_window,*self.win.qc_inputs.values()):
            self.assertEqual(spin.decimals(),1)
        for spin in (self.win.spin_cs_low,self.win.spin_ss_high,self.win.spin_template_lowpass):
            self.assertEqual(spin.decimals(),0)
        labels=[label.text() for label in self.win.findChildren(QtWidgets.QLabel)]
        self.assertNotIn('Spike polarity:',labels)
        self.assertNotIn('Spike Detection',labels)
        self.assertIn('Baseline correction:',labels)
        self.assertFalse(any(b.text()=='Reset App' for b in self.win.findChildren(QtWidgets.QPushButton)))
        advanced=SettingsDialog(self.win)
        self.assertTrue(any(b.text()=='Reset App' for b in advanced.findChildren(QtWidgets.QPushButton)))
        advanced.close()


if __name__=='__main__':
    unittest.main()
