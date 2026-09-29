"""Run with QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -m unittest discover -s tests."""
import json
import os
from pathlib import Path
import tempfile
import time
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/spike-detector-test-mpl')

import numpy as np
import pandas as pd
from PyQt6 import QtWidgets, QtCore
from spike_detector.gui import (MainWindow, StatsViewerDialog, DetectionViewerDialog,
    BatchDetectionDialog, SettingsDialog, TemplateViewerDialog)
from spike_detector.utils.batch import discover_batch_sessions, normalize_folder_paths


class BatchDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.folders = [self.root / 'a' / 'recording', self.root / 'b' / 'recording']
        rng = np.random.default_rng(6)
        self.time = np.arange(1500, dtype=float)
        for i, folder in enumerate(self.folders):
            folder.mkdir(parents=True)
            trace = 100 + rng.normal(0, .15, self.time.size)
            for peak in [250, 510, 800, 1220]:
                trace -= (4+i) * np.exp(-.5*((self.time-peak)/.8)**2)
            pd.DataFrame({'time': self.time, f'Cell{i}': trace}).to_csv(folder/'session.csv', index=False)
        self.win = MainWindow()
        self.win.params['SS_MAX_FWHM_FILTER_ENABLED'] = False
        self.win.spin_avg_frames.setValue(0)

    def tearDown(self):
        self.win.close()
        self.temp.cleanup()

    def test_ss_width_default_and_explicit_override(self):
        self.assertEqual(self.win.params['SS_MAX_FWHM_MS'], 4.5)
        dialog = SettingsDialog(self.win)
        self.assertEqual(dialog.spin_ss_max_fwhm.value(), 4.5)
        dialog.spin_ss_max_fwhm.setValue(3.0)
        dialog.save_and_close()
        self.assertEqual(self.win._current_settings_snapshot()['params']['SS_MAX_FWHM_MS'], 3.0)
        with patch('spike_detector.gui.QMessageBox.question',
                   return_value=QtWidgets.QMessageBox.StandardButton.Yes):
            self.win.reset_app()
        self.assertEqual(self.win.params['SS_MAX_FWHM_MS'], 4.5)

    def test_cs_filtered_peak_setting_applies_to_batch_and_viewer(self):
        self.win.tabs_detection.setCurrentIndex(1)
        self.assertEqual(self.win.spin_template_cs_peak.value(), 3.0)
        self.assertIn('filtered-trace peak', self.win.lbl_template_cs_peak.text())
        self.win.spin_template_cs_peak.setValue(0.0)
        self.assertEqual(self.win._current_settings_snapshot()['params']['TEMPLATE_CS_MIN_FILTERED_PEAK_SIGMA'], 0.0)
        wave = np.exp(-.5 * ((np.arange(41) - 20) / 5.0) ** 2)
        self.win.template_store['cs_templates'] = [wave]
        self.win.template_store['fs_cs'] = [1000.0]
        self.win.load_batch_folders(map(str, self.folders))
        self.assertEqual(self.win.run_detection_all(), (2, 2))
        for data in self.win.loaded_sessions.values():
            self.assertEqual(data['effective_settings']['cs_min_filtered_peak_sigma'], 0.0)
            self.assertEqual(data['results'][0]['cs_min_filtered_peak_sigma_used'], 0.0)
        viewer = DetectionViewerDialog(self.win.data, self.win)
        self.assertIn('filtered peak rejected', viewer.lbl_candidate_info.text())
        viewer.close()

    def test_discovery_duplicates_invalid_and_nonrecursive(self):
        (self.folders[0]/'bad.csv').write_text('not a trace\nhello\n')
        nested = self.folders[0]/'spike_detection'
        nested.mkdir()
        (nested/'ignored.csv').write_text('x,y\n0,1\n1,2\n')
        paths, names, loaded, issues, folders = discover_batch_sessions(
            [str(self.folders[0]), str(self.folders[0]/'..'/'recording'), str(self.folders[1]), str(self.root/'missing')])
        self.assertEqual(len(paths), 2)
        self.assertEqual(len(set(names)), 2)
        self.assertEqual(len(folders), 3)
        self.assertTrue(any('bad.csv' in x for x in issues))
        self.assertTrue(any('Path unavailable' in x for x in issues))

    @unittest.skipIf(os.name == 'nt', 'macOS volume-path normalization')
    def test_pasted_volume_paths_and_closable_batch_errors(self):
        pasted = r'Volumes\T7 Shield\Organized\20260903_B10_tri_MZ\3PCs'
        self.assertEqual(normalize_folder_paths([pasted]),
                         ['/Volumes/T7 Shield/Organized/20260903_B10_tri_MZ/3PCs'])
        self.assertEqual(normalize_folder_paths(['recording'], base_dir=str(self.root / 'a')),
                         [str(self.folders[0])])
        dialog = BatchDetectionDialog(self.win)
        dialog.paths.setPlainText(r'Volumes\T7 Shield\missing_batch_folder')
        dialog.run_batch()
        self.assertFalse(dialog._running)
        self.assertIn('No valid sessions', dialog.status.text())
        self.assertIn('/Volumes/T7 Shield/missing_batch_folder',
                      dialog.issue_details.toPlainText())
        self.assertTrue(dialog.close_button.isEnabled())
        dialog.show()
        dialog.close()
        self.assertFalse(dialog.isVisible())

    @unittest.skipIf(os.name == 'nt', 'POSIX backslash-path normalization')
    def test_backslash_batch_paths_run_and_dialog_closes_during_detection(self):
        dialog = BatchDetectionDialog(self.win)
        dialog.paths.setPlainText('\n'.join(str(folder).replace('/', '\\') for folder in self.folders))
        dialog.run_batch()
        self.assertEqual(self.win.batch_folders, [str(folder) for folder in self.folders])
        self.assertTrue(dialog.close_button.isEnabled())
        dialog.show()
        dialog.close()
        self.assertFalse(dialog.isVisible())
        deadline = time.monotonic() + 15
        while self.win._detection_running and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertFalse(self.win._detection_running)
        self.assertEqual(len(self.win.loaded_sessions), 2)

    def test_batch_matches_single_and_preserves_exports(self):
        self.win.master_folder = str(self.folders[0])
        self.win.refresh_sessions()
        self.assertEqual(self.win.run_detection_all(), (1, 1))
        expected = self.win.data['results'][0]
        original = self.folders[0]/'spike_detection'/'session_analyzed.npz'
        original_bytes = original.read_bytes()
        self.win.load_batch_folders(map(str, self.folders))
        self.assertEqual(self.win.run_detection_all(), (2, 2))
        got = self.win.loaded_sessions[self.win.session_names[0]]['results'][0]
        np.testing.assert_array_equal(got['cs_peaks'], expected['cs_peaks'])
        np.testing.assert_array_equal(got['ss_peaks'], expected['ss_peaks'])
        self.assertGreater(len(got['ss_peaks']), 0)
        self.assertEqual(original.read_bytes(), original_bytes)
        settings = []
        for name, data in self.win.loaded_sessions.items():
            self.assertTrue(Path(data['results_file']).is_file())
            self.assertTrue(Path(data['results_file']).is_relative_to(Path(data['source_folder'])/'spike_detection'))
            settings.append(json.loads(Path(data['settings_file']).read_text())['analysis_settings'])
        self.assertEqual(settings[0], settings[1])
        self.assertNotIn('two_step', settings[0])
        info = self.win.text_stats.toPlainText()
        self.assertIn('Batch paths: 2', info)
        self.assertIn('2 sessions, 2 cells', info)
        for folder in self.folders:
            self.assertIn(str(folder), info)
        self.assertTrue(self.win.centralWidget().isEnabled())
        # Standard viewer still renders after removing the six-panel branch.
        viewer = DetectionViewerDialog(self.win.data, self.win)
        self.assertEqual(len(viewer.fig.axes), 4)
        viewer.close()

    def test_path_session_cell_filter_and_reset_to_single(self):
        self.win.load_batch_folders(map(str, self.folders))
        self.win.run_detection_all()
        dlg = StatsViewerDialog(self.win.data, self.win)
        self.assertEqual(len(dlg._path_session_names()), 2)
        with patch.object(self.win, '_get_waveform_source_trace_for_stats',
                          wraps=self.win._get_waveform_source_trace_for_stats) as source:
            dlg.combo_path.setCurrentText(str(self.folders[1]))
            self.assertTrue(source.called)
            self.assertTrue(all(call.args[0]['source_folder'] == str(self.folders[1])
                                for call in source.call_args_list))
        self.assertEqual(dlg._path_session_names(), [self.win.session_names[1]])
        self.assertEqual(dlg.combo_session.count(), 2)
        self.assertEqual([dlg.combo_cell.itemText(i) for i in range(dlg.combo_cell.count())], ['All', 'Cell1'])
        dlg.combo_session.setCurrentIndex(1)
        dlg.combo_cell.setCurrentText('Cell1')
        dlg.combo_path.setCurrentText(str(self.folders[0]))
        self.assertEqual(dlg.combo_session.currentText(), 'All')
        self.assertEqual(dlg.combo_cell.itemText(1), 'Cell0')
        dlg.close()
        self.win.master_folder = str(self.folders[0])
        self.win.refresh_sessions()
        self.assertEqual(self.win.batch_folders, [])
        self.assertEqual(self.win.session_names, ['session.csv'])

    def test_failure_visible_and_next_session_completes(self):
        from spike_detector import gui
        self.win.load_batch_folders(map(str, self.folders))
        actual = gui.process_cell_simple
        calls = 0
        def fail_first(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError('synthetic processing failure')
            return actual(*args, **kwargs)
        with patch.object(gui, 'process_cell_simple', side_effect=fail_first):
            self.assertEqual(self.win.run_detection_all(), (1, 1))
        self.assertIn('synthetic processing failure', self.win.text_stats.toPlainText())
        self.assertEqual(self.win.loaded_sessions[self.win.session_names[0]]['results'], [None])
        with patch.object(gui.np, 'savez_compressed', side_effect=OSError('synthetic write failure')):
            self.win.run_detection_all()
        self.assertIn('Save failed:', self.win.text_stats.toPlainText())
        self.assertIn('synthetic write failure', self.win.text_stats.toPlainText())

    def test_template_mode_and_dialog(self):
        self.win.tabs_detection.setCurrentIndex(1)
        wave = 5*np.exp(-.5*((np.arange(21)-10)/1.2)**2)
        self.win.template_store['ss_templates'] = [wave]
        self.win.template_store['fs_ss'] = [1000.0]
        self.win.load_batch_folders(map(str, self.folders))
        for parallel, method in ((True, 'Normalized Similarity'),
                                 (False, 'LLR Probability Vector'),
                                 (False, 'Burst-aware LLR')):
            self.win.chk_template_parallel.setChecked(parallel)
            self.win.combo_template_method.setCurrentIndex(
                self.win.combo_template_method.findData(method))
            self.assertEqual(self.win.run_detection_all(), (2, 2))
            for data in self.win.loaded_sessions.values():
                result = data['results'][0]
                self.assertIn('Positive-core LLR' if method == 'Burst-aware LLR' else method,
                              result['det_method'])
                self.assertEqual(result['parallel_match'], parallel)
                self.assertEqual(len(result['cs_peaks']), 0)
                self.assertEqual(len(result['cs_similarity_trace']), len(data['time_ms']))
                self.assertEqual(len(result['ss_similarity_trace']), len(data['time_ms']))
                self.assertTrue(result['cs_candidate_diagnostics']['no_templates'])
            viewer = DetectionViewerDialog(self.win.data, self.win)
            self.assertIn('CS: no templates loaded', viewer.lbl_candidate_info.text())
            viewer.close()
        dialog = BatchDetectionDialog(self.win)
        dialog.paths.setPlainText('\n'.join(map(str, self.folders)))
        dialog.run_batch()
        deadline = time.monotonic() + 15
        while dialog._running and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertIn('Finished: 2 sessions', dialog.status.text())
        self.assertTrue(dialog.run_button.isEnabled())
        self.win.clear_templates()
        self.assertEqual(len(self.win.template_store['ss_templates']), 0)
        self.assertEqual(len(self.win.template_store['cs_templates']), 0)
        self.win.template_store['cs_templates'] = [wave]
        self.win.template_store['fs_cs'] = [1000.0]
        self.assertEqual(self.win.run_detection_all(), (2, 2))
        for data in self.win.loaded_sessions.values():
            result = data['results'][0]
            self.assertEqual(len(result['ss_peaks']), 0)
            self.assertTrue(result['ss_candidate_diagnostics']['no_templates'])
        dialog.close()

    def test_clear_button_empties_both_template_banks(self):
        self.win.template_store.update({
            'cs_templates': [np.ones(9)], 'fs_cs': [1000.0], 'cs_sources': ['cs.npz'],
            'ss_templates': [np.ones(9)], 'fs_ss': [1000.0], 'ss_sources': ['ss.npz'],
        })
        clear_buttons = [button for button in self.win.findChildren(QtWidgets.QPushButton)
                         if button.text() == 'Clear']
        self.assertEqual(len(clear_buttons), 1)
        clear_buttons[0].click()
        for key in ('cs_templates', 'fs_cs', 'cs_sources',
                    'ss_templates', 'fs_ss', 'ss_sources'):
            self.assertEqual(self.win.template_store[key], [])
        self.assertIn('CS [0]', self.win.lbl_template_status.text())
        self.assertIn('SS [0]', self.win.lbl_template_status.text())

    def test_parallel_view_selects_groups_and_batch_uses_saved_settings(self):
        self.win.tabs_detection.setCurrentIndex(1)
        self.win.chk_template_parallel.setChecked(True)
        self.win.spin_template_groups.setValue(2)
        wave = np.exp(-.5*((np.arange(41)-20)/2.0)**2)
        self.win.template_store['ss_templates'] = [wave, wave*2,
            np.exp(-.5*((np.arange(41)-20)/5.0)**2)]
        self.win.template_store['fs_ss'] = [1000.0]*3
        viewer = TemplateViewerDialog(self.win.template_store, self.win)
        self.assertEqual(len(viewer.group_checks['SS']), 2)
        viewer.group_checks['SS'][1].click()
        self.assertEqual(self.win.params['TEMPLATE_SS_SELECTED_GROUPS'], [1])
        viewer.close()
        settings = SettingsDialog(self.win)
        settings.spin_template_components.setValue(1)
        settings.save_and_close()
        self.assertIsNone(self.win.params['TEMPLATE_SS_SELECTED_GROUPS'])
        viewer = TemplateViewerDialog(self.win.template_store, self.win)
        viewer.group_checks['SS'][1].click()
        viewer.close()
        self.assertEqual(self.win._current_settings_snapshot()['params']['TEMPLATE_SS_SELECTED_GROUPS'], [1])
        self.win.load_batch_folders(map(str, self.folders))
        self.assertEqual(self.win.run_detection_all(), (2, 2))
        for data in self.win.loaded_sessions.values():
            self.assertEqual(data['effective_settings']['parallel_template_groups'], 2)
            self.assertEqual(data['effective_settings']['parallel_template_components'], 1)
            self.assertEqual(data['_control_context']['params']['TEMPLATE_SS_SELECTED_GROUPS'], [1])

    def test_polarity_control_uses_saved_settings_and_reports_in_viewer(self):
        self.win.tabs_detection.setCurrentIndex(1)
        wave = 5*np.exp(-.5*((np.arange(21)-10)/1.2)**2)
        self.win.template_store['ss_templates'] = [wave]
        self.win.template_store['fs_ss'] = [1000.0]
        self.win.load_batch_folders([str(self.folders[0])])
        self.assertEqual(self.win.run_detection_all(), (1, 1))
        data = self.win.data
        viewer = DetectionViewerDialog(data, self.win)
        self.assertTrue(viewer.btn_polarity_control.isEnabled())
        original_mask = data['results'][0]['ss_exclusion_mask'].copy()
        viewer.btn_polarity_control.click()
        deadline = time.monotonic() + 15
        while viewer._polarity_control_thread is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertIsNone(viewer._polarity_control_thread)
        summary = data['results'][0]['polarity_control']
        self.assertIn('ss_kept', summary)
        self.assertIn('Polarity reversed (same settings and masks', viewer.lbl_candidate_info.text())
        np.testing.assert_array_equal(data['results'][0]['ss_exclusion_mask'], original_mask)
        viewer.close()

    def test_advanced_ss_spacing_is_the_only_gui_source_for_template_detection(self):
        self.assertFalse(hasattr(self.win, 'spin_ss_mind'))
        settings = SettingsDialog(self.win)
        settings.spin_ss_mind.setValue(6.0)
        settings.save_and_close()
        self.assertEqual(self.win.params['SS_MIN_DIST_MS'], 6.0)
        self.win.tabs_detection.setCurrentIndex(1)
        self.assertEqual(self.win.spin_template_cs_sigma.maximum(), 100.0)
        self.assertEqual(self.win.spin_template_ss_sigma.maximum(), 100.0)
        self.win.spin_template_cs_sigma.setValue(80.0)
        self.assertEqual(self.win._current_settings_snapshot()['params']['TEMPLATE_CS_SIGMA'], 80.0)
        wave = 5*np.exp(-.5*((np.arange(21)-10)/1.2)**2)
        self.win.template_store['ss_templates'] = [wave]
        self.win.template_store['fs_ss'] = [1000.0]
        self.win.load_batch_folders([str(self.folders[0])])
        self.assertEqual(self.win.run_detection_all(), (1, 1))
        self.assertEqual(self.win.data['effective_settings']['ss_min_distance_samples'], 6)
        self.assertIn('SS spacing: 6.00 ms (Advanced Settings)', self.win.text_stats.toPlainText())

    def test_async_detection_keeps_event_loop_responsive(self):
        self.win.load_batch_folders(map(str, self.folders))
        main_thread = threading.get_ident()
        worker_threads = []
        save_threads = []
        fired = []
        save_fired = []
        completed = []
        gui_module = __import__('spike_detector.gui', fromlist=['process_cell_simple'])
        actual = gui_module.process_cell_simple
        actual_save = gui_module.np.savez_compressed

        def slow_detection(*args, **kwargs):
            worker_threads.append(threading.get_ident())
            time.sleep(.10)
            return actual(*args, **kwargs)

        def slow_save(*args, **kwargs):
            save_threads.append(threading.get_ident())
            time.sleep(.10)
            return actual_save(*args, **kwargs)

        self.win.detection_completed.connect(lambda sessions, cells: completed.append((sessions, cells)))
        QtCore.QTimer.singleShot(20, lambda: fired.append(not completed))
        self.win.detection_progress.valueChanged.connect(
            lambda value: QtCore.QTimer.singleShot(20, lambda: save_fired.append(not completed))
            if value == 2 else None)
        with (patch('spike_detector.gui.process_cell_simple', side_effect=slow_detection),
              patch('spike_detector.gui.np.savez_compressed', side_effect=slow_save)):
            self.assertTrue(self.win.start_detection_async())
            self.assertTrue(self.win._detection_running)
            deadline = time.monotonic() + 10
            while not completed and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.005)
        self.assertEqual(completed, [(2, 2)])
        self.assertEqual(fired, [True])
        self.assertEqual(save_fired, [True])
        self.assertTrue(all(tid != main_thread for tid in worker_threads))
        self.assertTrue(all(tid != main_thread for tid in save_threads))
        self.assertEqual(self.win.detection_progress.value(), 2)
        self.assertTrue(self.win.centralWidget().isEnabled())

    def test_colliding_stems_and_override(self):
        np.savez(self.folders[0]/'session.npz', time_ms=self.time,
                 raw_data=np.ones((len(self.time), 1)), cell_names=['flat'], fs=1000.)
        self.win.load_batch_folders([str(self.folders[0])])
        self.assertEqual(self.win.run_detection_all(), (2, 2))
        files = [data['results_file'] for data in self.win.loaded_sessions.values()]
        self.assertEqual(len(set(files)), 2)
        self.assertTrue(any('session.csv_analyzed.npz' in x for x in files))
        self.assertTrue(any('session.npz_analyzed.npz' in x for x in files))
        self.win.params['DETECTION_OVERRIDE'] = True
        self.assertEqual(self.win.run_detection_all(), (2, 2))
        self.assertEqual(files, [data['results_file'] for data in self.win.loaded_sessions.values()])

    def test_import_relative_path_list(self):
        file = self.root/'paths.txt'
        file.write_text('a/recording\nb/recording\na/recording\n', encoding='utf-8-sig')
        dlg = BatchDetectionDialog(self.win)
        with patch('spike_detector.gui.QFileDialog.getOpenFileName', return_value=(str(file), '')):
            dlg.import_paths()
        self.assertEqual(dlg.paths.toPlainText().splitlines(), list(map(str, self.folders)))
        dlg.close()

    def test_empty_batch_keeps_current_session(self):
        self.win.master_folder = str(self.folders[0])
        self.win.refresh_sessions()
        data = self.win.data
        with self.assertRaisesRegex(ValueError, 'No valid sessions'):
            self.win.load_batch_folders([str(self.root/'missing')])
        self.assertIs(self.win.data, data)


if __name__ == '__main__':
    unittest.main()
