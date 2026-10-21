"""Save consent, temporary ownership, single-class detection and polarity plots."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/spike-detector-test-mpl')
import numpy as np
import pandas as pd
from PyQt6 import QtWidgets
from spike_detector.gui import MainWindow, DetectionViewerDialog, DetectionSettingsDialog
from spike_detector.utils.results import TemporaryDetectionStore, cleanup_legacy_temporaries, publish_detection
from spike_detector.utils.batch import discover_batch_sessions
from spike_detector.utils.processing import process_cell_simple


class StorageTests(unittest.TestCase):
    def test_close_and_next_launch_cleanup_protect_active_and_unknown_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = TemporaryDetectionStore(root)
            file = first.allocate('recording')
            file.write_text('active viewing result')
            stale = root/('run-' + 'a'*32)
            stale.mkdir()
            (stale/'.spike_detector_viewing.json').write_text(json.dumps({'kind': 'SpikeDetectorViewing', 'version': 1}))
            (stale/'session.npz').write_text('abandoned')
            unknown = root/('run-' + 'b'*32)
            unknown.mkdir()
            (unknown/'sentinel').write_text('user file')
            malformed = root/('run-' + 'c'*32)
            malformed.mkdir()
            (malformed/'.spike_detector_viewing.json').write_text('[]')
            second = TemporaryDetectionStore(root)
            try:
                self.assertTrue(file.exists())
                self.assertFalse(stale.exists())
                self.assertTrue((unknown/'sentinel').exists())
                self.assertTrue(malformed.exists())
                first.close()
                self.assertFalse(file.exists())
                self.assertTrue(second.directory.exists())
                first.close()  # idempotent
            finally:
                second.close()
                first.close()

    def test_legacy_cleanup_is_narrow_and_batch_discovery_uses_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root/'spike_detection'/'_temporary_detection'
            directory.mkdir(parents=True)
            for name in ('cell_analyzed_temp_20261011_120000_123456.npz',
                         'cell_analyzed_temp_20261011_120000_123456_settings.json'):
                (directory/name).write_text('old viewing export')
            sentinel = directory/'keep_user_file.npz'
            sentinel.write_text('unrecognized')
            paths, _, _, _, _ = discover_batch_sessions([str(root)])
            self.assertFalse(paths)
            self.assertEqual(list(directory.iterdir()), [sentinel])
            sentinel.unlink()
            self.assertEqual(cleanup_legacy_temporaries(root), [])
            self.assertFalse(directory.exists())

    def test_cleanup_does_not_follow_directory_symlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            external = root/'outside'
            external.mkdir()
            file = external/'cell_analyzed_temp_20261011_120000_123456.npz'
            file.write_text('preserve')
            (root/'spike_detection').mkdir()
            try:
                (root/'spike_detection'/'_temporary_detection').symlink_to(external, target_is_directory=True)
            except OSError:
                self.skipTest('Directory symlinks unavailable')
            self.assertEqual(cleanup_legacy_temporaries(root), [])
            self.assertTrue(file.exists())

    def _outputs(self, root):
        stage = root/'stage.npz'
        np.savez(stage, value=[2])
        cfg = root/'stage_settings.json'
        cfg.write_text(json.dumps({'results_file': stage.name, 'params': {'SAVE_DETECTION_RESULTS': True}}))
        dest = root/'spike_detection'/'cell_analyzed.npz'
        dest.parent.mkdir()
        paths = [dest, dest.with_name(dest.stem+'_settings.json'), dest.with_suffix('.npy')]
        for i, path in enumerate(paths):
            path.write_bytes(f'previous-{i}'.encode())
        return stage, cfg, dest, paths

    def test_publish_replaces_pair_and_removes_legacy_only_after_success(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, cfg, dest, paths = self._outputs(Path(folder))
            publish_detection(stage, cfg, dest, {str(p.resolve()) for p in paths})
            with np.load(dest) as saved:
                np.testing.assert_array_equal(saved['value'], [2])
            self.assertEqual(json.loads(paths[1].read_text())['results_file'], dest.name)
            self.assertFalse(paths[2].exists())
            self.assertTrue(stage.exists())
            self.assertFalse(list(dest.parent.glob('.spike-detector-publish-*')))

    def test_failed_pair_replacement_rolls_back_every_previous_output(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, cfg, dest, paths = self._outputs(Path(folder))
            originals = {p: p.read_bytes() for p in paths}
            actual = os.replace
            def fail_settings(source, target):
                if Path(source).name == 'new.json':
                    raise OSError('simulated sidecar replacement failure')
                return actual(source, target)
            with patch('spike_detector.utils.results.os.replace', side_effect=fail_settings):
                with self.assertRaisesRegex(OSError, 'simulated'):
                    publish_detection(stage, cfg, dest, {str(p.resolve()) for p in paths})
            self.assertEqual({p: p.read_bytes() for p in paths}, originals)
            self.assertFalse(list(dest.parent.glob('.spike-detector-publish-*')))

    def test_unapproved_output_appearing_after_confirmation_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, cfg, dest, paths = self._outputs(Path(folder))
            originals = {p: p.read_bytes() for p in paths}
            with self.assertRaisesRegex(FileExistsError, 'after confirmation'):
                publish_detection(stage, cfg, dest, set())
            self.assertEqual({p: p.read_bytes() for p in paths}, originals)


class ThresholdTests(unittest.TestCase):
    def test_zero_disables_each_type_with_local_and_global_noise(self):
        t = np.arange(1600)
        trace = np.random.default_rng(18).normal(0, .1, len(t))
        trace += 8*np.exp(-.5*((t-500)/7)**2) + 4*np.exp(-.5*((t-1000)/.8)**2)
        for local in (False, True):
            for cs_cut, ss_cut in ((0, 3), (3, 0), (0, 0)):
                with self.subTest(local=local, cs=cs_cut, ss=ss_cut):
                    res = process_cell_simple(trace, 1000, negative_going=False,
                        cs_thresh_sigma=cs_cut, ss_thresh_sigma=ss_cut,
                        local_baseline=local)
                    if cs_cut == 0:
                        self.assertEqual(len(res['cs_peaks']), 0)
                        self.assertFalse(res['cs_detection_enabled'])
                        self.assertFalse(res['ss_exclusion_mask'].any())
                        self.assertEqual(res['cs_threshold_used'], 0)
                    else:
                        self.assertGreater(len(res['cs_peaks']), 0)
                    if ss_cut == 0:
                        self.assertEqual(len(res['ss_peaks']), 0)
                        self.assertFalse(res['ss_detection_enabled'])
                    else:
                        self.assertGreater(len(res['ss_peaks']), 0)

    def test_disabled_class_with_all_samples_masked_has_zero_threshold(self):
        with np.errstate(invalid='raise'):
            res = process_cell_simple(np.zeros(200), 1000, cs_thresh_sigma=0,
                ss_thresh_sigma=0, initial_blank_ms=200, local_baseline=True)
        np.testing.assert_array_equal(res['cs_threshold_trace'], np.zeros(200))
        np.testing.assert_array_equal(res['ss_threshold_trace'], np.zeros(200))

    def test_invalid_cutoffs_still_rejected(self):
        for cutoff in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                process_cell_simple(np.zeros(200), 1000, cs_thresh_sigma=cutoff)


class LifecycleGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.time = np.arange(1600, dtype=float)
        self.trace = 100 + np.random.default_rng(18).normal(0, .1, len(self.time))
        self.trace += 8*np.exp(-.5*((self.time-500)/7)**2) + 4*np.exp(-.5*((self.time-1000)/.8)**2)
        self.input = self.root/'session.csv'
        pd.DataFrame({'time_ms': self.time, 'cell': self.trace}).to_csv(self.input, index=False)
        self.win = MainWindow()
        self.win.params['NEGATIVE_GOING'] = False
        self.win.params['SS_MAX_FWHM_FILTER_ENABLED'] = False
        self.win.spin_avg_frames.setValue(0)
        self.win.master_folder = str(self.root)
        self.win.refresh_sessions()

    def tearDown(self):
        self.win.close()
        self.temp.cleanup()

    def test_default_temporary_first_run_and_close_cleanup(self):
        self.assertEqual(self.win.chk_detection_save.text(), 'save')
        self.assertFalse(self.win.chk_detection_save.isChecked())
        self.assertEqual(self.win.run_detection_all(), (1, 1))
        data = self.win.data
        result, cfg = Path(data['results_file']), Path(data['settings_file'])
        self.assertTrue(result.is_relative_to(self.win._temporary_store.directory))
        self.assertTrue(cfg.exists())
        self.assertFalse((self.root/'spike_detection').exists())
        self.assertTrue(data['temporary_result'])
        self.assertIn('Save: OFF', self.win.text_stats.toPlainText())
        self.win.close()
        self.assertFalse(result.exists())
        self.assertFalse(cfg.exists())

    def test_unsaved_rerun_preserves_existing_saved_pair(self):
        self.win.chk_detection_save.setChecked(True)
        self.assertEqual(self.win.run_detection_all(), (1, 1))
        result, cfg = Path(self.win.data['results_file']), Path(self.win.data['settings_file'])
        originals = result.read_bytes(), cfg.read_bytes()
        self.win.chk_detection_save.setChecked(False)
        with patch.object(self.win, '_confirm_result_replacement') as confirm:
            self.assertEqual(self.win.run_detection_all(), (1, 1))
        confirm.assert_not_called()
        self.assertEqual((result.read_bytes(), cfg.read_bytes()), originals)
        self.assertTrue(self.win.data['temporary_result'])
        self.assertNotEqual(Path(self.win.data['results_file']), result)

    def test_save_confirmation_cancel_then_accept_preserves_input(self):
        self.win.chk_detection_save.setChecked(True)
        self.assertEqual(self.win.run_detection_all(), (1, 1))
        data = self.win.data
        result, cfg = Path(data['results_file']), Path(data['settings_file'])
        original_pair = result.read_bytes(), cfg.read_bytes()
        input_bytes = self.input.read_bytes()
        prior_results = data['results']
        with patch.object(self.win, '_confirm_result_replacement', return_value=False) as confirm:
            self.assertEqual(self.win.run_detection_all(), (0, 0))
        confirm.assert_called_once()
        self.assertEqual((result.read_bytes(), cfg.read_bytes()), original_pair)
        self.assertIs(data['results'], prior_results)
        legacy = result.with_suffix('.npy')
        legacy.write_bytes(b'legacy result')
        self.win.spin_ss_thresh.setValue(0)
        with patch.object(self.win, '_confirm_result_replacement', return_value=True) as confirm:
            self.assertEqual(self.win.run_detection_all(), (1, 1))
        self.assertIn(legacy, confirm.call_args.args[0])
        self.assertFalse(legacy.exists())
        self.assertFalse(data['temporary_result'])
        self.assertEqual(Path(data['results_file']), result)
        self.assertNotEqual(result.read_bytes(), original_pair[0])
        self.assertEqual(self.input.read_bytes(), input_bytes)
        self.win.close()
        self.assertTrue(result.exists())

    def test_single_class_exports_keep_schema_and_settings_roundtrip(self):
        key_sets = []
        for cs_cut, ss_cut in ((0, 3), (3, 0)):
            self.win.spin_cs_thresh.setValue(cs_cut)
            self.win.spin_ss_thresh.setValue(ss_cut)
            self.assertEqual(self.win.run_detection_all(), (1, 1))
            data = self.win.data
            with np.load(data['results_file'], allow_pickle=True) as saved:
                key_sets.append(set(saved.files))
                kind = 'cs' if cs_cut == 0 else 'ss'
                for key in ('spike_times_', 'event_snr_', 'event_fwhm_'):
                    self.assertEqual(len(saved[key+kind][0]), 0)
                other = 'ss' if kind == 'cs' else 'cs'
                self.assertGreater(len(saved['spike_times_'+other][0]), 0)
                cfg = json.loads(str(saved['analysis_settings_json']))
                self.assertFalse(cfg['effective_settings'][kind+'_threshold_enabled'])
            with (patch('spike_detector.gui.QFileDialog.getOpenFileName', return_value=(data['settings_file'], '')),
                  patch('spike_detector.gui.QMessageBox.information')):
                self.win.load_all_settings()
            self.assertEqual(self.win.spin_cs_thresh.value(), cs_cut)
            self.assertEqual(self.win.spin_ss_thresh.value(), ss_cut)
        self.assertEqual(key_sets[0], key_sets[1])
        dialog = DetectionSettingsDialog(self.win)
        self.assertEqual(dialog.spin_cs_thresh.minimum(), 0)
        self.assertEqual(dialog.spin_ss_thresh.minimum(), 0)
        dialog.close()

    def test_legacy_override_settings_migrate_to_save(self):
        cfg = self.root/'old_settings.json'
        cfg.write_text(json.dumps({'params': {'DETECTION_OVERRIDE': True}}))
        with (patch('spike_detector.gui.QFileDialog.getOpenFileName', return_value=(str(cfg), '')),
              patch('spike_detector.gui.QMessageBox.information')):
            self.win.load_all_settings()
        self.assertTrue(self.win.chk_detection_save.isChecked())
        snapshot = self.win._current_settings_snapshot()['params']
        self.assertTrue(snapshot['SAVE_DETECTION_RESULTS'])
        self.assertNotIn('DETECTION_OVERRIDE', snapshot)

    def test_legacy_folder_cleanup_on_ordinary_load_preserves_saved_results(self):
        outputs = self.root/'spike_detection'
        old = outputs/'_temporary_detection'
        old.mkdir(parents=True)
        (old/'session_analyzed_temp_20261011_120000_123456.npz').write_bytes(b'old')
        permanent = outputs/'session_analyzed.npz'
        permanent.write_bytes(b'permanent')
        self.win.refresh_sessions()
        self.assertFalse(old.exists())
        self.assertEqual(permanent.read_bytes(), b'permanent')

    def test_marker_rail_uses_detected_polarity_even_after_settings_change(self):
        self.assertEqual(self.win.run_detection_all(), (1, 1))
        res = self.win.data['results'][0]
        # Exercise measured and uncertain CS widths as well as SS ticks.
        res['cs_peaks'] = np.array([500, 700])
        res['ss_peaks'] = np.array([1000])
        res['event_widths_cs'] = [dict(candidate_index=500, status='measured', left_index=494, right_index=506),
                                  dict(candidate_index=700, status='uncertain')]
        self.win.params['NEGATIVE_GOING'] = True
        viewer = DetectionViewerDialog(self.win.data, self.win)
        viewer.spin_window.setValue(1600)
        for negative in (False, True):
            res['negative_going'] = negative
            viewer.plot_detection()
            markers = [line for line in viewer.ax_raw_bot.lines if line.get_gid() in ('cs_event_marker', 'ss_event_marker')]
            self.assertEqual(len(markers), 3)
            for line in markers:
                ys = np.asarray(line.get_ydata())
                self.assertTrue(np.all(ys < .1) if negative else np.all(ys > .9))
                self.assertEqual(line.get_transform(), viewer.ax_raw_bot.get_xaxis_transform())
            self.assertEqual(viewer.ax_raw_top.get_ylim(), viewer.ax_raw_bot.get_ylim())
        viewer.close()

    def test_raw_legend_above_axes_without_data_overlap(self):
        self.win.update_plot()
        self.win.canvas.draw()
        axis = self.win.canvas.figure.axes[0]
        legend = axis.get_legend()
        renderer = self.win.canvas.get_renderer()
        self.assertIsNotNone(legend)
        self.assertGreater(legend.get_window_extent(renderer).y0, axis.get_window_extent(renderer).y1)
        self.assertLess(legend.get_window_extent(renderer).y1, axis.title.get_window_extent(renderer).y0)


if __name__ == '__main__':
    unittest.main()
