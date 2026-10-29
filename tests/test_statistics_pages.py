"""Lag rates and CV2 respect exposure, gaps and independent recordings."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/spike-detector-test-mpl')
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets
from spike_detector.utils.stats import triggered_rate_components, local_cv2_values
from spike_detector.gui import MainWindow, StatsViewerDialog, DenoisingSettingsDialog


class TemporalStatsTests(unittest.TestCase):
    def test_trigger_counts_and_edge_exposure_have_hand_calculated_rates(self):
        counts, exposure, n = triggered_rate_components([20,80], [15,25,30,85,95],
            [-10,0,10,40], [[0,100]], [[0,100]])
        np.testing.assert_array_equal(counts, [1,2,2])
        np.testing.assert_allclose(exposure, [.02,.02,.04])
        np.testing.assert_allclose(counts/exposure, [50,100,50])
        self.assertEqual(n,2)

    def test_ss_blanking_is_unobserved_not_a_pause_and_does_not_remove_cs_trigger(self):
        counts, exposure, n = triggered_rate_components([20], [15,25,35],
            [-10,0,10,20], [[0,100]], [[0,20],[30,100]])
        np.testing.assert_array_equal(counts, [1,0,1])
        np.testing.assert_allclose(exposure, [.01,0,.01])
        self.assertEqual(n,1)
        counts, exposure, _ = triggered_rate_components([20], [25,29], [0,10],
            [[0,100]], [[0,25.5],[28,100]])
        self.assertEqual(counts[0],2)
        self.assertAlmostEqual(exposure[0],.0075)

    def test_independent_recordings_never_form_cross_recording_pairs(self):
        a, e, _ = triggered_rate_components([50], [], [-10,10], [[0,100]], [[0,100]])
        b, _, _ = triggered_rate_components([], [55], [-10,10], [[0,100]], [[0,100]])
        self.assertEqual((a+b)[0],0)
        self.assertGreater(e[0],0)
        combined, _, _ = triggered_rate_components([50], [55], [-10,10], [[0,100]], [[0,100]])
        self.assertEqual(combined[0],1)

    def test_cv2_regular_and_alternating_intervals(self):
        np.testing.assert_allclose(local_cv2_values([0,10,20,30],[[0,31]]),[0,0])
        np.testing.assert_allclose(local_cv2_values([0,10,30,40],[[0,41]]),[2/3,2/3])
        self.assertEqual(len(local_cv2_values([0,10],[[0,100]])),0)

    def test_cv2_does_not_bridge_gaps_or_include_unobserved_spikes(self):
        values=local_cv2_values([0,10,30,35,40,50,100],[[0,31],[39,101]])
        np.testing.assert_allclose(values,[2/3,4/3])
        np.testing.assert_allclose(local_cv2_values([200,210,230,240],[[200,241]]),[2/3,2/3])
        self.assertEqual(len(local_cv2_values([10,20,30],[])),0)

    def test_invalid_intervals_and_lags_are_rejected(self):
        with self.assertRaises(ValueError):
            local_cv2_values([1,2,3],[[0,10],[5,20]])
        with self.assertRaises(ValueError):
            triggered_rate_components([1],[2],[0,0],[[0,10]],[[0,10]])
        counts, exposure, n=triggered_rate_components([-1,10,100,np.nan], [10,100],
            [0,10], [[0,100]],[[0,100]])
        self.assertEqual(n,1);self.assertEqual(counts[0],1)
        self.assertAlmostEqual(exposure[0],.01)


class StatisticsPagesGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name)
        t=np.arange(3000)
        for name in ('a','b'):
            pd.DataFrame({'time_ms':t,'Cell1':np.full(3000,100.),
                          'Cell2':np.full(3000,100.)}).to_csv(root/(name+'.csv'),index=False)
        self.win=MainWindow();self.win.master_folder=str(root);self.win.refresh_sessions()
        self.win.run_detection_all()
        for data in self.win.loaded_sessions.values():
            for cell,res in enumerate(data['results']):
                res['cs_peaks']=np.array([500,1000,1500,2000]) if cell==0 else np.array([],dtype=int)
                res['ss_peaks']=np.arange(205,2700,25) if cell==0 else np.array([],dtype=int)
                for kind in ('cs','ss'):
                    res[kind+'_exclusion_mask']=res['recording_exclusion_mask'].copy()
                    peaks=res[kind+'_peaks']
                    res['event_widths_'+kind]=[dict(candidate_index=int(p),fwhm_ms=5 if kind=='cs' else 1,
                        status='measured') for p in peaks]
                    res['amplitude_snr_indices_'+kind]=peaks.copy()
                    res['event_amplitude_snr_'+kind]=np.full(len(peaks),3.)
        with patch.object(QtGui.QScreen,'availableGeometry',return_value=QtCore.QRect(0,0,1512,982)):
            self.stats=StatsViewerDialog(self.win.data,self.win)

    def tearDown(self):
        self.stats.close();self.win.close();self.tmp.cleanup()

    def test_page_toggle_and_filters_recompute_only_within_selected_records(self):
        self.assertEqual(len(self.stats.fig.axes),12)
        self.stats.btn_next_page.click()
        self.assertEqual(self.stats.btn_next_page.text(),'Previous Page')
        self.assertEqual(len(self.stats.fig.axes),9)
        self.assertEqual(self.stats.temporal_statistics['cs_references'],8)
        np.testing.assert_allclose(self.stats.temporal_statistics['edges_ms'],np.arange(-20,61))
        self.assertEqual(next(ax for ax in self.stats.fig.axes if ax.get_gid()=='cs_ss_peth').get_xlim(),(-20,60))
        self.stats.combo_session.setCurrentText('a.csv')
        self.assertEqual(self.stats.temporal_statistics['cs_references'],4)
        np.testing.assert_allclose(self.stats.temporal_statistics['cv2']['CS'],[0,0])
        self.stats.combo_cell.setCurrentText('Cell2')
        self.assertEqual(self.stats.temporal_statistics['cs_references'],0)
        self.assertFalse(np.any(np.isfinite(self.stats.temporal_statistics['rate_hz'])))
        self.stats.btn_next_page.click()
        self.assertEqual(len(self.stats.fig.axes),12)
        self.assertEqual(self.stats.combo_session.currentText(),'a.csv')
        self.assertEqual(self.stats.combo_cell.currentText(),'Cell2')

    def test_bins_affect_all_distributions_and_save_load(self):
        self.stats.spin_distribution_bins.setValue(11)
        for index in (2,3,4,8,9,10):
            self.assertEqual(len(self.stats.fig.axes[index].patches),11)
        snapshot=self.win._current_settings_snapshot()
        self.assertEqual(snapshot['params']['STATS_DISTRIBUTION_BINS'],11)
        self.win.params['STATS_DISTRIBUTION_BINS']=30
        self.win.load_all_settings(snapshot=snapshot,quiet=True)
        reopened=StatsViewerDialog(self.win.data,self.win)
        self.assertEqual(reopened.spin_distribution_bins.value(),11)
        reopened.close()
        self.stats.btn_next_page.click()
        for axis in self.stats.fig.axes:
            if axis.get_gid() in ('ss_cv2','cs_cv2','ss_dff','cs_dff'):
                self.assertEqual(len(axis.patches),11)

    def test_masks_are_unavailable_and_disabled_ss_is_not_silent_ss(self):
        for data in self.win.loaded_sessions.values():
            for res in data['results']:
                for p in res['cs_peaks']:
                    res['ss_exclusion_mask'][p-10:p+11]=True
        self.stats.btn_next_page.click()
        result=self.stats.temporal_statistics
        unavailable=(result['edges_ms'][:-1]>=-10)&(result['edges_ms'][1:]<=11)
        self.assertTrue(np.all(np.isnan(result['rate_hz'][unavailable])))
        self.assertTrue(next(ax for ax in self.stats.fig.axes if ax.get_gid()=='cs_ss_peth').patches)
        for data in self.win.loaded_sessions.values():
            data['_control_context']['params']['SS_THRESHOLD_SIGMA']=0
            for res in data['results']:
                res['ss_detection_enabled']=False
        self.stats.compute_stats()
        self.assertEqual(self.stats.temporal_statistics['cs_references'],0)

    def test_default_size_controls_and_summary_fit_without_rate_exclusion_text(self):
        self.stats.show();self.app.processEvents();self.stats.canvas.draw()
        self.assertLessEqual(self.stats.width(),1472)
        self.assertLessEqual(self.stats.height(),902)
        self.assertLess(self.stats.btn_next_page.x(),self.stats.btn_save.x())
        self.assertLessEqual(self.stats.btn_save.geometry().right(),self.stats.width())
        self.assertTrue(self.stats.spin_distribution_bins.toolTip())
        renderer=self.stats.canvas.get_renderer()
        for axis in (self.stats.fig.axes[5],self.stats.fig.axes[11]):
            for text in axis.texts:
                self.assertNotIn('Rate excludes',text.get_text())
                box=text.get_window_extent(renderer)
                self.assertGreaterEqual(box.x0,0)
                self.assertLessEqual(box.x1,self.stats.fig.bbox.x1)
        self.assertLessEqual(self.stats.fig.subplotpars.left,.04)

    def test_page_specific_settings_and_saved_dff_nan_precedence(self):
        self.stats.show();self.app.processEvents()
        self.assertTrue(self.stats.spin_cs_acg_window.isVisible())
        self.stats.btn_next_page.click()
        self.assertFalse(self.stats.spin_cs_acg_window.isVisible())
        self.assertTrue(self.stats.spin_peth_bin_ms.isVisible())
        self.stats.spin_peth_pre_ms.setValue(15)
        self.stats.spin_peth_post_ms.setValue(50)
        self.stats.spin_peth_bin_ms.setValue(2)
        self.stats.spin_ss_trend_s.setValue(1)
        self.stats.spin_cs_trend_s.setValue(2)
        snapshot=self.win._current_settings_snapshot()
        self.assertEqual(snapshot['params']['STATS_PETH_PRE_MS'],15)
        self.assertEqual(snapshot['params']['STATS_SS_TREND_S'],1)
        reopened=StatsViewerDialog(self.win.data,self.win)
        self.assertEqual(reopened.spin_peth_post_ms.value(),50)
        self.assertEqual(reopened.spin_cs_trend_s.value(),2)
        reopened.close()
        res=self.win.data['results'][0]
        res['event_minus_dff_percent_cs']=np.array([7,np.nan,8,np.nan])
        res['dff_indices_cs']=res['cs_peaks'].copy()
        self.stats.compute_stats()
        # The other session contributes four zeros; saved NaNs are not replaced.
        self.assertEqual(len(self.stats.temporal_statistics['dff_percent']['CS']),6)
        self.assertIn(7,self.stats.temporal_statistics['dff_percent']['CS'])
        res['event_minus_dff_percent_cs']=np.array([7,np.nan])
        values=self.stats._event_amplitudes(self.win.data,0,res,'CS')
        np.testing.assert_allclose(values,[7,np.nan,0,0],equal_nan=True)

    def test_batch_all_paths_leaves_time_courses_blank_and_svg_text_is_arial(self):
        self.win.loaded_sessions['b.csv']['source_folder']='/tmp/other_recording'
        self.stats.btn_next_page.click()
        self.assertIn('one Path',self.stats.temporal_statistics['courses']['SS']['reason'])
        from spike_detector.utils.export import save_figure_with_dialog
        import xml.etree.ElementTree as ET
        dest=Path(self.tmp.name)/'page2.svg'
        save_figure_with_dialog(lambda *args:(str(dest),''),self.stats.fig)
        root=ET.parse(dest).getroot();texts=root.findall('.//{http://www.w3.org/2000/svg}text')
        self.assertGreater(len(texts),20)
        self.assertTrue(all('Arial' in text.get('style','') for text in texts))
        self.assertIn('SS local CV2',''.join(root.itertext()))


    def test_closed_denoising_preview_cannot_interrupt_later_navigation(self):
        dialog=DenoisingSettingsDialog(self.win)
        dialog._preview_timer.setInterval(10000)
        dialog.show();self.app.processEvents()
        self.assertTrue(dialog._preview_timer.isActive())
        dialog.close()
        self.assertFalse(dialog._preview_timer.isActive())
        with patch('spike_detector.gui.QMessageBox.warning') as warning:
            self.win.slider_time.setValue(50)
            dialog.run_preview()
            self.assertFalse(dialog._preview_timer.isActive())
            warning.assert_not_called()



if __name__=='__main__':unittest.main()
