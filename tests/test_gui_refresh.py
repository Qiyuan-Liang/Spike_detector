"""Polarity wiring and conditional ACG rate units, including observed exposure."""
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
from PyQt6 import QtWidgets
from spike_detector.gui import MainWindow, SettingsDialog, StatsViewerDialog
from spike_detector.utils.stats import autocorrelogram_components, recording_intervals_ms


class AcgRateTests(unittest.TestCase):
    def test_reference_exposure_rate_and_recording_edges(self):
        edges, counts, exposure = autocorrelogram_components([20, 40, 60, 80], 30, 10, [[0, 100]])
        np.testing.assert_array_equal(edges, [-30,-20,-10,0,10,20,30])
        np.testing.assert_array_equal(counts, [0,3,0,0,3,0])
        np.testing.assert_allclose(exposure, [.03,.04,.04,.04,.04,.03])
        np.testing.assert_allclose(counts/exposure, [0,75,0,0,75,0])

    def test_masked_target_time_is_not_treated_as_observed_silence(self):
        _, counts, exposure = autocorrelogram_components([20,40,60,80],30,10,[[0,45],[55,100]])
        self.assertEqual(counts[4],3)
        self.assertAlmostEqual(exposure[4], .035)
        self.assertAlmostEqual(counts[4]/exposure[4], 3/.035)

    def test_short_final_bin_uses_its_actual_observed_duration(self):
        _, _, exposure=autocorrelogram_components([100,200],25,10,[[0,300]])
        np.testing.assert_allclose(exposure,[.01,.02,.02,.02,.02,.01])

    def test_empty_and_outside_observation_events_have_no_reference_exposure(self):
        _, counts, exposure=autocorrelogram_components([0,5,100,np.nan],20,10,[[10,100]])
        self.assertFalse(counts.any())
        self.assertFalse(exposure.any())
        _, counts, exposure=autocorrelogram_components([20,40],20,10,[])
        self.assertFalse(counts.any())
        self.assertFalse(exposure.any())

    def test_pool_components_without_creating_cross_session_pairs(self):
        _, c1,e1=autocorrelogram_components([50],30,10,[[0,100]])
        _, c2,e2=autocorrelogram_components([70],30,10,[[0,100]])
        self.assertFalse((c1+c2).any())
        self.assertTrue(np.all(e1+e2>0))
        _, combined,_=autocorrelogram_components([50,70],30,10,[[0,100]])
        self.assertGreater(combined.sum(),0)

    def test_mask_intervals_preserve_time_origin_and_last_sample_duration(self):
        intervals=recording_intervals_ms(100+np.arange(6)*.2,5000,[False,True,True,False,False,True])
        np.testing.assert_allclose(intervals,[[100,100.2],[100.6,101]])
        with self.assertRaises(ValueError):
            recording_intervals_ms([0,1],1000,[False])
        with self.assertRaises(ValueError):
            autocorrelogram_components([],20,10,[[0,20],[10,30]])


class GuiRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        t=np.arange(1200,dtype=float)
        wave=np.random.default_rng(3).normal(0,.03,len(t))
        wave+=6*np.exp(-.5*((t-600)/1.2)**2)
        pd.DataFrame({'time_ms':t,'cell':100+wave}).to_csv(self.root/'recording.csv',index=False)
        self.win=MainWindow()
        self.win.spin_avg_frames.setValue(0)
        self.win.master_folder=str(self.root)
        self.win.refresh_sessions()

    def tearDown(self):
        self.win.close()
        self.tmp.cleanup()

    def test_main_polarity_buttons_load_reset_and_batch_share_one_setting(self):
        self.assertEqual(self.win.tabs_polarity.currentIndex(), 1)
        self.win.tabs_polarity.setCurrentIndex(0)
        self.assertFalse(self.win.params['NEGATIVE_GOING'])
        self.assertEqual(self.win.tabs_polarity.currentIndex(), 0)
        advanced=SettingsDialog(self.win)
        self.assertFalse(hasattr(advanced,'chk_negative'))
        advanced.close()
        self.win.tabs_polarity.setCurrentIndex(1)
        self.assertTrue(self.win.params['NEGATIVE_GOING'])
        self.win.tabs_polarity.setCurrentIndex(0)
        snapshot=self.win._current_settings_snapshot()
        cfg=self.root/'settings.json';cfg.write_text(json.dumps(snapshot))
        self.win.tabs_polarity.setCurrentIndex(1)
        with (patch('spike_detector.gui.QFileDialog.getOpenFileName',return_value=(str(cfg),'')),
              patch('spike_detector.gui.QMessageBox.information')):
            self.win.load_all_settings()
        self.assertEqual(self.win.tabs_polarity.currentIndex(), 0)
        self.win.spin_cs_thresh.setValue(0)
        self.win.load_batch_folders([str(self.root)])
        self.assertEqual(self.win.run_detection_all(),(1,1))
        res=self.win.data['results'][0]
        self.assertFalse(res['negative_going'])
        self.assertTrue(np.any(abs(res['ss_peaks']-600)<=2))
        with patch('spike_detector.gui.QMessageBox.question',return_value=QtWidgets.QMessageBox.StandardButton.Yes):
            self.win.reset_app()
        self.assertEqual(self.win.tabs_polarity.currentIndex(), 1)

    def test_threshold_group_cutoffs_keep_highpass_lowpass_parameter_mapping(self):
        for kind in ('CS','SS'):
            grid=self.win.threshold_groups[kind].layout()
            prefix=kind.lower()
            self.assertIs(grid.itemAtPosition(0,1).widget(),getattr(self.win,f'spin_{prefix}_low'))
            self.assertIs(grid.itemAtPosition(0,3).widget(),getattr(self.win,f'spin_{prefix}_high'))
            self.assertIs(grid.itemAtPosition(1,1).widget(),getattr(self.win,f'spin_{prefix}_order'))
            self.assertIs(grid.itemAtPosition(1,3).widget(),getattr(self.win,f'spin_{prefix}_thresh'))
            getattr(self.win,f'spin_{prefix}_low').setValue(5)
            getattr(self.win,f'spin_{prefix}_high').setValue(100)
            self.assertEqual(self.win.params[kind+'_LOW_CUT_HZ'],5)
            self.assertEqual(self.win.params[kind+'_HIGH_CUT_HZ'],100)
            for label in self.win.threshold_groups[kind].findChildren(QtWidgets.QLabel):
                self.assertNotIn(kind,label.text())
        self.win.chk_combo.setChecked(True)
        self.assertIn('Seed', self.win.lbl_cs_threshold.text())
        self.assertIn('Seed', self.win.lbl_ss_threshold.text())

    def test_statistics_acg_pools_counts_and_exposure_and_displays_hz(self):
        self.win.spin_cs_thresh.setValue(0)
        self.win.spin_ss_thresh.setValue(0)
        self.win.run_detection_all()
        data=self.win.data; res=data['results'][0]
        res['ss_peaks']=np.array([200,220,240,260])
        res['ss_exclusion_mask']=np.zeros(1200,dtype=bool)
        stats=StatsViewerDialog(data,self.win)
        stats.spin_ss_acg_window.setValue(30)
        stats.spin_ss_acg_bin.setValue(10)
        axis=stats.fig.axes[7]
        self.assertEqual(axis.get_ylabel(),'Spike rate (Hz)')
        # Four interior reference spikes, each with 10 ms of exposure per bin.
        np.testing.assert_allclose(axis.patches[0].get_data().values,[0,75,0,0,75,0])
        stats.close()


if __name__=='__main__':
    unittest.main()
