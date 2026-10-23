"""Boundary artifacts cannot affect valid data; annotations stay with their source."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('MPLCONFIGDIR','/tmp/spike-detector-test-mpl')
import numpy as np
import pandas as pd
from PyQt6 import QtWidgets
from spike_detector.gui import MainWindow, SettingsDialog, StatsViewerDialog, DetectionViewerDialog, _detect_cell
from spike_detector.utils.denoise import default_denoise_config
from spike_detector.utils.boundaries import observation_mask
from spike_detector.utils.saved_session import load_saved_detection, load_saved_detection_folder


class AnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.time=200+np.arange(1800,dtype=float)
        wave=100+np.random.default_rng(16).normal(0,.08,len(self.time))
        wave+=6*np.exp(-.5*((self.time-800)/5)**2)+3*np.exp(-.5*((self.time-1300)/1)**2)
        for name in ('a','b'):
            pd.DataFrame({'time_ms':self.time,'cell1':wave,'cell2':wave*1.1}).to_csv(self.root/(name+'.csv'),index=False)
        self.win=MainWindow();self.win.master_folder=str(self.root)
        self.win.tabs_polarity.setCurrentIndex(0);self.win.refresh_sessions()
        self.win.spin_avg_frames.setValue(3)

    def tearDown(self):
        with patch('spike_detector.gui.QMessageBox.question',return_value=QtWidgets.QMessageBox.StandardButton.Yes):
            self.win.close()
        self.tmp.cleanup()

    def mode(self,kind):
        self.win.template_selection_active=True;self.win.template_selection_type=kind
        self.win.template_selection_window_ms=8 if kind=='SS' else 30
        self.win._reset_template_selection_button_text();self.win.update_plot()

    def click(self,time,button=1,key=None):
        ax=self.win.canvas.figure.axes[0]
        pixel=ax.transData.transform((time,100))
        self.win.on_main_plot_click(SimpleNamespace(inaxes=ax,xdata=time,x=float(pixel[0]),button=button,key=key))

    def test_end_and_start_artifacts_do_not_change_detection_or_noise(self):
        raw=self.win.data['raw_data'][:,0].copy();params=copy.deepcopy(self.win.params)
        for method in ('Threshold','Template Matching','Combo'):
            params.update(DETECTION_METHOD='Template Matching' if method=='Template Matching' else 'Threshold',
                          COMBO_DETECTION=method=='Combo',TEMPLATE_SS_LOWPASS_HZ=0)
            bank={'cs_templates':[np.exp(-.5*((np.arange(31)-15)/5)**2)],'fs_cs':[1000],
                  'ss_templates':[np.exp(-.5*((np.arange(9)-4)/1)**2)],'fs_ss':[1000]}
            a=_detect_cell(raw,1000,params,self.win.baseline_params,3,'Rolling average',bank,default_denoise_config())
            corrupted=raw.copy();corrupted[:150]=-1e8;corrupted[-50:]=1e9
            b=_detect_cell(corrupted,1000,params,self.win.baseline_params,3,'Rolling average',bank,default_denoise_config())
            valid=~a['recording_exclusion_mask']
            self.assertEqual(valid.sum(),1600)
            self.assertTrue(np.isnan(a['stats_source_trace'][~valid]).all())
            for kind in ('cs','ss'):
                np.testing.assert_array_equal(a[kind+'_peaks'],b[kind+'_peaks'])
                np.testing.assert_allclose(a[kind+'_trace'][valid],b[kind+'_trace'][valid])
                self.assertEqual(a['sigma_'+kind],b['sigma_'+kind])
                np.testing.assert_allclose(a['event_amplitude_snr_'+kind],b['event_amplitude_snr_'+kind],equal_nan=True)
                self.assertTrue(np.all((a[kind+'_peaks']>=150)&(a[kind+'_peaks']<1750)))
                self.assertTrue(a[kind+'_exclusion_mask'][:150].all())
                self.assertTrue(a[kind+'_exclusion_mask'][-50:].all())

    def test_clicks_remove_and_move_between_cells_without_projecting_labels(self):
        self.mode('LABEL');self.assertEqual(self.win.btn_select_templates.text(),'Save Labels')
        self.click(800);self.click(800);self.click(800,key='control')
        self.assertEqual(self.win._label_draft()['events'][0],{'CS':[600],'SS':[]})
        self.click(800);self.click(1300,key='control')
        draft=self.win._label_draft()
        self.assertEqual(draft['events'][0],{'CS':[1100],'SS':[600]})
        self.win.combo_cell.setCurrentIndex(1)
        self.assertNotIn(1,self.win._label_draft()['visited'])
        self.assertEqual(self.win._label_draft()['events'][1],{'CS':[],'SS':[]})
        self.assertFalse(any(line.get_gid()=='manual_ss' for line in self.win.canvas.figure.axes[0].lines))
        self.click(900)
        self.win.list_sessions.setCurrentText('b.csv');self.click(1000,key='ctrl')
        self.win.list_sessions.setCurrentText('a.csv');self.win.combo_cell.setCurrentIndex(0)
        self.assertEqual(self.win._label_draft()['events'][0]['SS'],[600])
        self.click(800,button=3);self.assertEqual(draft['events'][0]['SS'],[])
        self.click(200);self.click(1990)
        self.assertEqual(draft['events'][0],{'CS':[1100],'SS':[]})

    def test_template_selection_is_cell_scoped_removable_and_keeps_source_fs(self):
        self.mode('SS');self.click(800)
        self.win.combo_cell.setCurrentIndex(1);self.click(1000)
        self.win.list_sessions.setCurrentText('b.csv');self.click(1300)
        self.assertEqual(len(self.win._template_records),3)
        self.click(1300,button=3);self.assertEqual(len(self.win._template_records),2)
        self.win.list_sessions.setCurrentText('a.csv');self.win.combo_cell.setCurrentIndex(0)
        self.click(800,button=3);self.assertEqual(len(self.win._template_records),1)
        dest=self.root/'bank.npz'
        with (patch('spike_detector.gui.QFileDialog.getSaveFileName',return_value=(str(dest),'')),
              patch('spike_detector.gui.QMessageBox.information')):
            self.win.save_selected_templates()
        with np.load(dest,allow_pickle=True) as z:
            meta=json.loads(str(z['ss_selection_metadata_json'].item()))
            self.assertEqual(meta[0]['cell'],1)
            self.assertEqual(meta[0]['fs'],1000)
            self.assertEqual(len(z['ss_waveforms']),1)

    def test_saved_labels_are_standard_archives_and_stats_use_new_labels(self):
        self.win.chk_detection_save.setChecked(True)
        self.assertEqual(self.win.run_detection_all(),(2,4))
        automatic={name:Path(data['results_file']).read_bytes() for name,data in self.win.loaded_sessions.items()}
        self.mode('LABEL');self.assertEqual(self.win._label_draft()['events'][0],{'CS':[],'SS':[]})
        self.click(800);self.click(1300,key='control')
        self.win.combo_cell.setCurrentIndex(1);self.click(1000)
        self.win.list_sessions.setCurrentText('b.csv');self.click(1100)
        with (patch('spike_detector.gui.QMessageBox.information'),patch('spike_detector.gui.QMessageBox.warning') as warning):
            self.win.save_selected_labels()
            deadline=time.monotonic()+20
            while getattr(self.win, '_label_save_worker', None) is not None and time.monotonic()<deadline:
                self.app.processEvents();time.sleep(.01)
            self.assertIsNone(self.win._label_save_worker)
        warning.assert_not_called()
        for name,data in self.win.loaded_sessions.items():
            original=self.root/'spike_detection'/(Path(name).stem+'_analyzed.npz')
            self.assertEqual(original.read_bytes(),automatic[name])
            self.assertTrue(data['is_manual_labels'])
            self.assertTrue(Path(data['results_file']).name.endswith('_labels.npz'))
            loaded=load_saved_detection(data['results_file'])
            self.assertTrue(loaded['label_metadata']['completeness'].startswith('partial'))
            for kind in ('cs','ss'):
                np.testing.assert_array_equal(loaded['results'][0][kind+'_peaks'],data['results'][0][kind+'_peaks'])
            with np.load(original,allow_pickle=True) as a,np.load(data['results_file'],allow_pickle=True) as b:
                self.assertEqual(set(a.files),set(b.files)-{'label_metadata_json'})
        self.win.list_sessions.setCurrentText('a.csv')
        self.assertEqual(self.win.data['results'][0]['ss_peaks'].tolist(),[600])
        self.assertEqual(self.win.data['results'][0]['cs_peaks'].tolist(),[1100])
        stats=StatsViewerDialog(self.win.data,parent=self.win);stats.close()
        viewer=DetectionViewerDialog(self.win.data,parent=self.win)
        self.assertIn('manual labels',viewer.lbl_candidate_info.text())
        self.assertEqual(len(viewer.ax_cs.lines),1);viewer.close()
        self.mode('LABEL')
        self.assertEqual(self.win._label_draft()['events'][0]['SS'],[600])
        _,sessions,issues=load_saved_detection_folder(self.root)
        self.assertFalse(issues);self.assertEqual(len(sessions),2)
        self.assertTrue(all(d['is_manual_labels'] for d in sessions.values()))

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'Analysis/spike_analysis/io.py').is_file(),
                         'Optional local shared-analysis package is not installed')
    def test_shared_analysis_loader_crops_before_means_and_correlation(self):
        self.win.data['raw_data'][:150]=1e8;self.win.data['raw_data'][-50:]=-1e8
        self.win.run_detection_all()
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Analysis'))
        from spike_analysis.io import load_session
        session=load_session(self.win.data['results_file'],self.root)
        np.testing.assert_array_equal(session['time_ms'],self.time[150:-50])
        np.testing.assert_array_equal(session['raw_data'],self.win.data['raw_data'][150:-50])
        self.assertTrue(session['analysis_blanking_applied'])
        np.testing.assert_allclose(np.median(session['raw_data'],axis=0),np.median(self.win.data['raw_data'][150:-50],axis=0))
        np.testing.assert_allclose(np.corrcoef(session['raw_data'].T),np.corrcoef(self.win.data['raw_data'][150:-50].T))

    def wait_labels(self):
        deadline=time.monotonic()+20
        while getattr(self.win,'_label_save_worker',None) is not None and time.monotonic()<deadline:
            self.app.processEvents();time.sleep(.01)
        self.assertIsNone(getattr(self.win,'_label_save_worker',None))

    def test_save_failure_keeps_drafts_and_previous_label_file(self):
        self.mode('LABEL');self.click(800)
        with patch('spike_detector.gui.QMessageBox.information'):
            self.win.save_selected_labels();self.wait_labels()
        dest=Path(self.win.data['results_file']);before=dest.read_bytes()
        self.mode('LABEL');self.click(900)
        with (patch.object(self.win,'_confirm_result_replacement',return_value=True),
              patch('spike_detector.gui.publish_detection',side_effect=OSError('simulated disk failure')),
              patch('spike_detector.gui.QMessageBox.warning') as warning):
            self.win.save_selected_labels();self.wait_labels()
        warning.assert_called_once()
        self.assertEqual(dest.read_bytes(),before)
        self.assertEqual(self.win._label_draft()['events'][0]['SS'],[600,700])
        self.assertTrue(self.win.template_selection_active)
        self.assertTrue(self.win.centralWidget().isEnabled())
        self.assertEqual(self.win.data['results'][0]['ss_peaks'].tolist(),[600])

    def test_save_cancel_and_manual_qc_does_not_reject_ground_truth(self):
        self.win.qc_inputs['CS_MIN_AMPLITUDE_SNR'].setValue(1000)
        self.win.qc_inputs['SS_MAX_FWHM_MS'].setValue(.1)
        self.mode('LABEL');self.click(800,key='control');self.click(1300)
        with patch('spike_detector.gui.QMessageBox.information'):
            self.win.save_selected_labels();self.wait_labels()
        self.assertEqual(self.win.data['results'][0]['cs_peaks'].tolist(),[600])
        self.assertEqual(self.win.data['results'][0]['ss_peaks'].tolist(),[1100])
        self.mode('LABEL');self.click(900)
        with patch.object(self.win,'_confirm_result_replacement',return_value=False):
            self.win.save_selected_labels()
        self.assertIsNone(self.win._label_save_worker)
        self.assertEqual(self.win.data['results'][0]['ss_peaks'].tolist(),[1100])
        self.assertIn(700,self.win._label_draft()['events'][0]['SS'])

    def test_mixed_sampling_rate_templates_are_resampled_independently(self):
        fast_time=200+np.arange(3600)*.5
        raw=np.interp(fast_time,self.time,self.win.data['raw_data'][:,0])
        pd.DataFrame({'time_ms':fast_time,'cell':raw}).to_csv(self.root/'fast.csv',index=False)
        self.win.refresh_sessions();self.mode('SS');self.click(800)
        self.win.list_sessions.setCurrentText('fast.csv');self.click(800)
        self.assertEqual([r['fs'] for r in self.win._template_records],[1000,2000])
        dest=self.root/'mixed.npz'
        with (patch('spike_detector.gui.QFileDialog.getSaveFileName',return_value=(str(dest),'')),
              patch('spike_detector.gui.QMessageBox.information')):
            self.win.save_selected_templates()
        with np.load(dest,allow_pickle=True) as z:
            self.assertEqual(float(z['fs']),5000)
            self.assertEqual([len(w) for w in z['ss_waveforms']],[41,41])

    def test_select_as_dialog_type_options_and_label_button(self):
        def accept(dialog):
            combo=dialog.findChildren(QtWidgets.QComboBox)[0]
            self.assertEqual([combo.itemText(i) for i in range(combo.count())],['CS template','SS template','Spike label'])
            form=dialog.layout()
            self.assertEqual(form.labelForField(combo).text(),'Type:')
            combo.setCurrentIndex(2)
            return QtWidgets.QDialog.DialogCode.Accepted
        with patch.object(QtWidgets.QDialog,'exec',accept):
            self.win.toggle_template_selection_or_save()
        self.assertEqual(self.win.btn_select_templates.text(),'Save Labels')
        self.assertEqual(self.win._label_draft()['events'][0],{'CS':[],'SS':[]})

    def test_all_blank_is_rejected_and_zero_disables_both_ends(self):
        params=dict(self.win.params,INITIAL_BLANK_MS=5000)
        with self.assertRaisesRegex(ValueError,'Boundary blanks leave'):
            _detect_cell(self.win.data['raw_data'][:,0],1000,params,self.win.baseline_params,0,
                'Rolling average',{},default_denoise_config())
        params.update(INITIAL_BLANK_MS=0,FINAL_BLANK_MS=0)
        result=_detect_cell(self.win.data['raw_data'][:,0],1000,params,self.win.baseline_params,0,
                'Rolling average',{},default_denoise_config())
        self.assertFalse(result['recording_exclusion_mask'].any())

    def test_boundary_smoothing_does_not_create_spikes_on_a_flat_trace(self):
        raw=np.full(1800,100.0);raw[:150]=-10000;raw[-50:]=-10000
        params=dict(self.win.params,NEGATIVE_GOING=True)
        result=_detect_cell(raw,1000,params,self.win.baseline_params,5,'Rolling average',{},default_denoise_config())
        self.assertEqual(len(result['cs_peaks']),0)
        self.assertEqual(len(result['ss_peaks']),0)
        np.testing.assert_allclose(result['stats_source_trace'][150:-50],0,atol=1e-12)

    def test_rerunning_automatic_detection_does_not_import_it_as_manual_labels(self):
        self.mode('LABEL');self.click(900)
        with patch('spike_detector.gui.QMessageBox.information'):
            self.win.save_selected_labels();self.wait_labels()
        self.win.chk_detection_save.setChecked(False)
        self.assertEqual(self.win.run_detection_all(),(2,4))
        self.assertFalse(self.win.data.get('is_manual_labels'))
        self.mode('LABEL')
        self.assertEqual(self.win._label_draft()['events'][0]['SS'],[700])

    def test_blank_controls_roundtrip(self):
        dialog=SettingsDialog(self.win)
        self.assertEqual(dialog.spin_final_blank.value(),50)
        dialog.spin_final_blank.setValue(75.5);dialog.save_and_close();dialog.close()
        cfg=self.win._current_settings_snapshot()
        self.win.params['FINAL_BLANK_MS']=0
        self.win.load_all_settings(snapshot=cfg,quiet=True)
        self.assertEqual(self.win.params['FINAL_BLANK_MS'],75.5)


    def test_blank_display_smoothing_is_independent_and_preserves_valid_processing(self):
        from spike_detector.utils.boundaries import protected_preprocess, smooth_blanked_display
        raw = self.win.data['raw_data'][:,0].copy()
        raw[:150] = np.arange(150)*1000
        raw[-50:] = np.arange(50)*-1000
        for count in (4, 5):
            display, base, valid = protected_preprocess(raw, 1000, self.win.params,
                self.win.baseline_params, count)
            clean, clean_base, _ = protected_preprocess(raw[150:-50], 1000,
                dict(INITIAL_BLANK_MS=0, FINAL_BLANK_MS=0), self.win.baseline_params, count)
            np.testing.assert_allclose(display[valid], clean)
            np.testing.assert_allclose(base[valid], clean_base)
            for start, stop in ((0,150),(1750,1800)):
                expected = np.convolve(np.pad(raw[start:stop],(count//2,(count-1)//2),mode='reflect'),
                                       np.ones(count)/count,mode='valid')
                np.testing.assert_allclose(display[start:stop],expected)
            # Old saved display arrays get the same blank-only smoothing without changing valid samples.
            old = raw.copy(); old[valid] = 42
            saved = smooth_blanked_display(raw, old, valid, count)
            np.testing.assert_array_equal(saved[valid], old[valid])
            np.testing.assert_allclose(saved[~valid], display[~valid])
            np.testing.assert_array_equal(smooth_blanked_display(raw,old,valid,count,'Downsampling frames'),old)

    def test_default_views_start_at_observation_and_allow_blank_navigation(self):
        self.assertEqual(self.win.spin_window.value(),1000)
        self.win.update_plot()
        np.testing.assert_allclose(self.win.canvas.figure.axes[0].get_xlim(),[350,1350])
        self.win.slider_time.setValue(1)
        np.testing.assert_allclose(self.win.canvas.figure.axes[0].get_xlim(),[200,1200])
        self.assertFalse(self.win._current_settings_snapshot()['ui']['start_at_observation'])
        self.win.slider_time.setValue(0)
        self.win._main_view_auto_start=True
        snapshot=self.win._current_settings_snapshot()
        self.win.slider_time.setValue(50)
        self.win.load_all_settings(snapshot=snapshot,quiet=True)
        self.assertTrue(self.win._main_view_auto_start)
        self.win.run_detection_all()
        viewer=DetectionViewerDialog(self.win.data,self.win)
        self.assertEqual(viewer.spin_window.value(),1000)
        np.testing.assert_allclose(viewer.ax_raw_bot.get_xlim(),[350,1350])
        viewer.slider_time.setValue(1)
        np.testing.assert_allclose(viewer.ax_raw_bot.get_xlim(),[200,1200])
        self.assertFalse(self.win.params['DETECTION_VIEW_AUTO_START'])
        viewer.close()

    def test_end_blank_artifact_does_not_set_visible_scale(self):
        self.win.data['raw_data'][-50:,0]=-1e8
        self.win.spin_window.setValue(2000)
        self.win.update_plot()
        self.assertGreater(self.win.canvas.figure.axes[0].get_ylim()[0],90)
        self.win.run_detection_all()
        viewer=DetectionViewerDialog(self.win.data,self.win)
        viewer.spin_window.setValue(2000)
        self.assertGreater(viewer.ax_raw_bot.get_ylim()[0],90)
        viewer.close()


if __name__=='__main__':unittest.main()
