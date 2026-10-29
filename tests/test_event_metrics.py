import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('MPLCONFIGDIR','/tmp/spike-detector-test-mpl')
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
from spike_detector.utils.event_metrics import event_dff_percent,peth_edges,session_time_courses

class EventMetricsTests(unittest.TestCase):
    def test_polarity_percent_and_invalid_baseline_preserve_event_order(self):
        trace=np.full(200,100.);base=trace.copy();trace[50]=70;trace[100]=80;base[100]=0
        values=event_dff_percent(trace,base,[5,50,100,195],1000,True)
        self.assertTrue(np.isnan(values[[0,2,3]]).all());self.assertEqual(values[1],30)
        trace[50]=130
        self.assertEqual(event_dff_percent(trace,np.full(200,100.),[50],1000,False)[0],30)
        mask=np.ones(200,bool);mask[49]=False
        self.assertTrue(np.isnan(event_dff_percent(trace,np.full(200,100.),[50],1000,False,mask)[0]))

    def test_natural_session_order_and_no_cross_session_bins(self):
        sessions=[dict(key=10,name='trial10.csv',folder='/tmp/a',duration_s=6),
                  dict(key=2,name='trial2.csv',folder='/tmp/a',duration_s=6)]
        records=[dict(session_key=2,times_s=[1,5.5],dff_percent=[10,20],intervals_s=[[.5,6]]),
                 dict(session_key=10,times_s=[1],dff_percent=[30],intervals_s=[[0,6]])]
        result=session_time_courses(sessions,records,5)
        self.assertEqual(result['names'],['trial2.csv','trial10.csv'])
        self.assertEqual(result['boundaries_s'],[6]);self.assertEqual(result['duration_s'],12)
        np.testing.assert_allclose(result['segments'][0]['rate_hz'],[1/4.5,1])
        np.testing.assert_allclose(result['segments'][0]['dff_percent'],[10,20])
        np.testing.assert_allclose(result['segments'][1]['time_s'],[8.5,11.5])

    def test_missing_cell_keeps_session_slot_and_multiple_batch_paths_are_not_joined(self):
        sessions=[dict(key=i,name=f'trial{i}',folder='/tmp/a',duration_s=10) for i in (1,2,3)]
        records=[dict(session_key=3,times_s=[1],dff_percent=[5],intervals_s=[[0,10]])]
        result=session_time_courses(sessions,records,5)
        self.assertTrue(np.isnan(result['segments'][1]['rate_hz']).all())
        np.testing.assert_allclose(result['segments'][2]['time_s'],[22.5,27.5])
        sessions[2]['folder']='/tmp/b'
        self.assertIn('one Path',session_time_courses(sessions,records,5)['reason'])

    def test_pooling_uses_cell_exposure_not_summed_rates_or_nan_amplitudes(self):
        sessions=[dict(key=1,name='a',folder='/tmp/a',duration_s=10)]
        records=[dict(session_key=1,times_s=[1,2],dff_percent=[10,np.nan],intervals_s=[[0,10]]),
                 dict(session_key=1,times_s=[1],dff_percent=[20],intervals_s=[[0,5]])]
        result=session_time_courses(sessions,records,5)['segments'][0]
        np.testing.assert_allclose(result['rate_hz'],[.3,0])
        self.assertEqual(result['dff_percent'][0],15);self.assertTrue(np.isnan(result['dff_percent'][1]))
        np.testing.assert_allclose(result['exposure_s'],[10,5])

    def test_masked_events_and_time_are_both_excluded(self):
        sessions=[dict(key=1,name='a',folder='/tmp/a',duration_s=10)]
        records=[dict(session_key=1,times_s=[1,4,6,9],dff_percent=[10,999,20,999],intervals_s=[[0,3],[5,8]])]
        segment=session_time_courses(sessions,records,5)['segments'][0]
        np.testing.assert_allclose(segment['rate_hz'],[1/3,1/3])
        np.testing.assert_allclose(segment['dff_percent'],[10,20])

    def test_peth_range_includes_zero_and_exact_partial_bin_durations(self):
        np.testing.assert_allclose(peth_edges(20,60,1),np.arange(-20,61))
        np.testing.assert_allclose(peth_edges(5,7,3),[-5,-3,0,3,6,7])
        np.testing.assert_allclose(peth_edges(0,2,3),[0,2])
        with self.assertRaises(ValueError):peth_edges(0,0,1)

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'Analysis/spike_analysis/comparison.py').is_file(),
                         'Optional local comparison package not present')
    def test_fallback_matches_compare_notebook_extractor(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Analysis'))
        from spike_analysis.comparison import event_extractor
        trace=np.full(200,100.);trace[50]=70;trace[100]=80
        config=dict(baseline_window_ms=30,waveform_windows_ms={'CS':30,'SS':30},spike_types=['SS'],snr_source='saved')
        with tempfile.TemporaryDirectory() as folder:
            np.savez(Path(folder)/'trace_analyzed.npz',fs=1000,time_ms=200+np.arange(200),raw_data=trace[:,None],
                baseline_traces=np.full((200,1),100.),cell_names=['ROI1'],spike_times_ss=np.array([[250,300]],object))
            rows,_=event_extractor(config,dict(polarity='negative'))(folder,'Test')
        expected=np.array([row['minus_dff_percent'] for row in rows])
        np.testing.assert_allclose(event_dff_percent(trace,np.full(200,100.),[50,100],1000,True),expected)

if __name__=='__main__':unittest.main()
