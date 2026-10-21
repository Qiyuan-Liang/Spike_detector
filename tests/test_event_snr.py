"""Event SNR uses shared windows, rejects baseline contamination and keeps order."""
import unittest
import numpy as np
from spike_detector.utils.processing import compute_event_snrs, EVENT_SNR_POLICY
from spike_detector.utils.stats import mean_std_count


class EventSnrTests(unittest.TestCase):
    def test_shared_windows_mask_neighboring_spikes_for_both_types(self):
        x = np.random.default_rng(7).normal(0, 1, 600)
        x[140:159] += 100  # CS tail, outside the amplitude baseline.
        x[231:238] += 100  # SS inside the amplitude baseline.
        x[250] = 8
        result = dict(cs_peaks=np.array([140, 250]), ss_peaks=np.array([233, 250]))
        cs = compute_event_snrs(result, 'CS', 1000, trace_override=x)
        ss = compute_event_snrs(result, 'SS', 1000, trace_override=x)
        clean = np.ones(x.size, bool)
        clean[131:159] = False
        clean[241:269] = False
        clean[231:238] = False
        clean[248:255] = False
        baseline = np.median(x[225:245][clean[225:245]])
        noise = x[100:245][clean[100:245]]
        sigma = 1.4826 * np.median(abs(noise-np.median(noise)))
        self.assertAlmostEqual(cs[1], (8-baseline)/sigma)
        self.assertAlmostEqual(ss[1], cs[1])
        self.assertGreater(cs[1], 5)

    def test_insufficient_baseline_is_nan_in_event_order(self):
        x = np.random.default_rng(2).normal(0, 1, 600)
        mask = np.zeros(x.size, bool)
        mask[225:245] = True
        result = dict(ss_peaks=[250, 500], cs_peaks=[], ss_exclusion_mask=mask)
        values = compute_event_snrs(result, 'SS', 1000, trace_override=x)
        self.assertEqual(len(values), 2)
        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(np.isfinite(values[1]))

    def test_sparse_local_noise_uses_global_masked_mad(self):
        x = np.random.default_rng(9).normal(0, 1, 600)
        mask = np.zeros(x.size, bool)
        mask[100:245] = True
        mask[225:235] = False  # Baseline exists; only ten local noise samples.
        result = dict(ss_peaks=[250], cs_peaks=[], ss_exclusion_mask=mask)
        clean = ~mask
        clean[248:255] = False
        noise = x[clean]
        sigma = 1.4826*np.median(abs(noise-np.median(noise)))
        expected = (x[250]-np.median(x[225:235]))/sigma
        self.assertAlmostEqual(compute_event_snrs(result, 'SS', 1000, trace_override=x)[0], expected)

    def test_zero_noise_and_recording_edge_are_unavailable_not_shifted(self):
        result = dict(ss_peaks=[2, 250], cs_peaks=[])
        values = compute_event_snrs(result, 'SS', 1000, trace_override=np.zeros(500))
        self.assertEqual(len(values), 2)
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(mean_std_count([3, np.nan, 5, np.inf]), (4., 1., 2))
        self.assertEqual(EVENT_SNR_POLICY['amplitude_baseline_interval_ms'], [-25., -5.])
        self.assertEqual(EVENT_SNR_POLICY['noise_estimation_interval_ms'], [-150., -5.])


if __name__ == '__main__':
    unittest.main()
