"""Template matcher checks that do not require a Qt runtime."""
import unittest

import numpy as np

from spike_detector.utils.processing import process_cell_template_matching, _similarity_score


class TemplateMatchingCoreTests(unittest.TestCase):
    def test_cs_filtered_peak_gate_is_reported_and_can_be_disabled(self):
        fs = 5000.0
        trace = (-1.0) ** np.arange(5000)
        axis = np.arange(-75, 76)
        template = np.exp(-0.5 * (axis / 18.0) ** 2)
        trace[2425:2576] += 1.5 * template
        settings = dict(negative_going=False, cs_high_cut=0,
                        template_cs_bank=[template], template_cs_fs_bank=[fs],
                        cs_thresh_sigma=4, cs_min_fwhm_ms=0,
                        cs_min_dist_ms=10, initial_blank_ms=0)
        enabled = process_cell_template_matching(trace, fs, cs_min_filtered_peak_sigma=3, **settings)
        disabled = process_cell_template_matching(trace, fs, cs_min_filtered_peak_sigma=0, **settings)
        self.assertEqual(len(enabled['cs_peaks']), 0)
        self.assertEqual(disabled['cs_peaks'].tolist(), [2500])
        self.assertGreater(enabled['cs_candidate_diagnostics']['peak_rejected'], 0)
        self.assertEqual(enabled['cs_candidate_diagnostics']['response_rejected'], 0)
        self.assertEqual(disabled['cs_candidate_diagnostics']['peak_rejected'], 0)
        self.assertEqual(disabled['cs_min_filtered_peak_sigma_used'], 0)

    def test_positive_core_accepts_isolated_above_cutoff_spike(self):
        fs = 5000.0
        rng = np.random.default_rng(4)
        trace = rng.normal(0, .1, 5000)
        axis = np.arange(-20, 21)
        template = np.exp(-.5 * (axis / 3)**2)
        trace[2480:2521] += .15 * template
        result = process_cell_template_matching(
            trace, fs, negative_going=False, use_preprocessed=True,
            pre_detrended=trace, pre_baseline=np.zeros_like(trace),
            pre_detrended_cs=trace, pre_detrended_ss=trace,
            cs_high_cut=0, template_ss_bank=[template],
            template_ss_fs_bank=[fs], template_match_method='Burst-aware LLR',
            ss_thresh_sigma=4, similarity_min_response_sigma=2.2,
            ss_min_dist_ms=3, initial_blank_ms=0, template_ss_lowpass_hz=0)
        self.assertEqual(result['ss_peaks'].tolist(), [2500])
        self.assertGreater(result['ss_similarity_trace'][2500], .5 * 4**2)
        self.assertLess(result['ss_similarity_trace'][2500], .5 * 5.5**2)
        self.assertNotIn('support_rejected', result['ss_candidate_diagnostics'])

    def test_positive_core_match_recovers_close_spikes_without_negative_rebound(self):
        fs = 5000.0
        rng = np.random.default_rng(41)
        trace = rng.normal(0, .05, 4000)
        axis = np.arange(-20, 21)
        template = np.exp(-.5 * (axis / 3)**2)
        for peak in (1000, 1019, 1038):
            trace[peak-20:peak+21] += .8 * template
        trace[1790:1831] -= 2.5 * template
        result = process_cell_template_matching(
            trace, fs, negative_going=False, use_preprocessed=True,
            pre_detrended=trace, pre_baseline=np.zeros_like(trace),
            pre_detrended_cs=trace, pre_detrended_ss=trace,
            cs_high_cut=0, template_ss_bank=[template],
            template_ss_fs_bank=[fs], template_match_method='Burst-aware LLR',
            ss_thresh_sigma=3.5, similarity_min_response_sigma=2.2,
            ss_min_dist_ms=3, initial_blank_ms=0, template_ss_lowpass_hz=0)
        for peak in (1000, 1019, 1038):
            self.assertTrue(np.any(np.abs(result['ss_peaks'] - peak) <= 1))
        self.assertFalse(np.any((result['ss_peaks'] >= 1780) & (result['ss_peaks'] <= 1840)))
        self.assertNotIn('support_rejected', result['ss_candidate_diagnostics'])

    def test_amplitude_fitted_llr_recovers_dim_and_bright_events(self):
        fs = 3333.0
        rng = np.random.default_rng(34)
        n = 5000
        positions = (900, 1100, 2500, 2700)
        x = rng.normal(0, .2, n)
        axis = np.arange(-13, 14)
        template = np.exp(-.5 * (axis / 2.0)**2)
        for peak, amp in zip(positions, (1.2, 3.0, 1.2, 3.0)):
            x[peak-13:peak+14] += amp * template
        result = process_cell_template_matching(
            x, fs, negative_going=False, cs_high_cut=0,
            template_ss_bank=[3.0 * template], template_ss_fs_bank=[fs],
            ss_thresh_sigma=4, similarity_min_response_sigma=2,
            template_ss_lowpass_hz=0, initial_blank_ms=0,
            template_match_method='LLR Probability Vector')
        for peak in positions:
            self.assertTrue(np.any(np.abs(result['ss_peaks'] - peak) <= 3))
        self.assertTrue(result['cs_candidate_diagnostics']['no_templates'])
        self.assertEqual(len(result['cs_peaks']), 0)

    def test_single_cs_bank_keeps_empty_ss_track(self):
        fs = 1000.0
        x = np.zeros(1000)
        template = np.exp(-.5 * (np.arange(-20, 21) / 5)**2)
        x[280:321] = 5 * template
        result = process_cell_template_matching(
            x, fs, negative_going=False, cs_high_cut=0,
            template_cs_bank=[template], template_cs_fs_bank=[fs],
            template_match_method='Normalized Similarity',
            cs_similarity_threshold=.8, initial_blank_ms=0)
        self.assertTrue(result['ss_candidate_diagnostics']['no_templates'])
        self.assertEqual(len(result['ss_peaks']), 0)
        np.testing.assert_array_equal(result['ss_similarity_trace'], 0)

    def test_truncated_ss_core_is_not_rejected_by_full_template_shape(self):
        fs = 5000.0
        rng = np.random.default_rng(7)
        x = rng.normal(0, .04, 2500)
        axis = np.arange(-25, 26)
        template = np.exp(-.5*(axis/3)**2) + .9*np.exp(-.5*((axis-16)/7)**2)
        peak = 1250
        x[peak-25:peak+26] += 1.3*np.exp(-.5*(axis/3)**2)
        full_similarity, _, _ = _similarity_score(x, [template], [fs], fs, short_core=False)
        self.assertLess(full_similarity[peak], .5)
        result = process_cell_template_matching(
            x, fs, negative_going=False, template_ss_bank=[template],
            template_ss_fs_bank=[fs], ss_thresh_sigma=4,
            template_ss_lowpass_hz=0, initial_blank_ms=0)
        self.assertTrue(np.any(np.abs(result['ss_peaks']-peak) <= 2))
        self.assertNotIn('shape_rejected', result['ss_candidate_diagnostics'])

    def test_fixed_control_masks_keep_the_same_evaluation_interval(self):
        fs = 5000.0
        x = np.zeros(1500)
        axis = np.arange(-15, 16)
        template = np.exp(-.5*(axis/2)**2)
        x[735:766] = 3*template
        fixed_cs = np.zeros(x.size, dtype=bool)
        fixed_ss = np.zeros(x.size, dtype=bool)
        fixed_ss[700:800] = True
        result = process_cell_template_matching(
            x, fs, negative_going=False, template_ss_bank=[template],
            template_ss_fs_bank=[fs], ss_thresh_sigma=3,
            template_ss_lowpass_hz=0, initial_blank_ms=0,
            fixed_exclusion_masks={'cs': fixed_cs, 'ss': fixed_ss})
        np.testing.assert_array_equal(result['cs_exclusion_mask'], fixed_cs)
        np.testing.assert_array_equal(result['ss_exclusion_mask'], fixed_ss)
        self.assertEqual(len(result['ss_peaks']), 0)
        self.assertGreater(result['ss_candidate_diagnostics']['masked'], 0)

    def test_template_ss_spacing_is_applied_once_to_close_candidates(self):
        fs = 5000.0
        rng = np.random.default_rng(3)
        x = rng.normal(0, .02, 2000)
        axis = np.arange(-20, 21)
        template = np.exp(-.5*(axis/2)**2)
        for peak in (1000, 1020):  # 4 ms apart
            x[peak-20:peak+21] += template
        kwargs = dict(negative_going=False, template_ss_bank=[template],
            template_ss_fs_bank=[fs], template_ss_lowpass_hz=0,
            ss_thresh_sigma=5, initial_blank_ms=0)
        narrow = process_cell_template_matching(x, fs, ss_min_dist_ms=2, **kwargs)
        wide = process_cell_template_matching(x, fs, ss_min_dist_ms=6, **kwargs)
        np.testing.assert_array_equal(narrow['ss_peaks'], [1000, 1020])
        self.assertEqual(len(wide['ss_peaks']), 1)
        self.assertEqual(wide['ss_candidate_diagnostics']['refractory_rejected'], 1)


if __name__ == '__main__':
    unittest.main()
