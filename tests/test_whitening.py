"""Whitening must preserve alignment, mask safety and the original controls."""
import json
import unittest
from unittest.mock import patch
import numpy as np
from scipy.signal import lfilter
from spike_detector.utils.processing import process_cell_template_matching, _amplitude_fitted_llr_score, estimate_noise_mad
from spike_detector.utils.whitening import fit_whitening, whiten_trace, whiten_kernel


class WhiteningTests(unittest.TestCase):
    def signal(self):
        fs = 5000.
        x = lfilter([1], [1, -.8], np.random.default_rng(21).normal(0, .15, 30000))
        wave = np.exp(-.5*(np.arange(-20, 21)/3)**2)
        return fs, x, wave

    def test_background_correlation_reduced_with_bounded_symmetric_filter(self):
        fs, x, _ = self.signal()
        state = fit_whitening(x, fs, np.zeros(x.size, bool), 'ss')
        self.assertTrue(state['info']['applied'])
        y = whiten_trace(x, state)
        before = np.corrcoef(x[100:-101], x[101:-100])[0, 1]
        after = np.corrcoef(y[100:-101], y[101:-100])[0, 1]
        self.assertGreater(before, .75)
        self.assertLess(abs(after), .2)
        np.testing.assert_allclose(state['fir'], state['fir'][::-1], atol=1e-12)
        json.dumps(state['info'], allow_nan=False)

    def test_trace_and_template_share_transform_and_score_is_gaussian_gain(self):
        fs, x, wave = self.signal()
        state = fit_whitening(x, fs, np.zeros(x.size, bool), 'ss')
        # A complete, mean-centered kernel planted on white background.
        wave -= np.mean(wave)
        signal = np.zeros_like(x)
        signal[14980:15021] = wave
        transformed = whiten_trace(signal, state)
        expected = whiten_kernel(wave, state)
        half = len(expected)//2
        np.testing.assert_allclose(transformed[15000-half:15000+half+1], expected, atol=1e-12)
        y = whiten_trace(x+4*signal, state)
        gain, response, _, _ = _amplitude_fitted_llr_score(y, [wave], [fs], fs,
            state['noise_mask'], whitening=state)
        self.assertLessEqual(abs(np.argmax(gain)-15000), 1)
        sigma = estimate_noise_mad(y[~state['noise_mask']])
        np.testing.assert_allclose(gain, .5*(np.maximum(response, 0)/sigma)**2)

    def test_all_methods_burst_recovery_and_frozen_control(self):
        fs, x, wave = self.signal()
        peaks = (5000, 5050, 5100, 15000)
        for p in peaks:
            x[p-20:p+21] += 4*wave
        settings = dict(negative_going=False, cs_high_cut=0, template_ss_bank=[wave],
            template_ss_fs_bank=[fs], ss_thresh_sigma=4, initial_blank_ms=0,
            template_ss_lowpass_hz=0, template_whitening=True, ss_similarity_threshold=.2)
        for method in ('LLR Probability Vector', 'Burst-aware LLR', 'Normalized Similarity'):
            with self.subTest(method=method):
                result = process_cell_template_matching(x, fs, template_match_method=method, **settings)
                original = process_cell_template_matching(x, fs, template_match_method=method,
                    **dict(settings, template_whitening=False))
                self.assertTrue(result['template_whitening']['ss']['applied'])
                self.assertEqual(result['template_whitening']['cs']['reason'], 'no templates')
                for p in peaks:
                    self.assertTrue(np.any(abs(result['ss_peaks']-p) <= 2))
                np.testing.assert_array_equal(result['ss_trace'], original['ss_trace'])
                masks = {kind: result[kind+'_exclusion_mask'] for kind in ('cs', 'ss')}
                with patch('spike_detector.utils.whitening.fit_whitening', side_effect=AssertionError('refitted')):
                    reverse = process_cell_template_matching(-x, fs, template_match_method=method,
                        fixed_whitening=result['_template_whitening'], fixed_exclusion_masks=masks, **settings)
                np.testing.assert_array_equal(reverse['ss_exclusion_mask'], masks['ss'])

    def test_mask_covers_trace_and_kernel_firs_and_excluded_artifact_cannot_leak(self):
        fs, x, wave = self.signal()
        mask = np.zeros(len(x), bool)
        mask[10000:10100] = True
        state = fit_whitening(x, fs, mask, 'ss')
        settings = dict(negative_going=False, cs_high_cut=0, template_ss_bank=[wave],
            template_ss_fs_bank=[fs], initial_blank_ms=2000, template_ss_lowpass_hz=0,
            template_whitening=True, fixed_whitening={'ss': state}, use_preprocessed=True,
            pre_baseline=np.zeros_like(x))
        reference = process_cell_template_matching(x, fs, pre_detrended=x, **settings)
        altered = x.copy()
        altered[:10000] = 1e5
        artifact = process_cell_template_matching(altered, fs, pre_detrended=altered, **settings)
        # Initial blank plus SS core half width plus TWO whitening half widths.
        guard = 7 + 2*(len(state['fir'])//2)
        self.assertTrue(np.all(reference['ss_exclusion_mask'][:10000+guard]))
        safe = ~reference['ss_exclusion_mask']
        np.testing.assert_allclose(reference['ss_similarity_trace'][safe],
                                   artifact['ss_similarity_trace'][safe], atol=1e-7)

    def test_insufficient_background_falls_back_explicitly(self):
        fs, _, wave = self.signal()
        x = np.zeros(1000)
        x[480:521] = wave
        settings = dict(negative_going=False, cs_high_cut=0, template_ss_bank=[wave],
                        template_ss_fs_bank=[fs], initial_blank_ms=0, template_ss_lowpass_hz=0)
        original = process_cell_template_matching(x, fs, **settings)
        whitened = process_cell_template_matching(x, fs, template_whitening=True, **settings)
        self.assertFalse(whitened['template_whitening']['ss']['applied'])
        self.assertIn('zero', whitened['template_whitening']['ss']['reason'])
        np.testing.assert_array_equal(original['ss_peaks'], whitened['ss_peaks'])
        np.testing.assert_array_equal(original['ss_similarity_trace'], whitened['ss_similarity_trace'])


if __name__ == '__main__':
    unittest.main()
