"""Shared amplitude and candidate-centred width QC for all GUI detectors.

Amplitude uses the original frame-processed, baseline-corrected, polarity-oriented
trace, before denoising, detection filters or whitening. SNR follows the shared
EVENT_SNR_POLICY. Candidate noise exclusions are frozen before QC, so rejection
cannot make its own SNR improve. Unknown measurements remain flagged, not passes.
"""
import numpy as np
from .processing import compute_event_snrs
from .widths import measure_widths

QC_DEFAULTS = dict(CS_MIN_AMPLITUDE_SNR=0.0, SS_MIN_AMPLITUDE_SNR=0.0,
                   CS_MIN_FWHM_MS=4.0, CS_MAX_FWHM_MS=0.0,
                   SS_MIN_FWHM_MS=0.0, SS_MAX_FWHM_MS=4.5)
LEGACY_QC_KEYS = ('TEMPLATE_CS_MIN_FILTERED_PEAK_SIGMA', 'TEMPLATE_MIN_RESPONSE_SIGMA',
                  'TEMPLATE_SS_MIN_FWHM_MS', 'SS_MAX_FWHM_FILTER_ENABLED')


def migrate_qc_settings(params):
    """Keep width choices; do not reinterpret projection MAD as amplitude SNR."""
    params = dict(params)
    if params.get('SS_MAX_FWHM_FILTER_ENABLED') is False:
        params['SS_MAX_FWHM_MS'] = 0.0
    for key in LEGACY_QC_KEYS:
        params.pop(key, None)
    for key, value in QC_DEFAULTS.items():
        params.setdefault(key, value)
    return params


def validate_qc(params):
    limits = {key: float(params.get(key, value)) for key, value in QC_DEFAULTS.items()}
    for key, value in limits.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative.')
    for kind in ('CS', 'SS'):
        lo, hi = limits[f'{kind}_MIN_FWHM_MS'], limits[f'{kind}_MAX_FWHM_MS']
        if lo and hi and lo > hi:
            raise ValueError(f'{kind} minimum FWHM cannot exceed maximum FWHM.')
    return limits


def _evaluate(res, kind, source, fs, limits):
    peaks = np.asarray(res[f'{kind}_peaks'], dtype=int)
    label = kind.upper()
    snrs = np.asarray(compute_event_snrs(res, label, fs, trace_override=source), dtype=float)
    rows = measure_widths(res[f'{kind}_trace'], peaks, fs, 100 if kind == 'cs' else 50,
                          neighbors=res['cs_peaks'] if kind == 'ss' else (),
                          align_ms=res.get('width_alignment_ms', 0))
    minimum = limits[f'{label}_MIN_AMPLITUDE_SNR']
    lo, hi = limits[f'{label}_MIN_FWHM_MS'], limits[f'{label}_MAX_FWHM_MS']
    keep, kept_snrs = [], []
    counts = dict(amplitude_rejected=0, amplitude_unavailable=0, width_rejected=0, width_uncertain=0)
    for row, snr in zip(rows, snrs):
        width = row['fwhm_ms']
        amplitude_fail = bool(minimum > 0 and np.isfinite(snr) and snr < minimum)
        amplitude_unknown = bool(minimum > 0 and not np.isfinite(snr))
        width_fail = bool(np.isfinite(width) and ((lo > 0 and width < lo) or (hi > 0 and width > hi)))
        row['amplitude_snr'] = float(snr)
        row['amplitude_decision'] = 'fail' if amplitude_fail else 'uncertain' if amplitude_unknown else 'pass' if minimum else 'off'
        row['decision'] = 'fail' if width_fail else 'uncertain' if not np.isfinite(width) else 'pass'
        counts['amplitude_rejected'] += int(amplitude_fail)
        counts['amplitude_unavailable'] += int(amplitude_unknown)
        counts['width_rejected'] += int(not amplitude_fail and not amplitude_unknown and width_fail)
        if not amplitude_fail and not amplitude_unknown and not width_fail:
            keep.append(row['candidate_index'])
            kept_snrs.append(float(snr))
            counts['width_uncertain'] += int(not np.isfinite(width))
    return np.asarray(keep, dtype=int), rows, kept_snrs, counts


def run_with_qc(detect, source, fs, params):
    """Run detector, QC CS, rebuild SS exclusions if necessary, then QC SS.

    detect(cs_keep_indices=None) must filter CS before constructing SS masks.
    Only one extra detection is required when CS QC rejects candidates. Combo
    uses this separately for threshold seeds and its frozen-template pass.
    """
    limits = validate_qc(params)
    result = detect()
    cs_peaks, cs_rows, cs_snrs, cs_counts = _evaluate(result, 'cs', source, fs, limits)
    first_diagnostics = result.get('cs_candidate_diagnostics')
    if not np.array_equal(cs_peaks, result['cs_peaks']):
        result = detect(cs_keep_indices=cs_peaks)
        # Report the proposals entering QC, not only the surviving CS subset.
        if first_diagnostics is not None:
            result['cs_candidate_diagnostics'] = first_diagnostics
    ss_peaks, ss_rows, ss_snrs, ss_counts = _evaluate(result, 'ss', source, fs, limits)
    result.update(cs_peaks=cs_peaks, ss_peaks=ss_peaks, stats_source_trace=np.asarray(source).copy(),
                  cs_width_candidates=cs_rows, ss_width_candidates=ss_rows,
                  event_widths_cs=[r for r in cs_rows if r['candidate_index'] in cs_peaks],
                  event_widths_ss=[r for r in ss_rows if r['candidate_index'] in ss_peaks],
                  event_amplitude_snr_cs=cs_snrs, event_amplitude_snr_ss=ss_snrs,
                  amplitude_snr_indices_cs=cs_peaks.copy(), amplitude_snr_indices_ss=ss_peaks.copy(),
                  spike_qc=dict(limits=limits, cs=cs_counts, ss=ss_counts,
                    amplitude_source='frame-processed baseline-corrected trace before denoising/filtering/whitening',
                    width_source='filtered detection trace before whitening',
                    unknown_amplitude_policy='reject when minimum SNR enabled',
                    unknown_width_policy='retain and flag'),
                  ss_width_source='filtered SS detection trace',
                  ss_fwhm_filter_enabled=limits['SS_MAX_FWHM_MS'] > 0,
                  ss_fwhm_filter_max_ms=limits['SS_MAX_FWHM_MS'],
                  ss_fwhm_filter_removed=ss_counts['width_rejected'],
                  ss_fwhm_filter_uncertain=ss_counts['width_uncertain'])
    return result
