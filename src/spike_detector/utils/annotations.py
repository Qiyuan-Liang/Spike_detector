"""Session/cell-scoped manual annotations and reproducible metric calculation."""
import copy
import numpy as np
from .qc import run_with_qc, QC_DEFAULTS


def manual_result(raw, fs, settings, events, detect, denoise_cfg):
    params = dict(settings['params'])
    # Prepare the same traces without collecting any automatic candidates.
    preparation = dict(params, DETECTION_METHOD='Threshold', COMBO_DETECTION=False,
                       CS_THRESHOLD_SIGMA=0, SS_THRESHOLD_SIGMA=0)
    res = detect(raw, fs, preparation, settings['baseline_params'], settings['ui']['avg_frames'],
                 params.get('FRAME_PROCESSING_MODE', 'Rolling average'), {}, denoise_cfg)
    for kind in ('cs', 'ss'):
        peaks = np.unique(np.asarray(events.get(kind.upper(), []), dtype=int))
        if np.any(peaks < 0) or np.any(peaks >= len(raw)) or np.any(res['recording_exclusion_mask'][peaks]):
            raise ValueError('A manual label lies outside the observed recording interval. Review the boundary blanks before saving.')
        res[kind+'_peaks'] = peaks
        res[kind+'_exclusion_mask'] = res['recording_exclusion_mask'].copy()
        res.pop('event_amplitude_snr_'+kind, None)
        res.pop('amplitude_snr_indices_'+kind, None)
    if np.intersect1d(res['cs_peaks'], res['ss_peaks']).size:
        raise ValueError('A sample cannot be labelled as both CS and SS; resolve its class before saving.')
    traces = {kind: res[kind+'_trace'].copy() for kind in ('cs', 'ss')}
    # Do not measure widths using artificial padding outside the observed interval.
    for kind in ('cs', 'ss'):
        res[kind+'_trace'][res['recording_exclusion_mask']] = np.nan
    res = run_with_qc(lambda **unused: copy.deepcopy(res), res['stats_source_trace'], fs,
                      {key: 0 for key in QC_DEFAULTS})
    for kind in ('cs', 'ss'):
        res[kind+'_trace'] = traces[kind]
        res[kind+'_detection_enabled'] = True
        res[kind+'_candidate_diagnostics'] = {'manual': True}
    res['det_method'] = 'Manual labels'
    res['threshold_mode'] = 'Manual labels; no detection/QC rejection'
    res['local_baseline'] = False
    res['cs_threshold_used'] = res['ss_threshold_used'] = np.nan
    res['manual_labels'] = True
    return res
