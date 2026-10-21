"""Restore archived GUI detections without running or reinterpreting a detector."""
from pathlib import Path
import copy
import json
import numpy as np
from .validation import validate_session
from .processing import estimate_noise_mad


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _json(archive, key, default):
    if key not in archive:
        return copy.deepcopy(default)
    value = archive[key]
    if value.size != 1:
        raise ValueError(f'{key} must contain one JSON document.')
    return json.loads(str(value.item()))


def _matrix(archive, key, shape, optional=False):
    if key not in archive:
        if optional:
            return None
        raise ValueError(f'Missing saved {key}; original preprocessing cannot be restored.')
    array = np.asarray(archive[key], dtype=float)
    if array.shape != shape or (not optional and not np.all(np.isfinite(array))):
        raise ValueError(f'Saved {key} must match trace dimensions and contain valid samples.')
    return array


def _cell_values(archive, key, cell, cells):
    if key not in archive:
        return None
    array = archive[key]
    if array.ndim < 1 or len(array) != cells:
        raise ValueError(f'{key} must contain one array per cell.')
    return np.asarray(array[cell], dtype=float).ravel()


def _indices(times, time_ms, fs):
    times = np.asarray(times, dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('Saved spike times must be finite, unique and increasing.')
    right = np.clip(np.searchsorted(time_ms, times), 0, len(time_ms)-1)
    left = np.maximum(0, right-1)
    indices = np.where(np.abs(time_ms[left]-times) < np.abs(time_ms[right]-times), left, right)
    if np.any(np.abs(time_ms[indices]-times) > max(1e-6, 1e-4 * 1000/fs)):
        raise ValueError('Saved spike times do not lie on the recording time axis.')
    return indices.astype(int)


def load_saved_detection(path):
    path = Path(path).resolve()
    warnings = []
    with np.load(path, allow_pickle=True) as z:
        for key in ('time_ms', 'raw_data', 'cell_names', 'fs', 'spike_times_cs', 'spike_times_ss'):
            if key not in z:
                raise ValueError(f'Not a saved GUI detection: missing {key}.')
        data = dict(time_ms=np.asarray(z['time_ms'], dtype=float),
                    raw_data=np.asarray(z['raw_data'], dtype=float),
                    cell_names=[str(v) for v in z['cell_names']], fs=float(z['fs']))
        validate_session(data)
        shape = data['raw_data'].shape
        cells = shape[1]
        processed = _matrix(z, 'processed_raw_data', shape)
        baseline = _matrix(z, 'baseline_traces', shape)
        traces = {kind: _matrix(z, kind+'_traces', shape) for kind in ('cs', 'ss')}
        analysis = _json(z, 'analysis_settings_json', {})
        if not isinstance(analysis, dict):
            raise ValueError('Invalid saved analysis settings.')
        snapshot = _json(z, 'gui_settings_json', {})
        sidecar = path.with_name(path.stem+'_settings.json')
        if sidecar.is_file():
            try:
                side = json.loads(sidecar.read_text())
                if not isinstance(side, dict) or side.get('results_file', path.name) != path.name:
                    raise ValueError('Sidecar does not identify this result file.')
                if not snapshot:
                    snapshot = side
                if not analysis:
                    analysis = side.get('analysis_settings', {})
            except (ValueError, OSError) as exc:
                warnings.append(f'Ignored settings sidecar: {exc}')
        if not isinstance(snapshot, dict):
            raise ValueError('Invalid saved GUI settings.')
        # Embedded run settings are authoritative over editable sidecar/UI values.
        params = _json(z, 'detection_params_json', analysis.get('params', snapshot.get('params', {})))
        base = _json(z, 'baseline_params_json', analysis.get('baseline_params', snapshot.get('baseline_params', {})))
        if not isinstance(params, dict) or not params or not isinstance(base, dict) or not base:
            raise ValueError('Saved detection/preprocessing settings are missing.')
        params = dict(params)
        params['FRAME_PROCESSING_MODE'] = analysis.get('frame_processing_mode', params.get('FRAME_PROCESSING_MODE', 'Rolling average'))
        for section in ('ui', 'colors', 'template_sources'):
            if section in snapshot and not isinstance(snapshot[section], dict):
                raise ValueError(f'Invalid saved {section}.')
        for kind, sources in snapshot.get('template_sources', {}).items():
            if kind not in ('CS', 'SS') or not isinstance(sources, list) or not all(isinstance(v, str) for v in sources):
                raise ValueError('Invalid saved template source list.')
        ui = dict(snapshot.get('ui', {}))
        ui.update(baseline_method=base.get('method', 'Median'),
                  baseline_window_ms=base.get('window_ms', 30), baseline_percentile=base.get('percentile', 20),
                  avg_frames=analysis.get('avg_frames', ui.get('avg_frames', 0)),
                  detection_tab_index=int(params.get('DETECTION_METHOD') == 'Template Matching'))
        snapshot.update(params=params, baseline_params=base, ui=ui)
        snapshot.setdefault('template_sources', {'CS': [], 'SS': []})
        runtime = _json(z, 'detection_runtime_json', [{} for _ in range(cells)])
        widths = _json(z, 'width_quality_json', [{} for _ in range(cells)])
        qc = _json(z, 'spike_qc_json', [{} for _ in range(cells)])
        for label, records in (('runtime', runtime), ('width reports', widths), ('QC', qc)):
            if not isinstance(records, list) or len(records) != cells or not all(isinstance(r, dict) for r in records):
                raise ValueError(f'Saved {label} must match the cell count.')
        matrices = {key: _matrix(z, key, shape, optional=True) for key in
                    ('cs_exclusion_mask', 'ss_exclusion_mask', 'cs_similarity_traces',
                     'ss_similarity_traces', 'cs_threshold_traces', 'ss_threshold_traces')}
        sign = -1 if params.get('NEGATIVE_GOING', True) else 1
        data['results'] = []
        data['spike_times_cs'], data['spike_times_ss'] = [], []
        for cell in range(cells):
            res = dict(runtime[cell])
            method = analysis.get('detection_method', params.get('DETECTION_METHOD', 'Threshold'))
            if params.get('COMBO_DETECTION'):
                method = 'Template Matching + Combo'
            res.setdefault('det_method', method)
            for key, archive_key in (('combo', 'combo_cells'), ('template_whitening', 'template_whitening_cells')):
                records = analysis.get(archive_key, [])
                if key not in res and len(records) == cells:
                    res[key] = records[cell]
            res.update(baseline=baseline[:, cell], detrended=(processed[:, cell]-baseline[:, cell])*sign,
                       stats_source_trace=(processed[:, cell]-baseline[:, cell])*sign,
                       negative_going=bool(params.get('NEGATIVE_GOING', True)))
            raw_sigmas = np.asarray(z['raw_sigmas'], dtype=float) if 'raw_sigmas' in z else []
            res['raw_sigma'] = float(raw_sigmas[cell]) if len(raw_sigmas) == cells else estimate_noise_mad(res['detrended'])
            for kind in ('cs', 'ss'):
                times = _cell_values(z, 'spike_times_'+kind, cell, cells)
                peaks = _indices(times, data['time_ms'], data['fs'])
                data['spike_times_'+kind].append(times)
                res[kind+'_peaks'], res[kind+'_trace'] = peaks, traces[kind][:, cell]
                mask = matrices[kind+'_exclusion_mask']
                if mask is None:
                    raise ValueError('Saved observation/exclusion masks are missing; cannot restore statistics reliably.')
                if not np.all(np.isin(mask[:, cell], [0, 1])):
                    raise ValueError('Saved exclusion masks must be Boolean.')
                res[kind+'_exclusion_mask'] = mask[:, cell].astype(bool)
                for archived, result_key in ((kind+'_similarity_traces', kind+'_similarity_trace'),
                                              (kind+'_threshold_traces', kind+'_threshold_trace')):
                    matrix = matrices[archived]
                    if matrix is not None and np.all(np.isfinite(matrix[:, cell])):
                        res[result_key] = matrix[:, cell]
                snrs = _cell_values(z, 'event_snr_'+kind, cell, cells)
                fwhms = _cell_values(z, 'event_fwhm_'+kind, cell, cells)
                for label, values in (('SNR', snrs), ('FWHM', fwhms)):
                    if values is not None and len(values) != len(peaks):
                        raise ValueError(f'Saved {kind.upper()} {label} does not match event count.')
                res['event_amplitude_snr_'+kind] = snrs if snrs is not None else np.full(len(peaks), np.nan)
                res['amplitude_snr_indices_'+kind] = peaks.copy()
                rows = widths[cell].get('event_widths_'+kind)
                if rows is None:
                    rows = [dict(candidate_index=int(p), peak_index=int(p), fwhm_ms=float(w),
                        status='measured' if np.isfinite(w) else 'uncertain',
                        decision='pass' if np.isfinite(w) else 'uncertain', reason='legacy_saved_width')
                        for p, w in zip(peaks, fwhms if fwhms is not None else np.full(len(peaks), np.nan))]
                rows = copy.deepcopy(rows)
                if [r.get('candidate_index') for r in rows] != peaks.tolist():
                    raise ValueError('Saved width rows do not match event sample indices.')
                for row in rows:
                    if row.get('fwhm_ms') is None:
                        row['fwhm_ms'] = np.nan
                res['event_widths_'+kind] = rows
                res[kind+'_width_candidates'] = widths[cell].get(kind+'_width_candidates', rows)
                for row in res[kind+'_width_candidates']:
                    if row.get('fwhm_ms') is None:
                        row['fwhm_ms'] = np.nan
                res.setdefault(kind+'_candidate_diagnostics', {'unavailable': True})
                res.setdefault('sigma_'+kind, estimate_noise_mad(traces[kind][~res[kind+'_exclusion_mask'], cell]))
                res.setdefault(kind+'_threshold_used', np.nan)
                if res[kind+'_threshold_used'] is None:
                    res[kind+'_threshold_used'] = np.nan
                res.setdefault(kind+'_detection_enabled', params.get(kind.upper()+'_THRESHOLD_SIGMA', 1) > 0)
            if qc[cell]:
                res['spike_qc'] = qc[cell]
            res.setdefault('local_baseline', bool(params.get('LOCAL_BASELINE', False)))
            res['saved_score_unavailable'] = 'Template' in str(res['det_method']) and any(
                kind+'_similarity_trace' not in res for kind in ('cs', 'ss'))
            res['loaded_local_threshold_unavailable'] = bool(res['local_baseline']) and any(
                kind+'_threshold_trace' not in res for kind in ('cs', 'ss'))
            res['ss_fwhm_filter_uncertain'] = sum(not np.isfinite(r['fwhm_ms']) for r in res['event_widths_ss'])
            data['results'].append(res)
        data.update(processed_raw_data=processed, baseline_traces=baseline,
                    loaded_detection_result=True, results_file=str(path), settings_file=str(sidecar),
                    _saved_settings_snapshot=snapshot, _saved_settings_base=str(path.parent),
                    _saved_detection_params=copy.deepcopy(params), input_time_unit=analysis.get('input_time_unit', 'ms'),
                    effective_settings=analysis.get('effective_settings', {}), load_warnings=warnings,
                    temporary_result=False, analysis_settings=analysis)
    return data


def load_saved_detection_folder(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError('Select a folder containing saved detection results.')
    result_dir = folder / 'spike_detection'
    if folder.name == 'spike_detection':
        result_dir, source_folder = folder, folder.parent
    elif result_dir.is_dir():
        source_folder = folder
    else:
        result_dir, source_folder = folder, folder
    files = sorted(result_dir.glob('*_analyzed.npz'))
    if not files:
        raise ValueError('No saved *_analyzed.npz results found in this folder or its spike_detection subfolder.')
    loaded, issues = {}, []
    for path in files:
        try:
            data = load_saved_detection(path)
            stem = path.name.removesuffix('_analyzed.npz')
            sources = [source_folder/(stem+ext) for ext in ('.csv', '.xlsx', '.npz')]
            name = next((p.name for p in sources if p.is_file()), path.name)
            data.update(source_folder=str(source_folder), session_name=name,
                        session_path=str(next((p for p in sources if p.is_file()), path)))
            loaded[name] = data
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            issues.append(f'{path.name}: {exc}')
    if not loaded:
        raise ValueError('No compatible saved detections found.\n'+'\n'.join(issues[:5]))
    return str(source_folder), loaded, issues
