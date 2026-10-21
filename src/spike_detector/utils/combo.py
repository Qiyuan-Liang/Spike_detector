"""Learn frozen, cell-specific templates from conservative threshold seeds."""
import numpy as np

from .processing import (process_cell_simple, process_cell_template_matching,
                         apply_filter, _build_parallel_template_banks)
from .widths import measure_widths


def _learn_bank(trace, peaks, mask, fs, window_ms, *, max_width=None,
                neighbors=(), groups=1, components=2):
    half = int(round(float(window_ms) * fs / 2000))
    quality = dict(seeds=len(peaks), clean=0, edge_or_mask=0,
                   overlapping=0, width_rejected=0, invalid_amplitude=0,
                   clean_seed_indices=[])
    waves = []
    if half < 2:
        quality['edge_or_mask'] = len(peaks)
        return [], quality
    rows = measure_widths(trace, peaks, fs, window_ms, neighbors, align_ms=2.0)
    all_peaks = np.unique(np.r_[peaks, neighbors]).astype(int)
    for row in rows:
        candidate, center = row['candidate_index'], row['peak_index']
        start, end = center - half, center + half + 1
        if start < 0 or end > len(trace) or np.any(mask[start:end]):
            quality['edge_or_mask'] += 1
            continue
        others = all_peaks[all_peaks != candidate]
        if np.any((others >= start) & (others < end)):
            quality['overlapping'] += 1
            continue
        if not np.isfinite(row['fwhm_ms']) or (max_width is not None and row['fwhm_ms'] > max_width):
            quality['width_rejected'] += 1
            continue
        wave = np.asarray(trace[start:end], dtype=float).copy()
        edge = max(1, len(wave) // 10)
        wave -= np.median(np.r_[wave[:edge], wave[-edge:]])
        height = wave[half]
        if not np.isfinite(height) or height <= 0:
            quality['invalid_amplitude'] += 1
            continue
        waves.append(wave / height)
        quality['clean_seed_indices'].append(int(candidate))
    quality['clean'] = len(waves)
    if len(waves) < 3:
        return [], quality
    if groups > 1:
        grouped = _build_parallel_template_banks(waves, [fs] * len(waves), fs,
            force_peak_positive=True, max_use_types=min(groups, len(waves)//3), n_components=components)
    else:
        grouped = [(waves, [fs] * len(waves))]
    # Robust equal-amplitude median per shape group; no second-pass events
    # enter this bank. Grouping may resample by a sample, so center-align rows.
    bank = []
    quality['unsupported_groups'] = 0
    for group, _ in grouped:
        if len(group) < 3:
            quality['unsupported_groups'] += 1
            continue
        size = min(len(wave) for wave in group)
        aligned = [wave[(len(wave)-size)//2:(len(wave)-size)//2+size] for wave in group]
        bank.append(np.median(np.stack(aligned), axis=0))
    return bank, quality


def process_cell_combo(raw, fs, *, common, seed_options, template_options,
                       fallback_templates, cs_window_ms=30, ss_window_ms=8,
                       ss_max_width=None, qc_params=None, qc_source=None):
    template_options = dict(template_options)
    if qc_params is None:
        seeds = process_cell_simple(raw, fs, **common, **seed_options)
    else:
        from .qc import run_with_qc
        seeds = run_with_qc(lambda **qc: process_cell_simple(raw, fs, **common, **seed_options, **qc),
                            qc_source, fs, qc_params)
    cs = seeds['cs_trace']
    ss = seeds['ss_trace']
    lowpass = template_options.get('template_ss_lowpass_hz', 0)
    if lowpass > 0:
        ss = apply_filter(ss, fs, high=min(lowpass, .45 * fs), order=common['ss_order'])
    groups = template_options.get('parallel_groups', 3) if template_options.get('parallel_match') else 1
    components = template_options.get('parallel_components', 2)
    learned, metadata = {}, {}
    for kind, trace, window in (('cs', cs, cs_window_ms), ('ss', ss, ss_window_ms)):
        bank, quality = _learn_bank(trace, seeds[f'{kind}_peaks'], seeds[f'{kind}_exclusion_mask'],
            fs, window, max_width=ss_max_width if kind == 'ss' else None,
            neighbors=seeds['cs_peaks'] if kind == 'ss' else (), groups=groups, components=components)
        if bank:
            rates = [float(fs)] * len(bank)
            quality['source'] = 'auto'
            template_options[f'{kind}_selected_groups'] = None
        else:
            bank = list(fallback_templates.get(f'{kind}_templates', []))
            rates = list(fallback_templates.get(f'fs_{kind}', []))
            quality['source'] = 'loaded fallback' if bank else 'unavailable'
        learned[f'{kind}_templates'], learned[f'fs_{kind}'] = bank, rates
        quality['kernels'] = len(bank)
        quality['seed_indices'] = np.asarray(seeds[f'{kind}_peaks'], dtype=int).tolist()
        metadata[kind] = quality
    options = dict(template_options)
    # When no CS template can be learned, confident CS seeds still protect SS
    # from CS contamination. Otherwise SS exclusions follow final detected CS.
    if not learned['cs_templates']:
        options['ss_mask_reference_cs'] = seeds['cs_peaks']
    def match(**qc):
        return process_cell_template_matching(raw, fs, **common, **options, **qc,
        template_cs_bank=learned['cs_templates'], template_cs_fs_bank=learned['fs_cs'],
        template_ss_bank=learned['ss_templates'], template_ss_fs_bank=learned['fs_ss'])
    result = match() if qc_params is None else run_with_qc(match, qc_source, fs, qc_params)
    result['det_method'] += ' + Combo'
    result['combo'] = dict(min_clean_seeds=3, seed_cs_cutoff=seed_options['cs_thresh_sigma'],
        seed_ss_cutoff=seed_options['ss_thresh_sigma'], cs=metadata['cs'], ss=metadata['ss'],
        cs_seed_mask_fallback=not bool(learned['cs_templates']),
        templates={key: [np.asarray(wave).tolist() for wave in values]
                   if key.endswith('_templates') else list(values)
                   for key, values in learned.items()})
    result['_combo_templates'] = learned
    return result
