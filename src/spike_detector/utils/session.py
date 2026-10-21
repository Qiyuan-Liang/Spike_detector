import os
import glob
import re
import numpy as np
import pandas as pd
from .validation import validate_time, validate_session


def is_excel_lock_file(path):
    """Excel owner files share the workbook extension but contain no trace data."""
    name = os.path.basename(os.fspath(path))
    return name.startswith('~$') and os.path.splitext(name)[1].lower() == '.xlsx'


def _list_table_files_in_dir(folder_path):
    files = []
    try:
        for name in os.listdir(folder_path):
            fp = os.path.join(folder_path, name)
            if not os.path.isfile(fp) or is_excel_lock_file(fp):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in {'.xlsx', '.csv'}:
                files.append(fp)
    except Exception:
        return []
    return sorted(files)


def infer_time_unit(time_vec_raw, column_name=None, default_fs=1000.0):
    """Infer table units from a labelled header, then plausible sampling rates."""
    label = str(column_name or '').strip().lower()
    if re.search(r'(?<![a-z])(ms|msec|millisecond|milliseconds)(?![a-z])', label):
        return 'ms'
    if re.search(r'(?<![a-z])(s|sec|second|seconds)(?![a-z])', label):
        return 's'
    t = np.asarray(time_vec_raw, dtype=float)
    if t.ndim != 1 or t.size < 2 or not np.all(np.isfinite(t)):
        raise ValueError('Time must contain at least two finite samples.')
    step = float(np.median(np.diff(t)))
    if step <= 0:
        raise ValueError('Time must be strictly increasing.')
    plausible = [u for u, rate in (('ms', 1000.0 / step), ('s', 1.0 / step))
                 if 50.0 <= rate <= 30000.0]
    if len(plausible) == 1:
        return plausible[0]
    raise ValueError('Ambiguous table time units: label the first column time_ms or time_s.')


def normalize_time_and_fs(time_vec_raw, default_fs=1000.0, time_unit='auto', column_name=None):
    """Convert inferred or explicit units, retaining origin and every sample."""
    if time_unit == 'auto':
        time_unit = infer_time_unit(time_vec_raw, column_name, default_fs)
    if time_unit not in ('ms', 's'):
        raise ValueError('Table time units must be auto, ms or s.')
    t = np.asarray(time_vec_raw, dtype=float)
    time_ms = t * (1000.0 if time_unit == 's' else 1.0)
    fs = validate_time(time_ms)
    return time_ms, time_ms, fs, np.ones(t.size, dtype=bool)


def load_table_session_file(file_path, default_fs=1000.0, time_unit='auto'):
    if is_excel_lock_file(file_path):
        raise ValueError('Temporary Excel lock file is not a recording; select the workbook without the ~$ prefix.')
    stem = os.path.splitext(os.path.basename(file_path))[0].lower()
    if stem.endswith(('_time_offsets', '_coordinates')):
        raise ValueError('Auxiliary time-offset/coordinate table is not a spike trace recording.')
    if file_path.lower().endswith('.xlsx'):
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1', engine='openpyxl')
        except Exception:
            # Fallback to first sheet when Sheet1 is absent or malformed.
            df = pd.read_excel(file_path, engine='openpyxl')
    elif file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f'Unsupported file type: {file_path}')

    if df is None or int(df.shape[1]) < 2:
        raise ValueError(
            f'Invalid table format in {os.path.basename(file_path)}: '
            'expected at least 2 columns (time + >=1 cell trace).'
        )

    # A table can contain trace values below the last timestamp (for example,
    # one ROI column pasted longer than the shared time axis). The rectangular
    # detector can only analyze rows with timestamps. Keep the valid timed
    # prefix and report any excluded untimed values; missing rows *inside* the
    # timed region still fail the strict checks below.
    numeric_time = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy(dtype=float)
    timed_rows = np.flatnonzero(np.isfinite(numeric_time))
    if timed_rows.size == 0:
        raise ValueError(f'Invalid time column in {os.path.basename(file_path)}: no numeric timestamps found.')
    last_timed_row = int(timed_rows[-1])
    tail = df.iloc[last_timed_row + 1:]
    if not tail.empty and tail.iloc[:, 0].notna().any():
        raise ValueError('Non-numeric time labels occur after the last timestamp; inspect the table before detection.')
    untimed_counts = tail.iloc[:, 1:].notna().sum() if not tail.empty else pd.Series(dtype=int)
    ignored_untimed_tail_rows = int(len(tail))
    load_warnings = []
    if ignored_untimed_tail_rows:
        affected = [(str(name), int(count)) for name, count in untimed_counts.items() if count]
        if affected:
            detail = ', '.join(f'{name}: {count:,}' for name, count in affected[:3])
            if len(affected) > 3:
                detail += f', and {len(affected) - 3} more cell(s)'
            load_warnings.append(
                f'Excluded {ignored_untimed_tail_rows:,} rows after the last timestamp; '
                f'untimed trace values were present ({detail}). Add timestamps to analyze that tail.'
            )
        else:
            load_warnings.append(f'Ignored {ignored_untimed_tail_rows:,} completely blank trailing rows.')
        df = df.iloc[:last_timed_row + 1]

    time_vec_raw = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy(dtype=float)
    raw_matrix_full = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    resolved_unit = infer_time_unit(time_vec_raw, df.columns[0], default_fs) if time_unit == 'auto' else time_unit
    _, time_vec_ms, fs, mask_valid = normalize_time_and_fs(time_vec_raw, default_fs=default_fs, time_unit=resolved_unit)

    if raw_matrix_full.ndim != 2 or raw_matrix_full.shape[1] < 1:
        raise ValueError(
            f'Invalid table format in {os.path.basename(file_path)}: '
            'no valid cell trace columns were found.'
        )

    if np.sum(mask_valid) < 2:
        raise ValueError(
            f'Invalid time column in {os.path.basename(file_path)}: '
            'need at least 2 valid numeric time samples.'
        )

    raw_matrix_full = raw_matrix_full[mask_valid, :]

    if not np.all(np.isfinite(raw_matrix_full)):
        raise ValueError('Trace contains missing/nonfinite samples; repair or exclude them explicitly before detection.')

    data = {
        'input_time_unit': resolved_unit,
        'time_ms': time_vec_ms,
        'raw_data': raw_matrix_full,
        'cell_names': df.columns[1:].tolist(),
        'fs': fs,
        'session_path': file_path,
        'ignored_untimed_tail_rows': ignored_untimed_tail_rows,
        'load_warnings': load_warnings,
    }
    validate_session(data)
    return data


def load_session_path(session_path, default_fs=1000.0, time_unit='auto'):
    if os.path.isfile(session_path):
        if session_path.lower().endswith('.npz'):
            with np.load(session_path, allow_pickle=True) as npz:
                data = {'time_ms': npz['time_ms'], 'raw_data': npz['raw_data'],
                        'cell_names': list(npz['cell_names']), 'fs': float(npz['fs']),
                        'session_path': session_path, 'input_time_unit': 'ms'}
            validate_session(data)
            return data
        if session_path.lower().endswith(('.xlsx', '.csv')):
            return load_table_session_file(session_path, default_fs, time_unit)
        raise ValueError(f'Unsupported file: {session_path}')
    npz_files = sorted(glob.glob(os.path.join(session_path, '*_analyzed.npz')))
    if not npz_files:
        sd_dir = os.path.join(os.path.dirname(session_path), 'spike_detection')
        npz_files = sorted(glob.glob(os.path.join(sd_dir, os.path.basename(session_path) + '*_analyzed.npz')))
    files = npz_files or _list_table_files_in_dir(session_path)
    if not files:
        raise FileNotFoundError(f'No .npz, .xlsx, or .csv found in {session_path}')
    return load_session_path(files[0], default_fs, time_unit)
