import os
import glob
import numpy as np
import pandas as pd


def _list_table_files_in_dir(folder_path):
    files = []
    try:
        for name in os.listdir(folder_path):
            fp = os.path.join(folder_path, name)
            if not os.path.isfile(fp):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in {'.xlsx', '.csv'}:
                files.append(fp)
    except Exception:
        return []
    return sorted(files)


def normalize_time_and_fs(time_vec_raw, default_fs=1000.0):
    time_vec_raw = np.asarray(time_vec_raw, dtype=float)
    mask_valid = ~np.isnan(time_vec_raw)
    time_vec_raw = time_vec_raw[mask_valid]

    frac_secs = np.mean(time_vec_raw < 1.0) if len(time_vec_raw) > 0 else 0.0
    frac_ms = np.mean(time_vec_raw > 10.0) if len(time_vec_raw) > 0 else 0.0
    if frac_secs > 0.5 and frac_ms < 0.5:
        time_vec = time_vec_raw * 1000.0
    elif frac_ms > 0.5:
        time_vec = time_vec_raw
    else:
        med = float(np.median(np.abs(np.diff(time_vec_raw)))) if len(time_vec_raw) > 1 else 0.0
        time_vec = time_vec_raw * 1000.0 if med < 0.01 else time_vec_raw

    dt = float(np.median(np.diff(time_vec_raw))) if len(time_vec_raw) > 1 else 0.0
    if dt > 0 and dt < 0.01:
        time_vec_ms = time_vec_raw * 1000.0
    else:
        time_vec_ms = time_vec_raw

    dt_ms = float(np.median(np.diff(time_vec_ms))) if len(time_vec_ms) > 1 else 1.0
    fs = 1000.0 / dt_ms if dt_ms > 0 else float(default_fs)
    return time_vec, time_vec_ms, fs, mask_valid


def load_table_session_file(file_path, default_fs=1000.0):
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

    time_vec_raw = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy(dtype=float)
    raw_matrix_full = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    _, time_vec_ms, fs, mask_valid = normalize_time_and_fs(time_vec_raw, default_fs=default_fs)

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

    # Replace NaN/Inf in traces to keep downstream filtering stable.
    if not np.all(np.isfinite(raw_matrix_full)):
        col_med = np.nanmedian(raw_matrix_full, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        bad = ~np.isfinite(raw_matrix_full)
        if np.any(bad):
            raw_matrix_full[bad] = col_med[np.where(bad)[1]]

    return {
        'time_ms': time_vec_ms,
        'raw_data': raw_matrix_full,
        'cell_names': df.columns[1:].tolist(),
        'fs': fs,
        'session_path': file_path,
    }


def load_session_path(session_path, default_fs=1000.0):
    if os.path.isfile(session_path):
        if session_path.lower().endswith('.npz'):
            npz = np.load(session_path, allow_pickle=True)
            return {
                'time_ms': npz['time_ms'],
                'raw_data': npz['raw_data'],
                'cell_names': list(npz['cell_names']),
                'fs': float(npz['fs']),
                'session_path': session_path,
            }
        if session_path.lower().endswith('.xlsx') or session_path.lower().endswith('.csv'):
            return load_table_session_file(session_path, default_fs=default_fs)
        raise ValueError(f'Unsupported file: {session_path}')

    npz_files = glob.glob(os.path.join(session_path, '*_analyzed.npz'))
    # Also check the centralised spike_detection/ folder in the parent directory
    if not npz_files:
        parent = os.path.dirname(session_path)
        sd_dir = os.path.join(parent, 'spike_detection')
        folder_name = os.path.basename(session_path)
        candidate = os.path.join(sd_dir, folder_name + '_analyzed.npz')
        if os.path.isfile(candidate):
            npz_files = [candidate]
        else:
            # try any matching prefix in spike_detection/
            npz_files = sorted(glob.glob(os.path.join(sd_dir, folder_name + '*_analyzed.npz')))
    if len(npz_files) > 0:
        npz = np.load(npz_files[0], allow_pickle=True)
        return {
            'time_ms': npz['time_ms'],
            'raw_data': npz['raw_data'],
            'cell_names': list(npz['cell_names']),
            'fs': float(npz['fs']),
            'session_path': npz_files[0],
        }

    table_files = _list_table_files_in_dir(session_path)
    if not table_files:
        raise FileNotFoundError(f'No .npz, .xlsx, or .csv found in {session_path}')

    data = load_table_session_file(table_files[0], default_fs=default_fs)
    time_vec = np.asarray(data['time_ms'], dtype=float)
    raw_data = np.asarray(data['raw_data'], dtype=float)
    if not np.all(np.diff(time_vec) >= 0):
        order = np.argsort(time_vec)
        data['time_ms'] = time_vec[order]
        data['raw_data'] = raw_data[order, :]
    data['session_path'] = session_path
    return data
