"""Folder discovery for batch detection; no detection or output writes here."""
import os
from pathlib import Path

import numpy as np

from .session import load_session_path


def normalize_folder_paths(paths, base_dir=None):
    """Expand folder paths, including pasted Windows separators on macOS."""
    folders = []
    for value in paths:
        value = str(value).strip().strip('\"').strip("'")
        if not value:
            continue
        if os.name != 'nt':
            value = value.replace('\\', '/')
            # Finder's /Volumes paths are often copied without the first slash.
            if value.startswith('Volumes/'):
                value = '/' + value
        value = os.path.expanduser(value)
        if base_dir is not None and not os.path.isabs(value):
            value = os.path.join(base_dir, value)
        folder = os.path.realpath(value)
        if folder not in folders:
            folders.append(folder)
    return folders


def discover_batch_sessions(paths, default_fs=1000.0, time_unit='auto'):
    folders = normalize_folder_paths(paths)
    sessions, names, loaded, issues = [], [], {}, []
    for folder in folders:
        try:
            files = sorted(Path(folder).iterdir())
        except OSError as exc:
            issues.append(f'Path unavailable: {folder}: {exc}')
            continue
        count = 0
        for source in files:
            if not source.is_file() or source.suffix.lower() not in {'.csv', '.xlsx', '.npz'}:
                continue
            try:
                data = load_session_path(str(source), default_fs=default_fs, time_unit=time_unit)
                raw = np.asarray(data.get('raw_data', []), dtype=float)
                time = np.asarray(data.get('time_ms', []), dtype=float)
                fs = float(data.get('fs', 0))
                if (raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 1 or
                        time.ndim != 1 or len(time) != raw.shape[0] or
                        len(data.get('cell_names', [])) != raw.shape[1] or
                        not np.isfinite(fs) or fs <= 0):
                    raise ValueError('Invalid trace, time, cell-name dimensions, or sampling rate')
                data['source_folder'] = folder
                # The full source path also disambiguates repeated folder basenames.
                name = str(source)
                sessions.append(str(source))
                names.append(name)
                loaded[name] = data
                count += 1
            except Exception as exc:
                issues.append(f'Skipped: {source}: {exc}')
        if count == 0:
            issues.append(f'No valid sessions: {folder}')
    for folder in folders:
        group = [data for data in loaded.values() if data['source_folder'] == folder]
        stems = [Path(data['session_path']).stem for data in group]
        if len(set(stems)) < len(stems):
            # If extensions share a stem, retain extensions for this entire folder.
            for data in group:
                data['output_basename'] = Path(data['session_path']).name
    return sessions, names, loaded, issues, folders
