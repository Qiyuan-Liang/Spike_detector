"""Managed viewing exports, narrowly scoped cleanup, and staged result replacement."""
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid
from PyQt6.QtCore import QLockFile

_MARKER = '.spike_detector_viewing.json'
_LEGACY = re.compile(r'.+_analyzed_temp_\d{8}_\d{6}_\d{6}(?:\.np[zy]|_settings\.json)$')


def cleanup_legacy_temporaries(folder):
    """Delete only known old temporary exports, without following symlinks."""
    directory = Path(folder) / 'spike_detection' / '_temporary_detection'
    errors = []
    if directory.parent.is_symlink() or directory.is_symlink() or not directory.is_dir():
        return errors
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        return [f'Temporary cleanup failed: {directory}: {exc}']
    for path in entries:
        if _LEGACY.fullmatch(path.name) and path.is_file() and not path.is_symlink():
            try:
                path.unlink()
            except OSError as exc:
                errors.append(f'Temporary cleanup failed: {path}: {exc}')
    try:
        if not any(directory.iterdir()):
            directory.rmdir()
    except OSError as exc:
        errors.append(f'Temporary cleanup failed: {directory}: {exc}')
    return errors


class TemporaryDetectionStore:
    """A lock protects exports belonging to another running GUI instance."""
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else Path(tempfile.gettempdir()) / 'spike_detector_viewing'
        if self.root.is_symlink():
            raise ValueError('Temporary export root must not be a symbolic link.')
        self.root.mkdir(parents=True, exist_ok=True)
        self.cleanup_errors = self.cleanup_stale()
        self.directory = self.root / ('run-' + uuid.uuid4().hex)
        self.directory.mkdir()
        self.lock = QLockFile(str(self.directory / 'owner.lock'))
        self.lock.setStaleLockTime(0)
        if not self.lock.tryLock(0):
            raise RuntimeError('Cannot lock temporary viewing exports.')
        # Publish the ownership marker only after acquiring the lock, so another
        # launching instance cannot mistake initialization for an abandoned run.
        (self.directory / _MARKER).write_text(json.dumps({'kind': 'SpikeDetectorViewing', 'version': 1}))

    def cleanup_stale(self):
        errors = []
        for directory in self.root.iterdir():
            if directory.is_symlink() or not re.fullmatch(r'run-[0-9a-f]{32}', directory.name) or not directory.is_dir():
                continue
            try:
                marker = directory / _MARKER
                if marker.is_symlink():
                    continue
                metadata = json.loads(marker.read_text())
                if (not isinstance(metadata, dict) or
                        metadata.get('kind') != 'SpikeDetectorViewing' or metadata.get('version') != 1):
                    continue
                if (directory / 'owner.lock').is_symlink():
                    continue
                lock = QLockFile(str(directory / 'owner.lock'))
                lock.setStaleLockTime(0)
                if not lock.tryLock(0):
                    continue
                lock.unlock()
                shutil.rmtree(directory)
            except (OSError, ValueError) as exc:
                errors.append(f'Temporary cleanup failed: {directory}: {exc}')
        return errors

    def allocate(self, basename):
        if not self.directory.exists():
            raise RuntimeError('Temporary export store is closed.')
        return self.directory / f'{Path(basename).name}_analyzed_temp_{uuid.uuid4().hex}.npz'

    def close(self):
        self.lock.unlock()
        if self.directory.exists():
            shutil.rmtree(self.directory)


def output_paths(data, fallback_folder):
    source = Path(data['session_path'])
    folder = Path(data.get('source_folder') or fallback_folder or source.parent)
    base = data.get('output_basename') or source.stem
    if data.get('loaded_detection_result') and source.suffix.lower() == '.npz' and not data.get('output_basename'):
        base = base.removesuffix('_analyzed').removesuffix('_labels')
    destination = folder / 'spike_detection' / (base + '_analyzed.npz')
    return destination, [destination, destination.with_suffix('.npy'),
        destination.with_name(destination.stem + '_settings.json')]


def publish_detection(staged_result, staged_settings, destination, approved_existing):
    """Stage on the destination filesystem before replacing a saved result pair.

    Existing outputs are backed up during replacement and restored on failure.
    Only the exact session result/sidecar names are touched, never input traces.
    """
    destination = Path(destination)
    sidecar = destination.with_name(destination.stem + '_settings.json')
    legacy = destination.with_suffix('.npy')
    candidates = [destination, legacy, sidecar]
    existing = [p for p in candidates if p.exists()]
    if any(str(p.resolve()) not in approved_existing for p in existing):
        raise FileExistsError('A saved result appeared after confirmation; kept the new result temporary. Run again to approve replacement.')
    if any(p.is_symlink() for p in candidates):
        raise ValueError('Refusing to replace a detection output symbolic link.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.spike-detector-publish-', dir=destination.parent))
    backups, installed = {}, []
    try:
        result_copy, settings_copy = staging/'new.npz', staging/'new.json'
        shutil.copyfile(staged_result, result_copy)
        settings = json.loads(Path(staged_settings).read_text())
        settings['results_file'] = destination.name
        settings_copy.write_text(json.dumps(settings, indent=2))
        for i, path in enumerate(existing):
            backup = staging / f'old-{i}'
            shutil.copyfile(path, backup)
            backups[path] = backup
        os.replace(result_copy, destination)
        installed.append(destination)
        os.replace(settings_copy, sidecar)
        installed.append(sidecar)
        if legacy.exists():
            legacy.unlink()
    except Exception:
        # Restore by rename on the same filesystem. Retain backups if recovery
        # itself fails, so a filesystem error cannot silently discard them.
        try:
            for path in installed:
                if path not in backups:
                    path.unlink(missing_ok=True)
            for path, backup in backups.items():
                os.replace(backup, path)
        except OSError as recovery:
            raise RuntimeError(f'Replacement recovery failed; original backups retained in {staging}: {recovery}')
        shutil.rmtree(staging)
        raise
    else:
        shutil.rmtree(staging)
