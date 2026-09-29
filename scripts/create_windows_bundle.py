"""Create a small, explicit Windows source bundle for the current GUI.

Run from the repository root with Python 3.11 or later:
    python3 scripts/create_windows_bundle.py
"""

import hashlib
import json
from pathlib import Path
import re
import tomllib
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'spike_detector_windows_build.zip'
REQUIRED = (
    '.python-version',
    'pyproject.toml',
    'requirements.txt',
    'README.md',
    'WINDOWS_BUILD.md',
    'build_windows_uv.bat',
    'spike_detector_launcher.py',
    'packaging/pyinstaller/spike_detector.spec',
    'scripts/create_windows_bundle.py',
    'scripts/verify_bundle.py',
)


def bundle_files():
    files = [ROOT / item for item in REQUIRED]
    files.extend(sorted((ROOT / 'src' / 'spike_detector').rglob('*.py')))
    if not (ROOT / 'src' / 'spike_detector' / 'gui.py') in files:
        raise ValueError('Spike Detector GUI source is missing.')
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'Bundle input missing or outside repository: {path}')
    return sorted(set(files), key=lambda p: p.relative_to(ROOT).as_posix())


def check_version():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    version = project['project']['version']
    gui = (ROOT / 'src' / 'spike_detector' / 'gui.py').read_text(encoding='utf-8')
    match = re.search(r"^APP_VERSION\s*=\s*['\"]([^'\"]+)['\"]", gui, re.MULTILINE)
    if match is None or version != match.group(1):
        raise ValueError('pyproject.toml version does not match GUI APP_VERSION.')
    if (ROOT / '.python-version').read_text(encoding='utf-8').strip() != '3.11':
        raise ValueError('Expected .python-version to request Python 3.11.')
    return version


def add_bytes(archive, name, data):
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=9)


def create_bundle():
    version = check_version()
    files = bundle_files()
    hashes = {}
    payloads = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        content = path.read_bytes()
        hashes[relative] = hashlib.sha256(content).hexdigest()
        payloads.append((relative, content))
    manifest = {'app_version': version, 'python_version': '3.11', 'sha256': hashes}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode('utf-8')
    with ZipFile(OUTPUT, 'w') as archive:
        for name, content in payloads:
            add_bytes(archive, name, content)
        add_bytes(archive, 'BUNDLE_MANIFEST.json', manifest_bytes)
    with ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP integrity check failed.')
        expected = set(hashes) | {'BUNDLE_MANIFEST.json'}
        if set(archive.namelist()) != expected:
            raise RuntimeError('ZIP file list differs from the explicit manifest.')
        for name, digest in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise RuntimeError(f'ZIP hash mismatch: {name}')
    print(f'Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes, {len(expected)} files, GUI {version})')
    return OUTPUT


if __name__ == '__main__':
    create_bundle()
