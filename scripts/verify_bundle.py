"""Check extracted bundle files against the creation-time SHA-256 manifest."""

import hashlib
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def verify_bundle(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / 'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
    version = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    if manifest['app_version'] != version or manifest['python_version'] != '3.11':
        raise ValueError('Bundle version or Python version does not match its manifest.')
    for relative, expected in manifest['sha256'].items():
        path = root / relative
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Bundle file missing or outside directory: {relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Bundle file changed or damaged: {relative}')
    return len(manifest['sha256'])


if __name__ == '__main__':
    try:
        count = verify_bundle()
    except (OSError, ValueError, KeyError) as exc:
        print(f'Bundle verification failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
    print(f'Verified {count} bundle source files.')
