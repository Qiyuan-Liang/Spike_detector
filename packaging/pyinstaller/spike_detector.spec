# -*- mode: python ; coding: utf-8 -*-
"""Windows one-folder / macOS application-bundle GUI build. Invoke from the extracted bundle root."""

from pathlib import Path
import sys
import tomllib
from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPEC).resolve().parents[2]
ASSETS = ROOT / 'src' / 'spike_detector' / 'assets'
VERSION = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']

analysis = Analysis(
    [str(ROOT / 'spike_detector_launcher.py')],
    pathex=[str(ROOT / 'src')],
    binaries=[],
    datas=collect_data_files('matplotlib') + [(str(ASSETS), 'spike_detector/assets')],
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'matplotlib.backends.backend_qtagg',
        'openpyxl',
        'pywt',
        'sklearn.decomposition',
        'sklearn.cluster',
        'sklearn.metrics',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6', 'PyQt5'],
    noarchive=False,
)

pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name='spike_detector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ASSETS / 'spike_detector.ico') if sys.platform == 'win32' else None,
)
collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name='spike_detector',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        collection,
        name='Spike Detector.app',
        icon=str(ASSETS / 'spike_detector.icns'),
        bundle_identifier='org.spikedetector.gui',
        version=VERSION,
        info_plist={
            'CFBundleDisplayName': 'Spike Detector',
            'CFBundleVersion': VERSION,
            'NSHighResolutionCapable': True,
        },
    )
