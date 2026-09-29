# -*- mode: python ; coding: utf-8 -*-
"""Windows one-folder GUI build. Invoke from the extracted bundle root."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPEC).resolve().parents[2]

analysis = Analysis(
    [str(ROOT / 'spike_detector_launcher.py')],
    pathex=[str(ROOT / 'src')],
    binaries=[],
    datas=collect_data_files('matplotlib'),
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
)
collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name='spike_detector',
)
