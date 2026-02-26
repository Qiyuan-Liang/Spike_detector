# Spike Detector GUI

Spike detector GUI for detecting CS and SS spiking activity in Purkinje neurons from AOD 2P microscope recordings.

## Repository structure

- `src/spike_detector/gui.py`: GUI application (layout and interactions preserved)
- `src/spike_detector/core/detection.py`: extracted core detection and waveform/stat functions
- `src/spike_detector/io/session.py`: shared session loading and time/fs normalization
- `src/spike_detector/utils/export.py`: shared figure export helper
- `src/spike_detector/utils/stats.py`: shared stats summary helper
- `tests/core/`: core regression/smoke tests
- `packaging/pyinstaller/spike_detector.spec`: PyInstaller build spec

## Install

```bash
pip install -e .
```

## Run

```bash
spike-detector
```

or

```bash
python -m spike_detector
```

## Build `.exe`/app with PyInstaller

```bash
pyinstaller packaging/pyinstaller/spike_detector.spec
```

## Publish + CI build

See [GITHUB_PUBLISH.md](GITHUB_PUBLISH.md) for:
- VS Code GitHub publishing steps
- automated Windows `.exe` build via GitHub Actions
- release/tag workflow
