# Spike Detector GUI

Spike detector GUI for detecting complex spikes (CS) and simple spikes (SS) in Purkinje neurons from AOD 2P microscope recordings of mouse cerebellum.

## Environment dependencies

- Python 3.10+
- PyQt6
- numpy
- scipy
- pandas
- matplotlib
- openpyxl

## Installation

### Option A (recommended): conda environment

```bash
conda create -n spike_detector python=3.11 -y
conda activate spike_detector
pip install -e .
```

### Option B: existing Python environment

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Notes:
- `pip install -e .` installs the project in editable mode, so code changes in `src/` are used immediately without reinstall.
- If dependencies change later, rerun `pip install -e .`.

## How to run

From repository root:

```bash
python -m spike_detector
```

Alternative launcher:

```bash
python gui_preprocess_V3.3.py
```

Do not run package files directly by path (for example `src/spike_detector/gui.py`), because relative imports require package execution.

## Software operation logic

### 1) Load data

1. Click **Select Folder**.
2. Choose the master directory containing sessions.
3. Pick a session from **Session** dropdown.
4. Pick a cell from **Cell** dropdown.

Supported session inputs:
- analyzed `*_analyzed.npz`
- `.xlsx`
- `.csv`
- folder containing `.xlsx` or analyzed `.npz`

### 2) Configure preprocessing

Use baseline controls:
- **Baseline correction method**: Disable / Percentile / Median / Savitzky-Golay
- **Baseline correction window**
- **Percentile** (for percentile baseline)
- **SGolay Polyorder** (for Savitzky-Golay)

Use averaging controls:
- **Frame processing**: Rolling average / Downsampling frames
- **Averaging frames (0 = off)**

### 3) Choose detection method

In the **Spike Detection** tab panel:
- **Threshold** tab (default): set CS/SS filter bands, filter orders, and sigma thresholds.
- **Template Matching** tab: load CS/SS templates and set template sigma thresholds.

Template workflow:
1. Click **Load CS templates** and **Load SS templates**.
2. (Optional) Click **View** to inspect loaded templates.
3. Run detection.

### 4) Run detection

- Click **Spike Detection** to run over all loaded sessions/cells with current settings.
- Click **Two-step Detection** for template matching + threshold verification.

Results are stored per session and summarized in the **Info** panel.

### 5) Inspect results

Main plot controls:
- **Center** slider: move viewing window along time.
- **Window**: adjust visible duration.
- **Y-Range**: set vertical range (`0` = auto).
- Toggle baseline / CS / SS visual overlays.

Viewer tools:
- **Spike Statistics**: waveform, FWHM, SNR summaries.
- **Export as templates** (Detection Viewer): save detected spikes as CS/SS templates.
- **Save Figure**: export current plot.

### 6) Advanced and reset

- **Advanced Settings**: negative-going mode, timing windows, color scheme.
- **Reset App**: clear loaded sessions and return parameters to defaults.

## Minimal troubleshooting

- If GUI does not start, ensure `PyQt6` is installed in the active environment.
- If module import errors occur, run from repository root with `python -m spike_detector`.
- If an Excel session fails, verify the first column is time and remaining columns are cell traces.
