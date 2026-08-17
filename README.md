# Spike Detector GUI

Spike Detector is a PyQt application for detecting complex spikes (CS) and simple spikes (SS) from voltage-imaging recordings of mouse cerebellar Purkinje neurons, including AOD two-photon random-access imaging data.

## Installation

Requirements:

- Python 3.10+
- PyQt6
- numpy, scipy, pandas, matplotlib
- PyWavelets, scikit-learn
- openpyxl

Recommended editable install:

```bash
conda create -n spike_detector python=3.11 -y
conda activate spike_detector
pip install -e .
```

You can also install into an existing Python environment:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Running The App

From the repository root:

```bash
python -m spike_detector
```

The package entry point is also available after installation:

```bash
spike-detector
```

Do not run `src/spike_detector/gui.py` directly by path; package-relative imports require module execution.

## Data Inputs

Click **Select Folder** and choose the master directory containing recordings. Supported inputs include:

- Raw `.xlsx` or `.csv` trace files, with time in the first column and cells/ROIs in the remaining columns.
- Folders containing `.xlsx` files.
- Existing Spike Detector `*_analyzed.npz` result files.

Detected outputs are saved in a `spike_detection/` subfolder under the selected master folder.

## Default Detection Algorithm

The default spike detection method is **Threshold** detection. This is the recommended starting point for AOD/ASAP Purkinje-cell recordings.

The default workflow is:

1. Apply the selected frame processing and baseline correction to each trace.
2. Build CS and SS detection traces using the configured filter bands. A cutoff value of `0` disables that side of the filter.
3. Estimate noise using a robust MAD-based sigma.
4. Detect CS candidates by threshold crossing on the CS trace, using the CS threshold, minimum distance, and minimum FWHM settings.
5. Blank SS detection around CS events using **SS blank after CS**.
6. Detect SS candidates by threshold crossing on the SS trace, using the SS threshold and SS minimum distance.
7. Save event times, processed traces, per-event SNR, FWHM, `-dF/F (%)`, waveform snippets, and the exact settings snapshot used for that run.

Current default values include:

- Detection method: `Threshold`
- Baseline correction: `Median`, `30 ms`
- CS filter: low cut `0 Hz`, high cut `150 Hz`
- SS filter: low cut `0 Hz`, high cut `0 Hz` (unfiltered)
- CS threshold: `6.0 x MAD`
- SS threshold: `2.5 x MAD`
- CS minimum FWHM: `4 ms`
- SS minimum distance: `4 ms`
- SS blank after CS: `18 ms`
- Negative-going detection: enabled

Template matching and two-step detection are available, but they are optional workflows rather than the default.

## Basic Workflow

1. Click **Select Folder**.
2. Select a session and cell for preview.
3. Adjust preprocessing settings if needed:
   - Baseline correction method, window, and percentile.
   - Frame processing mode and averaging/downsampling frames.
   - Optional wavelet denoising.
4. Use the **Threshold** tab to adjust CS/SS filter bands and sigma thresholds.
5. Use **Advanced Settings** for timing windows, negative-going mode, denoised CS detection, color settings, and scale-bar units.
6. Click **Spike Detection** to process all loaded sessions/cells.
7. Inspect results with **Detection Viewer** and **Spike Statistics**.

## Saved Outputs

Each detection run saves results into `spike_detection/`.

Main result file:

- `SESSION_analyzed.npz`

Automatic settings sidecar:

- `SESSION_analyzed_settings.json`

The sidecar JSON is written every time detection results are saved. It uses the same structure as **Save Settings**, so another run can reload the exact GUI parameters, baseline settings, colors, frame-processing settings, and detection method used for that result.

If **override** is off and a result already exists, Spike Detector preserves the existing file and writes the new result plus matching settings sidecar under:

- `spike_detection/_temporary_detection/`

## Manual Settings

- **Save Settings** writes the current GUI configuration to a JSON file.
- **Load Settings** restores parameters from a saved JSON file.
- Detection result sidecars can also be loaded through **Load Settings** to reproduce a previous run.

## Inspection And Export

Useful viewer tools:

- **Spike Statistics**: SS/CS waveform, SNR, FWHM, and instantaneous-rate summaries.
- **Detection Viewer**: raw traces, detection traces, thresholds, and detected events.
- **Export as templates**: save detected waveforms for template matching.
- **Save Figure**: export publication figures.

## Troubleshooting

- If the GUI does not start, confirm `PyQt6` is installed in the active environment.
- If imports fail, run from the repository root with `python -m spike_detector`.
- If Excel loading fails, check that the first column is time and remaining columns are cell traces.
- If old parameters reappear, check whether an older settings JSON was loaded; saved settings override current defaults.
