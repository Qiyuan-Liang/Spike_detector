# Spike Detector GUI

Spike Detector is a PyQt application for detecting complex spikes (CS) and simple spikes (SS) from voltage-imaging recordings of mouse cerebellar Purkinje neurons, including AOD two-photon random-access imaging data.

## Installation

Requirements:

- Python 3.10+
- PyQt6
- numpy, scipy, pandas, matplotlib
- PyWavelets, scikit-learn
- openpyxl
- Jupyter/IPython packages for the analysis notebooks

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

### UV Environment

For a UV-managed environment on another machine:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

`requirements.txt` installs the project in editable mode with the full `all` extra, including GUI, notebook, and packaging dependencies. See `UV.md` for exact-environment replication from the ASAP7 conda environment and optional `uv.lock` creation.

## Running The App

From the repository root:

```bash
python -m spike_detector
```

The package entry point is also available after installation:

```bash
spike-detector
```

With UV:

```bash
uv run spike-detector
```

Do not run `src/spike_detector/gui.py` directly by path; package-relative imports require module execution.

## Data Inputs

Click **Select Folder** and choose the master directory containing recordings. Supported inputs include:

- Raw `.xlsx` or `.csv` trace files, with time in the first column and cells/ROIs in the remaining columns.
- Folders containing `.xlsx` files.
- Existing Spike Detector `*_analyzed.npz` result files.

The loader skips auxiliary `*_time_offsets.csv`/`.xlsx` and `*_coordinates.csv`/`.xlsx` files, plus recordings with fewer than 128 time samples. It reports skipped inputs in the GUI; an otherwise valid trace with no detected spikes can still be processed.

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
7. Optionally discard broad SS events whose final measured FWHM exceeds the SS max-FWHM threshold.
8. Save event times, processed traces, per-event SNR, FWHM, `-dF/F (%)`, waveform snippets, and the exact settings snapshot used for that run.

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
- SS max FWHM filter: enabled, discard finite SS FWHM values `> 4.5 ms`
- Negative-going detection: enabled

Template matching is available as an alternative to threshold detection.

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

- Hover over a setting input to see a short description of what it controls.
- **Spike Statistics**: SS/CS waveform, autocorrelogram (ACG), FWHM, SNR, and instantaneous-rate summaries. The upper controls set and remember separate ACG half-windows and bin widths (CS ±1000 ms in 10 ms bins; SS ±100 ms in 1 ms bins by default). ACGs show symmetric raw event-pair counts within each cell and session, excluding self and zero-lag pairs; selecting **All** pools those counts without pairing events across cells or recordings.
- **Detection Viewer**: raw traces, detection traces, thresholds, and detected events.
- **Export as templates**: save detected waveforms for template matching.
- **Save Figure**: export publication figures. SVG exports preserve text as editable text when opened in tools such as Adobe Illustrator.


## Troubleshooting

- If the GUI does not start, confirm `PyQt6` is installed in the active environment.
- If imports fail, run from the repository root with `python -m spike_detector`.
- If Excel loading fails, check that the first column is time and remaining columns are cell traces.
- If old parameters reappear, check whether an older settings JSON was loaded; saved settings override current defaults.


### Batch detection

Set the detection method and all processing controls in the main GUI, then click **Batch detection**. Paste one folder path per line, use **Add folder…**, or **Load path list…** to import a UTF-8 text file. Relative paths in imported lists are resolved from the list file’s directory. Click **Run batch detection** to process all top-level CSV, XLSX and NPZ sessions with the same settings and templates.

On macOS, pasted backslash-separated paths such as `Volumes\T7 Shield\Organized\...` are resolved as `/Volumes/T7 Shield/Organized/...`. The batch dialog shows skipped or unavailable paths in a scrollable details box. **Close** dismisses the dialog even while detection continues in the main window.

Results go into each input folder’s `spike_detection/` directory. The existing override setting applies; with override off, repeat outputs go into `_temporary_detection/`. Duplicate paths are collapsed and sessions with identical names in different folders remain separate. If different input extensions share a filename stem in a batch folder, output names retain the input extension to avoid collisions.

The Info panel reports every path, completed sessions/cells, pooled statistics and input/detection/save errors. Close the batch dialog and open **Spike Statistics** to filter by **Path → Session → Cell**, including **All**. Selecting a new master folder returns to ordinary single-folder mode. Batch processing loads all sessions into memory, then detects each cell and compresses the result file on worker threads. The bottom status bar shows completed cells and the saving stage; settings are locked during detection. The same template method, bank, thresholds and masks apply to each batch session.

See [the workflow review and improvement priorities](docs/SPIKE_DETECTION_REVIEW.md) for the detection-method assessment and comparison with voltage-imaging and electrophysiology approaches.


### Template matching (3.6.25)

Load threshold-derived SS/CS templates, then choose **Amplitude-fit LLR (Gaussian)**, **Positive-core LLR (experimental)**, or **Normalized Similarity** in the Template Matching tab. **Clear** empties both template banks. The old `LLR Probability Vector` name remains the stored settings key for compatibility, but the score is a positive-amplitude-fitted Gaussian likelihood gain, not a probability. Its visible cutoff is `positive template response / global, per-sample noise MAD`; it is not a calibrated false-positive z-score. The plotted gain cutoff is `cutoff² / 2`. Both cutoff boxes allow values through 100. **Min CS filtered-trace peak** requires the polarity-corrected, filtered CS trace at the score candidate to reach the specified multiple of its unmasked per-sample MAD; its default is `3.0`, and `0` disables this additional gate. It applies to CS under every template method and is distinct from the matched-response cutoff. The Detection Viewer and Info now count candidates rejected by these two checks separately. SS matches use a short 3 ms core to tolerate overlapping or truncated tails. The former extra full-waveform correlation rejection for CS and isolated SS has been removed. **Normalized Similarity** uses Pearson correlation with defaults CS `r ≥ 0.90`, SS `r ≥ 0.80`. All three methods also require a minimum matched response (`2.2 × MAD` for SS; at least `3 × MAD` for CS). Old saved LLR cutoff values use a different score scale and should be reviewed before reuse.

**Positive-core LLR (experimental)** keeps the existing CS detector and changes only SS scoring. It fits a positive-only 3 ms core of the pooled SS template to the baseline-corrected, optionally low-pass-filtered trace. This avoids a neighboring negative trough adding evidence for a positive SS through the negative weights of a mean-centered template. The visible SS response/MAD cutoff, minimum response, Advanced Settings SS spacing, CS masks, optional maximum SS width, parallel groups, and batch detection still apply. There is no full-waveform correlation, strong-anchor, or 25 ms neighborhood gate. The former **Burst-aware LLR** settings key is retained so saved settings select this method, but the behavior changed in version 3.6.25; rerun detection to compare results. The cutoff is not a calibrated false-positive probability. Assess additions with held-out manual annotations across cells, bursts and quiet periods, and run the polarity-reversed control. That control is diagnostic rather than a calibrated specificity estimate.

The Detection Viewer shades excluded score regions and lists how many score peaks were removed by masks, response, spacing and width decisions. A score peak above the line is therefore a candidate, not necessarily a detected event. **Run polarity-reversed control** reruns the selected cell in a worker thread with its saved detection settings and original CS/SS masks, then shows reversed versus original kept counts beside the candidate diagnostics. Reversed detections are a control, not an estimate of the false-positive count. Parallel matching groups amplitude-normalized template shapes using PCA and k-means. Set the number of groups on the second Template Matching row; View shows each group mean, template count, and checkboxes for enabling it. PCA-component count is in Advanced Settings. All requested groups are scored unless deselected, and the enabled groups are saved and reused in batch detection. With multiple groups, the displayed score is excess above each group's own cutoff, with zero as the line. More active groups can recover minority shapes but also add opportunities for false positives. Template detection retains the adjustable 700 Hz SS low-pass used to suppress the 1,112 Hz interference found in the supplied recording; set it to 0 to disable. Template SS widths use this filtered trace. The template-only minimum-width rejection has been removed; the optional maximum SS width in Advanced Settings remains. The original baseline-corrected trace remains the waveform/SNR review source. **Advanced Settings** is the only SS spacing control; its interval is applied once across all template groups. CS-exclusion settings also remain in effect. Inspect low-amplitude additions against manual marks; a higher count alone does not establish better accuracy.

### Time, widths and exclusions (3.6.17)

CSV/Excel time units are inferred automatically from first-column labels such as `t(ms)` or `time_s`, then from plausible sampling rates when the header has no units. Ambiguous unlabelled columns are rejected with a request to label the column. The inferred unit appears in the Info panel and exported settings. NPZ files always use `time_ms` and must have a matching `fs`. The loader rejects nonfinite, duplicate, decreasing or irregular timestamps (0.1% spacing tolerance), and nonfinite trace samples, rather than dropping rows or filling missing data. Exported spike times preserve the input recording’s time origin.

In **Advanced Settings**, set **SS exclusion before CS** and **SS exclusion after CS** independently. Legacy `SS_BLANK_MS=18` maps to 9 ms before and 9 ms after CS. Endpoints are included and durations round upward to whole samples, so masks can differ by a sample from older results. Excluded intervals are never replaced with zeros. Template scoring also excludes windows that touch masked samples or recording boundaries; this can enlarge the effective exclusion relative to the requested pre/post values. NPZ exports include the actual `cs_exclusion_mask`, `ss_exclusion_mask`, full recording duration and valid duration per cell/type. Displayed rates use the full duration (`samples/fs`), including excluded time.

Widths are measured at the detected event, with search bounds between neighboring candidates. Template candidates may align to the nearest waveform peak within 2 ms, rounded upward to whole native samples so low-rate recordings can inspect the nearest frame. CS width uses the filtered CS waveform in both methods. An unmeasurable width is retained as **uncertain**, with a reason, instead of silently passing the width filter. The Detection Viewer draws a measured CS FWHM as a green horizontal line and a retained CS with uncertain width as a short green dash at the event time; the dash length does not represent a measured duration. Info and Spike Statistics report uncertainty counts; FWHM summaries exclude unknown widths. `event_fwhm_cs/ss` retain one slot per final event, including NaN. `width_quality_json` records candidate indices, aligned peaks, half-height crossings, widths, uncertainty reasons and width decisions, including rejected candidates. Statistical waveform sources are frozen at detection time.

Filters must use valid bands strictly below Nyquist; zero disables a cutoff. Invalid filters or requested preprocessing failures are reported per session instead of silently falling back. The effective baseline window, mask sample counts, local-noise windows and filter settings are saved in `analysis_settings_json.effective_settings`. With denoising enabled, its upper frequency must not exceed 95% of Nyquist; adjust the denoise settings for low-sampling-rate recordings. These corrections can change detections compared with older versions.


## Building a Windows EXE

Run `python3 scripts/create_windows_bundle.py` in the repository to create `spike_detector_windows_build.zip`. Extract that ZIP on Windows, then double-click `build_windows_uv.bat`. The batch file uses uv to install Python 3.11, the GUI dependencies, and PyInstaller, checks the extracted source, and builds `dist\spike_detector\spike_detector.exe` in one-folder mode. The complete `dist\spike_detector` folder is needed to run the app. See `WINDOWS_BUILD.md` inside the ZIP for the exact steps. The source ZIP does not contain an EXE or downloaded dependencies; the first Windows build requires internet access.
