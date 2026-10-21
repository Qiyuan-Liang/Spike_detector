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

With **Save** checked, outputs are written to each input folder’s `spike_detection/` directory. Otherwise viewing results are temporary and cleaned automatically.

## Default Detection Algorithm

The default spike detection method is **Threshold** detection. This is the recommended starting point for AOD/ASAP Purkinje-cell recordings.

The default workflow is:

1. Apply the selected frame processing and baseline correction to each trace.
2. Build CS and SS detection traces using the configured filter bands. A cutoff value of `0` disables that side of the filter.
3. Estimate noise using a robust MAD-based sigma.
4. Detect CS local maxima above the CS threshold, apply minimum spacing, then the shared amplitude-SNR and width QC limits.
5. Exclude SS around accepted CS using the separate before/after windows in **Advanced Settings** (default 9 ms per side).
6. Detect SS local maxima above the SS threshold, using SS minimum spacing. CS QC rejections rebuild SS exclusions before this final SS pass.
7. Apply the same **Spike QC** amplitude-SNR and minimum/maximum FWHM limits to SS.
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
- SS exclusion around CS: `9 ms` before and `9 ms` after (`18 ms` total)
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
5. Set **Spike polarity** below Cell, and **Spike QC** beneath the detection tabs. Use **Advanced Settings** for event spacing, exclusion/template windows, denoised CS detection, colors and scale-bar units.
6. Click **Spike Detection** to process all loaded sessions/cells.
7. Inspect results with **Detection Viewer** and **Spike Statistics**.

## Saved Outputs

The **Save** checkbox controls both ordinary and batch detection; it is off by default.

- **Save off:** every run creates temporary NPZ results and matching settings sidecars for viewing, outside the recording folders. Existing saved results are preserved. Temporary viewing exports are deleted when the main GUI closes; abandoned exports are cleaned on the next launch. Another running GUI's exports are protected by an ownership lock.
- **Save on:** results are saved in each input folder's `spike_detection/` directory as `SESSION_analyzed.npz` and `SESSION_analyzed_settings.json`. If those outputs or the legacy `SESSION_analyzed.npy` exist, a warning lists the affected files before detection. Cancel preserves them; proceed replaces the result pair and removes the superseded legacy NPY. Replacement uses staging and rollback on write failure. Input recordings are never overwritten.

Recognized older viewing exports in `spike_detection/_temporary_detection/` are cleaned when a data folder is loaded. Unrecognized files and symbolic links are preserved. Settings files from earlier versions translate **override** to **Save**; review this checkbox after loading an old configuration.

Each settings sidecar has the same structure as **Save Settings**, so it can reload the run's GUI configuration. Temporary files are intended for inspection during the current GUI session; enable **Save** for results needed after closing the app.

In **Threshold** mode, set **CS threshold** or **SS threshold** to **0** to disable that spike type. The export retains both CS and SS fields, with empty event arrays for the disabled type. Both traces remain available for inspection. In **Combo**, zero disables seed collection for that type; a loaded template bank can still supply the final matching pass.

In Detection Viewer, SS ticks and CS duration bars sit above the raw trace for positive-going indicators and below for negative-going indicators. A retained CS with uncertain FWHM still has a short green dash. The main raw-trace legend is placed above the plotting area.

## Manual Settings

- **Save Settings** writes the current GUI configuration to a JSON file.
- **Load Settings** restores parameters from a saved JSON file.
- Detection result sidecars can also be loaded through **Load Settings** to reproduce a previous run.

## Inspection And Export

The main window has native segmented **Positive-going** / **Negative-going** tabs below Cell. The tabs set the existing polarity parameter used by settings files, ordinary detection, Combo, and batch detection. The duplicate Advanced Settings switch has been removed. Raw plots retain the recorded direction; the choice controls the sign used for detecting events.

The Threshold tab groups **CS** and **SS** controls into two rows each: high-pass/low-pass cutoffs, then filter order/threshold. Filter cutoff `0` disables that cutoff (explained in hover help); threshold `0` still disables the corresponding spike type (or its Combo seeds). The controls keep the same saved parameter names and detection calculations. The upper panel follows the active tab’s height, with consistent spacing between rows and groups. **Parallel** and **Save**, like other GUI controls, start with capital letters.

Useful viewer tools:

- Hover over a setting input to see a short description of what it controls.
- **Save Settings** stores detection, all six shared QC limits, baseline, advanced, denoising (including preview size and advanced-panel visibility), statistics, colors, main-view controls, Detection Viewer window/Y-range/time position, and Slider Viewer Y-range/time position, plus paths to the currently loaded CS/SS template NPZ files. **Load Settings** restores those template banks when the files are available and reports missing sources. The JSON references the NPZ files; keep them accessible when moving settings between computers. Data folders and Path/Session/Cell navigation choices remain specific to the loaded recordings. Confirm dialog edits with **OK** before saving the main configuration.
- **Spike Statistics**: SS/CS waveform, autocorrelogram (ACG), FWHM, SNR, and instantaneous-rate summaries. The upper controls set and remember separate ACG half-windows and bin widths (CS ±1000 ms in 10 ms bins; SS ±100 ms in 1 ms bins by default). ACGs show conditional spike rate in Hz: each bin’s within-cell/session pair count is divided by the total observed target time contributed by its reference spikes. This uses the actual bin width, corrects recording boundaries, excludes masked observation gaps, and excludes self/zero-lag pairs. Selecting **All** pools counts and exposure separately, without pairing events across cells or recordings.
- **Detection Viewer**: raw traces, detection traces, thresholds, and detected events.
- The Detection Viewer raw-trace scale bar uses $\sigma_{\mathrm{MAD}}=1.4826\,\mathrm{median}(|x-\mathrm{median}(x)|)$ of the baseline-corrected trace. This is an SD-equivalent noise estimate for Gaussian noise, not the unscaled MAD or the sample SD; a zero-MAD trace falls back to its SD.
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

The **Save** checkbox applies to every batch folder: unchecked runs use temporary viewing exports; checked runs save into each input folder’s `spike_detection/` directory and ask before replacing existing outputs. Duplicate paths are collapsed and sessions with identical names in different folders remain separate. If different input extensions share a filename stem in a batch folder, output names retain the input extension to avoid collisions.

The Info panel reports every path, completed sessions/cells, pooled statistics and input/detection/save errors. Close the batch dialog and open **Spike Statistics** to filter by **Path → Session → Cell**, including **All**. Selecting a new master folder returns to ordinary single-folder mode. Batch processing loads all sessions into memory, then detects each cell and compresses the result file on worker threads. The bottom status bar shows completed cells and the saving stage; settings are locked during detection. The same template method, bank, thresholds and masks apply to each batch session.

See [the workflow review and improvement priorities](docs/SPIKE_DETECTION_REVIEW.md) for the detection-method assessment and comparison with voltage-imaging and electrophysiology approaches.

Excel's temporary owner/lock files (`~$*.xlsx`) are ignored during ordinary and batch folder discovery. They are not recording workbooks and are left on disk for Excel to manage. Selecting one directly produces an explanatory message; format errors in genuine workbook files are still reported.


### Spike QC and template matching (3.6.42)

**Local noise** in the Threshold tab (previously mislabeled Local baseline) selects a centered rolling, Gaussian-scaled MAD estimate for threshold cutoffs and Combo seeds. It does not subtract a second baseline or change template scores, amplitude-SNR windows or the whitening spectrum. In Denoising Settings, **Envelope threshold (SD)** instead uses the local cluster-envelope mean plus the selected multiplier times local SD; increasing it attenuates more frames. Hover descriptions and **Tips** distinguish these quantities.

Event amplitude SNR uses the same intervals for CS and SS: amplitude baseline −25 to −5 ms, local noise −150 to −5 ms, detected-SS exclusion −2 to +4 ms and detected-CS exclusion −9 to +18 ms. Both amplitude-baseline and noise estimates exclude these detected-event intervals and respect existing detector masks (which can exclude additional samples). The baseline is the median of remaining samples, requiring at least three finite samples; its absence produces NaN SNR rather than a contaminated or invented baseline. Noise remains Gaussian-scaled MAD, with at least 20 samples and approximately 5 ms of retained samples for a local estimate, then the existing recording-wide fallback. NaNs preserve event order in exported SNR arrays and are excluded from histogram/summary counts. Broad CS tails and undetected burst spikes can still contaminate these fixed windows. These intervals describe event SNR, not the recording-wide whitening spectrum or the GUI baseline correction. Exported analysis settings include `event_snr_policy`; rerun detection to regenerate exported metrics with this policy.

**Spike QC** is fixed beneath both detection tabs. Each CS/SS row has a minimum amplitude SNR, minimum FWHM and maximum FWHM. Zero disables that individual limit; hover over inputs for definitions. New amplitude SNR limits and the new CS maximum/SS minimum widths start at zero; the existing CS minimum of 4 ms and SS maximum of 4.5 ms are preserved. Minimum and maximum widths are inclusive and contradictory enabled limits are rejected before detection. All six values are saved, loaded, reset and applied in ordinary, batch and Combo detection, including threshold seeds used to learn temporary templates.

Amplitude QC uses the same event metric as Spike Statistics and exported event SNR: the polarity-oriented, frame-processed, baseline-corrected signal at the candidate, minus its local amplitude baseline, divided by Gaussian-scaled noise MAD. It uses the original signal before denoising, detection filtering or whitening. Candidate-event noise exclusions are frozen for each QC stage, and accepted events retain those SNR values for statistics/export. Unavailable SNR is reported separately and rejected when its minimum is enabled. FWHM uses each type's filtered detection waveform before whitening; this also makes threshold SS widths reflect the selected SS filter. Unmeasurable widths are retained and explicitly flagged as uncertain. If CS QC rejects proposals, SS is rescored with exclusions rebuilt from accepted CS; this can require one extra pass. Combo freezes its learned bank before final matching and uses the same QC for seeds and final events. No final events are fed back into template learning.

The duplicate template response/filtered-height floors and Advanced Settings width controls have been removed from the GUI. Legacy projection cutoffs are not converted numerically into amplitude SNR: old settings files load with new amplitude gates disabled and a review notice, while preserving width limits and translating a disabled legacy SS maximum to zero. Existing detection results are not rewritten. `spike_qc_json` and `width_quality_json` record limits, measurement policy, rejected/unavailable counts and candidate metrics. Detection Viewer shows separate amplitude-SNR and width losses.

**Whiten noise** (beside Shape groups) optionally accounts for temporal noise correlations in every template method, including Combo's final pass and batch detection. It is off by default and saved/loaded/reset with the other settings. Each cell/type estimates its own background spectrum using overlapping Hann windows confined to contiguous, unmasked stretches. Positive excursions above an independent 3-MAD cutoff are excluded from this noise fit (CS −20/+80 ms; SS −2/+4 ms); these noise-fit exclusions do not reject final candidates. Median periodograms are smoothed in log power, floored at 1% of the maximum PSD, and converted to an inverse-square-root spectral gain capped at 10. A symmetric FIR approximates this gain with up to 5 ms support on each side. The same FIR transforms the recording and matching kernels: full CS waveform or the existing nominal 3 ms SS core. Positive-core clipping occurs before whitening; whitening may introduce signed kernel coefficients. Score noise MAD is measured on the whitened background with FIR-contaminated samples excluded. Score masks cover the support of both the trace FIR and transformed kernel, so extra neighboring samples are excluded near recording boundaries and CS masks.

Whitening is approximate and regularized, not a conversion of cutoffs into calibrated significance levels. Reassess response and correlation cutoffs when enabling it; they refer to transformed signals and are not interchangeable with unwhitened values. Four clean spectral windows of at least 32 samples and enough background after FIR padding are required; otherwise the track explicitly falls back to ordinary matching. Info and Detection Viewer show applied/skipped status, and exported analysis settings record the fit parameters and FIR coefficients per cell/type. Polarity-reversed controls freeze the original whitening filters and noise-fit masks as well as the templates and detection masks. Spike QC, original waveform plots, SNR and FWHM measurements are not whitened. Validate accuracy on independent annotations or electrophysiology; reduced noise correlation alone does not establish improved detection.

**Combo** (checkbox beside the detection tabs) runs **Threshold seeds → temporary templates → template matching** for both ordinary and batch detection, regardless of the selected tab. The Threshold tab's CS/SS cutoffs select seed events; choose conservative values. The Template Matching tab selects the final scoring method and cutoffs. Advanced Settings controls extraction windows, event spacing and exclusions; Spike QC controls amplitude SNR and width limits. Combo aligns and amplitude-normalizes complete, isolated seed windows and forms robust median templates per cell/session. It requires at least three clean seeds for an automatic kernel; parallel matching uses only shape groups with enough seeds. Seed-quality exclusions affect learning, not a new final-event shape gate. When a type lacks sufficient seeds, its loaded template bank is used if available; otherwise that track reports no available templates and has no detected events. Confident CS seeds still exclude SS contamination if no CS bank is available. Final template detections can occur below the initial seed threshold.

Combo does not replace the user-loaded banks or iteratively retrain on its final events. Manual enabled-group choices apply to loaded fallback banks; all supported automatic groups are scored. The Detection Viewer and Info report seed/clean/kernel counts and automatic/fallback/unavailable sources. Per-cell learning diagnostics and kernels are included in exported analysis settings, and polarity-reversed controls reuse the original frozen templates. Assess sensitivity and false detections using held-out annotations; additional event counts alone do not establish accuracy. Combo is disabled by default and its checkbox is included in Save/Load Settings.

Load threshold-derived SS/CS templates, then choose **Amplitude-fit LLR (Gaussian)**, **Positive-core LLR (experimental)**, or **Normalized Similarity** under **Matching score** in the Template Matching tab. **Clear** empties both banks. The stored `LLR Probability Vector` name remains for compatibility; its score is a positive-amplitude-fitted Gaussian likelihood gain, not a probability. **CS/SS match response / noise MAD** controls the positive template response divided by global per-sample noise MAD, with a plotted gain cutoff of `cutoff² / 2`. It is distinct from Spike QC's local amplitude SNR and is not a calibrated false-positive z-score. These score cutoff boxes allow values through 100. SS matches use a short nominal 3 ms core to tolerate overlapping or truncated tails. **Normalized Similarity** uses Pearson correlation with defaults CS `r ≥ 0.90`, SS `r ≥ 0.80`. All methods then use the shared Spike QC limits; there are no additional hidden 3-MAD CS or minimum-SS-response gates in the GUI pipeline. Review old detection settings before reuse.

**Positive-core LLR (experimental)** keeps the existing CS detector and changes only SS scoring. It fits a positive-only 3 ms core of the pooled SS template to the baseline-corrected, optionally low-pass-filtered trace. This avoids a neighboring negative trough adding evidence for a positive SS through the negative weights of a mean-centered template. The SS matched-response cutoff, shared Spike QC, Advanced Settings SS spacing, CS masks, parallel groups and batch detection apply. There is no full-waveform correlation, strong-anchor, or 25 ms neighborhood gate. The former **Burst-aware LLR** settings key is retained so saved settings select this method, but the behavior changed in version 3.6.25; rerun detection to compare results. The cutoff is not a calibrated false-positive probability. Assess additions with held-out manual annotations across cells, bursts and quiet periods, and run the polarity-reversed control. That control is diagnostic rather than a calibrated specificity estimate.

The Detection Viewer shades excluded score regions and lists how many score peaks were removed by masks, spacing, amplitude-SNR and width decisions. A score peak above the line is therefore a candidate, not necessarily a detected event. **Run polarity-reversed control** reruns the selected cell in a worker thread with its saved detection settings and original CS/SS masks, then shows reversed versus original kept counts beside the candidate diagnostics. Reversed detections are a control, not an estimate of the false-positive count. Parallel matching groups amplitude-normalized template shapes using PCA and k-means. Set the number of groups on the second Template Matching row; View shows each group mean, template count, and checkboxes for enabling it. PCA-component count is in Advanced Settings. All requested groups are scored unless deselected, and the enabled groups are saved and reused in batch detection. With multiple groups, the displayed score is excess above each group's own cutoff, with zero as the line. More active groups can recover minority shapes but also add opportunities for false positives. Template detection retains the adjustable 700 Hz SS low-pass used to suppress the 1,112 Hz interference found in the supplied recording; set it to 0 to disable. SS widths use this filtered trace; both width limits are in the shared Spike QC panel. The original baseline-corrected trace remains the waveform/SNR review source. **Advanced Settings** is the only SS spacing control; its interval is applied once across all template groups. CS-exclusion settings also remain in effect. Inspect low-amplitude additions against manual marks; a higher count alone does not establish better accuracy.

### Time, widths and exclusions (3.6.17)

CSV/Excel time units are inferred automatically from first-column labels such as `t(ms)` or `time_s`, then from plausible sampling rates when the header has no units. Ambiguous unlabelled columns are rejected with a request to label the column. The inferred unit appears in the Info panel and exported settings. NPZ files always use `time_ms` and must have a matching `fs`. The loader rejects nonfinite, duplicate, decreasing or irregular timestamps (0.1% spacing tolerance), and nonfinite trace samples within the timed recording. If trace values extend below the last timestamp, only the fully timed prefix is analyzed; Info, batch issues, and exported settings warn about the excluded untimed tail. Add timestamps to include that tail. No source workbook is changed. Exported spike times preserve the input recording’s time origin.

In **Advanced Settings**, set **SS exclusion before CS** and **SS exclusion after CS** independently. Legacy `SS_BLANK_MS=18` maps to 9 ms before and 9 ms after CS. Endpoints are included and durations round upward to whole samples, so masks can differ by a sample from older results. Excluded intervals are never replaced with zeros. Template scoring also excludes windows that touch masked samples or recording boundaries; this can enlarge the effective exclusion relative to the requested pre/post values. NPZ exports include the actual `cs_exclusion_mask`, `ss_exclusion_mask`, full recording duration and valid duration per cell/type. Displayed rates use the full duration (`samples/fs`), including excluded time.

Widths are measured at the detected event, with search bounds between neighboring candidates. Template candidates may align to the nearest waveform peak within 2 ms, rounded upward to whole native samples so low-rate recordings can inspect the nearest frame. Both CS and SS widths use their filtered detection waveforms in every method. An unmeasurable width is retained as **uncertain**, with a reason, instead of silently passing the width filter. The Detection Viewer draws a measured CS FWHM as a green horizontal line and a retained CS with uncertain width as a short green dash at the event time; the dash length does not represent a measured duration. Info and Spike Statistics report uncertainty counts; FWHM summaries exclude unknown widths. `event_fwhm_cs/ss` retain one slot per final event, including NaN. `width_quality_json` records candidate indices, aligned peaks, half-height crossings, widths, uncertainty reasons and width decisions, including rejected candidates. Statistical waveform sources are frozen at detection time.

Filters must use valid bands strictly below Nyquist; zero disables a cutoff. Invalid filters or requested preprocessing failures are reported per session instead of silently falling back. The effective baseline window, mask sample counts, local-noise windows and filter settings are saved in `analysis_settings_json.effective_settings`. With denoising enabled, its upper frequency must not exceed 95% of Nyquist; adjust the denoise settings for low-sampling-rate recordings. These corrections can change detections compared with older versions.


## Building a Windows EXE

Run `python3 scripts/create_windows_bundle.py` in the repository to create `spike_detector_windows_build.zip`. Extract that ZIP on Windows, then double-click `build_windows_uv.bat`. The batch file uses uv to install Python 3.11, the GUI dependencies, and PyInstaller, checks the extracted source, and builds `dist\spike_detector\spike_detector.exe` in one-folder mode. The complete `dist\spike_detector` folder is needed to run the app. See `WINDOWS_BUILD.md` inside the ZIP for the exact steps. The source ZIP does not contain an EXE or downloaded dependencies; the first Windows build requires internet access.


## Application icon and macOS app

The supplied spike-trace picture is the program icon. Package assets include the unchanged source PNG, a multi-resolution Windows ICO, and a rounded macOS PNG/ICNS with transparent corners and margins. The macOS Qt application icon uses the rounded artwork too, so it agrees with the native bundle icon instead of overriding it with a square. The GUI sets its application/window icon when launched from Python or a packaged build; the Windows EXE embeds the ICO, and the macOS bundle uses the ICNS for Finder and Dock.

On macOS, run `uv sync --python 3.11 --extra packaging`, then `.venv/bin/python -m PyInstaller --noconfirm --clean packaging/pyinstaller/spike_detector.spec`. Open `dist/Spike Detector.app`. Build on the target platform; the Windows builder still produces `dist\spike_detector\spike_detector.exe`. These locally built macOS bundles are not Developer-ID notarized.

To regenerate the native icon files from the checked-in Windows and macOS PNGs, run `python scripts/create_app_icons.py` with Pillow and PyQt6 installed, then rebuild the application and refresh the Windows source ZIP. The original Windows artwork is 190×190; the macOS SVG places this unchanged image inside a rounded white tile and renders it with transparent margins. No trace detail is redrawn.
