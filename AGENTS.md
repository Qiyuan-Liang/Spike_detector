# Spike Detector Repository Notes

Read this file before making repository changes. Keep long change history in `LOG.md`, not here, so this file stays compact.

## Repository Structure

- `src/spike_detector/gui.py` - main PyQt spike detector GUI, threshold/template detection, plotting, detection preview, and spike statistics UI.
- `src/spike_detector/utils/` - shared processing, denoising, statistics, exporting, session, and mask helpers used by the GUI and analysis workflows.
- `Analysis/` - Jupyter analysis pipelines, including bleaching, synchrony/correlation, subthreshold analysis, locomotion, and related notebooks.
- `data/` - local project data, templates, and analysis outputs used by notebooks or GUI workflows.
- `packaging/pyinstaller/` - PyInstaller packaging assets for building a standalone application.
- `README.md`, `pyproject.toml`, `requirements.txt` - project documentation, package metadata, and Python dependencies.

## Working Notes

- Prefer editing the package GUI at `src/spike_detector/gui.py`; legacy or exported scripts should not be treated as the primary app unless the user explicitly points to them.
- For notebooks, verify that edits are applied to the notebook on disk and remind the user to reload the file in the IDE if it was already open.
- Preserve user data paths and external mounted paths. Do not move or rewrite data files unless directly requested.
- Use `rg` for repository searches and keep changes scoped to the requested analysis/GUI behavior.
- Update `LOG.md` whenever changing repository behavior.
- Maintain the spike detector version in real time in the GUI title, `pyproject.toml`, and `LOG.md`.
- Current version after this revision is `3.5.41`.
- For later patch revisions in this series, increment only the patch version: `3.5.42`, `3.5.43`, and so on.
- Do not change the minor or major version line yourself. Only move to versions such as `3.6` or `4.0` when the user explicitly asks.

## Change Log Location

See `LOG.md` for the dated change log.
