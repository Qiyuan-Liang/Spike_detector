# Build Spike Detector for Windows

This source bundle contains Spike Detector GUI **3.6.20** and a one-click Windows builder. It does not contain an EXE or downloaded Python packages.

1. Extract **all** files from `spike_detector_windows_build.zip` into one ordinary folder on a Windows PC. Do not run the batch file from inside the ZIP.
2. Double-click `build_windows_uv.bat`. Keep the computer online during the first build. The batch file installs uv into the extracted folder if needed, then uv downloads Python 3.11 and Windows-compatible dependencies. A project-local `.venv` and `uv.lock` are created as part of that build.
3. After the batch file reports success, launch `dist\spike_detector\spike_detector.exe`. Keep the complete `dist\spike_detector` folder together when copying the finished application.

The builder stops if uv installation, dependency sync, source verification, PyInstaller, or the final EXE check fails. The expected output, relative to the extracted folder, is exactly `dist\spike_detector\spike_detector.exe`. Re-running the batch file builds the current extracted source again. If dependencies have not been downloaded on that Windows PC, the first run needs internet access.

The batch file uses the package-based launcher so Python package-relative imports work. The existing legacy `gui_preprocess_V3.3.py` script is not used. No research data, notebooks, templates, or local machine paths are in this source bundle. The GUI asks the user to choose their own data or templates at run time. No custom Windows icon exists in the project, so the default PyInstaller icon is used.

For diagnosing a failure, open Command Prompt in the extracted folder and run `build_windows_uv.bat`; read the first error above the final message. The Windows build is created on Windows because PyInstaller does not cross-compile between macOS and Windows.
