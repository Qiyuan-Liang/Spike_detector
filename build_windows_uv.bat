@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 goto directory_failed

set "UV_EXE=%~dp0.tools\uv\uv.exe"
if exist "%UV_EXE%" goto uv_ready
where uv >nul 2>nul
if not errorlevel 1 (
    set "UV_EXE=uv"
    goto uv_ready
)

echo uv was not found. Installing uv into this extracted folder...
set "UV_INSTALL_DIR=%~dp0.tools\uv"
set "UV_NO_MODIFY_PATH=1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; irm https://astral.sh/uv/install.ps1 | iex"
if errorlevel 1 goto uv_install_failed
if not exist "%UV_EXE%" goto uv_install_failed

:uv_ready
echo Syncing Python 3.11, the GUI dependencies, and PyInstaller...
"%UV_EXE%" sync --python 3.11 --extra packaging
if errorlevel 1 goto sync_failed
if not exist ".venv\Scripts\python.exe" goto sync_failed

echo Verifying the extracted source bundle...
".venv\Scripts\python.exe" scripts\verify_bundle.py
if errorlevel 1 goto verify_failed

echo Building the Windows application...
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean packaging\pyinstaller\spike_detector.spec
if errorlevel 1 goto build_failed
if not exist "dist\spike_detector\spike_detector.exe" goto build_failed

 echo.
echo Build complete: "%CD%\dist\spike_detector\spike_detector.exe"
echo Keep the entire dist\spike_detector folder together when moving the app.
pause
exit /b 0

:directory_failed
echo ERROR: Could not enter the extracted bundle directory.
goto failed
:uv_install_failed
echo ERROR: uv installation failed or uv.exe was not created.
goto failed
:sync_failed
echo ERROR: uv sync failed; PyInstaller was not started.
goto failed
:verify_failed
echo ERROR: Source verification failed; PyInstaller was not started.
goto failed
:build_failed
echo ERROR: PyInstaller failed or dist\spike_detector\spike_detector.exe is missing.
:failed
echo Review the error above, then run this batch file again.
pause
exit /b 1
