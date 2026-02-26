# GitHub publish + Windows `.exe` build

## 1) First-time GitHub setup from VS Code

1. Open Source Control panel in VS Code.
2. Initialize git repo (if not initialized yet).
3. Stage all files and commit (example message: `refactor: package spike detector GUI`).
4. Use **Publish to GitHub** from VS Code (or create an empty repo on GitHub then add remote).

Equivalent commands:

```bash
git init
git add .
git commit -m "refactor: package spike detector GUI"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

## 2) Build `.exe` locally (Windows only)

> `.exe` should be built on Windows for best compatibility.

```bash
pip install -e .
pip install pyinstaller
pyinstaller packaging/pyinstaller/spike_detector.spec --noconfirm --clean
```

Output:
- `dist/spike_detector.exe`

## 3) Build `.exe` automatically on GitHub

This repo includes workflow:
- `.github/workflows/build-windows-exe.yml`

Triggers:
- Manual run (`workflow_dispatch`)
- Tag push (`v*`, e.g. `v3.3.1`)

Create a release tag:

```bash
git tag v3.3.1
git push origin v3.3.1
```

Then download artifact from Actions:
- `spike_detector_windows_exe`

## 4) Suggested release workflow

1. Edit code and run tests:
   - `pytest tests/core -q`
2. Bump version in `pyproject.toml`.
3. Commit and push.
4. Tag release (`vX.Y.Z`) and push tag.
5. Download EXE artifact from Actions.
6. Create GitHub Release and attach the EXE.

## 5) Editable install (`pip install -e .`) recap

- Creates an editable link to your local source.
- Code edits in `src/` are used immediately.
- Re-run only when dependencies/metadata change (`pyproject.toml`).
