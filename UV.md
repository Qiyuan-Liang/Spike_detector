# UV Environment Setup

This project can be installed with UV on another machine while keeping the same package set needed by the Spike Detector GUI and analysis notebooks.

## Fresh Environment

From the repository root:

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

Run the GUI:

```bash
uv run spike-detector
```

or:

```bash
uv run python -m spike_detector
```

## Notebooks

The `requirements.txt` file installs the notebook extras, including Jupyter, IPython, ipywidgets, seaborn, tqdm, and nbconvert.

To start Jupyter:

```bash
uv run jupyter lab
```

## Reproduce An Existing ASAP7 Conda Environment

If you want to capture the exact package versions from the working ASAP7 conda environment, run this inside that environment on the machine where it exists:

```bash
python -m pip freeze > requirements-asap7-freeze.txt
```

Then on the UV-managed machine:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements-asap7-freeze.txt
uv pip install -e .
```

Use the freeze file only when you need exact version replication. For normal development, prefer:

```bash
uv pip install -r requirements.txt
```

## Locking With UV

After installing successfully on the UV machine, you can create a lock file:

```bash
uv lock
```

Commit `uv.lock` if you want future machines to sync identical resolved versions:

```bash
uv sync --all-extras
```
