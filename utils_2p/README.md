# `utils_2p`

`utils_2p` is the shared utility package for this repository. It is meant to replace copy-pasted helper functions across the different 2p analysis subprojects.

## What it solves

Without installation, each project has to either:

- duplicate helper functions locally, or
- manually modify `sys.path` so Python can find `utils_2p`

Installing `utils_2p` into your environment lets any script or notebook import it directly:

```python
from utils_2p.stats import norm01, get_mean_sem
from utils_2p.timing import get_trigger_time
from utils_2p.alignment import trim_seq
```

## Recommended install

From the root of the `2p_imaging` repo:

```bash
python -m pip install -e .
```

This performs an editable install, which is the best option during development.

Benefits:

- `utils_2p` becomes importable from anywhere in that Python environment
- changes to files under `utils_2p/` are picked up without reinstalling
- subprojects can depend on one shared implementation instead of copied code

## Non-editable install

If you want a standard install instead:

```bash
python -m pip install .
```

## Verify the install

Run:

```bash
python -c "import utils_2p; print(utils_2p.__file__)"
```

If installation worked, Python will print the package location instead of raising `ModuleNotFoundError`.

## Upgrading after dependency changes

If `pyproject.toml` changes, reinstall:

```bash
python -m pip install -e .
```

## Using it from notebooks

Make sure the notebook kernel uses the same environment where you installed the package.

Then imports work normally:

```python
from utils_2p import norm01, get_frame_idx_from_time
```

## Current modules

- `utils_2p.stats`
- `utils_2p.timing`
- `utils_2p.alignment`
- `utils_2p.matlab`
- `utils_2p.voltage`

## Current limitation

Some existing project files in this repo still add the repo root to `sys.path` manually before importing `utils_2p`. That was kept for compatibility during the first refactor pass.

Once all scripts are updated to rely on the installed package, those manual path edits can be removed.
