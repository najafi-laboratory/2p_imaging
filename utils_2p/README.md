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
- `utils_2p.slurm_pipeline`

## Staged Slurm 2p Pipeline

`utils_2p.slurm_pipeline` submits a generic five-stage Slurm chain for each
session in a CSV/TSV manifest:

```text
prep CPU -> suite2p GPU -> qc CPU -> label GPU -> dff CPU
```

Create a manifest with `session_name,data_path,save_path` columns. A
`bpod_mat_path` column is optional.

```csv
session_name,data_path,save_path,bpod_mat_path
example_session,/path/to/raw/session,/path/to/output/session,/path/to/bpod.mat
```

Preview the generated `sbatch` commands without submitting:

```bash
utils-2p-slurm-pipeline submit \
  --manifest sessions.csv \
  --processing-root /path/to/2p_processing_pipeline_202401 \
  --postprocess-root /path/to/2p_post_process_module_202404 \
  --sbatch-dir /path/to/sbatch_runs \
  --python-bin /path/to/env/bin/python \
  --dry-run
```

Remove `--dry-run` to submit the dependency chains.

To resume partially processed sessions, add `--resume`. The launcher inspects
the output directory and starts each session at the first missing stage:

```bash
utils-2p-slurm-pipeline submit \
  --manifest sessions.csv \
  --processing-root /path/to/2p_processing_pipeline_202401 \
  --postprocess-root /path/to/2p_post_process_module_202404 \
  --sbatch-dir /path/to/sbatch_runs \
  --python-bin /path/to/env/bin/python \
  --resume
```

The file markers used for resume/status inference are:

| Stage | Required output markers |
| --- | --- |
| `prep` | `raw_voltages.h5`, `suite2p/plane0/ops.npy` |
| `suite2p` | `suite2p/plane0/F.npy`, `Fneu.npy`, `spks.npy`, `stat.npy`, `iscell.npy`, `redcell.npy` |
| `qc` | `qc_results/fluo.npy`, `neuropil.npy`, `stat.npy`, `masks.npy`, `move_offset.h5`, `ops.npy` |
| `label` | `masks.h5` |
| `dff` | `dff.h5` |

Check a processed session, or all direct children of a processed root:

```bash
utils-2p-slurm-pipeline status /path/to/output/session
utils-2p-slurm-pipeline status --children /path/to/processed_root
```

## Current limitation

Some existing project files in this repo still add the repo root to `sys.path` manually before importing `utils_2p`. That was kept for compatibility during the first refactor pass.

Once all scripts are updated to rely on the installed package, those manual path edits can be removed.
