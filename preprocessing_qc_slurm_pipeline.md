# Preprocessing QC Slurm Pipeline

`utils_2p.preprocessing_qc_pipeline` creates and optionally submits a linked
Slurm processing chain for each raw two-photon imaging session. Jobs read raw
data in place and write all processed results under a selected output root.

## Stages And Outputs

| Stage | Resource | Main output |
|---|---|---|
| `prep` | CPU | `raw_voltages.h5`, copied `bpod_session_data.mat` when available, provenance JSON |
| `suite2p` | High-memory CPU | `suite2p/plane0/ops.npy`, fluorescence traces, ROI statistics, registered projections |
| `qc` | CPU | `qc_results/fluo.npy`, `neuropil.npy`, `stat.npy`, `masks.npy`, `qc_parameters.json`, `move_offset.h5` |
| `label` | GPU | `masks.h5` and anatomical Cellpose outputs; Cellpose is invoked with GPU enabled and is omitted for single-channel data or with `--no-label` |
| `dff` | CPU | `dff.h5` containing raw, non-z-scored dF/F traces |
| `summary` | CPU | `{session}_preprocessing_summary.pdf`, `{session}_interactive_fov_roi_dff.html` |

Each session has its own dependency chain. A failed session does not prevent
other submitted sessions from proceeding.

The orchestration, voltage extraction, raw dF/F creation, and figure generation
live in `utils_2p`. The utility intentionally reuses the versioned
`2p_processing_pipeline_202401/config_*.json` Suite2p configurations and the
existing `2p_post_process_module_202404/modules/QualControlDataIO.py` and
`LabelExcInh.py` algorithms already on `main`, so these established algorithms
do not need duplicate copies.

The prior scratch wrapper requested a GPU for Suite2p, but the checked
`suite2p 0.14.6` configuration does not expose a GPU execution option. This
submitter requests high-memory CPU resources for Suite2p and reserves a GPU
for the Cellpose anatomical-labeling stage, which explicitly enables GPU use.

## Submit A Run

Two-channel neuronal sessions use the defaults:

```bash
python -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session_1 \
  --session /path/to/raw/session_2 \
  --output-root /path/to/processed_outputs
```

For a cerebellar dendritic session:

```bash
python -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/processed_outputs \
  --target-structure dendrite
```

For a functional-only, single-channel session:

```bash
python -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/processed_outputs \
  --nchannels 1 \
  --target-structure dendrite \
  --no-label
```

Use `generate` instead of `submit` to review the `.sbatch` scripts first:

```bash
python -m utils_2p.preprocessing_qc_pipeline generate \
  --sessions-file raw_sessions.txt \
  --output-root /path/to/processed_outputs
```

This writes a job manifest, stage `.sbatch` scripts, logs directory, and a
`submit_jobs.sh` command script beneath:

```text
{output_root}/.preprocessing_qc_jobs/{timestamp}_{username}/
```

Select a subset of stages with `--stages`. Stages are ordered according to the
pipeline and chained only among those selected. This is useful when upstream
outputs already exist; for example, to regenerate dF/F and QC figures:

```bash
python -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/existing_processed_outputs \
  --stages dff,summary
```

Available stages are `prep`, `suite2p`, `qc`, `label`, `dff`, and `summary`.
Selected downstream stages assume that their required upstream output files
already exist. The `label` stage is not valid when anatomical labeling is
disabled for a functional-only recording.

## Python API

The Python API permits per-session modality and target-structure settings in
one submission:

```python
from utils_2p.preprocessing_qc_pipeline import SessionSpec, submit_preprocessing_qc_jobs

jobs = submit_preprocessing_qc_jobs(
    [
        SessionSpec("/raw/SM07_session", target_structure="dendrite"),
        SessionSpec("/raw/EBC_single_channel", target_structure="dendrite",
                    nchannels=1, functional_chan=1, run_label=False,
                    stages=("dff", "summary")),
    ],
    "/processed/preprocessing_qc",
)
```

Do not rely on the default `neuron` preset for dendritic recordings. The target
structure is not reliably inferable from a raw path.

The Suite2p configuration corresponding to the selected target structure is
used unchanged for `denoise` and `spatial_scale` by default. Use
`--denoise` or `--spatial-scale` only when intentionally overriding that
configuration; in particular, the supplied dendrite configuration uses
`spatial_scale=2`.

## User Configuration

The generated jobs have no hard-coded user home path. Configure site/user
values either with CLI options or environment variables:

| Setting | CLI option | Environment variable |
|---|---|---|
| Suite2p-capable Python | `--python-bin` | `TWO_P_PYTHON` |
| Slurm account | `--account` | `TWO_P_SLURM_ACCOUNT` |
| Failure email | `--mail-user` | `TWO_P_SLURM_MAIL_USER` |
| Numba cache location | `--numba-cache-dir` | `TWO_P_NUMBA_CACHE_DIR` |

Example:

```bash
export TWO_P_PYTHON=/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_v1/bin/python
export TWO_P_SLURM_ACCOUNT=gts-fnajafi3
export TWO_P_SLURM_MAIL_USER="$USER@gatech.edu"
export TWO_P_NUMBA_CACHE_DIR="$TMPDIR/2p_numba_cache"
```

If neither `--python-bin` nor `TWO_P_PYTHON` is specified, the generator will
only use the currently running Python when it can import `suite2p`; it fails
early instead of creating Slurm jobs with an unusable system Python.

## Shared Conda Environment

A shared environment can be stored on project storage and referenced through
`TWO_P_PYTHON`. Existing Suite2p environments in user project directories are
about 8 to 9 GB, so capacity should be checked before creating another copy.
The repository includes [environment-preprocessing-qc.yml](environment-preprocessing-qc.yml),
which pins the currently validated core package versions. The environment
currently used for pipeline tests has a CUDA 12.8 PyTorch build
(`torch 2.9.1+cu128`); validate GPU availability on a compute node after
creating a shared environment.

Recommended lab-managed setup:

```bash
module load anaconda3/2023.03
umask 0022
mkdir -p /storage/project/r-fnajafi3-0/grubin6/shared_envs

PYTHONNOUSERSITE=1 \
conda env create \
  --prefix /storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_v1 \
  --file environment-preprocessing-qc.yml

chmod -R g+rX,o-rwx \
  /storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_v1

PYTHONNOUSERSITE=1 NUMBA_CACHE_DIR="$TMPDIR/2p_numba_cache" \
MPLCONFIGDIR="$TMPDIR/2p_matplotlib_cache" \
/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_v1/bin/python -c \
  "import suite2p, cellpose, torch; print(torch.__version__, torch.version.cuda)"
```

The environment directory must be created by someone with write permission to
the shared project area. After validation, normal users should execute it but
not modify it; updates should create a new versioned environment path. Per-job
caches should remain in each user's scratch space, not inside the shared
environment directory.

`PYTHONNOUSERSITE=1` during creation and validation is important: without it,
pip may treat packages installed in the creator's personal `~/.local` directory
as satisfying requirements, producing an environment that fails for other lab
members. Generated Slurm scripts also export `PYTHONNOUSERSITE=1`, so the
processing run uses packages from this environment rather than packages from
the submitting user's home directory.

### Environment Created May 27, 2026

The current shared environment was installed at:

```text
/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_v1
```

It is group-readable/executable by `pace-fnajafi3` and occupies approximately
`7.7G`. During initial creation, pip detected several compatible packages in
the creator's personal user site. The missing in-environment dependencies were
installed with `PYTHONNOUSERSITE=1`, and the creation command above includes
that setting so a future versioned install is self-contained from the start.
Validation with personal user packages disabled imported `suite2p 0.14.6`,
`cellpose 4.0.7`, `torch 2.9.1+cu128`, and `numba 0.62.1`; `pip check`
reported no broken requirements. The login node reports no available CUDA
device, so execution against a GPU remains a compute-job validation step.

Use it in the job generator with:

```bash
export TWO_P_PYTHON=/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_v1/bin/python
```

### Numba Cache

`NUMBA_CACHE_DIR` contains compiled machine-code cache files for numerical
functions used by Suite2p and related modules. It is disposable acceleration
state, not imaging output. The shared environment is kept read-only for users,
so each job must use a writable user-specific cache location such as scratch or
`$TMPDIR`; the submission utility sets this automatically unless overridden.
