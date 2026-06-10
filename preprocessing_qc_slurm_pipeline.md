# Preprocessing QC Slurm Pipeline

`utils_2p.preprocessing_qc_pipeline` creates and optionally submits a linked
Slurm processing chain for each raw two-photon imaging session. Jobs read raw
data in place and write all processed results under a selected output root.

## Stages And Outputs

| Stage | Resource | Main output |
|---|---|---|
| `prep` | CPU | `raw_voltages.h5`, copied `bpod_session_data.mat` when available, provenance JSON |
| `suite2p` | High-memory CPU; optional GPU request | `suite2p/plane0/ops.npy`, fluorescence traces, ROI statistics, registered projections |
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

Suite2p writes its temporary binary movie to node-local `$TMPDIR` by default
and deletes that binary when Suite2p finishes. This is the canonical execution
mode for raw TIFF sessions because it avoids writing tens of GB of intermediate
data to Cedar or project storage. Use `--fast-disk /path/to/workdir` only when
node-local tmp is too small or unavailable.

For Suite2p 1.x runs, the default batch tuning separates binary conversion from
GPU processing: TIFF-to-binary conversion uses `5000` frames per batch, while
registration and extraction use `500` frames per batch. These can be overridden
with `--suite2p-binary-batch-size`, `--suite2p-registration-batch-size`, and
`--suite2p-extraction-batch-size`.

Suite2p requests a GPU allocation by default. TIFF conversion is still
CPU/storage-bound, but Suite2p 1.x can use CUDA for registration and extraction
after the temporary binary is written. Use `--no-suite2p-gpu` for CPU-only
Suite2p runs. The Cellpose anatomical labeling stage also explicitly enables
GPU use.

## Manual ROI Review

The interactive HTML summary includes manual **Good** and **Bad** controls for
each displayed ROI. Use the buttons or the `G` and `B` keyboard shortcuts;
left/right arrow keys move between ROIs. FOV masks remain white outlines; the
selected ROI is filled translucent white. Stacked traces retain their
per-ROI color palette.

The **Morphology filter sandbox** uses the same `stat.npy` fields and threshold
logic as the preprocessing `qc` stage: `skew`, connectivity components,
`aspect_ratio`, `compact`, and `footprint`. The controls initialize from the
session's saved `qc_results/qc_parameters.json` when that file is present.
Built-in `neuron`, `dendrite`, and `cerebellum_lax` presets come directly from
the pipeline's `QC_PRESETS`. Changing thresholds previews the pass/fail count.
The preview does not change labels until **Apply filter to labels** is clicked.
Custom presets can be named and saved in the browser's local storage.

ROI review uses three states: Good, Bad, and Unlabeled. Sessions initialize
with morphology failures marked Bad and morphology passers Unlabeled. Good and
Unlabeled ROI outlines/traces are visible; Bad ROIs are hidden. Marking an ROI
Bad or applying morphology thresholds updates the FOV and trace list
immediately. **Show excluded ROIs** temporarily reveals Bad ROIs, and
**Mark all visible as good** accepts the current visible set. The exclusion
report is generated from the current morphology settings and current labels.
Unlabeled exports preserve the original Suite2p `iscell.npy` row in the
separate reviewed output.

The target-structure morphology parameters belong to the post-Suite2p QC stage;
they are not passed into Suite2p ROI detection. The HTML embeds all original
Suite2p ROIs. **Clear all labels / show all ROIs** removes both manual and
morphology-derived reviewer labels, restores the pending export rows from the
original `iscell.npy`, and reveals the complete Suite2p ROI set. Select a
preset and click **Apply filter to labels** to restore morphology filtering.

Suite2p stores ROI classifications in `suite2p/plane0/iscell.npy`, an
`N x 2` floating-point array:

- Column 0 is the binary classification used by downstream code: `1` means
  cell/good ROI and `0` means non-cell/bad ROI.
- Column 1 is Suite2p's classifier probability. Manual labels set this to
  `1.0` for good and `0.0` for bad so the final manual decision is explicit.

The preprocessing summary contains the post-QC subset of ROIs, while
`iscell.npy` contains every original Suite2p ROI. Summary generation therefore
matches each displayed ROI back to its original `stat.npy` row by pixel
coordinates. Export preserves all Suite2p rows that were not displayed.

The **Save iscell_qc.npy** button creates a complete Suite2p-shaped reviewed
file without replacing Suite2p's classifier output. Browsers cannot silently
overwrite local files, so supported browsers show a save-file prompt;
otherwise the file is downloaded. Save or move it to
`suite2p/plane0/iscell_qc.npy`.

The **Export label JSON** button creates a smaller auditable label file. Apply
it from the repository with:

```bash
python -m utils_2p.roi_labels \
  SESSION_manual_roi_labels.json \
  /path/to/session/suite2p/plane0
```

This validates ROI indices, preserves unreviewed rows, reads the original
`iscell.npy`, and writes `iscell_qc.npy`. An existing QC file is backed up as
`iscell_qc.npy.bak_YYYYMMDD_HHMMSS`.
Downstream analysis can then select good ROIs with:

```python
import numpy as np

plane = "/path/to/session/suite2p/plane0"
iscell_qc = np.load(f"{plane}/iscell_qc.npy")
good = iscell_qc[:, 0].astype(bool)
F_good = np.load(f"{plane}/F.npy")[good]
Fneu_good = np.load(f"{plane}/Fneu.npy")[good]
stat_good = np.load(f"{plane}/stat.npy", allow_pickle=True)[good]
```

## Submit A Run

Neuronal sessions use the defaults. Channel count is detected from TIFF names;
two-channel recordings use Ch2 as functional and Ch1 as anatomical, while
single-channel recordings use the only detected channel as functional:

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
  --target-structure dendrite
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
| Slurm QOS for all stages | `--qos` | `TWO_P_SLURM_QOS` |
| Slurm QOS for CPU stages | `--qos-cpu` | `TWO_P_SLURM_QOS_CPU` |
| Slurm QOS for GPU stages | `--qos-gpu` | `TWO_P_SLURM_QOS_GPU` |
| Failure email | `--mail-user` | `TWO_P_SLURM_MAIL_USER` |
| Numba cache location | `--numba-cache-dir` | `TWO_P_NUMBA_CACHE_DIR` |

Example:

```bash
export TWO_P_PYTHON=/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_suite2p_1x/bin/python
export TWO_P_SLURM_ACCOUNT=gts-fnajafi3
export TWO_P_SLURM_QOS=embers
export TWO_P_SLURM_MAIL_USER="$USER@gatech.edu"
export TWO_P_NUMBA_CACHE_DIR="$TMPDIR/2p_numba_cache"
```

Generated jobs request the `embers` QOS by default. On PACE, this keeps the
single-session stage jobs free but preemptible after the QOS runtime limit. Use
`--qos inferno` when a run should be guaranteed to complete and the paid QOS is
appropriate:

```bash
python -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/processed_outputs \
  --qos inferno
```

If CPU and GPU stages should use different QOS values, use `--qos-cpu` and
`--qos-gpu` instead of `--qos`.

If neither `--python-bin` nor `TWO_P_PYTHON` is specified, the generator will
use the version selected by `--suite2p-version` and falls back to the versioned
shared environment alias for that Suite2p release. The default is `1.x`.

## Shared Conda Environment

A shared environment can be stored on project storage and referenced through
`TWO_P_PYTHON`. Existing Suite2p environments in user project directories are
about 8 to 9 GB, so capacity should be checked before creating another copy.
The repository includes versioned YAMLs in `utils_2p/`:

- `utils_2p/environment-preprocessing-qc-suite2p-0x.yml`
- `utils_2p/environment-preprocessing-qc-suite2p-1x.yml`

The default shared alias is `.../2p_preprocessing_qc_suite2p_1x`, and the
legacy alias is `.../2p_preprocessing_qc_suite2p_0x`. The environment
currently used for pipeline tests has a CUDA 12.8 PyTorch build
(`torch 2.9.1+cu128`); validate GPU availability on a compute node after
creating or updating a shared environment.

Recommended lab-managed setup:

```bash
module load anaconda3/2023.03
umask 0022
mkdir -p /storage/project/r-fnajafi3-0/grubin6/shared_envs

PYTHONNOUSERSITE=1 \
conda env create \
  --prefix /storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_suite2p_1x \
  --file utils_2p/environment-preprocessing-qc-suite2p-1x.yml

chmod -R g+rX,o-rwx \
  /storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_suite2p_1x

PYTHONNOUSERSITE=1 NUMBA_CACHE_DIR="$TMPDIR/2p_numba_cache" \
MPLCONFIGDIR="$TMPDIR/2p_matplotlib_cache" \
/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_suite2p_1x/bin/python -c \
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
/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_suite2p_1x
```

It is group-readable/executable by `pace-fnajafi3` and occupies approximately
`7.7G`. During initial creation, pip detected several compatible packages in
the creator's personal user site. The missing in-environment dependencies were
installed with `PYTHONNOUSERSITE=1`, and the creation command above includes
that setting so a future versioned install is self-contained from the start.
Validation with personal user packages disabled imported `suite2p 1.0.0.1`,
`cellpose 4.0.7`, `torch 2.9.1+cu128`, and `numba 0.62.1`; `pip check`
reported no broken requirements. The login node reports no available CUDA
device, so execution against a GPU remains a compute-job validation step.

Use it in the job generator with:

```bash
export TWO_P_PYTHON=/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_suite2p_1x/bin/python
```

### Numba Cache

`NUMBA_CACHE_DIR` contains compiled machine-code cache files for numerical
functions used by Suite2p and related modules. It is disposable acceleration
state, not imaging output. The shared environment is kept read-only for users,
so each job must use a writable user-specific cache location such as scratch or
`$TMPDIR`; the submission utility sets this automatically unless overridden.
