# Processing Quickstart

This page covers the Slurm-based preprocessing/QC utility in `utils_2p`.
It is intended for launching one or more raw imaging sessions through the
standard staged pipeline without editing hard-coded session paths.

## Requirements

Run from a checkout of `2p_imaging/main` that includes
`utils_2p.preprocessing_qc_pipeline`:

```bash
git checkout main
git pull upstream main
```

Use the shared Suite2p-capable environment:

```bash
export TWO_P_PYTHON=/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_v1/bin/python
export TWO_P_SLURM_ACCOUNT=gts-fnajafi3
```

Jobs use the `embers` QOS by default. `embers` is free but preemptible after
the QOS runtime limit. Use `--qos inferno` for paid, non-preemptible jobs when
needed.

Suite2p writes its large temporary binary movie to node-local `$TMPDIR` by
default and deletes it when Suite2p finishes. This avoids writing tens of GB of
intermediate `.bin` data back to Cedar or project storage. Override with
`--fast-disk /path/to/workdir` only when node-local tmp does not have enough
free space.

For Suite2p 1.x runs, the pipeline defaults to a larger TIFF-to-binary batch
and smaller GPU processing batches: binary conversion uses `5000` frames per
batch, while registration and extraction use `500` frames per batch. Override
with `--suite2p-binary-batch-size`, `--suite2p-registration-batch-size`, or
`--suite2p-extraction-batch-size` when benchmarking a different node type.

## Pipeline Stages

The full processing chain is:

```text
prep -> suite2p -> qc -> label -> dff -> summary
```

| Stage | Resource | Main outputs |
|---|---|---|
| `prep` | CPU | `raw_voltages.h5`, copied `bpod_session_data.mat` when available, provenance JSON |
| `suite2p` | high-memory CPU; optional GPU request | `suite2p/plane0/ops.npy`, ROI statistics, fluorescence and neuropil traces, registered projections |
| `qc` | CPU | `qc_results/fluo.npy`, `neuropil.npy`, `stat.npy`, `masks.npy`, `qc_parameters.json`, `move_offset.h5` |
| `label` | GPU | `masks.h5` and anatomical Cellpose outputs; skipped for functional-only recordings |
| `dff` | CPU | `dff.h5` containing raw, non-z-scored dF/F traces |
| `summary` | CPU | `{session}_preprocessing_summary.pdf`, `{session}_interactive_fov_roi_dff.html` |

Each session gets its own linked Slurm chain. A failed session does not block
other submitted sessions.

## Submit Common Runs

The repository includes a reproducible YH24 pipeline test script that calls the
pipeline command-line interface with prefilled arguments. By default it
generates the Slurm scripts without submitting:

```bash
python utils_2p/scripts/run_yh24_preprocessing_pipeline_test.py
```

Submit the same test chain with:

```bash
python utils_2p/scripts/run_yh24_preprocessing_pipeline_test.py --submit
```

Neuronal sessions use the defaults. Channel count is detected from TIFF names;
two-channel recordings use Ch2 as functional and Ch1 as anatomical, while
single-channel recordings use the only detected channel as functional:

```bash
python -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session_1 \
  --session /path/to/raw/session_2 \
  --output-root /path/to/processed_outputs
```

Dendritic sessions should explicitly select the dendrite target structure:

```bash
python -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/processed_outputs \
  --target-structure dendrite
```

Suite2p requests a GPU by default. TIFF conversion is still CPU/storage-bound,
but Suite2p 1.x can use CUDA for registration and extraction after the
temporary binary is written. Add `--no-suite2p-gpu` for CPU-only Suite2p runs.

Functional-only, single-channel sessions do not need channel arguments unless
the TIFF naming is unusual:

```bash
python -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/processed_outputs \
  --target-structure dendrite
```

Use `generate` instead of `submit` to write the `.sbatch` files without
submitting them:

```bash
python -m utils_2p.preprocessing_qc_pipeline generate \
  --sessions-file raw_sessions.txt \
  --output-root /path/to/processed_outputs
```

Generated Slurm files are written below:

```text
{output_root}/.preprocessing_qc_jobs/{timestamp}_{username}/
```

## Rerun Selected Stages

Use `--stages` when upstream outputs already exist. For example, to regenerate
raw dF/F and the QC summary files:

```bash
python -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/existing_processed_outputs \
  --stages dff,summary
```

Available stages are:

```text
prep
suite2p
qc
label
dff
summary
```

Selected stages are always ordered according to the pipeline. Downstream-only
runs assume the required upstream files already exist.

## QOS Controls

Default generated jobs include:

```bash
#SBATCH --qos=embers
```

Use inferno for all stages:

```bash
python -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/processed_outputs \
  --qos inferno
```

Or split CPU and GPU stages:

```bash
python -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/processed_outputs \
  --qos-cpu embers \
  --qos-gpu inferno
```

## Notes

The utility reuses the versioned Suite2p configuration files in
`2p_processing_pipeline_202401/config_*.json` and the existing QC/label
algorithms in `2p_post_process_module_202404/modules/QualControlDataIO.py` and
`LabelExcInh.py`. The orchestration, voltage extraction, raw dF/F creation, and
preprocessing summary outputs live in `utils_2p`.
