# Processing Quickstart

This guide covers the staged two-photon processing pipeline in
`utils_2p.processing_pipeline`, including shared environment use,
environment installation, single-session submission, and multi-session
submission.

## Running on PACE

### Process One Session

This is the shortest path for processing one session on PACE. Replace
`RAW_SESSION` with the raw session directory and run the commands from a PACE
login node:

```bash
module load anaconda3/2023.03
# After creating the shorter symlink described below, this can be replaced with:
# conda activate ~/suite2p_1x
conda activate /storage/project/r-fnajafi3-0/shared/shared_envs/2p_processing_suite2p_1x

# Expected output:
# - utils_2p: a path inside the activated shared environment, not an import error.
# - suite2p: the installed Suite2p version number.
python -c "from importlib.metadata import version; import utils_2p; print('utils_2p:', utils_2p.__file__); print('suite2p:', version('suite2p'))"

RAW_SESSION="/path/to/raw/session"
OUTPUT_ROOT=~/scratch/2p_processing_results
TARGET_STRUCTURE="soma"  # soma or dendrite

python -m utils_2p.processing_pipeline submit \
  --session "$RAW_SESSION" \
  --output-root "$OUTPUT_ROOT" \
  --target-structure "$TARGET_STRUCTURE" \
  --qos embers
```

For one-session runs, the default run name is the raw session directory name
plus `_processing`, for example `MC11_20260330_processing`. This names the
generated job/provenance directory and is used in the generated job files. Pass
`--run-name` only when you want a different label.

Generated Slurm jobs call `python -m utils_2p.processing_pipeline` from
the installed package and use bundled Suite2p config/QC helper files by
default. When you run the launcher from the activated shared environment, you
do not need to set a Python environment variable or pass `--python-bin`.

The launcher defaults to the lab Slurm account `gts-fnajafi3` and Suite2p
version `1.x`, so those arguments are omitted from the standard examples.

If the session path was not given to you directly, common places to check are:

```text
/storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging/
/storage/project/r-fnajafi3-0/shared/2P_Imaging/
/storage/project/r-fnajafi3-0/
```

Use `find` only with a specific animal/session prefix so it does not crawl a
large shared tree:

```bash
find /storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging -maxdepth 4 -type d -name "MC11_20260330*"
```

Check job progress with:

```bash
squeue -u "$USER"
```

If `$USER` is not set correctly in your shell, replace it with your GT username
directly, for example `squeue -u <gt-username>`.

The launcher writes a run directory under:

```text
<output-root>/.processing_jobs/<run-name>_<username>/
```

Look in that directory for `manifest.json`, generated `.sbatch` files,
`submit_jobs.sh`, and stage logs.

To make activation faster in later sessions, create a shorter symlink once:

```bash
ln -s /storage/project/r-fnajafi3-0/shared/shared_envs/2p_processing_suite2p_1x ~/suite2p_1x
```

You can name the symlink whatever is memorable. After creating it, activate the
same environment with:

```bash
module load anaconda3/2023.03
conda activate ~/suite2p_1x
```

### Pipeline Stages

For new sessions, the default pipeline is:

```text
prep -> suite2p -> dff -> summary
```

Additional stages must be specified explicitly:

```text
prep -> suite2p -> roi_model_scores -> label -> dff -> spikes -> summary
```

These non-default stages are enabled with command-line flags. See
[Important Non-Default Arguments](#important-non-default-arguments) for the
available flags and constraints.

### Batch Submission Script

For repeated use, write a small Slurm submission wrapper. Save this as
`run_2p_pipeline_pace.sh`, edit the path variables, and submit it with
`sbatch run_2p_pipeline_pace.sh`. This wrapper is lightweight: it activates the
shared environment and asks the pipeline launcher to submit the real stage jobs.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=submit-2p-pipeline
#SBATCH --account=gts-fnajafi3
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=submit-2p-pipeline_%j.out
#SBATCH --error=submit-2p-pipeline_%j.err

set -euo pipefail

RAW_SESSION="/path/to/raw/session"
OUTPUT_ROOT=~/scratch/2p_processing_results
TARGET_STRUCTURE="soma"  # soma or dendrite
RUN_NAME="$(basename "${RAW_SESSION}")_processing"

module load anaconda3/2023.03
conda activate ~/suite2p_1x

python -c "import utils_2p; print('utils_2p:', utils_2p.__file__)"

python -m utils_2p.processing_pipeline submit \
  --session "${RAW_SESSION}" \
  --output-root "${OUTPUT_ROOT}" \
  --target-structure "${TARGET_STRUCTURE}" \
  --qos embers \
  --run-name "${RUN_NAME}"
```

For multiple sessions, put one raw session path per line in a text file and use
`--sessions-file` instead of `--session`:

```text
# soma_sessions.txt
/path/to/raw/session_1
/path/to/raw/session_2
/path/to/raw/session_3
```

```bash
python -m utils_2p.processing_pipeline submit \
  --sessions-file soma_sessions.txt \
  --output-root ~/scratch/2p_processing_results \
  --target-structure soma \
  --qos embers \
  --run-name soma_batch
```

The `--sessions-file` interface submits one independent chain per listed
session; it does not throttle how many session chains can be active at once.
Start with a small file, such as five to ten sessions, and monitor the batch
before submitting more.

Argument meanings:

| Argument | Meaning |
|---|---|
| `submit` | Generate the Slurm files and immediately submit all requested stages. |
| `--session` | Raw session directory containing the imaging TIFF files and associated session inputs. |
| `--output-root` | Parent directory where a processed directory named after the raw session will be created. |
| `--target-structure` | Suite2p target preset: `soma` or `dendrite`. This affects the default Suite2p arguments; channel count and functional channel are inferred from the raw session files unless overridden. |
| `--suite2p-version` | Advanced override for the generated job environment. The default is `1.x`, so users in the shared 1.x environment usually omit this. |
| `--python-bin` | Optional advanced override for the exact Python executable used inside every generated job. Omit this for the shared activated environment. |
| `--processing-root` | Optional override for Suite2p config JSON files. Omit this for the packaged defaults. |
| `--postprocess-root` | Optional override for packaged cell-labeling helper modules. Omit this for the packaged defaults. |
| `--account` | Advanced override for the Slurm allocation charged for the jobs. The default is `gts-fnajafi3`. |
| `--qos` | Slurm QOS for all stages. `embers` is preemptible; use `inferno` when paid, non-preemptible execution is required. |
| `--run-name` | Optional readable name for the generated job directory and provenance files. For one-session runs, the default is `<session-name>_processing`. |

### Raw Session and Output Layout

The raw session directory should be the directory for one imaging session. It
normally contains imaging movies and any synchronized non-imaging files that
were recorded with the session:

```text
<raw-session>/
├── *.tif or *.ome.tif
├── *.csv or raw voltage files
└── bpod_session_data.mat, when available
```

The exact filenames vary across rigs and acquisition versions. The important
point is that `--session` should point to the session directory itself, not to
an individual TIFF file and not to a parent directory containing many sessions.

For a default run, the processed output directory is created under
`--output-root` and should look broadly like:

```text
<output-root>/<session-name>/
├── raw_voltages.h5
├── bpod_session_data.mat, when available
├── processing_pipeline_parameters.json
├── suite2p/plane0/
│   ├── ops.npy
│   ├── stat.npy
│   ├── F.npy
│   ├── Fneu.npy
│   ├── iscell.npy
│   └── spks.npy
├── dff.h5
├── <session-name>_processing_summary.pdf
└── <session-name>_interactive_fov_roi_dff.html
```

The job-control files are written separately under:

```text
<output-root>/.processing_jobs/<run-name>_<username>/
```

### PACE Storage and Job-Submission Guidance

Run the processing jobs on PACE compute nodes, not on a login node. For
multi-session processing, use PACE scratch for staged input data, the
`--output-root`, and Suite2p temporary files whenever practical:

```text
~/scratch/
├── staged_raw_sessions/
└── 2p_processing_results/
```

Cedar and project storage are durable shared filesystems, but they should not be
used in the same way. Cedar is best treated as the record-keeping location for
raw files and retained results, not as the day-to-day working filesystem for
large analysis jobs. The main concern is extensive repeated reading from Cedar:
Suite2p repeatedly reads large TIFF stacks, and many simultaneous jobs can
saturate shared read bandwidth, trigger metadata or I/O throttling, and make
every job slower.

Running the processing pipeline from raw sessions that are already in project
storage is fine. The main project-storage concern is putting too many active
processing outputs and temporary intermediates there. Use scratch for the
`--output-root` while processing, then transfer the validated outputs back to
project storage or Cedar after QC is complete. Scratch has much more working
space and is faster for high-throughput temporary job I/O, so it is usually the
better place for pipeline outputs and staged working copies.

A practical workflow is:

1. Keep the durable raw recording on Cedar for record keeping.
2. If the session is on Cedar and will be processed heavily or as part of a
   large batch, stage a working copy to scratch first.
3. If the session is already in project storage, it is fine to use that as the
   input path.
4. Run the pipeline with a scratch `--output-root`.
5. Validate the final outputs.
6. Copy the retained processed results back to durable project or Cedar
   storage.

Use Globus for large-scale file transfers whenever possible, especially when
moving raw sessions or processed batches between Cedar, project storage,
scratch, or a local workstation. Globus is preferred over `rsync` for imaging
sessions because it is built for managed, resumable bulk transfers and is the
PACE-recommended method. Reserve `rsync` for small one-off copies or cases
where Globus is not available.

For Globus transfers, use the endpoint/path information for the source and
destination filesystems:

```bash
globus transfer SOURCE_ENDPOINT:/path/to/raw/session/ DEST_ENDPOINT:~/scratch/staged_raw_sessions/session/ --recursive
globus task list
globus task show <task-id>
```

Create the scratch directories before processing:

```bash
mkdir -p ~/scratch/staged_raw_sessions
mkdir -p ~/scratch/2p_processing_results
```

After validating the processed session, copy it to its durable destination:

```bash
globus transfer SOURCE_ENDPOINT:~/scratch/2p_processing_results/session/ DEST_ENDPOINT:/path/to/processed_results/session/ --recursive
```

Scratch is temporary, may be purged according to current PACE policy, and is
not a backup. Move validated outputs back to Cedar or project storage within 30
days because scratch is regularly emptied. Do not remove the durable source or
only copy of a result until the transfer back has been verified.

Each session creates four Slurm jobs by default: `prep`, `suite2p`, `dff`, and
`summary`. Specifying every non-default stage creates up to seven jobs per
session: `prep`, `suite2p`, `roi_model_scores`, `label`, `dff`, `spikes`, and
`summary`. Submitting 50 default sessions at once can therefore create roughly
200 queued jobs, while several Suite2p stages may
begin reading TIFFs at the same time. Large submissions increase scheduler
load, consume pending-job allowances, and can cause an I/O burst even when the
jobs are linked by dependencies.

Submit a small batch first, confirm its memory, runtime, and I/O behavior, then
process additional sessions in controlled groups. Five to ten sessions per
batch is a reasonable conservative starting point, but current PACE limits and
the size of the recordings should determine the final batch size. Monitor the
batch before submitting another:

```bash
squeue -u "$USER"
```

For unusually large recordings, use fewer concurrent sessions.

Channel count and functional channel are normally inferred from TIFF names.
For a functional-only, single-channel dendrite session, specify the channel
overrides:

```bash
python -m utils_2p.processing_pipeline submit \
  --session /path/to/raw/single_channel_session \
  --output-root /path/to/processed_outputs \
  --target-structure dendrite \
  --nchannels 1 \
  --functional-chan 1 \
  --qos embers
```

Additional argument meanings:

| Argument | Meaning |
|---|---|
| `--nchannels 1` | Override automatic channel detection and treat the recording as single-channel. |
| `--functional-chan 1` | Use channel 1 as the calcium-imaging channel. |

Suite2p requests a GPU by default. Add `--no-suite2p-gpu` to run Suite2p on
CPU-only resources. Add `--run-label` only when anatomical/cell-type labeling
is needed; that stage still requires a GPU when enabled.

### Launch Multiple Sessions

#### Repeat `--session`

For a small batch with the same processing settings, repeat `--session`:

```bash
python -m utils_2p.processing_pipeline submit \
  --session /path/to/raw/session_1 \
  --session /path/to/raw/session_2 \
  --session /path/to/raw/session_3 \
  --output-root /path/to/processed_outputs \
  --target-structure soma \
  --qos embers \
  --run-name soma_batch
```

Each session receives its own linked stage chain. A failed session does not
block the other sessions.

#### Use a Sessions File

For a larger batch, create a plain-text file with one raw session path per
line. Blank lines and lines beginning with `#` are ignored:

```text
# soma_sessions.txt
/path/to/raw/session_1
/path/to/raw/session_2
/path/to/raw/session_3
```

Submit all paths in the file:

```bash
python -m utils_2p.processing_pipeline submit \
  --sessions-file soma_sessions.txt \
  --output-root /path/to/processed_outputs \
  --target-structure soma \
  --qos embers \
  --run-name soma_manifest
```

All entries in one `--sessions-file` invocation share the command-line
settings. Use separate files or separate invocations when sessions require
different target structures, channel overrides, or stage selections.
Keep each file to a controlled batch size rather than putting an entire large
dataset into one submission.

The launcher writes its resolved JSON manifest, stage `.sbatch` files, logs,
and submission script below. The default run writes only the scripts for
`prep`, `suite2p`, `dff`, and `summary`; non-default stage scripts appear only
when those stages are specified:

```text
<output-root>/.processing_jobs/<run-name>_<username>/
├── manifest.json
├── prep.sbatch
├── suite2p.sbatch
├── roi_model_scores.sbatch
├── label.sbatch
├── dff.sbatch
├── spikes.sbatch
├── summary.sbatch
├── submit_jobs.sh
└── logs/
```

Only stages used by at least one session are written, so `roi_model_scores`,
`label`, and `spikes` scripts appear only when those stages are specified.

### Generate Jobs Without Submitting

Use `generate` to validate inputs and inspect the manifest and `.sbatch` files
before submitting:

```bash
python -m utils_2p.processing_pipeline generate \
  --sessions-file soma_sessions.txt \
  --output-root /path/to/processed_outputs \
  --target-structure soma \
  --qos embers \
  --run-name soma_manifest
```

The command prints the generated job directory and the corresponding
`submit_jobs.sh` path. On PACE, submit the generated chains with:

```bash
bash /path/to/processed_outputs/.processing_jobs/soma_manifest_${USER}/submit_jobs.sh
```

Run both `generate` and the resulting submission script on PACE so all
Python, session, and output paths are accessible to the compute nodes.

### Important Non-Default Arguments

| Argument | Meaning |
|---|---|
| `--stages prep,suite2p,dff,summary` | Run only the listed stages; they are reordered into pipeline dependency order automatically. |
| `--denoise 0` or `--denoise 1` | Override the denoising setting from the selected Suite2p configuration. |
| `--spatial-scale N` | Override Suite2p spatial scale instead of using the target configuration value. |
| `--qos-cpu` | QOS for CPU stages, overriding `--qos`. |
| `--qos-gpu` | QOS for GPU stages, overriding `--qos`. |
| `--mail-user` | Send Slurm failure notifications to this email address. |
| `--fast-disk` | Directory for Suite2p's temporary binary movie. The default uses node-local `$TMPDIR`. |
| `--suite2p-gpu` | Explicitly request a GPU for Suite2p; this is the default. |
| `--no-suite2p-gpu` | Run Suite2p without requesting a GPU. |
| `--suite2p-binary-batch-size` | Tune the Suite2p 1.x TIFF-to-binary batch size; default `5000`. |
| `--suite2p-registration-batch-size` | Tune the Suite2p 1.x registration batch size; default `500`. |
| `--suite2p-extraction-batch-size` | Tune the Suite2p 1.x extraction/deconvolution batch size; default `500`. |
| `--run-label` | Specify the non-default anatomical/cell-type labeling stage. Not run by default for new sessions. |
| `--run-roi-model-scores` | Specify the non-default trained ROI model scoring stage. This writes `roi_model_scores.h5` and can update `ROI_label.h5`; the currently available checkpoint is only for cerebellar dendrite ROIs. |
| `--roi-model-path` | Fallback path to a trained ROI model checkpoint. |
| `--roi-model-registry` | JSON mapping target structures to trained ROI model checkpoints. |
| `--roi-target-model target=/path/model.pt` | Register a target-specific ROI model checkpoint from the command line; repeat for multiple targets. |
| `--roi-model-good-threshold` | Probability threshold for model-labeled good ROIs; default `0.8`. |
| `--roi-model-bad-threshold` | Probability threshold for model-labeled bad ROIs; default `0.2`. |
| `--initialize-summary-labels-from-roi-model-scores` | Initialize reviewer labels from `ROI_label.h5` / `roi_model_scores.h5`. Without this, model outputs are available as metrics but the reviewer starts with ROIs not manually labeled. |
| `--summary-input-layout` | Select the summary input layout: `suite2p`, `qc_results`, `manual_qc_results`, `external_rois`, or `auto`. The pipeline default is `suite2p`; `qc_results` and `manual_qc_results` are deprecated legacy layouts. |

### Use a Trained ROI Model Checkpoint

The `roi_model_scores` stage is optional and runs after Suite2p ROI detection.
It scores each original Suite2p ROI and writes `roi_model_scores.h5`; the HTML
reviewer then loads those probabilities as a filter/sort metric. By default,
the reviewer still opens with ROIs **not labeled** unless
`--initialize-summary-labels-from-roi-model-scores` is supplied.

The currently available trained checkpoint is intended for cerebellar dendrite
ROIs only. Do not use it as a general soma, PPC, or non-cerebellar classifier
unless a model has been trained and validated for that target.

For one frozen checkpoint, point directly at the model file:

```bash
python -m utils_2p.processing_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/processed_outputs \
  --target-structure dendrite \
  --run-roi-model-scores \
  --roi-model-path /path/to/frozen_cerebellar_dendrite_model.pt
```

For multiple trained models, register a model per target structure on the
command line:

```bash
python -m utils_2p.processing_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/processed_outputs \
  --target-structure dendrite \
  --run-roi-model-scores \
  --roi-target-model dendrite=/path/to/frozen_cerebellar_dendrite_model.pt \
  --roi-target-model soma=/path/to/frozen_soma_model.pt
```

Or keep the mapping in a JSON registry:

```json
{
  "models": {
    "dendrite": "/path/to/frozen_cerebellar_dendrite_model.pt",
    "soma": "/path/to/frozen_soma_model.pt"
  }
}
```

Then pass the registry path:

```bash
python -m utils_2p.processing_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/processed_outputs \
  --target-structure dendrite \
  --run-roi-model-scores \
  --roi-model-registry /path/to/roi_model_registry.json
```

Model lookup order is: explicit `--roi-model-path`, repeated
`--roi-target-model target=/path/model.pt` entries, `--roi-model-registry` or
`TWO_P_ROI_MODEL_REGISTRY`, target-specific environment variables such as
`TWO_P_ROI_MODEL_DENDRITE`, then the fallback `TWO_P_ROI_MODEL_PATH`.

The model is reused when `roi_model_scores.h5` already exists. Add
`--force-roi-model-scores` only when the checkpoint, thresholds, or prediction
code changed and scores should be regenerated.

Suite2p's temporary binary movie is deleted when processing completes. Keeping
it in node-local `$TMPDIR` avoids writing a large intermediate file to project
or Cedar storage.

### Rerun Selected Stages

When upstream outputs already exist, use `--stages`. For example, regenerate
dF/F and the PDF/interactive summaries:

```bash
python -m utils_2p.processing_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/existing_processed_outputs \
  --target-structure soma \
  --stages dff,summary \
  --qos embers
```

The processed session must already be located at:

```text
/path/to/existing_processed_outputs/<raw-session-directory-name>/
```

Downstream-only runs assume all required upstream files are already present.

### Pipeline Outputs

For a graphical overview of how the main inputs and outputs move through each
stage, see the [processing pipeline documentation](processing-pipeline.md#full-processing-data-flow).

| Stage | Resource | Main outputs |
|---|---|---|
| `prep` | CPU | `raw_voltages.h5`, copied `bpod_session_data.mat` when available, provenance JSON |
| `suite2p` | High-memory CPU and optional GPU | `suite2p/plane0/ops.npy`, ROI statistics, fluorescence and neuropil traces, registered projections |
| `roi_model_scores` | CPU; must be specified with `--run-roi-model-scores` | `roi_model_scores.h5` with trained ROI model probabilities and labels; currently available only for cerebellar dendrite ROI scoring |
| `label` | GPU; must be specified with `--run-label` | `masks.h5` and anatomical Cellpose outputs |
| `dff` | CPU | `dff.h5` containing raw, non-z-scored dF/F traces computed from `suite2p/plane0/F.npy` and `Fneu.npy`; legacy `qc_results` traces are accepted only as a fallback |
| `spikes` | CPU; must be specified with `--run-oasis` | `spikes.h5` containing OASIS inferred spike amplitudes and event-threshold metadata |
| `summary` | CPU | `<session>_processing_summary.pdf`, `<session>_interactive_fov_roi_dff.html`; morphology, fluorescence, and inferred-spike metrics are calculated for the reviewer |

The pipeline reuses packaged Suite2p configuration files and packaged
postprocessing helpers from `utils_2p`. The older
`2p_post_process_module_202404` tree is retained as a fallback/reference copy,
but new sessions should keep the default Suite2p layout and use the interactive
HTML reviewer for ROI filtering. Existing legacy `qc_results/` inputs can still
be read as a fallback by downstream summary/loading code, but new pipeline runs
should not create separate QC directories.

## Running Locally

### Run Without Slurm

The `submit` command requires Slurm because it calls `sbatch`. On a local
workstation or other non-Slurm environment, use `generate` to write the same
manifest, then call each stage directly with `run-stage`. This is most useful
for testing one session or regenerating downstream outputs. Full Suite2p runs
can still be slow and memory-heavy on a laptop or desktop.

First install or activate an environment that has Suite2p and `utils_2p`
available. From a local repository checkout:

```bash
conda env create \
  --prefix ~/conda/envs/2p_processing_suite2p_1x \
  --file utils_2p/environment-processing-suite2p-1x.yml

conda activate ~/conda/envs/2p_processing_suite2p_1x
python -m pip install -e .
```

Then run the stages sequentially. This example skips anatomical labeling and
Suite2p GPU use, which is the safest default for CPU-only local testing:

```bash
#!/usr/bin/env bash
set -euo pipefail

RAW_SESSION="/path/to/raw/session"
OUTPUT_ROOT="/path/to/local/processed_outputs"
RUN_NAME="local_test"
USER_NAME="${USER:-$(id -un)}"
MANIFEST="${OUTPUT_ROOT}/.processing_jobs/${RUN_NAME}_${USER_NAME}/manifest.json"

python -m utils_2p.processing_pipeline generate \
  --session "${RAW_SESSION}" \
  --output-root "${OUTPUT_ROOT}" \
  --target-structure soma \
  --no-suite2p-gpu \
  --run-name "${RUN_NAME}"

for stage in prep suite2p dff summary; do
  python -m utils_2p.processing_pipeline run-stage \
    --manifest "${MANIFEST}" \
    --index 0 \
    --stage "${stage}"
done
```

For local Python-driven testing, the same workflow can be written without a
shell loop:

```python
from pathlib import Path
import sys

from utils_2p.processing_pipeline import (
    PipelineConfig,
    SessionSpec,
    generate_processing_jobs,
    run_stage,
)

raw_session = Path("/path/to/raw/session")
output_root = Path("/path/to/local/processed_outputs")

session = SessionSpec(
    raw_path=raw_session,
    target_structure="soma",
    run_label=False,
    stages=("prep", "suite2p", "dff", "summary"),
)

generated = generate_processing_jobs(
    [session],
    output_root,
    config=PipelineConfig(
        python_bin=sys.executable,
        suite2p_gpu=False,
    ),
    run_name="local_test",
)

for stage in ("prep", "suite2p", "dff", "summary"):
    run_stage(generated.manifest, index=0, stage=stage)
```

Use `run_label=True` or pass `--run-label` only when the session has an
anatomical channel and the local environment has the Cellpose/GPU setup needed
for labeling. Add `--run-oasis` and include `spikes` before `summary` when
local OASIS spike inference should be generated.

### Rebuild or Install the Environment

Most PACE users should use the shared environment shown above. Build a personal
environment only when the shared path is unavailable, different package versions
are required, or you are running outside PACE.

Suite2p 1.x is the default and recommended environment. The repository also
provides a legacy Suite2p 0.x environment for reproducing older processing.

From a repository checkout:

```bash
git clone https://github.com/najafi-laboratory/2p_imaging.git
cd 2p_imaging
git checkout main
git pull origin main

conda env create \
  --prefix ~/conda/envs/2p_processing_suite2p_1x \
  --file utils_2p/environment-processing-suite2p-1x.yml

conda activate ~/conda/envs/2p_processing_suite2p_1x
python -m pip install -e .
```

Without keeping a checkout, download the YAML first and install the package from
GitHub:

```bash
curl -L -o environment-processing-suite2p-1x.yml \
  https://raw.githubusercontent.com/najafi-laboratory/2p_imaging/main/utils_2p/environment-processing-suite2p-1x.yml

conda env create \
  --prefix ~/conda/envs/2p_processing_suite2p_1x \
  --file environment-processing-suite2p-1x.yml

conda activate ~/conda/envs/2p_processing_suite2p_1x
python -m pip install "git+https://github.com/najafi-laboratory/2p_imaging.git"
```

For legacy Suite2p 0.x, use
`utils_2p/environment-processing-suite2p-0x.yml` or download:

```bash
curl -L -o environment-processing-suite2p-0x.yml \
  https://raw.githubusercontent.com/najafi-laboratory/2p_imaging/main/utils_2p/environment-processing-suite2p-0x.yml
```

Activate the 0.x environment and use `--suite2p-version 0.x` only when
reproducing an older result. An explicit `--python-bin` is still available for
advanced debugging and takes precedence over `--suite2p-version`.
