# Processing Quickstart

This guide covers the complete staged preprocessing and QC pipeline in
`utils_2p.preprocessing_qc_pipeline`, including environment installation,
single-session submission, and multi-session submission.

The full pipeline is:

```text
prep -> suite2p -> qc -> label -> dff -> summary
```

The launcher generates linked Slurm jobs and is designed to run on PACE.

## Get the repository

Run the launcher from the root of a current `2p_imaging` checkout:

```bash
git clone https://github.com/najafi-laboratory/2p_imaging.git
cd 2p_imaging
git checkout main
git pull origin main
```

For an existing laboratory checkout configured with an `upstream` remote, use
`git pull upstream main`.

## Companion notebook

The
[`preprocessing_pipeline_quickstart.ipynb`](https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/preprocessing_pipeline_quickstart.ipynb)
notebook provides editable `SINGLE` versus `BATCH` variables. It builds the
corresponding PACE pipeline command, creates a sessions file for batch mode,
and keeps command execution disabled until `RUN_COMMAND = True`.

## Launch one session on PACE

Users with read and execute access can launch directly with the shared Suite2p
1.x Python. They do not need to build or activate a personal Conda
environment:

```bash
export TWO_P_PYTHON=/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_suite2p_1x/bin/python
export TWO_P_SLURM_ACCOUNT=gts-fnajafi3

"$TWO_P_PYTHON" -c "from importlib.metadata import version; print(version('suite2p'))"

cd /path/to/2p_imaging

"$TWO_P_PYTHON" -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session \
  --output-root "/storage/scratch1/3/$USER/2p_processing_results" \
  --target-structure neuron \
  --suite2p-version 1.x \
  --python-bin "$TWO_P_PYTHON" \
  --account "$TWO_P_SLURM_ACCOUNT" \
  --qos embers \
  --run-name example_neuron_session
```

Argument meanings:

| Argument | Meaning |
|---|---|
| `submit` | Generate the Slurm files and immediately submit all requested stages. |
| `--session` | Raw session directory containing the imaging TIFF files and associated session inputs. |
| `--output-root` | Parent directory where a processed directory named after the raw session will be created. |
| `--target-structure` | Morphology QC preset: `neuron`, `dendrite`, or `cerebellum_lax`. |
| `--suite2p-version` | Select the default versioned environment when `--python-bin` is not supplied. |
| `--python-bin` | Exact Python executable used inside every generated job. |
| `--account` | Slurm allocation charged for the jobs. |
| `--qos` | Slurm QOS for all stages. `embers` is preemptible; use `inferno` when paid, non-preemptible execution is required. |
| `--run-name` | Readable name for the generated job directory and provenance files. |

## PACE storage and job-submission guidance

Run the processing jobs on PACE compute nodes, not on a login node. For
multi-session processing, use PACE scratch for staged input data, the
`--output-root`, and Suite2p temporary files whenever practical:

```text
/storage/scratch1/3/<username>/
├── staged_raw_sessions/
└── 2p_processing_results/
```

Cedar and project storage are appropriate for durable source data and final
results, but they are shared network filesystems. Suite2p repeatedly reads
large TIFF stacks and writes a large binary movie. Many simultaneous sessions
performing those operations against Cedar can saturate shared read bandwidth,
trigger metadata or I/O throttling, and make every job slower. Scratch is
designed for high-throughput temporary job I/O and is usually a better working
location.

A practical workflow is:

1. Copy or stage the raw sessions needed for the current batch from Cedar to
   scratch.
2. Run the pipeline with the staged scratch paths and a scratch
   `--output-root`.
3. Validate the final outputs.
4. Copy the retained processed results back to durable project or Cedar
   storage.

For example:

```bash
mkdir -p "/storage/scratch1/3/$USER/staged_raw_sessions"
mkdir -p "/storage/scratch1/3/$USER/2p_processing_results"

rsync -a --info=progress2 \
  /storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging/path/to/session/ \
  "/storage/scratch1/3/$USER/staged_raw_sessions/session/"
```

After validating the processed session, copy it to its durable destination:

```bash
rsync -a --info=progress2 \
  "/storage/scratch1/3/$USER/2p_processing_results/session/" \
  /storage/project/path/to/processed_results/session/
```

Scratch is temporary, may be purged according to current PACE policy, and is
not a backup. Do not remove the durable source or only copy of a result until
the transfer back has been verified.

Each session creates up to six Slurm jobs. Submitting 50 sessions at once can
therefore create roughly 300 queued jobs, while several Suite2p stages may
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

For unusually large recordings, use fewer concurrent sessions. The built-in
`--sessions-file` submits one independent chain per listed session; it does not
throttle the number of active session chains.

Channel count and functional channel are normally inferred from TIFF names.
For a functional-only, single-channel dendrite session, specify the overrides
and skip anatomical labeling:

```bash
"$TWO_P_PYTHON" -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/single_channel_session \
  --output-root /path/to/processed_outputs \
  --target-structure dendrite \
  --nchannels 1 \
  --functional-chan 1 \
  --no-label \
  --python-bin "$TWO_P_PYTHON" \
  --account gts-fnajafi3 \
  --qos embers
```

Additional argument meanings:

| Argument | Meaning |
|---|---|
| `--nchannels 1` | Override automatic channel detection and treat the recording as single-channel. |
| `--functional-chan 1` | Use channel 1 as the calcium-imaging channel. |
| `--no-label` | Skip anatomical Cellpose labeling when no anatomical channel is available or labeling is not wanted. |

Suite2p requests a GPU by default. Add `--no-suite2p-gpu` to run Suite2p on
CPU-only resources. The anatomical `label` stage still requires a GPU when it
is enabled.

## Rebuild the environment if needed

Most PACE users should use the shared Python shown above. Build a personal
environment only when the shared path is unavailable or different package
versions are required.

Suite2p 1.x is the default and recommended environment. The repository also
provides a legacy Suite2p 0.x environment for reproducing older processing.

Make Conda available:

```bash
module load anaconda3/2023.03
```

From the repository checkout:

```bash
conda env create \
  --prefix ~/conda/envs/2p_preprocessing_qc_suite2p_1x \
  --file utils_2p/environment-preprocessing-qc-suite2p-1x.yml

export TWO_P_PYTHON=~/conda/envs/2p_preprocessing_qc_suite2p_1x/bin/python
```

Without a checkout, download the YAML first:

```bash
curl -L -o environment-preprocessing-qc-suite2p-1x.yml \
  https://raw.githubusercontent.com/najafi-laboratory/2p_imaging/main/utils_2p/environment-preprocessing-qc-suite2p-1x.yml

conda env create \
  --prefix ~/conda/envs/2p_preprocessing_qc_suite2p_1x \
  --file environment-preprocessing-qc-suite2p-1x.yml
```

For legacy Suite2p 0.x, use
`utils_2p/environment-preprocessing-qc-suite2p-0x.yml` or download:

```bash
curl -L -o environment-preprocessing-qc-suite2p-0x.yml \
  https://raw.githubusercontent.com/najafi-laboratory/2p_imaging/main/utils_2p/environment-preprocessing-qc-suite2p-0x.yml
```

Use `--suite2p-version 0.x` with the 0.x Python only when reproducing an older
result. An explicit `--python-bin` takes precedence over
`--suite2p-version`.

## Launch multiple sessions

### Repeat `--session`

For a small batch with the same processing settings, repeat `--session`:

```bash
"$TWO_P_PYTHON" -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session_1 \
  --session /path/to/raw/session_2 \
  --session /path/to/raw/session_3 \
  --output-root /path/to/processed_outputs \
  --target-structure neuron \
  --python-bin "$TWO_P_PYTHON" \
  --account gts-fnajafi3 \
  --qos embers \
  --run-name neuron_batch
```

Each session receives its own linked stage chain. A failed session does not
block the other sessions.

### Use a sessions file

For a larger batch, create a plain-text file with one raw session path per
line. Blank lines and lines beginning with `#` are ignored:

```text
# neuron_sessions.txt
/path/to/raw/session_1
/path/to/raw/session_2
/path/to/raw/session_3
```

Submit all paths in the file:

```bash
"$TWO_P_PYTHON" -m utils_2p.preprocessing_qc_pipeline submit \
  --sessions-file neuron_sessions.txt \
  --output-root /path/to/processed_outputs \
  --target-structure neuron \
  --python-bin "$TWO_P_PYTHON" \
  --account gts-fnajafi3 \
  --qos embers \
  --run-name neuron_manifest
```

All entries in one `--sessions-file` invocation share the command-line
settings. Use separate files or separate invocations when sessions require
different target structures, channel overrides, or stage selections.
Keep each file to a controlled batch size rather than putting an entire large
dataset into one submission.

The launcher writes its resolved JSON manifest, stage `.sbatch` files, logs,
and submission script below:

```text
<output-root>/.preprocessing_qc_jobs/<run-name>_<username>/
├── manifest.json
├── prep.sbatch
├── suite2p.sbatch
├── qc.sbatch
├── label.sbatch
├── dff.sbatch
├── summary.sbatch
├── submit_jobs.sh
└── logs/
```

Only stages used by at least one session are written.

## Generate jobs without submitting

Use `generate` to validate inputs and inspect the manifest and `.sbatch` files
before submitting:

```bash
"$TWO_P_PYTHON" -m utils_2p.preprocessing_qc_pipeline generate \
  --sessions-file neuron_sessions.txt \
  --output-root /path/to/processed_outputs \
  --target-structure neuron \
  --python-bin "$TWO_P_PYTHON" \
  --account gts-fnajafi3 \
  --qos embers \
  --run-name neuron_manifest
```

The command prints the generated job directory and the corresponding
`submit_jobs.sh` path. On PACE, submit the generated chains with:

```bash
bash /path/to/processed_outputs/.preprocessing_qc_jobs/neuron_manifest_${USER}/submit_jobs.sh
```

Run both `generate` and the resulting submission script on PACE so all
repository, Python, session, and output paths are accessible to the compute
nodes.

## Important optional arguments

| Argument | Meaning |
|---|---|
| `--stages prep,suite2p,qc,label,dff,summary` | Run only the listed stages; they are reordered into pipeline dependency order automatically. |
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

Suite2p's temporary binary movie is deleted when processing completes. Keeping
it in node-local `$TMPDIR` avoids writing a large intermediate file to project
or Cedar storage.

## Rerun selected stages

When upstream outputs already exist, use `--stages`. For example, regenerate
dF/F and the PDF/interactive summaries:

```bash
"$TWO_P_PYTHON" -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/existing_processed_outputs \
  --target-structure neuron \
  --stages dff,summary \
  --python-bin "$TWO_P_PYTHON" \
  --account gts-fnajafi3 \
  --qos embers
```

The processed session must already be located at:

```text
/path/to/existing_processed_outputs/<raw-session-directory-name>/
```

Downstream-only runs assume all required upstream files are already present.

## Pipeline outputs

| Stage | Resource | Main outputs |
|---|---|---|
| `prep` | CPU | `raw_voltages.h5`, copied `bpod_session_data.mat` when available, provenance JSON |
| `suite2p` | High-memory CPU and optional GPU | `suite2p/plane0/ops.npy`, ROI statistics, fluorescence and neuropil traces, registered projections |
| `qc` | CPU | `qc_results/fluo.npy`, `neuropil.npy`, `stat.npy`, `masks.npy`, `qc_parameters.json`, `move_offset.h5` |
| `label` | GPU | `masks.h5` and anatomical Cellpose outputs; skipped for functional-only recordings |
| `dff` | CPU | `dff.h5` containing raw, non-z-scored dF/F traces |
| `summary` | CPU | `<session>_preprocessing_summary.pdf`, `<session>_interactive_fov_roi_dff.html` |

The pipeline reuses the Suite2p configuration files in
`2p_processing_pipeline_202401` and the QC/label algorithms in
`2p_post_process_module_202404`.
