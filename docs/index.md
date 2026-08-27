# PACE / Phoenix Compute

## Basics

PACE is Georgia Tech's high-performance computing service. Phoenix is one of
the PACE clusters used for research compute. This guide is meant to save time
spent on trial-and-error by providing a basic understanding of where to store
different files for different purposes, how to use standardized lab Conda/Python environments,
and efficiently run different kinds of data-processing or analysis tasks. This will make
standard tasks like running the lab 2p processing pipeline and creating / adding
on your own extensions more intuitive.

## Prerequisities for this Guide

To access PACE Phoenix you need

1. A PACE account, which Dr.Najafi can request for you.

2. Access to the Georgia Tech VPN client, see the following link to download the official GlobalProtect VPN client:
[https://vpn.gatech.edu/global-protect/getsoftwarepage.esp](https://vpn.gatech.edu/global-protect/getsoftwarepage.esp)

To follow this guide, you should be comfortable with:

  - Using bash in the terminal to navigate a Linux filesystem (commands **cd**, **ls**, **mkdir** etc...)
  - Experience with an IDE like VS Code will make everything easier but not strictly necessary
  - Using a Georgia Tech account, including your GT username, password, Duo, and VPN access when off campus.
  - Can run Python scripts, and a basic understanding of the use/purpose of package management tools like Conda.
  - High level understanding that different tasks run faster on differ architectures (CPU vs. GPU)

### 1. Login nodes

This is the node that you use to connect to the cluster, submit jobs to compute
nodes, and check the status of your jobs. You can use this node to edit
scripts, inspect small files, and submit jobs that are run on compute nodes.


You should **not** use this node for long Suite2p runs, large file conversions,
or other heavy 2p processing (it will run out of memory quickly and become non-responsive).
You can also use an IDE with SSH support, such as VS Code, to connect and edit
files on the login node.

SSH to the following address to connect:

```bash
ssh <gt-username>@login-phoenix.pace.gatech.edu
```

Replace `<gt-username>` with your Georgia Tech username. If you are off campus,
connect to the GT VPN first, then run the SSH command and enter your password when prompted.

### 2. Compute nodes

Compute nodes have more resources and are where analysis and data processing jobs actually run.
Phoenix has multiple classes of compute resources for different workloads, such
as smaller CPU nodes, larger-memory CPU nodes, and GPUs. While our pipeline
and `utils_2p` functions largely handle compute-node selection automatically,
see [Phoenix Compute Node Resources](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0041976)
for reference on the available resources.

There are two main kinds of compute jobs, interactive and non-interactive,
both of which can be launched from a  terminal or a login node, or from
a browser connected to [Phoenix OnDemand dashboard](https://ondemand-phoenix.pace.gatech.edu/pun/sys/dashboard)
is the web home for starting these browser-based sessions, and it requires a GT
VPN connection.

1.**Non-interactive (sbatch)**: Batch jobs are submitted from the command line with `sbatch`,
which launch a request for a compute node and automatically run a specified script
when resources become available.

```
  #!/usr/bin/env bash
  #SBATCH --job-name=hello-world
  #SBATCH --account=gts-fnajafi3
  #SBATCH --qos=embers
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=1
  #SBATCH --mem=1G
  #SBATCH --time=00:05:00
  #SBATCH --output=hello-world_%j.out
  #SBATCH --error=hello-world_%j.err

  echo "Hello from Slurm"
  echo "Job ID: ${SLURM_JOB_ID}"
  echo "Running on node: $(hostname)"
  echo "Started at: $(date)"
```

  Submit it with:

  `sbatch hello_world.sbatch`

  Check whether it is queued or running:

  `squeue -u "$USER"`

  (replace $USER with your gt username if not already set as an env variable)

  After it finishes, inspect the output:

  `cat hello-world_<jobid>.out`

  Replace <jobid> with the job ID printed by sbatch.

2.**salloc**: Interactive jobs request a compute node for direct use through
`salloc` or through the Phoenix OnDemand web interface.

   - **Through the terminal*** - salloc in the terminal will print

The [Phoenix OnDemand dashboard](https://ondemand-phoenix.pace.gatech.edu/pun/sys/dashboard)
is the web home for starting these browser-based sessions, and it requires a GT
VPN connection. OnDemand is especially useful for workflows that need a visual
interface, such as manual ROI labeling: the GUI runs near the data on PACE, so
the user does not have to download an entire imaging session to a local machine
just to inspect or edit ROIs.

Useful references:

- [PACE home page](https://pace.gatech.edu/)
- [Phoenix OnDemand dashboard](https://ondemand-phoenix.pace.gatech.edu/pun/sys/dashboard)
- [Phoenix OnDemand documentation](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042133)
- [PACE service overview](https://oit.gatech.edu/oit-spotlight-partnership-advanced-computing-environment-pace)

## Storage locations

The same session may move through several storage systems during its lifetime.
Use each location for the job it is good at.

| Location | Typical path | Purpose |
|---|---|---|
| Home | `/home/<user>` or `~` | Shell configuration, small scripts, small text files. Do not store imaging datasets here. |
| CEDAR | `/storage/cedar/...` | Long-term, durable research data storage. Use this for archived original recordings and retained results that should not be purged. It is not the preferred place to run initial high-throughput processing from. |
| Project storage | `/storage/project/r-fnajafi3-0/...` | Shared lab/project storage for software, shared environments, active shared raw-session uploads, and active shared outputs. Good for common resources that multiple users need. |
| Scratch | `~/scratch/...` | Short-term high-throughput job workspace. Use this for large temporary processing runs and staged raw sessions. Scratch may be purged, so it is not a backup. |

For 2p processing, the preferred workflow is to do active processing from
project storage or scratch, then archive to CEDAR after the outputs have been
validated. Avoid launching large initial processing batches that repeatedly
read TIFF stacks directly from CEDAR when a working copy is available in
project storage or can be staged to scratch.

For a typical session:

1. Upload or keep the active raw session in shared project storage, or stage a
   working copy to scratch for large batches.
2. Run the processing pipeline using project storage or scratch as the input
   location.
3. Write temporary, intermediate, and first-pass processed outputs to scratch.
4. Validate and QC the processed outputs.
5. Copy retained raw data and final processed outputs to CEDAR for long-term
   record keeping, and keep any active shared results in project storage as
   needed.

This keeps heavy read/write activity on filesystems intended for active
compute work and reserves CEDAR for durable archival storage.

## Shared environments and Conda modules

The shared Suite2p 1.x processing environment is installed here:

```bash
/storage/project/r-fnajafi3-0/shared/shared_envs/2p_processing_suite2p_1x
```

It has `utils_2p` installed as a package, so most users do not need a local
repository checkout to run the processing pipeline.

On PACE, Conda is usually made available through the module system:

```bash
module avail anaconda
module load anaconda3/2023.03
conda env list
```

Use the shared environment directly when possible:

```bash
module load anaconda3/2023.03
conda activate /storage/project/r-fnajafi3-0/shared/shared_envs/2p_processing_suite2p_1x

python -c "import utils_2p; print(utils_2p.__file__)"
```

Create a personal Conda environment only when you need to test package changes,
install a different version, or work outside PACE. See the
[Processing Quickstart](processing-quickstart.md#rebuild-or-install-the-environment)
for the environment YAML workflow.

## Common Slurm commands

PACE jobs are submitted through Slurm. The commands below are the ones most
often used while running this repository's preprocessing jobs.

| Command | Purpose |
|---|---|
| `sbatch script.sbatch` | Submit a batch job script. The pipeline's `submit_jobs.sh` uses this internally. |
| `squeue -u "$USER"` | Show your queued and running jobs. |
| `sacct -j <jobid>` | Inspect completed job accounting and exit status. |
| `scancel <jobid>` | Cancel a queued or running job. |
| `salloc ...` | Request an interactive allocation on a compute node. Useful for debugging, not for large unattended batches. |

Examples:

```bash
squeue -u "$USER"

sbatch --account=gts-fnajafi3 --qos=embers my_job.sbatch

salloc --account=gts-fnajafi3 --qos=embers --cpus-per-task=4 --mem=32G --time=02:00:00
```

## QOS choices

The lab examples usually use:

- `embers`: preemptible QOS. Preemption means your job can be killed after
  about an hour if those resources are needed elsewhere, so use it for short
  jobs, jobs that are easy to rerun, or jobs where you do not care if they run
  longer than that and get stopped.
- `inferno`: paid, non-preemptible QOS. Use this when a long or expensive job
  should not be interrupted and the allocation supports it.

The preprocessing launcher accepts QOS arguments:

```bash
--qos embers
--qos-cpu embers
--qos-gpu embers
```

Use separate CPU/GPU QOS settings only when the cluster allocation or policy
requires it.

## Globus for file transfers

Use Globus for large transfers between local machines, CEDAR, project storage,
scratch, and other endpoints. It is more appropriate than browser upload or
ad-hoc `scp` for large imaging sessions because it supports managed transfers,
recursive directory copies, restart behavior, and task monitoring.

Useful references:

- [Globus CLI documentation](https://docs.globus.org/cli/)
- [Globus transfer command reference](https://docs.globus.org/cli/reference/transfer/)
- [PACE service overview, including Globus for file transfers](https://oit.gatech.edu/oit-spotlight-partnership-advanced-computing-environment-pace)

Install the Globus CLI with `pipx` when possible. On PACE, this installs into
your home directory, usually under `~/.local/`, and does not require modifying
the shared Conda environments:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
pipx install globus-cli
globus login
```

On a local laptop or workstation, the same `pipx install globus-cli` workflow
is preferred. If `pipx` is not available, install into an activated Python or
Conda environment:

```bash
python -m pip install globus-cli
globus login
```

For a recursive transfer, the command structure is:

```bash
globus transfer SOURCE_ENDPOINT:/path/to/source/ DEST_ENDPOINT:/path/to/destination/ --recursive
globus task list
globus task show <task-id>
```

Use Globus for bulk data movement, then run compute jobs on the filesystem
where the data has been staged.
