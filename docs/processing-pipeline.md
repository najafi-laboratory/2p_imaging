# Processing Pipeline Documentation

This page documents the staged `utils_2p.processing_pipeline` workflow.
Use the [Processing Quickstart](processing-quickstart.md) when you want a
copy-paste command to run the pipeline. Use this page when you want to
understand what each job does, which function is invoked, and which files are
created for downstream review.

## Current stage order

For new sessions, the default pipeline is:

```text
prep -> suite2p -> dff -> summary
```

Additional stages must be specified explicitly:

```text
prep -> suite2p -> roi_model_scores -> label -> dff -> spikes -> summary
```

`spikes` must be specified with `--run-oasis`. `label` must be specified with
`--run-label`. `roi_model_scores` must be specified with
`--run-roi-model-scores`. The currently available ROI model score
checkpoint is only for cerebellar dendrite ROIs, so this stage should not be
treated as a general-purpose soma or non-cerebellar classifier yet.

Morphology threshold filtering is not applied as a pipeline stage. New sessions
should keep the Suite2p ROI set and use the interactive HTML reviewer for
filtering. If morphology filtering is added back later, it should happen in the
summary/reviewer generation path rather than by creating separate pre-filtered
QC directories.

## How to read the invocation boxes

Each box in the diagram shows a stack of calls from the pipeline launcher down
to the work function that actually creates files.

`run_stage(manifest, index, stage)` is the generic stage entry point. Every
generated Slurm script calls this same function with a `manifest.json`, a
session `index`, and a stage name such as `suite2p` or `dff`. The manifest
contains the resolved paths, target structure, selected stages, Python
environment, Slurm settings, and per-session configuration.

`_run_<stage>(data, session)` is the pipeline wrapper for one stage. For
example, `_run_suite2p(data, session)` reads the manifest, prepares Suite2p
settings, applies command-line overrides, and then calls Suite2p. These wrapper
functions are private implementation details of
`utils_2p.processing_pipeline`, but they are useful landmarks when
debugging a failed stage.

Library or module calls such as `suite2p.run_s2p(...)`,
`LabelExcInh.run(...)`, and `dff_traces.run(...)` are the functions that
perform the domain-specific work. They are where motion correction, ROI
detection, anatomical labeling, trace generation, and summary generation
actually happen.

The stage names in the diagram match the allowed `--stage` values for
`python -m utils_2p.processing_pipeline run-stage`.

## Full Processing Data Flow

The chart below shows the current data flow when all staged preprocessing and
reviewer-output steps are enabled. It separates the high-level stages from the
main functions, input files, generated outputs, and downstream reviewer
exports. The interactive HTML reviewer can apply browser-side morphology,
fluorescence, ROI model score, inferred-spike, and manual-label filters. New
pipeline runs do not create `qc_results/` or `manual_qc_results/` directories.

<div class="overview-flowchart pipeline-dataflow">
  <img src="../images/workflow/preprocessing-pipeline-dataflow.svg" alt="Detailed processing pipeline data flow" />
</div>

The rendered diagram above is stored as
`docs/images/workflow/preprocessing-pipeline-dataflow.svg`. The editable
Mermaid source is stored beside it at
`docs/images/workflow/preprocessing-pipeline-dataflow.mmd`.

## Stage reference

| Stage | When it runs | Main work | Main outputs |
|---|---|---|---|
| `prep` | Default | Standardizes non-imaging session inputs and writes session metadata. | `raw_voltages.h5`, `bpod_session_data.mat` when available, `processing_pipeline_parameters.json` |
| `suite2p` | Default | Runs Suite2p registration, ROI detection, and fluorescence extraction. | `suite2p/plane0/ops.npy`, `stat.npy`, `F.npy`, `Fneu.npy`, `iscell.npy`, `spks.npy` |
| `roi_model_scores` | Must be specified with `--run-roi-model-scores`; currently available only for cerebellar dendrite ROIs | Applies a trained ROI classifier and records probability/state metadata. | `roi_model_scores.h5`, and `ROI_label.h5` when labels are generated |
| `label` | Must be specified with `--run-label` | Runs anatomical/functional cell-type labeling through `LabelExcInh`. | `masks.h5` |
| `dff` | Default after `suite2p` | Computes raw non-z-scored dF/F from native Suite2p fluorescence and neuropil traces. | `dff.h5` |
| `spikes` | Must be specified with `--run-oasis` | Runs OASIS/Suite2p inferred spike generation. | `spikes.h5` |
| `summary` | Default | Builds the PDF summary and interactive ROI reviewer. | `<session>_processing_summary.pdf`, `<session>_interactive_fov_roi_dff.html` |
