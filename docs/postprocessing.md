# Postprocessing

In older versions of the data processing pipeline, **postprocessing** referred
to the steps run after Suite2p. In the current staged pipeline, this mostly
corresponds to `dff` and `summary`, plus the non-default `roi_model_scores`,
`label`, and `spikes` stages when those are explicitly enabled.

The current staged pipeline calls postprocessing code from the installable
`utils_2p` package. Functions that were originally maintained under
`2p_post_process_module_202404/` have been copied into `utils_2p` so they can
travel with the package and be found without a separate checkout of the legacy
postprocessing folder.

## Current entry points

- `utils_2p.dff_traces`
- `utils_2p.oasis_spikes`
- `utils_2p.roi_model_scores`
- `utils_2p.processing_summary`
- `utils_2p/resources/postprocess_modules/LabelExcInh.py`

The default resolver in `utils_2p.processing_pipeline` uses the packaged
`utils_2p/resources/postprocess_modules/` copies first. The older
`2p_post_process_module_202404/modules/` directory is retained in the repository
as a fallback and reference copy, and can still be selected explicitly with
`--postprocess-root` when reproducing older behavior.

## Current pipeline responsibilities

The current staged pipeline runs Suite2p first, then uses postprocessing
helpers for trace generation and summary/reviewer output. Optional
post-Suite2p stages can add model scoring, anatomical labeling, inferred
spikes. The current post-Suite2p stages are:

1. `dff.h5` computation from native Suite2p `F.npy` and `Fneu.npy`
2. Summary PDF and interactive HTML reviewer generation
3. Optional trained ROI model scoring via `utils_2p.roi_model_scores`, currently available only for cerebellar dendrite ROIs
4. Optional cross-channel ROI labeling via `LabelExcInh`
5. Optional OASIS inferred spike generation

Morphology, fluorescence trace, and inferred-spike metrics are calculated for
the HTML reviewer. Filters are applied interactively in the browser and can be
saved or exported. By default, the summary stage uses the original Suite2p ROI
layout. Separate `qc_results/` and `manual_qc_results/` directories are legacy
layouts and should not be created for new sessions.

The older top-level orchestration still lives in
`2p_post_process_module_202404/run_postprocess.py`, where `process_session()`
runs the legacy modules in sequence. This script is retained for reference and
older workflows, but it is not the default entry point for new staged pipeline
runs.

## `run_postprocess.py`

Important responsibilities:

- parse QC thresholds from the command line
- read `suite2p/plane0/ops.npy`
- reattach `save_path0` to the current session directory
- execute the full postprocessing workflow for one or more sessions

The README update notes indicate that the postprocessing layer has evolved over time to support:

- batch processing of session lists
- improved PMT or shutter artifact handling
- default smoothing during ΔF/F preparation
- separation of raw `dff` saving from downstream filtering

## Legacy `QualControlDataIO`

The packaged copy at
`utils_2p/resources/postprocess_modules/QualControlDataIO.py` is retained for
compatibility but is deprecated for new sessions. It reads Suite2p outputs,
computes ROI-level QC metrics, filters ROIs using the target-structure preset,
and saves cleaned results for older downstream modules. It is not part of the
current staged pipeline.

Saved artifacts include:

- `qc_results/fluo.npy`
- `qc_results/neuropil.npy`
- `qc_results/stat.npy`
- `qc_results/masks.npy`
- `qc_results/ops.npy`
- `move_offset.h5`

The QC metrics described in the code and README include skew, connectivity, aspect ratio, footprint, and compactness thresholds.

## `LabelExcInh`

The packaged copy at `utils_2p/resources/postprocess_modules/LabelExcInh.py`
handles channel-aware ROI labeling, especially for dual-channel recordings.

Major tasks:

- reconstruct functional ROI masks from native Suite2p `stat.npy` for new sessions
- load legacy `qc_results/masks.npy` when labeling older processed sessions
- optionally run Cellpose on the anatomical channel
- estimate bleedthrough from functional to anatomical channels
- compare overlap between functional and anatomical masks
- save labeled masks to `masks.h5`

For single-channel recordings, the module falls back to a simpler labeling path.

## `DffTraces`

The current utility converts fluorescence and neuropil signals into ΔF/F
traces. In the default staged pipeline, it reads native Suite2p
`suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy`. Legacy
`qc_results/fluo.npy` and `qc_results/neuropil.npy` files are still accepted
as a fallback for older sessions.

Major tasks:

- apply PMT or LED artifact handling where needed
- compute a baseline-normalized trace
- optionally normalize traces
- save the resulting data to `dff.h5`

The later experiment directories generally treat `dff.h5` as the main starting point for trialization and plotting.

## CLI example

```bash
python run_postprocess.py \
  --session_data_path /path/to/session \
  --range_skew -5,5 \
  --max_connect 1 \
  --range_aspect 1,1.35 \
  --range_footprint 1,2 \
  --range_compact 0,1.05 \
  --diameter 6
```
