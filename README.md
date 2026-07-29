# Cross-Session ROI Tracking

This module tracks the same neurons across imaging sessions. It takes the Suite2p outputs for every session of a mouse, registers the fields of view onto a common frame, and clusters ROIs so that each neuron receives a single **UCID** (unique cluster ID) valid across all sessions. The tracking itself is done by [ROICaT](https://github.com/RichieHakim/ROICaT); what lives here is the lab's driver notebook, a session-screening helper, and a QC layer for eyeballing individual UCIDs before trusting them downstream.

## Attribution and license

`interactive_tracking.ipynb` is adapted from the ROICaT project by Rich Hakim (<https://github.com/RichieHakim/ROICaT>), licensed under **GPL-3.0**. Redistribution of this notebook — or of a larger work containing it — must preserve the copyright and license notices and apply GPL-3.0 to derivative works. See the repo-root `LICENSE`.

## Contents

| file | role |
| --- | --- |
| `interactive_tracking.ipynb` | The pipeline itself. Step-by-step, parameter-tunable, with a visualization after nearly every step. This is what you run. |
| `pipeline.py` | `filter_sessions_by_overlap()` — screens sessions for co-registerability before the real run, so poorly-overlapping sessions never enter the pipeline. |
| `roi_tracking_qc.py` | Per-UCID cross-session QC figures, exportable as a multipage PDF or a self-contained HTML viewer with a UCID picker. |

## Why the session-overlap filter exists

ROICaT aligns every session to a common template. If one session's FOV barely overlaps the others — a re-mount, a large stage drift, a different depth — the alignment for that session fails, and the failure contaminates the whole run rather than staying local to the bad session.

The stock notebook handles this with a manual `keep = [...]` list: run the aligner, read the alignment-score plot, decide by eye which sessions to drop, then re-run the data-loading cell with a subset. `pipeline.filter_sessions_by_overlap()` replaces that loop with a single call:

1. Load all sessions into a `Data_suite2p`.
2. Run a **silent geometric-only screening pass** (DISK_LightGlue, affine, CPU) purely to obtain the all-to-all alignment matrix. Nothing from this pass is reused.
3. Symmetrize the boolean alignment matrix, take its connected components, and keep the **largest** co-registerable group.
4. Rebuild `Data_suite2p` from only the kept sessions (skipped if nothing was dropped).

It returns `(data, keep)`, where `keep` indexes back into the original path lists — needed later for lining up per-session metadata (stim type, dates) with the filtered session order.

```
Session filter: 8/10 sessions kept
  keep  [0]  SA11_20250806
  ...
  drop  [4]  SA11_20250819  (poor overlap)
```

The screening pass costs one extra geometric fit. `z_threshold` (default 4.0) is the knob: higher is more stringent and will drop more sessions.

## Running the notebook

### Setup

```bash
conda env create -f ../environment.yml
conda activate 2p_postprocessing
jupyter lab
```

ROICaT (`roicat[all]`) comes in via pip in that environment file. The ROInet weights are downloaded on first run (`download_method='check_local_first'`) into the system temp directory.

Run the notebook from inside this directory — `import pipeline` and `import roi_tracking_qc` both resolve relative to the notebook's location.

**macOS note.** The first code cell sets `NUMBA_THREADING_LAYER=workqueue`, single-threaded `OMP`/`MKL`, and `KMP_DUPLICATE_LIB_OK=TRUE` *before* numpy/numba/roicat are imported, to avoid an Intel OpenMP crash. Keep that cell first and don't import anything above it.

### 1. Paths and session discovery

Point `dir_allOuterFolders` at the batch directory, e.g.

```
/Volumes/Elements/Najafi/2P_Imaging/SA11_LG/batches/sessions_01-10
```

The notebook globs for `stat.npy` at depth ≤ 10, drops anything under `EXCLUDE_DIRS` (`suite2p`, `qc_results` — the latter contains copies that would otherwise be treated as extra sessions), derives each `ops.npy` as `stat.npy`'s grandparent sibling, and asserts both files exist before handing anything to ROICaT.

A helper cell reads `bpod_session_data.mat` next to each session and tags it `VG` / `ST` / `unknown` from `SessionData.TrialSettings[0].GUI.SelfTimedMode`. These tags are cosmetic — they only feed panel titles in the QC figures and contact sheets — but they make it obvious at a glance when a batch mixes task types.

`um_per_pixel` is the one genuinely important parameter at this stage: a scalar, or a per-session list if resolution differs.

### 2. Alignment

The most important step, and the one worth stopping at. Four sub-steps:

1. **FOV augmentation** — blends the mean FOV with the ROI max-projection (`roi_FOV_mixing_factor=0.5`) and applies CLAHE. Turn CLAHE off for poor-quality or badly-drifting data.
2. **Geometric fit** — `DISK_LightGlue`, `constraint='affine'`, sequential templating (good for data that drifts across sessions). `RoMa` is more accurate but very slow on CPU; `LoFTR` and `ECC_cv2` sit in between. Check `plot_alignment_results_geometric()` before moving on.
3. **Non-rigid fit** — `DeepFlow` on top of the geometrically registered images, aligned to a single template image. Good in the middle of the FOV, weaker at the edges.
4. **Transform ROIs** — warps the spatial footprints through `remappingIdx_nonrigid`.

Then flip through the four image stacks (pre-alignment → geometric → non-rigid → transformed ROIs). If the aligned FOVs don't look aligned here, nothing downstream will save the run.

### 3. Embeddings and similarity

- **Blurring** (`kernel_halfWidth=4`) so that ROIs from different sessions with zero literal pixel overlap can still be matched.
- **ROInet** — pretrained network embedding of each cropped ROI image. ~15 min on CPU for ~40k ROIs, ~1 min on GPU. Check that a neuron fills roughly 25–50% of its cropped image.
- **Scattering wavelet transform** — a second, hand-designed appearance embedding (`J=2, L=12`).
- **Similarity graph** — blockwise (128×128 px blocks) pairwise similarities: `s_sf` (spatial-footprint overlap), `s_NN` (ROInet), `s_SWT` (wavelet), `s_sesh` (same-session mask). Then normalized against a local neighborhood distribution.

### 4. Clustering

The similarity matrices are mixed into a single conjunctive distance matrix, pruned at the estimated 50%-probability cross-over distance, and clustered. Mixing parameters come from `find_optimal_parameters_for_pruning()` by default; a commented-out manual block is available if the automatic fit misbehaves.

**Check the mixing plots.** You want a bimodal pairwise-distance distribution with a clean cross-over between the "same neuron" and "different neuron" modes. No bimodality means the run is not going to produce trustworthy clusters, whatever the downstream metrics say.

The clustering method switches on session count:

- **≥ 6 sessions** → `clusterer.fit()` (HDBSCAN, `min_cluster_size=2`, `rescue_noise=True`).
- **< 6 sessions** → `clusterer.fit_sequentialHungarian()` (`thresh_cost=0.8`). HDBSCAN needs enough sessions for density estimation to mean anything; below that the Hungarian matching is more reliable.

`compute_quality_metrics()` then produces per-cluster and per-ROI scores. **Skip it on very large datasets** — it is the slowest step. `cluster_silhouette` (`cs_sil`) is the one the QC layer sorts on.

### 5. Saving

Four artifacts land in `dir_save`, named from `name_save`:

| file | contents |
| --- | --- |
| `{name}.tracking.results_clusters.json` | labels, `labels_bySession`, `labels_dict`, quality metrics |
| `{name}.tracking.params_used.json` | every module's `params` dict |
| `{name}.tracking.results_all.richfile.zip` | clusters + aligned/raw ROI footprints + frame shape + input paths |
| `{name}.tracking.run_data.richfile.zip` | the full `__dict__` of every pipeline object (data, aligner, blurrer, roinet, swt, sim, clusterer) |

The split matters for QC: **`results_all` + `run_data` together are sufficient to rebuild every QC figure in a fresh kernel**, without re-running the pipeline. `run_data` carries the FOV images and the non-rigid remapping indices; `results_all` carries the labels and footprints.

## QC: `roi_tracking_qc.py`

Cluster-level metrics tell you the distribution is healthy; they don't tell you whether UCID 417 is actually the same neuron in all eight sessions. This module builds one figure per UCID so you can decide that by eye.

### Figure layout

Rows, top to bottom:

| row | contents |
| --- | --- |
| raw FOV | unregistered FOV per session (only when `fovs_raw` is supplied) |
| aligned FOV | non-rigidly registered FOV per session |
| zoom | ±`crop_halfwidth` px crop around the consensus centroid |

Columns, left to right: a superimposed projection across all sessions (`mean` or `max`), then one panel per session.

### Visual conventions

- **Red contour** — the ROI footprint, drawn at half-max on a σ = 1.5 px Gaussian-smoothed copy. The smoothing is deliberate: non-rigid warping of sparse Suite2p masks leaves small disconnected fragments, and contouring them raw produces a scatter of disjoint segments instead of one closed outline.
- **Dashed yellow box** — where the zoom row sits, drawn on both full-FOV rows. On the raw row each session gets its own box at that session's *pre-alignment* centroid, so you can see how far the ROI moved before registration.
- **Cyan contour** (raw row) — the boundary of valid tissue, i.e. where this session's aligned frame actually lands in raw coordinates. Computed by detecting tissue in the aligned image via local variance (robust to the uniform gray that `cv2.remap` fills out-of-bounds pixels with), pushing every tissue pixel through `remappingIdx_nonrigid` into raw coordinates, then consolidating the result — dilate, close, fill holes, keep the largest connected component — so the outline is one clean perimeter rather than speckle. Falls back to a centered mask when `remapping_idxs` isn't supplied.
- **`n/d`** — this session has no ROI for this UCID. Nothing is drawn.
- Suptitle carries mouse name, UCID, `detected k/n sessions`, and `cs_sil`.

### What to look for

A good UCID has its red contour landing on the same cell body in every session, with consistent size and shape, and detection in most or all sessions. Warning signs: contour drifting onto a neighboring soma between sessions; wildly varying footprint size; a contour sitting outside the cyan tissue boundary on the raw row (that session's ROI is in a region the alignment couldn't validate); low `detected k/n` on a cluster you expected to be stable.

### Exporters

```python
import roi_tracking_qc as qc

order = qc.order_ucids_by_quality(labels_bySession, cs_sil, ascending=True)  # worst-first

qc.export_html(
    "out_tracking.html", order[:200],
    fovs_aligned, rois_aligned, labels_bySession, H, W,
    cs_sil=cs_sil, crop_halfwidth=40,
    fovs_raw=fovs_raw, rois_raw=rois_raw,
    remapping_idxs=remapping_idxs,
    mouse_name=mouse_name, session_names=session_names,
)
```

- `export_html(path, ...)` — one self-contained HTML file: a dropdown, prev/next buttons, and every figure pre-rendered and base64-embedded. No server, no dependencies; hand the file to anyone. Because every PNG is inlined, there is a `max_ucids=400` safety cap — pass a worst-first slice rather than all clusters.
- `export_pdf(path, ...)` — one multipage PDF, one UCID per page. No cap, but no navigation either.
- `order_ucids_by_quality(labels_bySession, cs_sil, ascending=True)` — worst-first by silhouette score, so the first pages of the export are the clusters most likely to be wrong. Unclustered ROIs (label −1) are dropped by default; UCIDs with no score sort last.

Both exporters share `build_ucid_figure()`, so PDF and HTML pages are identical apart from DPI (110 vs. 90).

The module forces matplotlib's `Agg` backend at import, before `pyplot` is imported — it is intended to run headless, and importing it will override an interactive backend in the same kernel.

### Running QC from saved results

The QC cell near the end of the notebook reloads from the richfiles rather than reading live pipeline objects, so it works in a fresh kernel:

```python
_results_all = roicat.util.RichFile_ROICaT(path=paths_save['results_all']).load()
_run_data    = roicat.util.RichFile_ROICaT(path=paths_save['run_data']).load()
```

It needs `paths_save`, `dir_save`, `name_save`, `dir_allOuterFolders`, and `get_stim_type` in scope — run the paths and save cells above it, or set those five by hand.

### Other QC outputs

- **`{name}_FOVs_for_matching.png`** — one-row contact sheet of the raw mean FOV for each session that survived the overlap filter, titled with date and stim type.
- **All-sessions contact sheet** — the same idea over the *pre-filter* session list, with dropped sessions dimmed to 35% alpha. This is the fastest way to see *why* a session was dropped.
- **`FOV_clusters_highQuality.gif`** — the color-coded cluster FOV animated across sessions (`compute_colored_FOV`, one random color per cluster). Stable colors in the same locations across frames means the tracking held.

## Gotchas

- The overlap filter's screening pass and the notebook's real alignment run use the same parameters by default (`z_threshold=4.0`, `radius_in=4`, `radius_out=20`). If you tune the aligner in the notebook, tune the filter call to match, or the screen will be answering a different question than the run.
- `keep` indexes the *original* path lists. Any per-session metadata gathered before filtering (`stim_types_all`, `paths_allOps`) must be indexed through `keep`; anything read off `data` afterwards is already in filtered order.
- `EXCLUDE_DIRS` exists because QC output folders contain `stat.npy` copies. Adding a new output subfolder that contains Suite2p-shaped files means adding it here too.
- `um_per_pixel` is currently `1.0`, i.e. distances in the aligner's micrometer parameters are really pixels. Set it correctly if you want `radius_in`/`radius_out` to mean physical distance.
- Silhouette-based ordering is only as meaningful as the mixing fit. If the pairwise-distance plot was not bimodal, `cs_sil` ranking is not a reliable guide to which clusters to inspect — page through the PDF instead.
