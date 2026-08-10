# Cross-Session ROI Tracking

This module tracks the same neurons across imaging sessions. It takes the Suite2p outputs for every session of a mouse, registers the fields of view onto a common frame, and clusters ROIs so that each neuron receives a single **UCID** (unique cluster ID) valid across all sessions. The tracking itself is done by [ROICaT](https://github.com/RichieHakim/ROICaT); what lives here is the lab's driver notebook, a session-screening helper, and a QC layer for eyeballing individual UCIDs before trusting them downstream.

## Attribution and license

`interactive_tracking.ipynb` is adapted from the ROICaT project by Rich Hakim (<https://github.com/RichieHakim/ROICaT>), licensed under **GPL-3.0**. Redistribution of this notebook — or of a larger work containing it — must preserve the copyright and license notices and apply GPL-3.0 to derivative works. See the `LICENSE` in this directory.

**This directory is GPL-3.0; the rest of the repository is not.** The GPL covers this module and works derived from it. Other top-level directories are separate, independent programs that merely share a repository with it, which GPL-3.0 §5 treats as an aggregate — including `tracking/` here does not place them under the GPL.

## Contents

| file | role |
| --- | --- |
| `interactive_tracking.ipynb` | The pipeline itself. Step-by-step, parameter-tunable, with a visualization after nearly every step. This is what you run. |
| `pipeline.py` | `filter_sessions_by_overlap()` — screens sessions for co-registerability before the real run, so poorly-overlapping sessions never enter the pipeline. |
| `roi_tracking_qc.py` | Per-UCID cross-session QC figures, exportable as a multipage PDF or a self-contained HTML viewer with a UCID picker. |
| `results_table.py` | Flattens the nested label lists and quality-metric arrays into two pandas tables: one row per tracked ROI, and a UCID × session match matrix. |

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
conda env create -f environment.yml
conda activate roi_tracking
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

## Tabular results: `results_table.py`

The saved artifacts store labels as nested per-session lists and quality metrics as separate arrays with their own indexing conventions. Neither is convenient for joining tracking output to dF/F traces. This module flattens them.

### Long table — one row per tracked ROI

```python
import results_table as rt

roi_table = rt.build_roi_table(
    _results_all['clusters']['labels_bySession'],
    quality_metrics=_results_all['clusters']['quality_metrics'],
    paths_stat=_results_all['input_data']['paths_stat'],
    stim_types=stim_types,
    rois_aligned=_results_all['ROIs']['ROIs_aligned'],
    H=H, W=W,
)
```

| column | meaning |
| --- | --- |
| `ucid` | cluster ID, stable across sessions |
| `session_idx` | 0-based index in *filtered* (post-overlap-screen) session order |
| `session_name`, `date` | e.g. `SA11_20250811`, `20250811` — derived from `paths_stat` |
| `stim_type` | `VG` / `ST` / `unknown`, when passed in |
| `roi_idx` | **index into that session's `stat.npy`** — the join key for dF/F |
| `roi_idx_global` | index into the session-concatenated ROI vector, which is how `sample_silhouette` and `sample_probabilities` are indexed |
| `n_sessions_present` | distinct sessions this UCID appears in |
| `n_rois_in_cluster` | total ROIs in the cluster; greater than `n_sessions_present` means one session contributed two ROIs to the same cluster, which is worth inspecting |
| `cs_sil` | cluster silhouette (constant within a UCID) |
| `sample_sil` | per-ROI silhouette |
| `sample_prob` | per-ROI HDBSCAN membership probability; NaN on sequential-Hungarian runs, which don't produce it |
| `centroid_y`, `centroid_x` | intensity-weighted centroid of the **aligned** footprint |

ROIs with UCID −1 (unclustered) are dropped by default; pass `include_unclustered=True` to keep them for accounting.

The aligned centroids double as a cheap correctness check: a well-tracked UCID should have nearly the same aligned centroid in every session. On the SA11_LG 3-session run, within-UCID centroid standard deviation averages 2–4 px. Clusters far above that are worth opening in the HTML viewer.

### Wide match matrix — one row per UCID

```python
match_matrix = rt.build_match_matrix(roi_table)
```

```
ucid   cs_sil  n_rois_in_cluster  n_sessions_present  SA11_20250811  SA11_20250812  SA11_20250813
0    0.762147                  3                   3            8.0           31.0           22.0
1    0.254051                  2                   2           14.0            NaN           50.0
2   -0.014512                  2                   2           18.0           27.0            NaN
```

Cells hold the Suite2p ROI index for that neuron in that session, NaN where it wasn't detected. This is the table to join dF/F against: row = neuron, column = session, value = which ROI to pull. Columns are restored to acquisition order (`pivot_table` sorts alphabetically). If a session contributed two ROIs to one cluster, the cell becomes a comma-separated string rather than silently dropping one.

### Export

```python
rt.export_tables(roi_table, str(Path(dir_save) / name_save))
# -> {name}.roi_table.csv, {name}.match_matrix.csv
```

These are distinct from the pre-existing `*.matched_neurons_*.csv` and `*.quality_metrics_summary.csv` in the results folder, which are aggregate counts and metric distributions with no per-ROI rows.

### The cs_sil off-by-one

`quality_metrics['cluster_silhouette']` is aligned with `quality_metrics['cluster_labels_unique']`, and **that label array starts at −1**. So `cluster_silhouette[u]` is the score for cluster `u − 1`, not for UCID `u`, and position 0 holds the unclustered pseudo-cluster's score (near −1.0, so UCID 0 spuriously sorts to the front of a worst-first list).

`rt.cs_sil_by_ucid(quality_metrics)` returns a properly UCID-indexed array (NaN where unavailable). `roi_tracking_qc` now accepts either that array *or* the whole `quality_metrics` dict, and re-indexes internally — **pass the dict**. The notebook does. Earlier QC exports were built with the raw array and are shifted by one; regenerate them if you relied on the displayed `cs_sil` values or on the worst-first ordering.

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

# Pass the quality_metrics dict, not cluster_silhouette — see the off-by-one note above.
qm = _results_all['clusters']['quality_metrics']
order = qc.order_ucids_by_quality(labels_bySession, qm, ascending=True)  # worst-first

qc.export_html(
    "out_tracking.html", order[:200],
    fovs_aligned, rois_aligned, labels_bySession, H, W,
    cs_sil=qm, crop_halfwidth=40,
    fovs_raw=fovs_raw, rois_raw=rois_raw,
    remapping_idxs=remapping_idxs,
    mouse_name=mouse_name, session_names=session_names,
)
```

- `export_html(path, ...)` — one self-contained HTML file: a dropdown, prev/next buttons, and every figure pre-rendered and base64-embedded. No server, no dependencies; hand the file to anyone. Because every PNG is inlined, there is a `max_ucids=400` safety cap — pass a worst-first slice rather than all clusters.
- `export_pdf(path, ...)` — one multipage PDF, one UCID per page. No cap, but no navigation either.
- `order_ucids_by_quality(labels_bySession, quality_metrics, ascending=True)` — worst-first by silhouette score, so the first pages of the export are the clusters most likely to be wrong. Unclustered ROIs (label −1) are dropped by default; UCIDs with no score sort last.

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
- `cluster_silhouette` is indexed by position in `cluster_labels_unique`, which starts at −1 — never index it by UCID directly. Use `rt.cs_sil_by_ucid()`, or pass the whole `quality_metrics` dict to the QC functions.
- `roi_idx` in the tables is the index into that session's `stat.npy` — i.e. *all* Suite2p ROIs. The dF/F pipeline selects a subset, so joining tracking to dF/F means mapping through that subset's ROI indices, not against dF/F row order.
