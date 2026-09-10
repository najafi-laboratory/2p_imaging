# Cross-Session ROI Tracking

This module lives in `tracking/` at the repository root. Everything below assumes you are working from inside that directory — the notebook's `import pipeline` and `import roi_tracking_qc` resolve relative to its own location.

This module tracks the same neurons across imaging sessions. It takes the Suite2p outputs for every session of a mouse, registers the fields of view onto a common frame, and clusters ROIs so that each neuron receives a single **UCID** (unique cluster ID) valid across all sessions. The tracking itself is done by [ROICaT](https://github.com/RichieHakim/ROICaT); what lives here is the lab's driver notebook, a session-screening helper, and a QC layer for eyeballing individual UCIDs before trusting them downstream.

## Attribution and license

`interactive_tracking.ipynb` is adapted from the ROICaT project by Rich Hakim (<https://github.com/RichieHakim/ROICaT>), licensed under **GPL-3.0**. Redistribution of this notebook — or of a larger work containing it — must preserve the copyright and license notices and apply GPL-3.0 to derivative works. See [`tracking/LICENSE`](https://github.com/najafi-laboratory/2p_imaging/blob/main/tracking/LICENSE).

**This directory is GPL-3.0; the rest of the repository is not.** The GPL covers this module and works derived from it. Other top-level directories are separate, independent programs that merely share a repository with it, which GPL-3.0 §5 treats as an aggregate — including `tracking/` here does not place them under the GPL.

## Contents

| file | role |
| --- | --- |
| `interactive_tracking.ipynb` | The pipeline itself. Step-by-step, parameter-tunable, with a visualization after nearly every step. This is what you run. |
| `pipeline.py` | `filter_sessions_by_overlap()` — screens sessions for co-registerability before the real run, so poorly-overlapping sessions never enter the pipeline. |
| `roi_tracking_qc.py` | Per-UCID cross-session QC figures, exportable as a multipage PDF or a self-contained HTML viewer with a UCID picker that can be annotated with the manual review verdict. |
| `results_table.py` | Flattens the nested label lists and quality-metric arrays into two pandas tables: one row per tracked ROI, and a UCID × session match matrix. Also joins manual ROI-review labels onto the tracking output. |

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

Point `dir_allOuterFolders` at the mouse or batch directory, e.g.

```
/Volumes/Elements/Najafi/2P_Imaging/SA11_LG
```

The notebook globs for `stat.npy` at depth ≤ 10 and then has to decide which of the hits are actually sessions. A processed session can hold **three** copies of `stat.npy` — `suite2p/plane0/`, `qc_results/`, and `manual_qc_results/` — and the QC folders hold ROI *subsets*, not the full segmentation. On `SA11_LG`, 12 of 33 sessions carry all three, so a naive glob returns 114 paths for 33 sessions and hands ROICaT the same session up to three times.

Discovery therefore groups every hit by session folder and keeps exactly one per session, chosen by `STAT_SOURCE`:

```python
STAT_SOURCE = 'suite2p/plane0'   # the only source present in every session
EXCLUDE_DIRS = {'batches', 'results', 'memmap'}
```

**Leave `STAT_SOURCE` on `suite2p/plane0` unless you have a specific reason not to.** It is the only source present in every session, and it is the indexing that `roi_manual_labels.npy` is written against — see [Joining manual ROI review labels](#joining-manual-roi-review-labels).

`ops.npy` is resolved separately, because the QC folders contain `stat.npy` but no `ops.npy`: the notebook looks beside the chosen `stat.npy`, then at `<session>/suite2p/plane0/ops.npy`, then at `<session>/ops.npy`, and prints which fallback it used. Sessions with no usable source are collected and reported by name rather than tripping an assert, so one malformed session no longer stops the cell:

```
33 sessions found (from 114 stat files, source: suite2p/plane0)
  SA11_20250806
  ...
2 session(s) skipped:
  SA11_20250923: no suite2p/plane0 (has: manual_qc_results, qc_results)
```

A duplicate-path assert still fires if two sessions somehow resolve to the same file — a repeated session silently corrupts the clustering, so that one stays fatal.

Session folders are located by walking up from `stat.npy` to the directory containing `bpod_session_data.mat`, which handles both layouts on the drive (`<session>/qc_results/stat.npy` and `<session>/suite2p/plane0/stat.npy`).

A helper cell then reads that `bpod_session_data.mat` and tags each session by stimulus type:

| dataset | tag | read from |
| --- | --- | --- |
| joystick | `VG` / `ST` | `SessionData.TrialSettings[0].GUI.SelfTimedMode` |
| passive | `random` / `fix_jitter_odd` | `SessionData.RandomTypes` / `OddballTypes` |
| neither | `unknown` | no `bpod_session_data.mat`, or no recognized fields |

A passive `3331Random` session flags every trial random with no oddballs; `4131FixJitterOdd` interleaves ~950 oddball trials over a fix/jitter split and still carries ~100 random trials, so the discriminator is `all(RandomTypes == 1)`, not `any`. These tags are cosmetic — they only feed panel titles in the QC figures and contact sheets — but they make it obvious at a glance when a batch mixes protocols.

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
| `session_name`, `date` | e.g. `SA11_20250811`, `20250811` — the session folder name, and the first 8-digit run within it |
| `stim_type` | `VG` / `ST` / `random` / `fix_jitter_odd` / `unknown`, when passed in |
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

## Joining manual ROI review labels

The [interactive ROI reviewer](https://najafi-laboratory.github.io/2p_imaging/roi-reviewer-exports/) writes `roi_manual_labels.npy`: a 1-D float array with one entry per **original Suite2p ROI**, in `stat.npy` order — `NaN` not labeled, `0` bad, `1` good, `2` unsure. That is the same indexing ROICaT uses, so the array lines up element-for-element with `roi_idx` in the ROI table and no matching step is needed.

**Order does not matter.** Those indices never change, so tracking and review can happen in either order. Track first and review months later; re-running the join cell picks up whatever labels exist at that moment. Nothing needs re-clustering.

```python
roi_table = rt.attach_manual_labels(
    roi_table,
    _results_all['input_data']['paths_stat'],
    n_roi_bySession=[len(x) for x in _results_all['clusters']['labels_bySession']],
)
```

Sessions with no label file come back as `unlabeled`; the call prints which sessions were found and the resulting consensus breakdown. Added columns:

| column | meaning |
| --- | --- |
| `manual_label` | `NaN` / `0` / `1` / `2` for this ROI |
| `manual_label_str` | `unlabeled` / `bad` / `good` / `unsure` |
| `n_good`, `n_bad`, `n_unsure`, `n_labeled` | counts across every session in this UCID |
| `ucid_label` | consensus: `good`, `bad`, `unsure`, `conflict`, or `unlabeled` |

`conflict` means the cluster has a good label in one session and a bad one in another. With a single reviewed session it is unreachable; once two sessions are reviewed it becomes an independent check on the tracking, and it never passes any filter policy.

`ucid_label_display(roi_table)` turns those verdicts into a short `{ucid: str}` mapping for the QC picker. Most labels pass straight through; `conflict` is expanded to name the sessions on each side, because telling a merged cluster from an inconsistent review means knowing *which* session dissented:

```python
rt.ucid_label_display(roi_table)
# {0: 'good', 1: 'conflict: good@20250806,20250811 bad@20250903', 2: 'bad', ...}
```

Session names come from the `date` column (falling back to `session_idx` when absent), and each side collapses to `+N` past `max_names=3`. If two sessions share a date — a VG and an ST block the same day — the names are suffixed `#session_idx`, the same disambiguation `build_match_matrix` applies, since a collapsed `bad@20250806` would point at either one. Pass the result to `export_html(..., ucid_labels=...)` — see [Exporters](#exporters).

```python
good = rt.filter_by_manual_label(roi_table, policy='good', scope='cluster')
```

- `policy` — `'good'`, `'good_or_unsure'`, or `'not_bad'` (which also passes unlabeled ROIs).
- `scope='cluster'` keeps every session's ROI for a passing UCID. **This is the point of the whole exercise**: one session's review propagates to the others through the tracking. On the 7-session `YH03VT` run with one reviewed session, that turns 41 good UCIDs into 147 ROIs spanning all seven sessions.
- `scope='roi'` judges each ROI on its own label instead, leaving clusters partial — 41 rows for the same run.

### The label-alignment check

`attach_manual_labels` requires the label array length to **equal** the ROI count ROICaT tracked for that session, and raises otherwise. This is deliberately stricter than a bounds check: `manual_qc_results/stat.npy` holds 208 ROIs where `suite2p/plane0` holds 990, so if the tracking were run on the QC subset, every subset index would be trivially in range for the full-length label array and the join would silently pick the wrong neurons. There is no stored index mapping between the two, so the only correct answer is to refuse:

```
session 0 (SA11_20250806): ROICaT tracked 208 ROIs from manual_qc_results/stat.npy
but roi_manual_labels.npy has 990 entries. Labels are indexed by original suite2p
ROI, so the tracking must be run on suite2p/plane0/stat.npy
(set STAT_SOURCE='suite2p/plane0') for the two to line up.
```

Pass `n_roi_bySession` (free, from `labels_bySession`) or let it load each `stat.npy` to count.

The one case the check cannot catch is re-running **Suite2p** after labeling: ROI indices change, and if the ROI count happens to stay the same the lengths still match. Re-review after any Suite2p re-run.

### The cs_sil off-by-one

`quality_metrics['cluster_silhouette']` is aligned with `quality_metrics['cluster_labels_unique']`, and **that label array starts at −1**. So `cluster_silhouette[u]` is the score for cluster `u − 1`, not for UCID `u`, and position 0 holds the unclustered pseudo-cluster's score (near −1.0, so UCID 0 spuriously sorts to the front of a worst-first list).

`rt.cs_sil_by_ucid(quality_metrics)` returns a properly UCID-indexed array (NaN where unavailable). `roi_tracking_qc` now accepts either that array *or* the whole `quality_metrics` dict, and re-indexes internally — **pass the dict**. The notebook does. Earlier QC exports were built with the raw array and are shifted by one; regenerate them if you relied on the displayed `cs_sil` values or on the worst-first ordering.

### The session_name collapse

`session_labels_from_paths()` used to take the session name from `Path(stat_path).parts[-3]`. That is correct for `<session>/qc_results/stat.npy` but returns the literal string `suite2p` for `<session>/suite2p/plane0/stat.npy` — so on any run tracked from the Suite2p output, **every session was named `suite2p`**. The long table merely looked odd; the match matrix was actively wrong, because `build_match_matrix` pivots on `session_name` and collapsed all sessions into a single repeated column:

```
ucid,cs_sil,n_rois_in_cluster,n_sessions_present,suite2p,suite2p,suite2p,...
0,0.988,3,3,"392, 136, 125","392, 136, 125","392, 136, 125",...
```

Session names now come from the session folder, and dates from the first 8-digit run in that name (the old `name.split('_')[-1]` returned `3331Random` for the `VTYH03_PPC_20250106_3331Random` convention). `build_match_matrix` additionally suffixes `#<session_idx>` if names still collide, so a pivot can never silently merge sessions again. **Regenerate any `*.match_matrix.csv` produced before this fix** — the affected files have duplicate column headers, which is the tell.

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
    ucid_labels=rt.ucid_label_display(roi_table),
    fovs_raw=fovs_raw, rois_raw=rois_raw,
    remapping_idxs=remapping_idxs,
    mouse_name=mouse_name, session_names=session_names,
)
```

- `export_html(path, ...)` — one self-contained HTML file: a dropdown, prev/next buttons, and every figure pre-rendered and base64-embedded. No server, no dependencies; hand the file to anyone. Because every PNG is inlined, there is a `max_ucids=400` safety cap — pass a worst-first slice rather than all clusters.
- `export_pdf(path, ...)` — one multipage PDF, one UCID per page. No cap, but no navigation either.
- `order_ucids_by_quality(labels_bySession, quality_metrics, ascending=True)` — worst-first by silhouette score, so the first pages of the export are the clusters most likely to be wrong. Unclustered ROIs (label −1) are dropped by default; UCIDs with no score sort last.
- `ucid_labels={ucid: str}` (HTML only) — shown in brackets after each picker entry, e.g. `UCID 412  (cs_sil 0.310)  [conflict: good@20250806 bad@20250903]`. Optional; omit it and the picker reads exactly as before.

### Scoping QC to the clusters that matter

Once manual labels exist, the worst-200-by-`cs_sil` slice is the wrong 200 pages. A low-scoring cluster the reviewer already labeled **bad** is not worth inspecting — whether tracking correctly linked a cell you are discarding changes nothing. The clusters where an error actually costs something are **good** (a bad link silently contaminates the analysis set) and **conflict** (tracking and review disagree, and someone has to adjudicate):

```python
order = qc.order_ucids_by_quality(labels_bySession, qm, ascending=True)

REVIEW_LABELS = ('good', 'conflict')
if REVIEW_LABELS and roi_table.ucid_label.ne('unlabeled').any():
    keep = set(roi_table.loc[roi_table.ucid_label.isin(REVIEW_LABELS), 'ucid'])
    order = [u for u in order if u in keep]
```

Same worst-first ordering, but every page is a cluster whose correctness matters, so the `max_ucids` budget goes much further. This is what the notebook's HTML cell does by default; set `REVIEW_LABELS = ()` to inspect every cluster. The `ne('unlabeled')` guard matters — with no label files on disk every UCID is `unlabeled` and an unguarded filter would render an empty page.

Both exporters share `build_ucid_figure()`, so PDF and HTML pages are identical apart from DPI (110 vs. 90).

The module forces matplotlib's `Agg` backend at import, before `pyplot` is imported — it is intended to run headless, and importing it will override an interactive backend in the same kernel.

### Running QC from saved results

The table cell near the end of the notebook reloads from the richfiles rather than reading live pipeline objects, so it works in a fresh kernel:

```python
_results_all = roicat.util.RichFile_ROICaT(path=paths_save['results_all']).load()
_run_data    = roicat.util.RichFile_ROICaT(path=paths_save['run_data']).load()
```

It needs `paths_save`, `dir_save`, `name_save`, and `get_stim_type` in scope — run the paths and save cells above it, or set those four by hand.

**The table cell runs before the HTML cell**, and the HTML cell depends on it: it reuses the loaded arrays (`labels_bySession`, `fovs_aligned`, `cs_sil`, …) and needs `roi_table` for the review-label filter and picker annotations. Running the HTML cell alone in a fresh kernel raises `NameError`.

### Other QC outputs

- **`{name}_FOVs_for_matching.png`** — one-row contact sheet of the raw mean FOV for each session that survived the overlap filter, titled with date and stim type.
- **All-sessions contact sheet** — the same idea over the *pre-filter* session list, with dropped sessions dimmed to 35% alpha. This is the fastest way to see *why* a session was dropped.
- **`FOV_clusters_highQuality.gif`** — the color-coded cluster FOV animated across sessions (`compute_colored_FOV`, one random color per cluster). Stable colors in the same locations across frames means the tracking held.

## Gotchas

- The overlap filter's screening pass and the notebook's real alignment run use the same parameters by default (`z_threshold=4.0`, `radius_in=4`, `radius_out=20`). If you tune the aligner in the notebook, tune the filter call to match, or the screen will be answering a different question than the run.
- `keep` indexes the *original* path lists. Any per-session metadata gathered before filtering (`stim_types_all`, `paths_allOps`) must be indexed through `keep`; anything read off `data` afterwards is already in filtered order.
- `STAT_SOURCE` decides which of a session's `stat.npy` copies is tracked, and it must stay `suite2p/plane0` for manual labels to join. The QC folders hold ROI *subsets* with no stored index mapping back to the full segmentation, so pointing the tracking at one makes `roi_manual_labels.npy` unjoinable. `EXCLUDE_DIRS` is now only for non-session folders (`batches`, `results`, `memmap`).
- Sessions missing `STAT_SOURCE` or any `ops.npy` are reported in the skipped list, not raised. Read that list — a silently absent session is a session missing from the tracking.
- `um_per_pixel` is currently `1.0`, i.e. distances in the aligner's micrometer parameters are really pixels. Set it correctly if you want `radius_in`/`radius_out` to mean physical distance.
- Silhouette-based ordering is only as meaningful as the mixing fit. If the pairwise-distance plot was not bimodal, `cs_sil` ranking is not a reliable guide to which clusters to inspect — page through the PDF instead.
- `cluster_silhouette` is indexed by position in `cluster_labels_unique`, which starts at −1 — never index it by UCID directly. Use `rt.cs_sil_by_ucid()`, or pass the whole `quality_metrics` dict to the QC functions.
- `roi_idx` in the tables is the index into that session's `stat.npy` — i.e. *all* Suite2p ROIs. The dF/F pipeline selects a subset, so joining tracking to dF/F means mapping through that subset's ROI indices, not against dF/F row order. `roi_manual_labels.npy` uses this same full-`stat.npy` indexing, which is why it joins to `roi_idx` directly.
- Re-running Suite2p invalidates both the tracking and any manual labels for that session, and the label-length check only catches it if the ROI count also changed. Re-review and re-track after a re-segmentation.
