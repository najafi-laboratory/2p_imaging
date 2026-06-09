# Using Interactive ROI Reviewer Exports

The preprocessing summary reviewer can export either:

- `iscell_qc.npy`: a Suite2p-shaped reviewed array stored alongside the
  original `suite2p/plane0/iscell.npy`.
- `<session>_manual_roi_labels.json`: labels plus Suite2p ROI indices, the
  morphology settings shown in the reviewer, and a fingerprint used to reject
  exports from a different `stat.npy`.

For the standard workflow, place the reviewed NumPy file here:

```text
<session>/
└── suite2p/
    └── plane0/
        ├── F.npy
        ├── Fneu.npy
        ├── stat.npy
        ├── ops.npy
        ├── iscell.npy       # original Suite2p classification
        └── iscell_qc.npy    # morphology/manual review result
```

The companion notebook
[`utils_2p/roi_reviewer_exports.ipynb`](https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/roi_reviewer_exports.ipynb)
defines a small `load_session_dff` function for loading either format and
returning analysis-ready dF/F plus the original Suite2p ROI indices.

## Output formats

### `iscell_qc.npy`

`iscell_qc.npy` has the same `(N, 2)` shape as Suite2p's original
`iscell.npy`, where `N` is the number of original Suite2p ROIs:

```python
array([
    [1.0, 1.0],  # reviewed good
    [0.0, 0.0],  # reviewed bad
    [1.0, 0.83], # unlabeled: original Suite2p row preserved
])
```

- Column 0 is the binary ROI selection used by downstream code.
- Column 1 contains `1.0` or `0.0` for explicitly reviewed ROIs.
- Unlabeled ROIs retain both values from the original `iscell.npy`.

Because the NumPy format preserves original values for unlabeled rows, it
cannot independently indicate whether a row was reviewed or left unlabeled.
Use the JSON export when that distinction matters.

### Manual-label JSON

The JSON export is auditable and preserves all three review states. A shortened
example is:

```json
{
  "format": "utils_2p_manual_roi_labels_v1",
  "session": "YH26LG_CRBL_crus2_20251211_EBC-869",
  "suite2p_roi_count": 46,
  "suite2p_stat_fingerprint": "dbe856...",
  "morphology_filter": {
    "skewMin": 0.4,
    "skewMax": 2.5,
    "maxConnect": 30
  },
  "labels": [
    {"summary_roi": 0, "suite2p_roi": 0, "label": 1},
    {"summary_roi": 1, "suite2p_roi": 1, "label": 0},
    {"summary_roi": 2, "suite2p_roi": 2, "label": null}
  ]
}
```

Here, `1` means good, `0` means bad, and `null` means unlabeled.
`suite2p_roi` is the row in the original Suite2p arrays. `summary_roi` is the
row shown in the reviewer and should not be used to index original Suite2p
files.

On the lab cluster, the notebook defaults to this existing 46-ROI processed
test session:

```text
/storage/scratch1/3/grubin6/2p_pipeline_tests/yh26_staged_tiff_bin_benchmark/YH26LG_CRBL_crus2_20251211_EBC-869
```

The example passes its original `iscell.npy` explicitly only to demonstrate
the external `label_path` argument. After review, omit `label_path` and the
loader will use `suite2p/plane0/iscell_qc.npy`.

## Initial reviewer state

Suite2p first detects the complete ROI set and writes `stat.npy`, `F.npy`,
`Fneu.npy`, and its classifier output in `iscell.npy`. The preprocessing
pipeline then applies the morphology QC preset selected by `target_structure`,
such as `neuron`, `dendrite`, or `cerebellum_lax`.

These morphology parameters are passed to the repository's post-Suite2p QC
stage; they are not inputs to Suite2p's ROI detection or internal classifier.
The reviewer embeds the original Suite2p ROI set, then initializes:

- morphology failures as **Bad**
- morphology passers as **Unlabeled**

Bad ROIs are hidden initially, which makes the first display match the target
structure selected for preprocessing. The target structure and thresholds are
shown in the morphology settings section.

To inspect every ROI originally detected by Suite2p, click **Clear all labels /
show all ROIs**. This:

1. Changes every reviewer label to Unlabeled.
2. Restores every pending `iscell_qc.npy` row to its original `iscell.npy`
   values.
3. Makes the full Suite2p ROI set visible in the FOV and trace displays.

This action clears manual labels and morphology-derived Bad labels. It does not
delete or modify the original Suite2p files. To reapply morphology filtering,
choose the desired preset and click **Apply filter to labels**.

## Recommended workflow

Export the JSON when review is incomplete because JSON preserves all three
states:

- `1`: manually good
- `0`: manually bad
- `null`: unlabeled

The downloaded `iscell_qc.npy` is useful for software that reads Suite2p-shaped
classification arrays, but it cannot preserve a distinct unlabeled state.
Unlabeled rows keep their original Suite2p `iscell.npy` values.

Apply a JSON export from the repository root with:

```bash
python -m utils_2p.roi_labels \
  /path/to/session_manual_roi_labels.json \
  /path/to/session/suite2p/plane0
```

This verifies the ROI count and `stat.npy` fingerprint, starts with the
original `iscell.npy`, and writes `iscell_qc.npy`. The Suite2p file is not
modified. Unlabeled rows retain their original values.

Place a downloaded QC file in the Suite2p plane directory:

```bash
cp /path/to/downloaded/iscell_qc.npy /path/to/session/suite2p/plane0/iscell_qc.npy
```

Always confirm that the downloaded array has the same number of rows as
`stat.npy`. The JSON route performs the stronger fingerprint check and is
therefore safer.

## Directory assumptions found in this repository

### Ali Shamsnia: 2AFC

Example: `2p_2AFC_double_block_version/Modules/reader.py`, introduced in commit
`01194f0f`.

This reader treats each configured path as a processed session directory. It
loads `ops.npy`, `dff.h5`, `masks.h5`, and `neural_trials.h5` from the session
root. Its `dff.h5` loader does not inspect Suite2p `iscell.npy`.

Consequences:

1. Adding `suite2p/plane0/iscell_qc.npy` does not retroactively filter an
   existing root-level `dff.h5`.
2. The safest low-storage adaptation is to load Suite2p `F.npy` and `Fneu.npy`,
   select rows with the reviewed mask, and pass the resulting dF/F array into
   the analysis.
3. If the unmodified reader must be used, rerun the postprocessing stage that
   creates root-level `dff.h5`, `masks.h5`, and `neural_trials.h5` after
   installing the reviewed labels.

Do not apply an original Suite2p-length mask directly to an older `dff.h5`
unless that file is known to contain every original Suite2p ROI in the same
order.

### Sana Aminaji: joystick processing

Example: `JoystickProcessing2026/PostProcessing/spike_deconvolve/main.py`,
introduced in commit `301f5bb1`.

This script loads:

```text
<session>/suite2p/plane0/F.npy
<session>/suite2p/plane0/Fneu.npy
<session>/suite2p/plane0/iscell_qc.npy
```

Sana's script currently loads `iscell.npy`. To preserve the original classifier
output, point it at `iscell_qc.npy` or use `load_reviewed_dff`. An existing
`ROI_label.h5` takes precedence and must be removed, renamed, or regenerated if
the reviewer labels should control selection.

Sana's older manual workflow uses `ROI_label.h5` indices relative to
`qc_results/`, not necessarily original Suite2p indices. Those index spaces
must not be mixed. Prefer `iscell_qc.npy` or use the notebook to filter the
original Suite2p arrays.

### Yicong Huang: passive interval oddball

Example: `passive_interval_oddball_202412/modules/ReadResults.py`, introduced
in commit `2ccefbe3` and later updated by Yicong.

Session paths are constructed as:

```text
results/<session_folder>/<session_name>/
```

The loader reads `suite2p/plane0/ops.npy`, but neural traces and masks come
from root-level `dff.h5`, `masks.h5`, and `neural_trials.h5`. As with Ali's
reader, adding `iscell_qc.npy` alone does not alter files already generated at
the session root.

Use one of these approaches:

1. Apply the reviewer labels and rerun postprocessing/trialization so the
   root-level files all use the same ROI rows.
2. During exploratory analysis, use the notebook's direct Suite2p loader and
   pass the filtered dF/F array forward.

Regenerate `neural_trials.h5` whenever its dF/F dataset was built from an older
ROI set. Filtering only `dff.h5` would leave inconsistent ROI dimensions.

## Loading reviewed dF/F

### Default: `iscell_qc.npy` in the session

Use the shared loader after placing `iscell_qc.npy` in
`suite2p/plane0/`:

```python
from utils_2p.roi_labels import load_reviewed_dff

reviewed = load_reviewed_dff(
    "/path/to/processed/session",
    policy="good_only",
)
dff = reviewed["dff"]
suite2p_roi_indices = reviewed["roi_indices"]
```

The main returned values are:

| Key | Meaning |
| --- | --- |
| `dff` | Selected dF/F array shaped `(selected_rois, frames)` |
| `roi_indices` | Original Suite2p row for each returned trace |
| `stat` | Matching selected rows from `stat.npy` |
| `labels` | Full-length review labels |
| `ops` | Loaded Suite2p operations dictionary |
| `label_path` | Label file actually used |

### JSON or file stored elsewhere

Pass `label_path` when the export has not been moved into the session:

```python
reviewed = load_reviewed_dff(
    "/path/to/processed/session",
    label_path="/downloads/session_manual_roi_labels.json",
    policy="good_only",
)
```

The same argument accepts an external `iscell_qc.npy`:

```python
reviewed = load_reviewed_dff(
    "/path/to/processed/session",
    label_path="/shared/reviews/iscell_qc.npy",
)
```

The loader never silently uses Suite2p's original `iscell.npy`. Use
`policy="good_only"` for a finalized review or `policy="not_bad"` to retain
unlabeled JSON rows while excluding bad ROIs.

## Row-order rule

Reviewer indices always refer to the original row order in:

```text
suite2p/plane0/stat.npy
suite2p/plane0/F.npy
suite2p/plane0/Fneu.npy
suite2p/plane0/iscell.npy
suite2p/plane0/iscell_qc.npy
```

Any filtered `qc_results`, `manual_qc_results`, `dff.h5`, `masks.h5`, or
`neural_trials.h5` may use a shorter, renumbered ROI axis. Preserve the
original Suite2p indices alongside derived arrays, for example:

```python
np.save("reviewed_suite2p_roi_indices.npy", suite2p_roi_indices)
```

This makes results traceable to the reviewer and prevents accidental masking
of a differently ordered array.
