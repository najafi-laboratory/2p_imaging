# Using ROI Reviewer Output

The interactive preprocessing summary lets a reviewer mark each Suite2p ROI as
**Good**, **Bad**, or **Unlabeled**. It can export the review in two formats:

- `iscell_qc.npy` for direct use with Suite2p-style processing scripts.
- `<session>_manual_roi_labels.json` when the three label states need to be
  preserved explicitly.

## Default labels

The reviewer starts with the morphology QC target structure used by the
preprocessing pipeline already applied. For example, the pipeline may have
used the `neuron`, `dendrite`, or `cerebellum_lax` preset.

- ROIs excluded by that morphology QC preset start as **Bad**.
- ROIs that passed the preset start as **Unlabeled**.

The reviewer can change these labels manually, apply a different morphology
preset, or use **Clear all labels / show all ROIs** to return every Suite2p ROI
to the Unlabeled state.

## Output formats

### `iscell_qc.npy`

This file has the same two-column layout as Suite2p's `iscell.npy`:

```python
array([
    [1.0, 1.0],  # good
    [0.0, 0.0],  # bad
    [1.0, 0.83], # unlabeled; original Suite2p values retained
])
```

Column 0 is the binary ROI selection. Good rows are included and Bad rows are
excluded. Unlabeled rows retain their original values from `iscell.npy`.

Place the reviewed file beside the original Suite2p files:

```text
/path/to/session/
└── suite2p/
    └── plane0/
        ├── F.npy
        ├── Fneu.npy
        ├── iscell.npy
        └── iscell_qc.npy
```

### Manual-label JSON

The JSON file records Good, Bad, and Unlabeled separately:

```json
{
  "session": "example_session",
  "labels": [
    {"suite2p_roi": 0, "label": 1},
    {"suite2p_roi": 1, "label": 0},
    {"suite2p_roi": 2, "label": null}
  ]
}
```

`1` means Good, `0` means Bad, and `null` means Unlabeled.

## Loading reviewed dF/F

With `iscell_qc.npy` in `suite2p/plane0/`, a script or notebook can load the
ROIs selected by column 0:

```python
from utils_2p.roi_labels import load_reviewed_dff

session = load_reviewed_dff("/path/to/session")

dff = session["dff"]
roi_indices = session["roi_indices"]
```

`dff` has shape `(selected_rois, frames)`. `roi_indices` identifies the
corresponding ROI rows in the original Suite2p files. Use the JSON export when
the script needs to distinguish Good ROIs from Unlabeled ROIs.

If the JSON or `iscell_qc.npy` file is stored outside the session directory,
provide it directly:

```python
session = load_reviewed_dff(
    "/path/to/session",
    label_path="/path/to/session_manual_roi_labels.json",
)
```

The companion notebook contains the same example:
[`utils_2p/roi_reviewer_exports.ipynb`](https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/roi_reviewer_exports.ipynb).
