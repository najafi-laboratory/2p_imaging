# Interactive Manual ROI Labeler

The interactive preprocessing summary lets a reviewer mark each Suite2p ROI as
**Good**, **Bad**, **Unsure**, or **Unlabeled** by inspecting its morphology and
dF/F trace.

## 1. Generate the labeler and summary

The summary stage creates:

```text
<session>_preprocessing_summary.pdf
<session>_interactive_fov_roi_dff.html
```

Run the commands below from the `2p_imaging` repository root with the
preprocessing QC environment available.

### Required processed-session files

At minimum, the processed session must contain:

```text
/path/to/processed/session/
└── suite2p/
    └── plane0/
        ├── ops.npy
        ├── stat.npy
        ├── F.npy
        └── Fneu.npy
```

`ops.npy` must contain the functional mean image produced by Suite2p. An
existing `iscell.npy` is used when present; otherwise all Suite2p ROIs are
initially available.

To restore the morphology QC target structure and initial Bad/Unlabeled labels
from the preprocessing pipeline, the session should also contain:

```text
/path/to/processed/session/
├── preprocessing_pipeline_parameters.json
└── qc_results/
    └── qc_parameters.json
```

Without these morphology QC files, the summary can still be generated from the
Suite2p files, but it cannot recreate the pipeline's morphology-based starting
labels. `masks.h5` is optional and supplies anatomical images when available.

### Generate locally

```bash
python -m utils_2p.preprocessing_summary /path/to/processed/session
```

The PDF and interactive HTML are written into the processed session directory.

### Generate on PACE

Submit summary generation as a small CPU job instead of running it on a PACE
login node:

```bash
export TWO_P_PYTHON=/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_suite2p_1x/bin/python

sbatch \
  --account=gts-fnajafi3 \
  --qos=embers \
  --cpus-per-task=4 \
  --mem=24G \
  --time=02:00:00 \
  --job-name=preprocessing_summary \
  --wrap="$TWO_P_PYTHON -m utils_2p.preprocessing_summary /path/to/processed/session"
```

### Generate as part of the preprocessing pipeline

The full PACE preprocessing pipeline includes the `summary` stage by default:

```bash
python -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/processed_outputs \
  --target-structure neuron
```

Change `--target-structure` to the appropriate preset, such as `dendrite` or
`cerebellum_lax`.

To regenerate only the summaries for an existing pipeline output:

```bash
python -m utils_2p.preprocessing_qc_pipeline submit \
  --session /path/to/raw/session \
  --output-root /path/to/existing_processed_outputs \
  --target-structure neuron \
  --stages summary
```

The processed session must be located at
`/path/to/existing_processed_outputs/<raw-session-directory-name>/`.

## 2. Export format and downstream use

The interactive HTML contains the original Suite2p ROI set. The morphology QC
target structure used by the preprocessing pipeline provides the initial
filtering state:

- ROIs excluded by the `neuron`, `dendrite`, or `cerebellum_lax` preset start
  as **Bad**.
- ROIs that passed the preset start as **Unlabeled**.

The reviewer can change labels manually, apply another morphology preset, or
use **Clear all labels / show all ROIs** to return every Suite2p ROI to the
Unlabeled state.

```text
All ROIs detected by Suite2p
        |
        v
Morphology QC target structure
        |
        v
Manual Good / Bad / Unsure / Unlabeled review
        |
        v
reviewed HTML, roi_manual_labels.npy, or manual-label JSON
```

The reviewer can save the current labels back into a self-contained reviewed
HTML copy with **Save labels into HTML**. Reopening that saved HTML restores the
labels embedded in the file. The `.npy` and JSON buttons are export formats for
downstream scripts.

Custom morphology presets use the same explicit-save model. **Save preset**
adds the current threshold values to the open page, **Save preset into HTML**
saves a reviewed HTML copy that will reopen with that custom preset available,
and **Export preset JSON** / **Import preset JSON** move a preset between
sessions.

### `roi_manual_labels.npy`

This file is a three-column NumPy array with one row per original Suite2p ROI:

```python
array([
    [1.0, 1.0, 1.0],     # good, passed morphology
    [0.0, 0.0, 0.0],     # bad, passed morphology
    [0.0, 0.0, 1.0],     # unsure, passed morphology
    [0.0, nan, nan],     # morphology-excluded ROI
])
```

Column 0 is the full Suite2p good mask. Column 1 is the morphology-filtered
good mask, with morphology-excluded ROIs marked as `NaN`. Column 2 is the same
as column 1, but includes Unsure ROIs with Good ROIs.

Place the reviewed file beside the original Suite2p files:

```text
/path/to/session/
└── suite2p/
    └── plane0/
        ├── F.npy
        ├── Fneu.npy
        ├── iscell.npy
        └── roi_manual_labels.npy
```

### Manual-label JSON

The JSON export preserves all four label states explicitly:

```json
{
  "session": "example_session",
  "labels": [
    {"suite2p_roi": 0, "label": 1},
    {"suite2p_roi": 1, "label": 0},
    {"suite2p_roi": 2, "label": 2},
    {"suite2p_roi": 3, "label": null}
  ]
}
```

`1` means Good, `0` means Bad, `2` means Unsure, and `null` means Unlabeled.

### Load reviewed dF/F

With `roi_manual_labels.npy` in `suite2p/plane0/`, load morphology-filtered
Good ROIs in a script or notebook:

```python
from utils_2p.roi_labels import load_reviewed_dff

session = load_reviewed_dff("/path/to/session")
dff = session["dff"]
roi_indices = session["roi_indices"]
```

`dff` has shape `(selected_rois, frames)`. `roi_indices` contains the
corresponding original Suite2p ROI indices.

To include Unsure ROIs with Good ROIs, use:

```python
session = load_reviewed_dff("/path/to/session", policy="good_or_unsure")
```

When the JSON or `roi_manual_labels.npy` export is stored elsewhere, pass its path:

```python
session = load_reviewed_dff(
    "/path/to/session",
    label_path="/path/to/session_manual_roi_labels.json",
)
```

Use the JSON export when downstream code must distinguish Good, Bad, Unsure,
and Unlabeled ROIs. The companion notebook contains the same example:
[`utils_2p/roi_reviewer_exports.ipynb`](https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/roi_reviewer_exports.ipynb).
