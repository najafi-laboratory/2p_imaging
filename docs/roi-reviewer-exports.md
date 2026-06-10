# Using ROI Reviewer Output

The interactive preprocessing summary lets a reviewer mark each Suite2p ROI as
**Good**, **Bad**, or **Unlabeled**. It can export the review in two formats:

- `iscell_qc.npy` for direct use with Suite2p-style processing scripts.
- `<session>_manual_roi_labels.json` when the three label states need to be
  preserved explicitly.

## ROI filtering workflow

```text
All ROIs detected by Suite2p
        |
        | Morphology QC target structure
        | (neuron, dendrite, or cerebellum_lax)
        v
        +-------------------------------+
        |                               |
        v                               v
Pass morphology QC              Fail morphology QC
Start Unlabeled and visible      Start Bad and hidden
        |                               |
        +---------------+---------------+
                        |
                        | Manual review of ROI shape
                        | and dF/F activity
                        v
            Good / Bad / Unlabeled labels
                        |
                        v
iscell_qc.npy or manual-label JSON for downstream analysis
```

The interactive HTML contains the original Suite2p ROI set. Morphology QC
provides the initial filtering state, and the reviewer can then refine that
selection by inspecting each ROI and its dF/F trace.

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

## Generating Preprocessing QC Interactive `.html` ROI Labeler and Summary `.pdf`

The summary stage creates:

```text
<session>_preprocessing_summary.pdf
<session>_interactive_fov_roi_dff.html
```

Run the commands below from the `2p_imaging` repository root with the
preprocessing QC environment available.

### Processed session requirements

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

### Already processed session: local

```bash
python -m utils_2p.preprocessing_summary /path/to/processed/session
```

The PDF and interactive HTML are written into the processed session directory.

### Already processed session: PACE

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

### As part of the preprocessing pipeline

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
