# Passive Closed-Loop 2P Analysis

This folder contains analysis tools for passive visual closed-loop two-photon
experiments with sensorimotor mismatch, motor oddball, voltage recording, and
Suite2p-derived dF/F data.

The code was written for sessions where Bonsai receives a planned task CSV and
writes a performed stimulus-output CSV. For behavior-dependent blocks, especially
the motor oddball block, downstream analysis must use the performed Bonsai output
CSV rather than the planned input CSV.

## Main Workflow

Run the analysis in this order:

1. Compare the planned task CSV against the Bonsai output CSV.
2. Load all session files through the data pipeline and inspect the manifest.
3. Create the interactive ROI/dF/F viewer for field-of-view quality control.
4. Validate logger events, voltage edges, dF/F frame timing, and block-2
   performed stimulus events.
5. Use the validated trial-level analysis table to make event-aligned dF/F and
   wheel-speed summary figures. Standard 0 degree rows are trial starts, not
   every short stimulus presentation.

## Code Files

| File | Purpose |
| --- | --- |
| `closed_loop_data_pipeline.py` | Loads CSV, H5, NPY, Suite2p, QC, behavior, and voltage files into structured session objects. |
| `task_vs_bonsai_analysis.py` | Compares the task file Bonsai reads with the stimulus file Bonsai generated, including block-2 checks. |
| `interactive_dff_roi_viewer.py` | Generates a shareable HTML viewer for mean images, ROI masks, raw dF/F, and z-scored dF/F. |
| `event_alignment_validation.py` | Validates logger timestamps, voltage health, dF/F frame counts, and produces both the low-level performed-event QC table and the trial-level analysis table. |
| `aligned_dff_behavior_analysis.py` | Produces one ordered PDF with event-aligned population z-dF/F and wheel-speed traces. |

Each script also has a companion `*_README.md` with script-specific usage.

## Expected Dataset Structure

A session directory should contain the files below. Some QC and Suite2p files
are optional for specific steps, but the alignment workflow needs the stimulus,
logger, dF/F, ops, and voltage files.

```text
<session_dir>/
  sensorimotor_mismatch_example.csv              # Planned task CSV read by Bonsai
  orientations_orientations0.csv                 # Performed Bonsai stimulus output
  orientations_logger.csv                        # Start/stop/stim/photodiode/encoder timestamps
  raw_voltages.h5                                # Voltage recorder H5
  <session>_Cycle00001_VoltageRecording_001.csv  # Required raw analog voltage CSV for preferred frame alignment
  ops.npy                                        # Frame rate and imaging metadata
  masks.npy                                      # Optional ROI masks
  move_offset.h5                                 # Optional motion offsets
  qc_results/
    dff.h5                                      # Required dF/F matrix, ROI x frame
    fluo.npy
    masks.npy
    neuropil.npy
    smoothed.h5
    spikes.h5
    stat.npy
  suite2p/
    plane0/
      F.npy
      F_chan2.npy
      Fneu.npy
      Fneu_chan2.npy
      iscell.npy
      ops.npy
      redcell.npy
      spks.npy
      stat.npy
```

Important convention:

- `sensorimotor_mismatch_example.csv` is the planned input file.
- `orientations_orientations0.csv` is the actual Bonsai output file.
- For the second motor-oddball block, use `orientations_orientations0.csv`
  because that block depends on mouse behavior.

## Python Requirements

The scripts use standard scientific Python packages:

```text
numpy
pandas
h5py
matplotlib
seaborn
plotly
```

Notebook display examples also use IPython/Jupyter.

## Generated Outputs

Recommended output root:

```text
<figure_or_output_root>/<session_name>/
```

The scripts generate:

```text
task_vs_bonsai/
  task_vs_bonsai_timeline.pdf
  block2_task_vs_bonsai_comparison.pdf
  task_vs_bonsai_summary.csv
  task_vs_bonsai_block_summary.csv
  task_vs_bonsai_checks.csv

interactive_dff_roi_viewer.html

alignment_validation/
  block2_analysis_trials_with_dff_frames.csv
  block2_performed_events_with_dff_frames.csv
  event_alignment_validation_checks.csv
  dff_voltage_length_validation.csv
  raw_h5_voltage_channel_health.csv
  physical_voltage_input_health.csv
  logger_start_stop_voltage_alignment.csv
  stim_duration_summary.csv

aligned_dff_behavior/
  aligned_dff_behavior_all_conditions.pdf
  aligned_dff_behavior_condition_summary.csv
  aligned_dff_behavior_skipped_conditions.csv
```

The aligned dF/F behavior PDF has two pages:

1. Orientation-aligned events ordered as 0, 22.5, 45, 67.5, 90, 112.5, 135,
   and 157.5 degrees. With the reference session, 0 degrees uses standard
   trial starts between consecutive oddballs.
2. Block-2 motor oddball conditions ordered as standard 0 degree, omission,
   halt, orientation 90, and orientation 45.

For each subplot, the line is the mean and the shaded band is SEM. Legends are
placed beside the subplots and include neuron and trial counts.

## Voltage Alignment Source

The preferred frame-alignment voltage source is the raw Prairie
`*VoltageRecording*.csv` file, not the processed/binarized `raw_voltages.h5`
stream. The validator uses:

```text
Input 3 = scope exposure / 2P imaging frame TTL
```

The threshold for each raw TTL input is inferred from the analog voltage range
and saved in `physical_voltage_input_health.csv`. The processed HDF5 voltage
channels are still summarized as a secondary comparison and fallback.

## Notebook Usage

`2p_analysis.ipynb` is the interactive notebook entry point for stepping through
the workflow. Use it as a session-specific driver, not as the source of reusable
logic. The reusable code lives in the Python scripts listed above.

Recommended notebook setup cell:

```python
from pathlib import Path
import sys
import importlib
from IPython.display import display

code_dir = Path("/path/to/2p_imaging/passive_closed_loop")
data_dir = Path("/path/to/session_dir")
output_root = Path("/path/to/figures") / data_dir.name
output_root.mkdir(parents=True, exist_ok=True)

if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))
```

Then run each script from the notebook:

```python
import task_vs_bonsai_analysis
importlib.reload(task_vs_bonsai_analysis)

task_result = task_vs_bonsai_analysis.run_analysis(
    data_dir=data_dir,
    output_dir=output_root,
)

display(task_result.overall_summary)
display(task_result.block2_summary)
display(task_result.checks)
```

```python
import event_alignment_validation
importlib.reload(event_alignment_validation)

validation_result = event_alignment_validation.run_event_alignment_validation(
    data_dir=data_dir,
    output_dir=output_root / "alignment_validation",
    scan_voltage_csv_inputs=True,
)

display(validation_result.validation_checks)
display(validation_result.dff_length_checks)
display(validation_result.stim_duration_summary)
display(validation_result.analysis_trial_table.groupby("Analysis_Condition").size())
```

```python
import aligned_dff_behavior_analysis
importlib.reload(aligned_dff_behavior_analysis)

aligned_result = aligned_dff_behavior_analysis.run_aligned_dff_behavior_analysis(
    data_dir=data_dir,
    output_dir=output_root / "aligned_dff_behavior",
    event_table_path=output_root / "alignment_validation" / "block2_analysis_trials_with_dff_frames.csv",
    pre_sec=2.0,
    post_sec=4.0,
    exclude_duration_outliers=True,
)

print("Combined PDF:", aligned_result.pdf_path)
display(aligned_result.condition_summary)
display(aligned_result.skipped_conditions)
```

## Validation Notes

For the reference session used during development, block 2 was validated from
the Bonsai output and logger files:

- Bonsai output block 2 contained 45,470 performed events.
- The trial-level analysis table contained 278 events: 138 standard 0 degree
  trial starts and 35 trials for each motor oddball condition.
- Standard 0 degree trial starts are the first valid standard `StimStart`
  between two consecutive oddball rows. One inter-oddball interval in the
  reference session has no standard row, so the standard count is 138.
- Logger stimulus start/end IDs matched those Bonsai output IDs exactly.
- dF/F data contained 138 ROIs and 61,191 frames at 30 Hz.
- Raw VoltageRecording CSV `Input 3` had 61,195 rising edges, four more than
  the dF/F frame count, which is acceptable for this session.
- All block-2 performed events mapped to valid dF/F frames.
- Physical photodiode input was not usable for per-stimulus validation in that
  session, so alignment used logger stimulus timestamps and voltage timing
  checks instead.

These numbers are session-specific and should not be hard-coded into new
analyses. Use the validation CSVs produced by `event_alignment_validation.py`
for each session.
