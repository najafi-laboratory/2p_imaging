# Event Alignment Validation

`event_alignment_validation.py` validates the timing chain needed before
event-aligned neural analysis.

It checks:

1. Bonsai generated stimulus IDs against logger `StimStart` and `StimEnd` IDs.
2. Performed block-2 event timing using `orientations_orientations0.csv`.
3. Event start times mapped onto nearest 2P scope-exposure voltage edges and
   dF/F frame indices.
4. dF/F frame count against Suite2p `ops.npy`, movement offsets, and imaging
   voltage pulses.
5. Raw `*VoltageRecording*.csv` physical input health.
6. Processed/binarized HDF5 voltage channel health as a secondary comparison.

For block 2, this pipeline intentionally uses Bonsai output
`orientations_orientations0.csv`, not the planned rows in
`sensorimotor_mismatch_example.csv`, because the block is behavior-dependent.

The full Bonsai output has one row per short stimulus presentation. In this
session that is 45,470 rows over about 26 minutes, which is correct for the
visual stimulus stream but too dense to call "trials" for neural averaging. The
validator therefore writes two tables:

```text
block2_performed_events_with_dff_frames.csv      # low-level timing/QC table
block2_analysis_trials_with_dff_frames.csv       # trial-level alignment table
```

Use the trial-level table for dF/F and behavior alignment. It contains all
validated motor oddball events plus standard 0 degree trial starts. A standard
trial start is defined as the first valid standard `StimStart` between two
consecutive oddball rows.

The primary imaging-frame voltage source is the raw Prairie voltage CSV:

```text
VW01_20260520_Closed_Loop_test-1556_Cycle00001_VoltageRecording_001.csv
```

Specifically, the validator uses `Input 3`, which is the scope exposure channel
in the wiring diagram. This is preferred over `raw_voltages.h5/raw/vol_img`
because the HDF5 stream is already processed/binarized. The CSV threshold is
inferred from the raw analog voltage range for each TTL channel and saved in
`physical_voltage_input_health.csv`.

## Notebook Usage

```python
from pathlib import Path
import sys
import importlib
from IPython.display import display

code_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Code/2p")
data_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/2p_Data/VW01_20260520_Closed_Loop_test-1556")
output_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/VW01_20260520_Closed_Loop_test-1556/alignment_validation")
output_dir.mkdir(parents=True, exist_ok=True)

if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import event_alignment_validation
importlib.reload(event_alignment_validation)

result = event_alignment_validation.run_event_alignment_validation(
    data_dir=data_dir,
    output_dir=output_dir,
    scan_voltage_csv_inputs=True,  # required for raw CSV Input 3 frame alignment
)

print("Event table:", result.event_table_path)
print("Analysis trial table:", result.analysis_trial_table_path)
print("Validation checks:", result.validation_checks_path)
print("Report:", result.report_path)

display(result.validation_checks)
display(result.dff_length_checks)
display(result.raw_h5_health)
display(result.physical_input_health)
display(result.start_end_alignment.sort_values("abs_error_ms").head(12))
display(result.stim_duration_summary)
display(result.stim_duration_outliers.head())

# Use this table for later event-aligned dF/F analysis.
analysis_trials = result.analysis_trial_table
display(analysis_trials.groupby("Analysis_Condition").size())
display(analysis_trials.head())
```

## Important Output

The most important neural-alignment output is:

```text
block2_analysis_trials_with_dff_frames.csv
```

Each row is one analysis trial and includes:

```text
Analysis_Trial
Analysis_Condition
Selection_Rule
Id
Trial_Number
Trial_Type
Stim_Start_sec
Stim_End_sec
Nearest_Dff_Frame
Nearest_Dff_Time_sec
Stim_To_Dff_Frame_Delta_sec
Imaging_Frame_Source
```

For this session the expected trial-level counts are 138 standard-control trial
starts and 35 trials for each oddball condition: omission, halt,
orientation_90, and orientation_45. The count is 138 because one pair of
consecutive oddballs has no standard row between them.

The full low-level validation output is:

```text
block2_performed_events_with_dff_frames.csv
```

Each row is one performed block-2 stimulus presentation and includes:

```text
Id
Trial_Number
Trial_Type
Stim_Start_sec
Stim_End_sec
Logger_Duration_sec
Nearest_Dff_Frame
Nearest_Dff_Time_sec
Stim_To_Dff_Frame_Delta_sec
Imaging_Frame_Source
```

Use `Nearest_Dff_Frame` from the trial-level table as the first-pass frame
index for event-aligned dF/F.

## Current Session Result

The current session validates well for logger IDs, dF/F length, imaging TTLs,
and event-to-dF/F frame mapping. The physical photodiode voltage input is not
usable for per-event hardware validation in this recording because it only has
one rising and one falling edge, while the logger has 93,602 photodiode state
rows.

So for this session:

* Use Bonsai logger event times and Bonsai output IDs for event identity.
* Use raw `VoltageRecording_001.csv` `Input 3` scope exposure edges for mapping
  event times to dF/F frames.
* Use `block2_analysis_trials_with_dff_frames.csv` for trial-level neural and
  behavior alignment.
* Use `raw_voltages.h5/raw/vol_img` only as a secondary comparison/fallback.
* Do not use the physical photodiode voltage input as the per-stimulus event
  source unless the acquisition wiring/logging is fixed in a later session.

## Command-Line Usage

```bash
cd /home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Code/2p

.venv/bin/python event_alignment_validation.py \
  --data-dir "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/2p_Data/VW01_20260520_Closed_Loop_test-1556" \
  --output-dir "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/VW01_20260520_Closed_Loop_test-1556/alignment_validation"
```

To skip the large physical voltage CSV scan and fall back to processed
`raw_voltages.h5/raw/vol_img`:

```bash
.venv/bin/python event_alignment_validation.py \
  --data-dir "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/2p_Data/VW01_20260520_Closed_Loop_test-1556" \
  --output-dir "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/VW01_20260520_Closed_Loop_test-1556/alignment_validation" \
  --skip-voltage-csv-scan
```
