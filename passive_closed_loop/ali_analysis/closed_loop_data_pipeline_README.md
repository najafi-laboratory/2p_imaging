# Closed-Loop Data Pipeline

`closed_loop_data_pipeline.py` is the clean first step for deeper analysis of a
closed-loop two-photon session. It reads the session directory and organizes the
data into stimulus, two-photon, behavior, and voltage containers.

## What It Loads

Stimulus and behavior CSVs:

```text
sensorimotor_mismatch_example.csv
orientations_orientations0.csv
orientations_logger.csv
```

Two-photon processed data:

```text
qc_results/dff.h5
qc_results/smoothed.h5
qc_results/spikes.h5
qc_results/fluo.npy
qc_results/neuropil.npy
qc_results/masks.npy
qc_results/stat.npy
suite2p/plane0/*.npy
ops.npy
masks.npy
```

Behavior and voltages:

```text
move_offset.h5
raw_voltages.h5
*VoltageRecording*.csv
```

## Memory Behavior

Large data are not blindly loaded into RAM.

`npy` files are memory-mapped where possible:

```python
session.two_photon.qc_npy["fluo"].array
session.two_photon.suite2p["F"].array
```

HDF5 datasets are returned as references:

```python
dff_ref = session.two_photon.qc_h5["dff"]
dff_shape = dff_ref.shape
dff_first_1000_frames = dff_ref.read(np.s_[:, :1000])
```

Raw voltage streams are also HDF5 references:

```python
vol_time = session.voltages.raw_h5["raw/vol_time"].read(slice(0, 1000))
vol_img = session.voltages.raw_h5["raw/vol_img"].read(slice(0, 1000))
```

The large Prairie voltage CSV is not loaded by default. The pipeline reads only
its path and columns. To inspect a small sample:

```python
from ali_analysis.closed_loop_data_pipeline import read_voltage_csv_sample

sample = read_voltage_csv_sample(session.voltages.voltage_csv_path, nrows=1000)
```

## Notebook Snippet

```python
from pathlib import Path
import sys
import numpy as np
from IPython.display import display

code_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Code/2p")
data_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/2p_Data/VW01_20260520_Closed_Loop_test-1556")

if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from ali_analysis.closed_loop_data_pipeline import (
    load_closed_loop_session,
    h5_dataset_summary,
    npy_array_summary,
)

session = load_closed_loop_session(
    data_dir=data_dir,
    load_logger=True,
    include_qc=True,
    include_suite2p=True,
    include_voltages=True,
    npy_mmap_mode="r",
    voltage_csv_sample_rows=0,
)

display(session.summary())
display(session.manifest)
display(h5_dataset_summary(session.two_photon.qc_h5))
display(npy_array_summary(session.two_photon.qc_npy))
display(npy_array_summary(session.two_photon.suite2p))
display(h5_dataset_summary(session.behavior.move_offset))
display(h5_dataset_summary(session.voltages.raw_h5))

print("Task CSV:", session.stimulus.task.shape)
print("Bonsai output CSV:", session.stimulus.bonsai_output.shape)
print("Logger CSV:", session.stimulus.logger.shape)
print("Voltage CSV:", session.voltages.voltage_csv_path)
print("Voltage CSV columns:", session.voltages.voltage_csv_columns)

# Example: read only a small slice for deeper analysis.
dff_10_cells_1000_frames = session.two_photon.qc_h5["dff"].read(np.s_[:10, :1000])
voltage_time_first_1000 = session.voltages.raw_h5["raw/vol_time"].read(slice(0, 1000))
```

## Why This Structure

The loader separates reading from analysis. Later analysis code can use:

```python
session.stimulus.task
session.stimulus.bonsai_output
session.two_photon.qc_h5["dff"]
session.two_photon.suite2p["F"]
session.behavior.move_offset["xoff"]
session.voltages.raw_h5["raw/vol_time"]
```

without re-discovering file names or re-implementing HDF5/NumPy loading each
time.
