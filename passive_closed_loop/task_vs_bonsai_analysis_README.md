# Task vs Bonsai Analysis

`task_vs_bonsai_analysis.py` compares the CSV that Bonsai reads with the CSV
that Bonsai generates during the task.

For the current session, the expected files are in:

```text
/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/2p_Data/VW01_20260520_Closed_Loop_test-1556
```

Required files:

```text
sensorimotor_mismatch_example.csv
orientations_orientations0.csv
```

Optional but recommended:

```text
orientations_logger.csv
```

The logger file is used to align Bonsai output rows to actual `StimStart` and
`StimEnd` times. If the logger file is missing, the script falls back to timing
computed from the CSV `Duration` column.

## Outputs

The script writes these files to the output directory:

```text
task_vs_bonsai_timeline.pdf
block2_task_vs_bonsai_comparison.pdf
task_vs_bonsai_summary.csv
task_vs_bonsai_block_summary.csv
task_vs_bonsai_checks.csv
```

`task_vs_bonsai_timeline.pdf` shows the full task CSV and Bonsai output side by
side. Each side includes block structure, orientation events, oddball events,
and a compact summary.

`block2_task_vs_bonsai_comparison.pdf` focuses on block 2. It compares the
block duration, orientation rows, oddball placement, and a numeric difference
table.

The summary CSVs are meant for inspection or downstream analysis. The checks CSV
flags important mismatches, including the current block-2 boundary issue where
Bonsai output contains 9 more standard rows than the task CSV block-2 boundary.

## Command-Line Usage

```bash
python task_vs_bonsai_analysis.py \
  --data-dir "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/2p_Data/VW01_20260520_Closed_Loop_test-1556" \
  --output-dir "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/2p_Data/VW01_20260520_Closed_Loop_test-1556"
```

## Notebook Usage

Import `run_analysis` from the code directory and point it at the session data
folder. The returned object contains paths to the generated figures and the
summary DataFrames.

```python
from pathlib import Path
import sys
from IPython.display import display

code_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Code/2p")
data_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/2p_Data/VW01_20260520_Closed_Loop_test-1556")
output_dir = data_dir

sys.path.insert(0, str(code_dir))
from task_vs_bonsai_analysis import run_analysis

result = run_analysis(
    data_dir=data_dir,
    output_dir=output_dir,
)

print("Main timeline:", result.main_figure_path)
print("Block 2 comparison:", result.block2_figure_path)

display(result.overall_summary)
display(result.block2_summary)
display(result.block2_difference)
display(result.checks)
```

## Implementation Notes

The script normalizes the task CSV and Bonsai output to the same column names.
It also converts Bonsai output orientations from radians to degrees when needed,
so the plots and tables use degrees consistently.

Vertical grid lines are disabled in all plots. Top and right spines are hidden.
The figures use Matplotlib GridSpec for stable layout.
