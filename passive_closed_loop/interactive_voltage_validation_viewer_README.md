# Interactive Voltage Validation Viewer

`interactive_voltage_validation_viewer.py` creates a standalone HTML file for
inspecting the raw Prairie voltage CSV against the events used for alignment.

It plots all raw voltage inputs:

```text
Input 0  Trial sync / EBC
Input 1  Photodiode
Input 2  HiFi TTL
Input 3  Scope exposure / 2P frame TTL
Input 4  HiFi audio waveform
Input 5  FLIR output / camera strobe
Input 6  Encoder phase A
Input 7  Encoder phase B
```

The top event raster overlays:

```text
Analysis trial starts from logger
Analysis trial ends from logger
Nearest dF/F frame assigned to each analysis trial
Input 3 rising edges used as imaging-frame TTLs
Input 1 photodiode threshold crossings
```

By default the viewer uses:

```text
alignment_validation/block2_analysis_trials_with_dff_frames.csv
```

That means the vertical event ticks are the trial-level standard starts and
oddball events, not all 45,470 short stimulus presentations. If you explicitly want to
inspect every low-level stimulus presentation, pass
`block2_performed_events_with_dff_frames.csv`; explicit paths are honored.

For a full-session view, the raw traces are drawn as min/max envelopes per time
bin so the HTML remains usable. For raw-like detail, generate a short time
window with `time_start_sec` and `time_end_sec`.

## Notebook Usage

```python
from pathlib import Path
import sys
import importlib
from IPython.display import HTML, display

code_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Code/2p")
data_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/2p_Data/VW01_20260520_Closed_Loop_test-1556")
output_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/VW01_20260520_Closed_Loop_test-1556")
event_table = output_dir / "alignment_validation" / "block2_analysis_trials_with_dff_frames.csv"
physical_health = output_dir / "alignment_validation" / "physical_voltage_input_health.csv"

if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import interactive_voltage_validation_viewer
importlib.reload(interactive_voltage_validation_viewer)

result = interactive_voltage_validation_viewer.create_interactive_voltage_validation_viewer(
    data_dir=data_dir,
    output_path=output_dir / "interactive_voltage_validation_viewer_0_60s.html",
    event_table_path=event_table,
    physical_health_path=physical_health,
    time_start_sec=0,
    time_end_sec=60,
    max_bins=60_000,
)

display(result.summary)
display(HTML(f'<a href="{result.html_path}" target="_blank">Open voltage validation viewer</a>'))
```

For a full-session overview:

```python
result = interactive_voltage_validation_viewer.create_interactive_voltage_validation_viewer(
    data_dir=data_dir,
    output_path=output_dir / "interactive_voltage_validation_viewer_full.html",
    event_table_path=event_table,
    physical_health_path=physical_health,
    max_bins=80_000,
)
```

## Command-Line Usage

```bash
cd /home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Code/2p

.venv/bin/python interactive_voltage_validation_viewer.py \
  --data-dir "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/2p_Data/VW01_20260520_Closed_Loop_test-1556" \
  --output-path "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/VW01_20260520_Closed_Loop_test-1556/interactive_voltage_validation_viewer_0_60s.html" \
  --event-table "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/VW01_20260520_Closed_Loop_test-1556/alignment_validation/block2_analysis_trials_with_dff_frames.csv" \
  --physical-health "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/VW01_20260520_Closed_Loop_test-1556/alignment_validation/physical_voltage_input_health.csv" \
  --time-start-sec 0 \
  --time-end-sec 60
```
