# Interactive dF/F ROI Viewer

`interactive_dff_roi_viewer.py` creates a standalone HTML viewer for ROI traces.

The viewer shows:

1. Functional-channel mean image with ROI edge circles.
2. Anatomical-channel mean image with ROI edge circles.
3. ROI mask image.
4. Raw dF/F trace for the selected ROI.
5. Z-scored dF/F trace for the selected ROI.

The top ROI circles are clickable. The trace panels support mouse-wheel zoom,
click-drag panning, and double-click reset. The HTML is standalone and can be
shared directly.

## Notebook Usage

```python
from pathlib import Path
import sys
import importlib
from IPython.display import HTML, display

code_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Code/2p")
data_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/2p_Data/VW01_20260520_Closed_Loop_test-1556")
output_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/VW01_20260520_Closed_Loop_test-1556")
output_dir.mkdir(parents=True, exist_ok=True)

if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from ali_analysis import interactive_dff_roi_viewer
importlib.reload(interactive_dff_roi_viewer)

html_path = interactive_dff_roi_viewer.create_interactive_dff_viewer(
    data_dir=data_dir,
    output_path=output_dir / "interactive_dff_roi_viewer.html",
)

print(html_path)
display(HTML(f'<a href="{html_path}" target="_blank">Open interactive dF/F viewer</a>'))
```

## Command-Line Usage

```bash
cd /home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Code/2p

.venv/bin/python ali_analysis/interactive_dff_roi_viewer.py \
  --data-dir "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/2p_Data/VW01_20260520_Closed_Loop_test-1556" \
  --output "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/VW01_20260520_Closed_Loop_test-1556/interactive_dff_roi_viewer.html"
```

## File Size

The generated HTML embeds the full dF/F matrix as float32 data. For the current
session the file is about 44 MB. That is expected and keeps the file shareable
without requiring a Python server.
