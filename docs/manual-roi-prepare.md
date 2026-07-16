# Step 1: Prepare Suite2p-compatible inputs

Manual ROI extraction needs a Suite2p-compatible working folder. The GUI can
load ROI masks from `stat.npy` alone, but it cannot extract fluorescence for a
new manual ROI unless the folder also has trace files, settings, and the
registered movie binary.

For a normal Suite2p output, use:

```text
session/
└── suite2p/
    └── plane0/
        ├── stat.npy
        ├── F.npy
        ├── Fneu.npy
        ├── spks.npy
        ├── iscell.npy
        ├── ops.npy
        └── data.bin
```

For older `qc_results/` or `manual_qc_results/` folders, create a temporary
Suite2p GUI compatible directory:

```bash
python - <<'PY'
from utils_2p.manual_rois import create_manual_roi_workspace

result = create_manual_roi_workspace(
    qc_dir="/path/to/session/manual_qc_results",
    suite2p_plane_dir="/path/to/session/suite2p/plane0",
    workspace_dir="/path/to/session/manual_roi_workspace",
)
print(result)
PY
```

This workspace maps old QC names to Suite2p names:

```text
stat.npy     -> stat.npy
fluo.npy     -> F.npy
neuropil.npy -> Fneu.npy
```

Suite2p also needs `data.bin`. If it already exists elsewhere, symlink it into
the Suite2p plane folder before creating the temporary workspace:

```bash
ln -s /path/to/existing/data.bin /path/to/session/suite2p/plane0/data.bin
```

If `data.bin` must be rebuilt from TIFFs without rerunning the full pipeline,
use the binary prebuild helper:

```bash
utils_2p/scripts/run_suite2p_binary_prebuild.sh \
  /path/to/processed/session \
  /path/to/raw_tiff_folder \
  /path/to/output_root \
  8 \
  5000 \
  dendrite
```

Run these helper commands from the `2p_imaging` repo root:

```text
/storage/home/hcoda1/3/grubin6/2p_imaging
```

Full details, including two-channel behavior and scratch-storage guidance, are
in the Adding Manual ROIs overview.
