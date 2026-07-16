# Step 2: Add Manual ROIs in Suite2p GUI

Open a PACE interactive desktop through the PACE OnDemand dashboard:

```text
https://ondemand-phoenix.pace.gatech.edu/pun/sys/dashboard
```

You must be connected to the Georgia Tech VPN to access the dashboard.

Launch the patched Suite2p 1.x GUI:

```bash
~/suite2p1_gui/bin/suite2p
```

Use this specific Suite2p 1.x installation for manual ROI work. The manual
labelling flow has local patches for GUI extraction, ROI statistics, and manual
mask display behavior that are not present in the unpatched PyPI install.

The symlink points to:

```text
/storage/project/r-fnajafi3-0/grubin6/.conda/envs/suite2p1_gui
```

The command examples assume access to the `2p_imaging` repository:

```text
/storage/home/hcoda1/3/grubin6/2p_imaging
```

The helper package lives at:

```text
/storage/home/hcoda1/3/grubin6/2p_imaging/utils_2p
```

GUI steps:

1. Open the target `stat.npy`.
2. Open the manual labelling window.
3. Draw the custom ROI.
4. Click `Extract ROI` once and wait for extraction to finish.
5. Click `Save and quit`.

After adding, extracting, and saving a manual ROI in a temporary workspace,
export the workspace back to the older QC folder:

```bash
python - <<'PY'
from utils_2p.manual_rois import export_manual_roi_workspace

result = export_manual_roi_workspace(
    workspace_dir="/path/to/session/manual_roi_workspace",
    qc_dir="/path/to/session/manual_qc_results",
    cleanup_workspace=True,
)
print(result)
PY
```

The export updates `stat.npy`, `fluo.npy`, `neuropil.npy`, `masks.npy`, and
existing derived `dff.h5`, `denoised_dff.h5`, and `spikes.h5` files. It does
not regenerate experiment-specific `events.h5` or `onsets.h5`.

Full details, including removing manual ROIs and downstream regeneration
warnings, are in the Adding Manual ROIs overview.
