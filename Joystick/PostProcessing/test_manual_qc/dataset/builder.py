"""
dataset/builder.py

Builds ROI QC dataset for CNN training.

Output:
    patches.npy   -> (N, 2, H, W)
        channel 0: normalized mean image
        channel 1: ROI binary mask

    labels.npy    -> 0/1 labels
    roi.npy       -> ROI IDs
    session.npy   -> session names

FIXES:
- removed RGB rendering (no UI leakage into training)
- uses crop.make_two_channel_patch
- session-based grouping preserved
- safe ROI extraction via np.unique
"""

from __future__ import annotations

import os
import numpy as np
import h5py
from tqdm import tqdm

from crop import make_two_channel_patch


# -------------------------------------------------------------------------
# Labels
# -------------------------------------------------------------------------

def load_labels(session_path, roi_ids):
    """
    Map ROI IDs -> {1 good, 0 bad, -1 unlabeled}
    """
    labels = {int(r): -1 for r in roi_ids}

    path = os.path.join(session_path, "ROI_label.h5")

    if not os.path.exists(path):
        return labels

    with h5py.File(path, "r") as f:

        if "good_roi" in f:
            for r in f["good_roi"][:]:
                labels[int(r) + 1] = 1  # Suite2p offset

        if "bad_roi" in f:
            for r in f["bad_roi"][:]:
                labels[int(r) + 1] = 0

    return labels


# -------------------------------------------------------------------------
# Session loading
# -------------------------------------------------------------------------

def load_session(session_path):
    ops = np.load(
        os.path.join(session_path, "suite2p", "plane0", "ops.npy"),
        allow_pickle=True
    ).item()

    masks = np.load(
        os.path.join(session_path, "qc_results", "masks.npy")
    )

    mean_img = ops["meanImg"]

    return mean_img, masks

def count_labeled_rois(session_paths):
    """
    Count the total number of labeled ROIs across all sessions.
    """

    total = 0

    for sp in session_paths:

        _, masks = load_session(sp)

        roi_ids = np.unique(masks)
        roi_ids = roi_ids[roi_ids > 0].astype(int)

        y_map = load_labels(sp, roi_ids)

        total += sum(
            y_map.get(int(r), -1) >= 0
            for r in roi_ids
        )

    return total

# -------------------------------------------------------------------------
# Dataset builder
# -------------------------------------------------------------------------

def build_dataset(session_paths, output_dir, size=64):

    # patches = []
    # labels = []
    # rois = []
    # sessions = []
    print("Counting labeled ROIs...")

    n_samples = count_labeled_rois(session_paths)

    print(f"Found {n_samples} labeled ROIs.")

    patches = np.empty(
        (n_samples, 2, size, size),
        dtype=np.float32
    )

    labels = np.empty(
        n_samples,
        dtype=np.int8
    )

    rois = np.empty(
        n_samples,
        dtype=np.int32
    )

    sessions = np.empty(
        n_samples,
        dtype="<U32"
    )

    idx = 0

    for sp in session_paths:

        print(f"\nProcessing: {os.path.basename(sp)}")
    
        mean_img, masks = load_session(sp)
    
        # robust ROI extraction (native IDs)
        roi_ids = np.unique(masks)
        roi_ids = roi_ids[roi_ids > 0].astype(int)
    
        y_map = load_labels(sp, roi_ids)
    
        for roi_id in tqdm(roi_ids, desc="ROIs"):
    
            label = y_map.get(int(roi_id), -1)
            if label < 0:
                continue
    
            patch = make_two_channel_patch(
                mean_img,
                masks,
                roi_id,          # ✅ NO -1 SHIFT
                patch_size=size,
            )
    
            patches[idx] = patch
            labels[idx] = label
            rois[idx] = int(roi_id)
            sessions[idx] = os.path.basename(sp)

            idx += 1
            # patches.append(patch)
            # labels.append(label)
            # rois.append(int(roi_id))
            # sessions.append(os.path.basename(sp))
        
    # patches = np.stack(patches).astype(np.float32)
    # labels = np.array(labels, dtype=np.int8)
    # rois = np.array(rois, dtype=np.int32)
    # sessions = np.asarray(sessions, dtype=str)

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "patches.npy"), patches)
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    np.save(os.path.join(output_dir, "roi.npy"), rois)
    np.save(os.path.join(output_dir, "session.npy"), sessions)

    print("\nDONE")
    print("Patches :", patches.shape)
    print("Labels  :", labels.shape)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":

    # Example usage

    sessions = [
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA16_LG/SA16_20260103", # (538/1326) = 0.406
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA16_LG/SA16_20260115", # (547/907) = 0.603
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA16_LG/SA16_20260119", # (606/1023) = 0.592
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA16_LG/SA16_20260122", # (608/1220) = 0.498
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA16_LG/SA16_20260202", # (798/1178) = 0.677
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA16_LG/SA16_20260303", # (711/1155) = 0.616

        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA17_LG/SA17_20260116", # (629/1248) = 0.504
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA17_LG/SA17_20260114", # (616/1273) = 0.484

        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA19_LG/SA19_20260302", # (590/1354) = 0.436
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA19_LG/SA19_20260223", # (596/1209) = 0.493
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA19_LG/SA19_20260112", # (625/1320) = 0.473
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA19_LG/SA19_20260213", # (581/1330) = 0.437
    
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA11_LG/SA11_20250813", # (667/1138) = 0.437
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA11_LG/SA11_20250829", # (492/1170) = 0.421
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA11_LG/SA11_20250828", # (454/828) = 0.548






        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA09_LG/SA09_20250807",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA09_LG/SA09_20250813",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA09_LG/SA09_20250814",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA11_LG/SA11_20250811",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA11_LG/SA11_20250901",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA11_LG/SA11_20250903",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA11_LG/SA11_20250904",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA11_LG/SA11_20250910",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA16_LG/SA16_20251226",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA16_LG/SA16_20260112",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA16_LG/SA16_20260226",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA17_LG/SA17_20251230",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA17_LG/SA17_20260112",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA17_LG/SA17_20260126",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA17_LG/SA17_20260224",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA17_LG/SA17_20260227",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA19_LG/SA19_20260122",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA19_LG/SA19_20260202",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA19_LG/SA19_20260210",
        "/storage/project/r-fnajafi3-0/shared/2P_Imaging/SA19_LG/SA19_20260226",

    ]

    build_dataset(
        sessions,
        output_dir="./dataset_out",
        size=64,
    )