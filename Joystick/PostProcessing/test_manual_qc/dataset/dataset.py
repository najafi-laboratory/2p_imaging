"""
dataset/dataset.py

PyTorch Dataset for ROI QC classification.

Input:
    patches.npy  -> (N, 2, H, W)
    labels.npy   -> (N,)
    roi.npy      -> ROI IDs
    session.npy  -> session names

Key features:
- session-aware train/val split (NO ROI leakage)
- tensor conversion
- optional normalization hook
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset



# -------------------------------------------------------------------------
# Utility: session split
# -------------------------------------------------------------------------

def split_by_session(session_names, val_ratio=0.2, seed=42):
    """
    Splits dataset indices by session (not by ROI).

    Ensures no session appears in both train and val.
    For single-session datasets, falls back to an in-session split.
    """

    rng = np.random.default_rng(seed)
    session_names = np.asarray(session_names)

    unique_sessions = np.unique(session_names)
    if len(unique_sessions) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if len(unique_sessions) == 1:
        indices = np.arange(len(session_names), dtype=np.int64)
        if len(indices) <= 1:
            return indices, np.array([], dtype=np.int64)

        n_val = max(1, int(len(indices) * val_ratio))
        n_val = min(n_val, len(indices) - 1)
        rng.shuffle(indices)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        return train_idx.astype(np.int64), val_idx.astype(np.int64)

    rng.shuffle(unique_sessions)

    n_val = max(1, int(len(unique_sessions) * val_ratio))
    n_val = min(n_val, len(unique_sessions) - 1)

    val_sessions = set(unique_sessions[:n_val])
    train_sessions = set(unique_sessions[n_val:])

    train_idx = np.array(
        [i for i, s in enumerate(session_names) if s in train_sessions],
        dtype=np.int64,
    )
    val_idx = np.array(
        [i for i, s in enumerate(session_names) if s in val_sessions],
        dtype=np.int64,
    )

    if len(train_idx) == 0 and len(val_idx) > 0:
        train_idx = np.array([val_idx[0]], dtype=np.int64)
        val_idx = np.array(val_idx[1:], dtype=np.int64)

    return train_idx, val_idx


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class ROIQCDataset(Dataset):

    def __init__(
        self,
        patches_path: str,
        labels_path: str,
        roi_path: str = None,
        session_path: str = None,
        indices=None,
        augment=False,
    ):

        self.patches = np.load(patches_path).astype(np.float32)
        self.labels  = np.load(labels_path).astype(np.float32)

        self.rois = None
        if roi_path is not None:
            self.rois = np.load(roi_path)

        self.sessions = None
        if session_path is not None:
            self.sessions = np.load(session_path, allow_pickle=False)

        # -------------------------------------------------------------
        # Subset selection (train/val split)
        # -------------------------------------------------------------
        if indices is not None:
            indices = np.asarray(indices, dtype=np.int64)
            self.patches = self.patches[indices]
            self.labels  = self.labels[indices]

            if self.rois is not None:
                self.rois = self.rois[indices]

            if self.sessions is not None:
                self.sessions = self.sessions[indices]

        self.augment = augment
        self.patches = self.patches / 255.0

    # -----------------------------------------------------------------
    # length
    # -----------------------------------------------------------------

    def __len__(self):
        return len(self.labels)

    # -----------------------------------------------------------------
    # augmentation (light, safe for masks)
    # -----------------------------------------------------------------

    def _augment(self, x):
        """
        x: (2, H, W)
        """

        # random horizontal flip
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=2).copy()

        # random vertical flip
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=1).copy()

        return x

    # -----------------------------------------------------------------
    # get item
    # -----------------------------------------------------------------

    def __getitem__(self, idx):

        x = self.patches[idx]  # (2, H, W)
        y = self.labels[idx]

        if self.augment:
            x = self._augment(x)

        x = torch.tensor(np.ascontiguousarray(x), dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        assert x.ndim == 3, f"BAD SAMPLE SHAPE: {x.shape}, idx={idx}"

        return x, y


# -------------------------------------------------------------------------
# Convenience loader
# -------------------------------------------------------------------------

def load_datasets(
    data_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    augment: bool = True,
):
    """
    Returns:
        train_dataset, val_dataset
    """

    patches = f"{data_dir}/patches.npy"
    labels  = f"{data_dir}/labels.npy"
    rois    = f"{data_dir}/roi.npy"
    sessions = f"{data_dir}/session.npy"

    session_arr = np.load(sessions, allow_pickle=False)

    train_idx, val_idx = split_by_session(session_arr, val_ratio, seed)

    train_ds = ROIQCDataset(
        patches, labels, rois, sessions,
        indices=train_idx,
        augment=augment,
    )

    val_ds = ROIQCDataset(
        patches, labels, rois, sessions,
        indices=val_idx,
        augment=False,
    )

    return train_ds, val_ds