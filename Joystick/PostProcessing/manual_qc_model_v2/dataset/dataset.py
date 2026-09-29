"""
dataset/dataset.py

PyTorch Dataset for multimodal ROI QC classification.

Dataset files:
    patches.npy
        Shape: (N, 2, H, W)
        channel 0: normalized mean fluorescence image
        channel 1: ROI binary mask

    signals.npy
        Shape: (N, 2, T)
        channel 0: ROI fluorescence signal
        channel 1: neuropil signal

    metrics.npy
        Shape: (N, M)
        scalar morphology / fluorescence QC metrics

    labels.npy
        Shape: (N,)
        1 = Good
        0 = Bad

    roi.npy
        Shape: (N,)
        ROI IDs within their QC session

    session.npy
        Shape: (N,)
        session names

Key features:
    * session-aware train/validation split
    * no ROI leakage across sessions
    * image augmentation
    * independent fluorescence/neuropil normalization
    * training-set-only metric normalization
    * memory-efficient np.memmap loading
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset


# =========================================================================
# SESSION-AWARE TRAIN / VALIDATION SPLIT
# =========================================================================

def split_by_session(
    session_names,
    val_ratio=0.2,
    seed=42,
):
    """
    Split dataset indices by session rather than individual ROI.

    This prevents ROIs from the same imaging session appearing in both
    training and validation sets.

    For a single-session dataset, falls back to an ROI-level split.
    """

    rng = np.random.default_rng(seed)

    session_names = np.asarray(session_names)

    unique_sessions = np.unique(session_names)

    if len(unique_sessions) == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    # ---------------------------------------------------------------------
    # Single-session fallback
    # ---------------------------------------------------------------------

    if len(unique_sessions) == 1:

        indices = np.arange(
            len(session_names),
            dtype=np.int64,
        )

        if len(indices) <= 1:
            return (
                indices,
                np.array([], dtype=np.int64),
            )

        rng.shuffle(indices)

        n_val = max(
            1,
            int(len(indices) * val_ratio),
        )

        n_val = min(
            n_val,
            len(indices) - 1,
        )

        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        return (
            train_idx.astype(np.int64),
            val_idx.astype(np.int64),
        )

    # ---------------------------------------------------------------------
    # Multi-session split
    # ---------------------------------------------------------------------

    rng.shuffle(unique_sessions)

    n_val = max(
        1,
        int(len(unique_sessions) * val_ratio),
    )

    n_val = min(
        n_val,
        len(unique_sessions) - 1,
    )

    val_sessions = set(
        unique_sessions[:n_val]
    )

    train_sessions = set(
        unique_sessions[n_val:]
    )

    train_idx = np.array(
        [
            i
            for i, session in enumerate(session_names)
            if session in train_sessions
        ],
        dtype=np.int64,
    )

    val_idx = np.array(
        [
            i
            for i, session in enumerate(session_names)
            if session in val_sessions
        ],
        dtype=np.int64,
    )

    return train_idx, val_idx


# =========================================================================
# DATASET
# =========================================================================

class ROIQCDataset(Dataset):
    """
    Multimodal ROI QC dataset.

    Each sample contains:

        patch:
            (2, H, W)

        signal:
            (2, T)
            channel 0 = fluorescence
            channel 1 = neuropil

        metrics:
            (M,)

        label:
            scalar
    """

    def __init__(
        self,
        patches_path: str,
        signals_path: str,
        metrics_path: str,
        labels_path: str,
        roi_path: str | None = None,
        session_path: str | None = None,
        indices=None,
        augment=False,
        signal_augment=False,
        metric_mean=None,
        metric_std=None,
    ):

        # -----------------------------------------------------------------
        # Load arrays as memory-mapped files
        # -----------------------------------------------------------------

        self.patches = np.load(
            patches_path,
            mmap_mode="r",
        )

        self.signals = np.load(
            signals_path,
            mmap_mode="r",
        )

        self.metrics = np.load(
            metrics_path,
            mmap_mode="r",
        )

        self.labels = np.load(
            labels_path,
            mmap_mode="r",
        )

        # -----------------------------------------------------------------
        # Validate expected multimodal shapes
        # -----------------------------------------------------------------

        assert self.patches.ndim == 4, (
            f"Expected patches with shape (N, 2, H, W), "
            f"got {self.patches.shape}"
        )

        assert self.patches.shape[1] == 2, (
            f"Expected 2 image channels, "
            f"got {self.patches.shape}"
        )

        assert self.signals.ndim == 3, (
            f"Expected signals with shape (N, 2, T), "
            f"got {self.signals.shape}. "
            f"The builder must save both fluorescence and neuropil."
        )

        assert self.signals.shape[1] == 2, (
            f"Expected signals with 2 channels "
            f"(fluorescence, neuropil), "
            f"got {self.signals.shape}"
        )

        assert self.metrics.ndim == 2, (
            f"Expected metrics with shape (N, M), "
            f"got {self.metrics.shape}"
        )

        # -----------------------------------------------------------------
        # Optional metadata
        # -----------------------------------------------------------------

        self.rois = None

        if roi_path is not None:
            self.rois = np.load(
                roi_path,
                mmap_mode="r",
            )

        self.sessions = None

        if session_path is not None:
            self.sessions = np.load(
                session_path,
                allow_pickle=False,
                mmap_mode="r",
            )

        # -----------------------------------------------------------------
        # Alignment checks
        # -----------------------------------------------------------------

        n = len(self.labels)

        assert len(self.patches) == n, (
            f"Patch/label mismatch: "
            f"{len(self.patches)} vs {n}"
        )

        assert len(self.signals) == n, (
            f"Signal/label mismatch: "
            f"{len(self.signals)} vs {n}"
        )

        assert len(self.metrics) == n, (
            f"Metrics/label mismatch: "
            f"{len(self.metrics)} vs {n}"
        )

        if self.rois is not None:

            assert len(self.rois) == n, (
                f"ROI/label mismatch: "
                f"{len(self.rois)} vs {n}"
            )

        if self.sessions is not None:

            assert len(self.sessions) == n, (
                f"Session/label mismatch: "
                f"{len(self.sessions)} vs {n}"
            )

        # -----------------------------------------------------------------
        # Dataset indices
        #
        # Keep original arrays memory-mapped.
        # Only store the selected indices.
        # -----------------------------------------------------------------

        if indices is None:

            self.indices = np.arange(
                n,
                dtype=np.int64,
            )

        else:

            self.indices = np.asarray(
                indices,
                dtype=np.int64,
            )

        # -----------------------------------------------------------------
        # Augmentation settings
        # -----------------------------------------------------------------

        self.augment = augment
        self.signal_augment = signal_augment

        # -----------------------------------------------------------------
        # Metric normalization statistics
        #
        # IMPORTANT:
        # These MUST come from the training split only.
        # -----------------------------------------------------------------

        self.metric_mean = metric_mean
        self.metric_std = metric_std

        if self.metric_mean is not None:

            self.metric_mean = np.asarray(
                self.metric_mean,
                dtype=np.float32,
            )

        if self.metric_std is not None:

            self.metric_std = np.asarray(
                self.metric_std,
                dtype=np.float32,
            )

            self.metric_std = np.maximum(
                self.metric_std,
                1e-6,
            )

        # -----------------------------------------------------------------
        # Validate metric statistics
        # -----------------------------------------------------------------

        if (
            self.metric_mean is not None
            and self.metric_std is not None
        ):

            assert len(self.metric_mean) == self.metrics.shape[1], (
                f"Metric mean has shape {self.metric_mean.shape}, "
                f"but dataset contains {self.metrics.shape[1]} metrics."
            )

            assert len(self.metric_std) == self.metrics.shape[1], (
                f"Metric std has shape {self.metric_std.shape}, "
                f"but dataset contains {self.metrics.shape[1]} metrics."
            )

    # ---------------------------------------------------------------------
    # LENGTH
    # ---------------------------------------------------------------------

    def __len__(self):
        return len(self.indices)

    # ---------------------------------------------------------------------
    # IMAGE AUGMENTATION
    # ---------------------------------------------------------------------

    def _augment_patch(self, patch):
        """
        Apply identical spatial transformations to both image channels.

        patch:
            (2, H, W)

        channel 0:
            mean image

        channel 1:
            ROI mask
        """

        # Horizontal flip
        if np.random.rand() < 0.5:

            patch = np.flip(
                patch,
                axis=2,
            ).copy()

        # Vertical flip
        if np.random.rand() < 0.5:

            patch = np.flip(
                patch,
                axis=1,
            ).copy()

        return patch

    # ---------------------------------------------------------------------
    # SIGNAL NORMALIZATION
    # ---------------------------------------------------------------------

    @staticmethod
    def _normalize_signal(signal):
        """
        Normalize fluorescence and neuropil independently.

        Uses robust normalization:

            (x - median) / (1.4826 * MAD)

        If MAD is effectively zero, falls back to standard deviation.

        signal:
            (2, T)

        channel 0:
            fluorescence

        channel 1:
            neuropil

        returns:
            (2, T)
        """

        signal = signal.astype(
            np.float32,
            copy=True,
        )

        assert signal.ndim == 2, (
            f"Expected signal shape (2, T), "
            f"got {signal.shape}"
        )

        assert signal.shape[0] == 2, (
            f"Expected 2 signal channels "
            f"(fluorescence, neuropil), "
            f"got {signal.shape}"
        )

        for channel in range(2):

            x = signal[channel]

            median = np.median(x)

            mad = np.median(
                np.abs(x - median)
            )

            robust_std = 1.4826 * mad

            # Fallback for nearly constant traces
            if robust_std < 1e-6:

                robust_std = np.std(x)

            # Final safeguard
            if robust_std < 1e-6:

                robust_std = 1.0

            signal[channel] = (
                x - median
            ) / robust_std

        return signal

    # ---------------------------------------------------------------------
    # SIGNAL AUGMENTATION
    # ---------------------------------------------------------------------

    def _augment_signal(self, signal):
        """
        Light augmentation for fluorescence and neuropil traces.

        We do NOT reverse time because temporal direction matters.

        Amplitude scaling is applied independently to each channel.

        Small Gaussian noise is applied to both channels.
        """

        signal = signal.copy()

        # -------------------------------------------------------------
        # Independent amplitude scaling
        # -------------------------------------------------------------

        if np.random.rand() < 0.5:

            fluorescence_scale = np.random.uniform(
                0.9,
                1.1,
            )

            neuropil_scale = np.random.uniform(
                0.9,
                1.1,
            )

            signal[0] *= fluorescence_scale
            signal[1] *= neuropil_scale

        # -------------------------------------------------------------
        # Small Gaussian noise
        # -------------------------------------------------------------

        if np.random.rand() < 0.2:

            noise = np.random.normal(
                loc=0.0,
                scale=0.02,
                size=signal.shape,
            ).astype(np.float32)

            signal += noise

        return signal

    # ---------------------------------------------------------------------
    # METRIC NORMALIZATION
    # ---------------------------------------------------------------------

    def _normalize_metrics(self, metrics):
        """
        Apply training-set normalization to scalar metrics.

        metrics:
            (M,)
        """

        metrics = metrics.astype(
            np.float32,
            copy=True,
        )

        if (
            self.metric_mean is not None
            and self.metric_std is not None
        ):

            metrics = (
                metrics - self.metric_mean
            ) / self.metric_std

        return metrics

    # ---------------------------------------------------------------------
    # GET ITEM
    # ---------------------------------------------------------------------

    def __getitem__(self, idx):

        # Convert dataset-local index to original array index
        real_idx = self.indices[idx]

        # =============================================================
        # IMAGE
        # =============================================================

        patch = np.asarray(
            self.patches[real_idx],
            dtype=np.float32,
        ).copy()

        # Stored in 0-255 image range
        patch /= 255.0

        if self.augment:

            patch = self._augment_patch(
                patch
            )

        # =============================================================
        # SIGNAL
        # =============================================================

        signal = np.asarray(
            self.signals[real_idx],
            dtype=np.float32,
        ).copy()

        signal = self._normalize_signal(
            signal
        )

        if self.signal_augment:

            signal = self._augment_signal(
                signal
            )

        # =============================================================
        # METRICS
        # =============================================================

        metrics = np.asarray(
            self.metrics[real_idx],
            dtype=np.float32,
        ).copy()

        metrics = self._normalize_metrics(
            metrics
        )

        # =============================================================
        # LABEL
        # =============================================================

        label = np.float32(
            self.labels[real_idx]
        )

        # =============================================================
        # NUMPY -> TORCH
        # =============================================================

        patch = torch.from_numpy(
            np.ascontiguousarray(patch)
        )

        signal = torch.from_numpy(
            np.ascontiguousarray(signal)
        )

        metrics = torch.from_numpy(
            np.ascontiguousarray(metrics)
        )

        label = torch.tensor(
            label,
            dtype=torch.float32,
        )

        # =============================================================
        # SANITY CHECKS
        # =============================================================

        assert patch.ndim == 3, (
            f"Bad patch shape: {patch.shape}"
        )

        assert patch.shape[0] == 2, (
            f"Bad patch channels: {patch.shape}"
        )

        assert signal.ndim == 2, (
            f"Bad signal shape: {signal.shape}"
        )

        assert signal.shape[0] == 2, (
            f"Bad signal channels: {signal.shape}"
        )

        assert metrics.ndim == 1, (
            f"Bad metrics shape: {metrics.shape}"
        )

        return (
            patch,
            signal,
            metrics,
            label,
        )


# =========================================================================
# METRIC STATISTICS
# =========================================================================

def compute_metric_stats(
    metrics_path: str,
    train_indices,
):
    """
    Compute metric normalization statistics using ONLY training samples.

    This prevents validation-set information from leaking into training.

    Returns:
        mean:
            (M,)

        std:
            (M,)
    """

    metrics = np.load(
        metrics_path,
        mmap_mode="r",
    )

    train_indices = np.asarray(
        train_indices,
        dtype=np.int64,
    )

    train_metrics = np.asarray(
        metrics[train_indices],
        dtype=np.float32,
    )

    mean = train_metrics.mean(
        axis=0
    )

    std = train_metrics.std(
        axis=0
    )

    std = np.maximum(
        std,
        1e-6,
    )

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )


# =========================================================================
# CONVENIENCE LOADER
# =========================================================================

def load_datasets(
    data_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    augment: bool = True,
    signal_augment: bool = True,
):
    """
    Build train and validation datasets.

    Returns:
        train_dataset
        val_dataset
        metric_mean
        metric_std
    """

    # -----------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------

    patches = os.path.join(
        data_dir,
        "patches.npy",
    )

    signals = os.path.join(
        data_dir,
        "signals.npy",
    )

    metrics = os.path.join(
        data_dir,
        "metrics.npy",
    )

    labels = os.path.join(
        data_dir,
        "labels.npy",
    )

    rois = os.path.join(
        data_dir,
        "roi.npy",
    )

    sessions = os.path.join(
        data_dir,
        "session.npy",
    )

    # -----------------------------------------------------------------
    # Verify required files exist
    # -----------------------------------------------------------------

    required_files = {
        "patches": patches,
        "signals": signals,
        "metrics": metrics,
        "labels": labels,
        "roi": rois,
        "session": sessions,
    }

    for name, path in required_files.items():

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Missing {name} file:\n"
                f"    {path}\n\n"
                f"Run builder.py first and make sure the dataset "
                f"contains all required multimodal files."
            )

    # -----------------------------------------------------------------
    # Load session IDs
    # -----------------------------------------------------------------

    session_arr = np.load(
        sessions,
        allow_pickle=False,
    )

    # -----------------------------------------------------------------
    # Session-aware split
    # -----------------------------------------------------------------

    train_idx, val_idx = split_by_session(
        session_arr,
        val_ratio=val_ratio,
        seed=seed,
    )

    print(
        f"Train ROIs: {len(train_idx):,}"
    )

    print(
        f"Val ROIs:   {len(val_idx):,}"
    )

    print(
        f"Train sessions: "
        f"{len(np.unique(session_arr[train_idx]))}"
    )

    print(
        f"Val sessions: "
        f"{len(np.unique(session_arr[val_idx]))}"
    )

    # -----------------------------------------------------------------
    # Compute metric normalization using TRAIN ONLY
    # -----------------------------------------------------------------

    metric_mean, metric_std = compute_metric_stats(
        metrics_path=metrics,
        train_indices=train_idx,
    )

    print(
        f"Metric normalization computed from "
        f"{len(train_idx):,} training ROIs."
    )

    # -----------------------------------------------------------------
    # Build training dataset
    # -----------------------------------------------------------------

    train_ds = ROIQCDataset(
        patches_path=patches,
        signals_path=signals,
        metrics_path=metrics,
        labels_path=labels,
        roi_path=rois,
        session_path=sessions,
        indices=train_idx,
        augment=augment,
        signal_augment=signal_augment,
        metric_mean=metric_mean,
        metric_std=metric_std,
    )

    # -----------------------------------------------------------------
    # Build validation dataset
    #
    # NO augmentation.
    # Same training metric statistics.
    # -----------------------------------------------------------------

    val_ds = ROIQCDataset(
        patches_path=patches,
        signals_path=signals,
        metrics_path=metrics,
        labels_path=labels,
        roi_path=rois,
        session_path=sessions,
        indices=val_idx,
        augment=False,
        signal_augment=False,
        metric_mean=metric_mean,
        metric_std=metric_std,
    )

    return (
        train_ds,
        val_ds,
        metric_mean,
        metric_std,
    )