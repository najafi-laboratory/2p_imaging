"""
predict/prediction_utils.py

Utilities for running ROI QC predictions on a new imaging session.

Returns probabilities and predicted states for every ROI.

States:
    good  -> probability >= good_threshold
    bad   -> probability <= bad_threshold
    gray  -> otherwise
"""

from __future__ import annotations

import os
import numpy as np
import torch

from dataset.crop import make_two_channel_patch
from model.cnn import ROIModel
from dataset.builder import compute_metrics, downsample_trace


# ------------------------------------------------------------
# Session loading
# ------------------------------------------------------------

# def load_session(session_path):
#     """
#     Loads Suite2p outputs.

#     Returns
#     -------
#     mean_img : ndarray
#     masks    : ndarray
#     """

#     ops = np.load(
#         os.path.join(
#             session_path,
#             "suite2p",
#             "plane0",
#             "ops.npy",
#         ),
#         allow_pickle=True,
#     ).item()

#     masks = np.load(
#         os.path.join(
#             session_path,
#             "qc_results",
#             "masks.npy",
#         )
#     )

#     return ops["meanImg"], masks
def load_session(session_path):
    """
    Loads all data needed for multimodal ROI prediction.
    """

    ops = np.load(
        os.path.join(
            session_path,
            "suite2p",
            "plane0",
            "ops.npy",
        ),
        allow_pickle=True,
    ).item()

    masks = np.load(
        os.path.join(
            session_path,
            "qc_results",
            "masks.npy",
        )
    )

    fluo = np.load(
        os.path.join(
            session_path,
            "qc_results",
            "fluo.npy",
        ),
        mmap_mode="r",
    )

    neuropil = np.load(
        os.path.join(
            session_path,
            "qc_results",
            "neuropil.npy",
        ),
        mmap_mode="r",
    )

    stat = np.load(
        os.path.join(
            session_path,
            "qc_results",
            "stat.npy",
        ),
        allow_pickle=True,
    )

    return (
        ops["meanImg"],
        masks,
        fluo,
        neuropil,
        stat,
    )

# ------------------------------------------------------------
# Single ROI prediction
# ------------------------------------------------------------

# def predict_roi(
#     model,
#     mean_img,
#     masks,
#     roi_id,
#     patch_size=128,
# ):
#     """
#     Predict probability that one ROI is GOOD.

#     Returns
#     -------
#     probability : float
#     """
#     patch = make_two_channel_patch(
#         mean_img,
#         masks,
#         roi_id,
#         patch_size=patch_size,
#     )

#     patch = patch / 255.0

#     x = torch.tensor(
#         patch,
#         dtype=torch.float32,
#     ).unsqueeze(0)

#     # print("x.shape: ", x.shape)
#     # print("image channel:", x[:,0].min(), x[:,0].max())
#     # print("mask channel:", x[:,1].min(), x[:,1].max())


#     prob = float(model.predict_proba(x)[0][0])

#     return prob
def predict_roi(
    model,
    mean_img,
    masks,
    fluo,
    neuropil,
    stat,
    roi_id,
    patch_size=64,
    signal_length=2048,
    metric_mean=None,
    metric_std=None,
):
    """
    Predict probability that one ROI is GOOD.
    """

    # ------------------------------------------------------------
    # IMAGE
    # ------------------------------------------------------------

    patch = make_two_channel_patch(
        mean_img,
        masks,
        roi_id,
        patch_size=patch_size,
    )

    patch = patch / 255.0

    image = torch.tensor(
        patch,
        dtype=torch.float32,
    ).unsqueeze(0)


    # ------------------------------------------------------------
    # SIGNAL
    # ------------------------------------------------------------

    roi_idx = roi_id - 1

    fluo_trace = downsample_trace(
        fluo[roi_idx],
        signal_length,
    )

    neuropil_trace = downsample_trace(
        neuropil[roi_idx],
        signal_length,
    )

    signal = np.stack(
        [
            fluo_trace,
            neuropil_trace,
        ],
        axis=0,
    )


    # Same normalization used by dataset.py
    for channel in range(2):

        x = signal[channel]

        median = np.median(x)

        mad = np.median(
            np.abs(x - median)
        )

        robust_std = 1.4826 * mad

        if robust_std < 1e-6:
            robust_std = np.std(x)

        if robust_std < 1e-6:
            robust_std = 1.0

        signal[channel] = (
            x - median
        ) / robust_std


    signal = torch.tensor(
        signal,
        dtype=torch.float32,
    ).unsqueeze(0)


    # ------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------

    # metrics = compute_metrics(
    #     stat=stat[roi_idx],
    #     fluo=fluo[roi_idx],
    #     neuropil=neuropil[roi_idx],
    # )

    # metrics = torch.tensor(
    #     metrics,
    #     dtype=torch.float32,
    # ).unsqueeze(0)

    metrics = compute_metrics(
        stat=stat[roi_idx],
        fluo=fluo[roi_idx],
        neuropil=neuropil[roi_idx],
    )
    
    metrics = np.asarray(
        metrics,
        dtype=np.float32,
    )
    
    # Apply the SAME normalization used during training
    if metric_mean is not None and metric_std is not None:
    
        metrics = (
            metrics - metric_mean
        ) / metric_std
    
    metrics = torch.tensor(
        metrics,
        dtype=torch.float32,
    ).unsqueeze(0)


    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    prob = float(
        model.predict_proba(
            image=image,
            signal=signal,
            metrics=metrics,
        )[0][0]
    )

    return prob

# ------------------------------------------------------------
# Whole session prediction
# ------------------------------------------------------------

def predict_session(
    session_path,
    model_path,
    patch_size=64,
    good_threshold=0.90,
    bad_threshold=0.10,
):
    """
    Predict every ROI in a session.

    Returns
    -------
    predictions : list of dictionaries

    Example

    [
        {
            "roi": 13,
            "probability": 0.96,
            "state": "good"
        },
        ...
    ]
    """

    print("in predict..")

    model = ROIModel(num_metrics=24)
    checkpoint = model.load(model_path)

    metric_mean = checkpoint["metric_mean"]
    metric_std = checkpoint["metric_std"]

    # mean_img, masks = load_session(session_path)
    mean_img, masks, fluo, neuropil, stat = load_session(
        session_path
    )

    roi_ids = np.unique(masks)
    roi_ids = roi_ids[roi_ids > 0].astype(int)

    predictions = []

    for roi_id in roi_ids:

        # p = predict_roi(
        #     model,
        #     mean_img,
        #     masks,
        #     roi_id,
        #     patch_size=patch_size,
        # )
        p = predict_roi(
            model,
            mean_img,
            masks,
            fluo,
            neuropil,
            stat,
            roi_id,
            patch_size=patch_size,
            metric_mean=metric_mean,
            metric_std=metric_std
        )

        if p >= good_threshold:
            state = "good"

        elif p <= bad_threshold:
            state = "bad"

        else:
            state = "gray"

        predictions.append(
            {
                "roi": int(roi_id),
                "probability": p,
                "state": state,
            }
        )

    return predictions