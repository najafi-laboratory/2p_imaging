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


# ------------------------------------------------------------
# Session loading
# ------------------------------------------------------------

def load_session(session_path):
    """
    Loads Suite2p outputs.

    Returns
    -------
    mean_img : ndarray
    masks    : ndarray
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

    return ops["meanImg"], masks


# ------------------------------------------------------------
# Single ROI prediction
# ------------------------------------------------------------

def predict_roi(
    model,
    mean_img,
    masks,
    roi_id,
    patch_size=128,
):
    """
    Predict probability that one ROI is GOOD.

    Returns
    -------
    probability : float
    """
    patch = make_two_channel_patch(
        mean_img,
        masks,
        roi_id,
        patch_size=patch_size,
    )

    patch = patch / 255.0

    x = torch.tensor(
        patch,
        dtype=torch.float32,
    ).unsqueeze(0)

    # print("x.shape: ", x.shape)
    # print("image channel:", x[:,0].min(), x[:,0].max())
    # print("mask channel:", x[:,1].min(), x[:,1].max())


    prob = float(model.predict_proba(x)[0][0])

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

    model = ROIModel()
    model.load(model_path)

    mean_img, masks = load_session(session_path)

    roi_ids = np.unique(masks)
    roi_ids = roi_ids[roi_ids > 0].astype(int)

    predictions = []

    for roi_id in roi_ids:

        p = predict_roi(
            model,
            mean_img,
            masks,
            roi_id,
            patch_size=patch_size,
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