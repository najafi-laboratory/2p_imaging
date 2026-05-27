"""Create raw dF/F traces from QC-filtered Suite2p fluorescence results."""

from __future__ import annotations

import os
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, percentile_filter


def _correct_pmt_led(fluo: np.ndarray, dff: np.ndarray) -> np.ndarray:
    fluo_mean = np.percentile(fluo.copy(), 10, axis=0)
    fluo_base = percentile_filter(fluo_mean, 90, size=15, mode="nearest")
    fluo_correct = (fluo_mean - fluo_base) / fluo_base
    threshold = 2 * np.std(fluo_correct[fluo_correct < np.percentile(fluo_correct, 90)])
    dff[:, fluo_correct < -threshold] = np.nan
    return pd.DataFrame(dff).interpolate(method="linear", axis=1, limit_direction="both").to_numpy()


def compute_dff(
    signal: np.ndarray, *, normalize: bool = False, baseline_sigma_frames: float = 600.0
) -> np.ndarray:
    """Compute fractional dF/F; optionally z-score each ROI after baseline correction."""

    dff = signal.copy()
    baseline = gaussian_filter(dff, [0.0, baseline_sigma_frames])
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = (dff - baseline) / baseline
    if normalize:
        dff = (dff - np.nanmean(dff, axis=1, keepdims=True)) / (
            np.nanstd(dff, axis=1, keepdims=True) + 1e-5
        )
    return dff


def run(ops: dict[str, Any], *, normalize: bool = False, correct_pmt: bool = False) -> None:
    """Write `dff.h5`; raw non-z-scored dF/F is the default pipeline output."""

    result_dir = os.fspath(ops["save_path0"])
    fluo = np.load(os.path.join(result_dir, "qc_results", "fluo.npy"), allow_pickle=True)
    neuropil = np.load(os.path.join(result_dir, "qc_results", "neuropil.npy"), allow_pickle=True)
    signal = fluo.copy() - float(ops["neucoeff"]) * neuropil
    dff = compute_dff(signal, normalize=normalize)
    if correct_pmt:
        dff = _correct_pmt_led(fluo, dff)
    with h5py.File(os.path.join(result_dir, "dff.h5"), "w") as output:
        output["dff"] = dff
        output["fluo"] = fluo
