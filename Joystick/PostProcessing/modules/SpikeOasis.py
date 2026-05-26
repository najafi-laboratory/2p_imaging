# -*- coding: utf-8 -*-
"""
SpikeOasis.py  –  OASIS-based spike deconvolution + outlier ROI detection.

Key fixes vs. previous version
────────────────────────────────
1. find_flagged  : threshold computation and flagging moved *outside* the loop;
                   uses |skewness| so both over- and under-fit neurons are caught;
                   fully vectorised (no Python loop).
2. run           : neuron-wise OASIS loop parallelised with joblib so it scales
                   to 400+ neurons without blocking for minutes.
3. DffTraces caller note: the raw (non-z-scored) dff used inside run() is now
                   returned so the caller can use the *same* signal for QC and
                   for saving – avoiding the baseline mismatch that existed before.
"""

import os
import h5py
import numpy as np
import scipy.stats as stats
from scipy.ndimage import gaussian_filter
from scipy.signal import lfilter
from oasis.functions import deconvolve, estimate_parameters
from joblib import Parallel, delayed


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _deconvolve_one(y, threshold):
    """Deconvolve a single neuron trace.  Called in parallel by run()."""
    y = np.asarray(y, dtype=np.float64)
    try:
        g, sn = estimate_parameters(y, p=1)
        c, s, b, g, lam = deconvolve(y, g=g, sn=sn, penalty=1, b_nonneg=False)
    except Exception:
        # Fall back to zeros if OASIS fails (e.g. flat trace)
        return np.zeros_like(y), np.zeros_like(y)

    s_thresh = s.copy()
    s_thresh[s_thresh < threshold] = 0.0

    # Reconstruct calcium trace via fast AR(1) filter  c[t] = g*c[t-1] + s[t]
    c_filtered = lfilter([1], [1, -g], s_thresh)
    denoised = c_filtered + b
    return s_thresh, denoised


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def find_flagged(dff, denoised_traces, std_percentile=95, skew_percentile=95):
    """
    Flag ROIs whose deconvolution residual is an outlier in noise or skewness.

    Parameters
    ----------
    dff : ndarray, shape (n_neurons, n_frames)
        Raw (non-z-scored) ΔF/F used for deconvolution.
    denoised_traces : ndarray, shape (n_neurons, n_frames)
        Reconstructed calcium traces returned by run().
    std_percentile : float
        Neurons whose residual std exceeds this percentile are flagged.
    skew_percentile : float
        Neurons whose |residual skewness| exceeds this percentile are flagged.

    Returns
    -------
    flagged_neurons : ndarray of int
        Indices of ROIs that failed QC.
    residual_stds : ndarray
        Per-neuron residual standard deviations (useful for inspection).
    residual_skews : ndarray
        Per-neuron residual skewness values (useful for inspection).
    """
    # Vectorised residual computation – no Python loop needed
    residual = dff - denoised_traces                        # (n_neurons, n_frames)
    residual_stds  = residual.std(axis=1)                   # (n_neurons,)
    residual_skews = stats.skew(residual, axis=1)           # (n_neurons,)

    std_threshold  = np.percentile(residual_stds,           std_percentile)
    # Use absolute skewness: large positive *or* negative residual skew is bad
    skew_threshold = np.percentile(np.abs(residual_skews),  skew_percentile)

    flagged_neurons = np.where(
        (residual_stds > std_threshold) |
        (np.abs(residual_skews) > skew_threshold)
    )[0]

    return flagged_neurons, residual_stds, residual_skews


def binarize(inferred_spikes, threshold=0.05, use_binary=True):
    spikes = inferred_spikes.copy()
    if use_binary:
        print("Using BINARY spikes (threshold > {:.3f})".format(threshold))
        return (spikes > threshold).astype(float)
    else:
        print("Using HARD-thresholded continuous spikes")
        spikes = np.maximum(spikes, 0)
        spikes[spikes < threshold] = 0.0
        return spikes


def run(ops, fluo, neuropil, threshold=0.2, n_jobs=-1):
    """
    Compute ΔF/F, deconvolve spikes in parallel, and return results.

    Parameters
    ----------
    ops       : suite2p ops dict  (needs 'neucoeff', 'save_path0')
    fluo      : raw fluorescence,  shape (n_neurons, n_frames)
    neuropil  : neuropil signal,   shape (n_neurons, n_frames)
    threshold : spike amplitude threshold applied after OASIS
    n_jobs    : number of parallel jobs (-1 = use all CPU cores)

    Returns
    -------
    spikes          : ndarray (n_neurons, n_frames)  – thresholded spikes
    denoised_traces : ndarray (n_neurons, n_frames)  – reconstructed calcium
    dff             : ndarray (n_neurons, n_frames)  – baseline-subtracted ΔF/F
                      (same signal that was deconvolved – use this for QC)
    """
    # ── 1. Build non-z-scored ΔF/F (same baseline as before) ─────────────────
    dff = fluo.copy() - ops['neucoeff'] * neuropil
    sig_baseline = 600
    f0 = gaussian_filter(dff, [0., sig_baseline])
    dff = (dff - f0) / (f0 + 1e-10)          # avoid division by zero

    n_neurons = dff.shape[0]
    print(f"Deconvolving {n_neurons} neurons (n_jobs={n_jobs}) …")

    # ── 2. Parallelised per-neuron OASIS ──────────────────────────────────────
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_deconvolve_one)(dff[i], threshold)
        for i in range(n_neurons)
    )

    spikes          = np.array([r[0] for r in results], dtype=np.float64)
    denoised_traces = np.array([r[1] for r in results], dtype=np.float64)

    print("Thresholded deconvolution complete!")
    return spikes, denoised_traces, dff


# ──────────────────────────────────────────────────────────────────────────────
# Save helpers
# ──────────────────────────────────────────────────────────────────────────────

def save(ops, name, data):
    path = os.path.join(ops['save_path0'], 'qc_results', name + '.h5')
    with h5py.File(path, 'w') as f:
        f[name] = data


def save_manual(ops, name, data):
    path = os.path.join(ops['save_path0'], 'manual_qc_results', name + '.h5')
    with h5py.File(path, 'w') as f:
        f[name] = data
