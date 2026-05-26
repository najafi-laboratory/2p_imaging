# -*- coding: utf-8 -*-
"""
DffTraces.py  –  ΔF/F computation, spike deconvolution, and bad-ROI exclusion.

Key fixes vs. previous version
────────────────────────────────
1. run_denoising flag: if False, skips OASIS entirely (the slow step) and saves
   zero-filled placeholder arrays for spikes and denoised_dff so downstream
   code never breaks on missing files.
2. The z-scored dff saved to disk is always computed *after* bad ROIs have been
   removed, so dff / spikes / denoised_dff always share the same neuron count.
3. find_flagged receives the raw (non-z-scored) dff that was actually used for
   deconvolution — previously the QC signal and saved signal had different baselines.
"""

import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

from modules import SpikeDeconv
from modules import SpikeOasis
from .SpikeAnalysis import analyze_spike_traces


# ──────────────────────────────────────────────────────────────────────────────
# ΔF/F
# ──────────────────────────────────────────────────────────────────────────────

def get_dff(ops, fluo, neuropil, norm):
    """Neuropil-correct and baseline-subtract fluorescence to get ΔF/F."""
    dff = fluo.copy() - ops['neucoeff'] * neuropil
    f0  = gaussian_filter(dff, [0., ops['sig_baseline']])
    for j in range(dff.shape[0]):
        dff[j, :] = (dff[j, :] - f0[j, :]) / (f0[j, :] + 1e-10)
        if norm:
            dff[j, :] = (dff[j, :] - np.mean(dff[j, :])) / \
                        (np.std(dff[j, :]) + 1e-5)
    return dff


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


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def exclude_bad_roi(bad_roi_id, all_data):
    """Remove rows (neurons) at bad_roi_id from every array in all_data."""
    return [np.delete(d, bad_roi_id, axis=0) for d in all_data]


def read_data(ops, name, folder=False):
    sub   = 'qc_results' if folder else ''
    fpath = os.path.join(ops['save_path0'], sub, f"{name}.h5")
    if not os.path.isfile(fpath):
        print(f"Warning: file not found – {fpath}")
        return None
    try:
        with h5py.File(fpath, 'r') as f:
            key = name if name in f.keys() else ('name' if 'name' in f.keys() else None)
            if key is None:
                print(f"Warning: expected key '{name}' not found in {fpath}")
                return None
            return np.array(f[key])
    except Exception as e:
        print(f"Error reading {fpath}: {e}")
        return None


def read_ROI_label(ops):
    with h5py.File(os.path.join(ops['save_path0'], 'ROI_label.h5'), 'r') as f:
        return np.array(f['good_roi']), np.array(f['bad_roi'])


# ──────────────────────────────────────────────────────────────────────────────
# Main entry points
# ──────────────────────────────────────────────────────────────────────────────

def run(
        ops,
        norm=True,
        plotting_neurons=[5],
        taus=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50, 0.55, 0.6],
        plot_with_smoothed=False,
        plot_without_smoothed=False,
        new_spike=True,
        run_denoising=True,
        spike_std_percentile=95,
        spike_skew_percentile=95,
        n_jobs=-1):
    """
    Compute ΔF/F, optionally deconvolve spikes, flag outlier ROIs, and save.

    Parameters
    ----------
    run_denoising : bool
        If True  → run full OASIS spike deconvolution (slow, ~minutes per session).
        If False → skip deconvolution entirely; save zero-filled placeholder arrays
                   for spikes and denoised_dff so downstream code never breaks on
                   missing files. Use this for a fast first-pass ROI review.

    Returns
    -------
    spike_bad_roi : list of int
        Indices of neurons excluded by spike QC. Always [] when run_denoising=False.
    """
    print('===============================================')
    print('=========== dff trace normalization ===========')
    print('===============================================')

    fluo     = np.load(os.path.join(ops['save_path0'], 'qc_results', 'fluo.npy'),
                       allow_pickle=True)
    neuropil = np.load(os.path.join(ops['save_path0'], 'qc_results', 'neuropil.npy'),
                       allow_pickle=True)

    print(f"Loaded {fluo.shape[0]} neurons, {fluo.shape[1]} frames")
    dff = get_dff(ops, fluo, neuropil, norm)

    spike_bad_roi = []

    # ── Option A: tau parameter search (always skips denoising check) ─────────
    if len(taus) > 1:
        tau_spike_dict = {}
        neurons = np.arange(dff.shape[0])
        for tau in taus:
            _, spikes = SpikeDeconv.run(
                ops, dff, oasis_tau=tau, neurons=neurons,
                plotting_neurons=plotting_neurons,
                plot_with_smoothed=plot_with_smoothed,
                plot_without_smoothed=plot_without_smoothed)
            tau_spike_dict[tau] = spikes
        analyze_spike_traces(ops, dff, tau_spike_dict, neurons=neurons)
        save(ops, 'dff', dff)
        return spike_bad_roi

    # ── Option B: skip deconvolution, save placeholders ──────────────────────
    if not run_denoising:
        print("run_denoising=False → skipping OASIS deconvolution")
        print("Saving dff + zero-filled placeholders for spikes and denoised_dff")
        n_neurons, n_frames = dff.shape
        placeholders = np.zeros((n_neurons, n_frames), dtype=np.float32)
        save(ops, 'dff',          dff)
        save(ops, 'spikes',       placeholders)
        save(ops, 'denoised_dff', placeholders)
        print(f"  dff:          {dff.shape}")
        print(f"  spikes:       {placeholders.shape}  [zeros placeholder]")
        print(f"  denoised_dff: {placeholders.shape}  [zeros placeholder]")
        return spike_bad_roi

    # ── Option C: full OASIS spike detection ─────────────────────────────────
    if new_spike:
        spikes, denoised_traces, dff_raw = SpikeOasis.run(
            ops, fluo, neuropil, threshold=0.2, n_jobs=n_jobs)

        flagged, res_stds, res_skews = SpikeOasis.find_flagged(
            dff_raw, denoised_traces,
            std_percentile=spike_std_percentile,
            skew_percentile=spike_skew_percentile)

        spike_bad_roi = flagged.tolist()
        n_bad = len(spike_bad_roi)
        print(f"Spike QC flagged {n_bad} / {dff.shape[0]} neurons")
        if n_bad:
            print(f"  residual std  range of flagged: "
                  f"{res_stds[flagged].min():.4f} – {res_stds[flagged].max():.4f}")
            print(f"  residual skew range of flagged: "
                  f"{res_skews[flagged].min():.4f} – {res_skews[flagged].max():.4f}")

        all_data  = [dff, spikes, denoised_traces]
        all_names = ['dff', 'spikes', 'denoised_dff']
        clean_data = exclude_bad_roi(spike_bad_roi, all_data)
        for name, data in zip(all_names, clean_data):
            save(ops, name, data)
            print(f"Saved '{name}' with shape {data.shape}")

    # ── Option D: single-tau SpikeDeconv (legacy path) ───────────────────────
    else:
        neurons = np.arange(dff.shape[0])
        smoothed, spikes = SpikeDeconv.run(
            ops, dff, oasis_tau=taus[0], neurons=neurons,
            plotting_neurons=plotting_neurons,
            plot_with_smoothed=plot_with_smoothed,
            plot_without_smoothed=plot_without_smoothed)
        save(ops, 'dff',      dff)
        save(ops, 'spikes',   spikes)
        save(ops, 'smoothed', smoothed)
        print("Saved 'dff', 'spikes', 'smoothed'")

    return spike_bad_roi


def run_manual(ops,
               run_denoising=True,
               spike_std_percentile=95,
               spike_skew_percentile=95,
               n_jobs=-1):
    """
    Re-run after manual ROI curation.

    Parameters
    ----------
    run_denoising : bool
        Same as in run() — set False to skip OASIS and save placeholders.
    """
    print('===============================================')
    print('=========== manual dff normalisation ==========')
    print('===============================================')

    fluo     = np.load(os.path.join(ops['save_path0'], 'manual_qc_results', 'fluo.npy'),
                       allow_pickle=True)
    neuropil = np.load(os.path.join(ops['save_path0'], 'manual_qc_results', 'neuropil.npy'),
                       allow_pickle=True)

    _, bad_roi_id = read_ROI_label(ops)

    all_names = ['dff', 'spikes', 'smoothed', 'denoised_dff']
    all_data, all_existing_names = [], []
    for name in all_names:
        d = read_data(ops, name, folder=True)
        if d is not None:
            all_data.append(d)
            all_existing_names.append(name)

    all_data_clean = exclude_bad_roi(bad_roi_id, all_data)
    spike_bad_roi  = []
    n_neurons, n_frames = fluo.shape

    if not run_denoising:
        print("run_denoising=False → skipping OASIS; saving zero placeholders")
        placeholders = np.zeros((n_neurons, n_frames), dtype=np.float32)
        for pname in ['spikes', 'denoised_dff']:
            if pname not in all_existing_names:
                all_data_clean.append(placeholders)
                all_existing_names.append(pname)

    elif 'denoised_dff' not in all_existing_names:
        spikes, denoised_traces, dff_raw = SpikeOasis.run(
            ops, fluo, neuropil, threshold=0.2, n_jobs=n_jobs)

        flagged, res_stds, res_skews = SpikeOasis.find_flagged(
            dff_raw, denoised_traces,
            std_percentile=spike_std_percentile,
            skew_percentile=spike_skew_percentile)

        spike_bad_roi = flagged.tolist()
        print(f"Spike QC flagged {len(spike_bad_roi)} additional neurons")

        all_data_clean.extend([spikes, denoised_traces])
        all_existing_names.extend(['spikes', 'denoised_dff'])
        all_data_clean = exclude_bad_roi(spike_bad_roi, all_data_clean)

    for name, data in zip(all_existing_names, all_data_clean):
        save_manual(ops, name, data)
        print(f"Saved manual '{name}' with shape {data.shape}")

    return spike_bad_roi
