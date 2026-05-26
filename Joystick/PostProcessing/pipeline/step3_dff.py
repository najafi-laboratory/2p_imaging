#!/usr/bin/env python3
"""
Step 3 — Apply manual QC labels, compute ΔF/F, run E/I labeling.

Reads:   <session_path>/qc_results/   (from step 1)
         <session_path>/ROI_label.h5  (from step 2 notebook)
           — indices in ROI_label.h5 are 0-based relative to qc_results/ arrays

Writes:  <session_path>/manual_qc_results/
             fluo.npy, neuropil.npy, stat.npy, masks.npy
             dff.h5  (key: 'dff')
         <session_path>/masks.h5   (E/I labels, single-channel → all -1)

Usage (single session):
    python pipeline/step3_dff.py <session_path>

Called by cluster/submit_step34.sh for SLURM array jobs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
from scipy.ndimage import gaussian_filter

import config
from modules import LabelExcInh
from modules.QualControlDataIO import stat_to_masks


def read_ops(session_path):
    ops = np.load(
        os.path.join(session_path, 'suite2p', 'plane0', 'ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = session_path
    return ops


def read_roi_label(session_path):
    label_path = os.path.join(session_path, 'ROI_label.h5')
    if not os.path.isfile(label_path):
        raise FileNotFoundError(
            f'ROI_label.h5 not found at {session_path}\n'
            'Run step 2 (manual_review notebook) first.')
    with h5py.File(label_path, 'r') as f:
        good_roi = np.array(f['good_roi'], dtype=np.int64)
        bad_roi  = np.array(f['bad_roi'],  dtype=np.int64)
    return good_roi, bad_roi


def compute_dff(fluo, neuropil, ops):
    neucoeff     = float(ops.get('neucoeff', 0.7))
    sig_baseline = int(ops.get('sig_baseline', config.SIG_BASELINE))
    dff = fluo.astype(np.float64) - neucoeff * neuropil.astype(np.float64)
    f0  = gaussian_filter(dff, [0., sig_baseline])
    dff = (dff - f0) / (f0 + 1e-10)
    return dff


def save_manual_qc(session_path, fluo, neuropil, stat, masks):
    out = os.path.join(session_path, 'manual_qc_results')
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, 'fluo.npy'),     fluo)
    np.save(os.path.join(out, 'neuropil.npy'), neuropil)
    np.save(os.path.join(out, 'stat.npy'),     stat)
    np.save(os.path.join(out, 'masks.npy'),    masks)


def main(session_path):
    print('=' * 60)
    print(f'Step 3: DFF — {os.path.basename(session_path)}')
    print('=' * 60)

    ops = read_ops(session_path)
    good_roi, bad_roi = read_roi_label(session_path)

    qc = os.path.join(session_path, 'qc_results')
    F    = np.load(os.path.join(qc, 'fluo.npy'),     allow_pickle=True)
    Fneu = np.load(os.path.join(qc, 'neuropil.npy'), allow_pickle=True)
    stat = np.load(os.path.join(qc, 'stat.npy'),     allow_pickle=True)

    print(f'qc_results ROIs: {F.shape[0]}  |  keeping {len(good_roi)} good')

    fluo     = F[good_roi]
    neuropil = Fneu[good_roi]
    stat_clean = stat[good_roi]
    masks    = stat_to_masks(ops, stat_clean)

    save_manual_qc(session_path, fluo, neuropil, stat_clean, masks)
    print(f'Saved manual_qc_results/ ({len(good_roi)} ROIs)')

    dff = compute_dff(fluo, neuropil, ops)
    out = os.path.join(session_path, 'manual_qc_results')
    with h5py.File(os.path.join(out, 'dff.h5'), 'w') as f:
        f['dff'] = dff
    print(f'Saved dff.h5: shape {dff.shape}')

    LabelExcInh.run_manual(ops, config.DIAMETER)

    print(f'Step 3 complete: {session_path}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python pipeline/step3_dff.py <session_path>')
        sys.exit(1)
    main(sys.argv[1])
