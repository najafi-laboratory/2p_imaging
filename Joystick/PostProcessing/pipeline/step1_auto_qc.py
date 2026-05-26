#!/usr/bin/env python3
"""
Step 1 — Automated morphological QC.

Reads:   <session_path>/suite2p/plane0/
Writes:  <session_path>/qc_results/   (fluo.npy, neuropil.npy, stat.npy, masks.npy)
         <session_path>/ops.npy
         <session_path>/move_offset.h5

Usage (single session):
    python pipeline/step1_auto_qc.py <session_path>

Called by cluster/submit_step1.sh for SLURM array jobs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config
from modules import QualControlDataIO


def read_ops(session_path):
    ops = np.load(
        os.path.join(session_path, 'suite2p', 'plane0', 'ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = session_path
    return ops


def main(session_path):
    print('=' * 60)
    print(f'Step 1: Auto QC — {os.path.basename(session_path)}')
    print('=' * 60)

    if not os.path.isdir(os.path.join(session_path, 'suite2p', 'plane0')):
        print(f'[ERROR] suite2p output not found: {session_path}')
        sys.exit(1)

    ops = read_ops(session_path)

    QualControlDataIO.run(
        ops,
        range_skew=config.RANGE_SKEW,
        max_connect=config.MAX_CONNECT,
        max_aspect=config.MAX_ASPECT,
        range_compact=config.RANGE_COMPACT,
        range_footprint=config.RANGE_FOOTPRINT,
        stat_file_names=['stat'],
    )

    print(f'Step 1 complete: {session_path}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python pipeline/step1_auto_qc.py <session_path>')
        sys.exit(1)
    main(sys.argv[1])
