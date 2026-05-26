#!/usr/bin/env python3
"""
Event detection only — for sessions that already have manual_qc_results/dff.h5
from a PREVIOUS manual QC run (old pipeline). Skips steps 1–3 entirely.

Reads:   <session_path>/manual_qc_results/dff.h5
Writes:  <session_path>/manual_qc_results/
             spikes.h5        key 'spikes'       — binary event matrix
             denoised_dff.h5  key 'denoised_dff' — smoothed ΔF/F
             events.h5        detailed per-neuron info

Usage (one session):
    python pipeline/step4_events_only.py <session_path>

Usage (multiple sessions from a text file, one path per line):
    python pipeline/step4_events_only.py --list sessions.txt

Called by cluster/submit_events_only.sh for SLURM array jobs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import config
from modules.EventDetection import detect, save


def run_one(session_path):
    dff_path = os.path.join(session_path, 'manual_qc_results', 'dff.h5')
    if not os.path.isfile(dff_path):
        print(f'[SKIP] manual_qc_results/dff.h5 not found: {session_path}')
        return False

    with h5py.File(dff_path, 'r') as f:
        key = 'dff' if 'dff' in f else list(f.keys())[0]
        dff = np.array(f[key])

    print(f'[{os.path.basename(session_path)}] dff shape: {dff.shape}')

    event_binary, event_times, event_amplitudes, event_rates, smoothed_dff = detect(
        dff,
        fs=config.FS,
        smooth_sigma=config.SMOOTH_SIGMA,
        threshold_factor=config.THRESHOLD_FACTOR,
        min_isi_s=config.MIN_ISI_S,
        min_prominence_factor=config.MIN_PROMINENCE_FACTOR,
    )

    save(session_path, event_binary, event_times, event_amplitudes,
         event_rates, smoothed_dff)
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python pipeline/step4_events_only.py <session_path> [<session_path> ...]')
        print('       python pipeline/step4_events_only.py --list sessions.txt')
        sys.exit(1)

    if sys.argv[1] == '--list':
        if len(sys.argv) != 3:
            print('Usage: python pipeline/step4_events_only.py --list sessions.txt')
            sys.exit(1)
        with open(sys.argv[2]) as fh:
            paths = [line.strip() for line in fh if line.strip()]
    else:
        paths = sys.argv[1:]

    ok = 0
    for path in paths:
        if run_one(path):
            ok += 1
    print(f'\nDone: {ok}/{len(paths)} sessions processed.')
