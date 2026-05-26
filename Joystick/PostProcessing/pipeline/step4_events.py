#!/usr/bin/env python3
"""
Step 4 — Complex-spike event detection (GCaMP8s, Purkinje dendrites, 30 Hz).

Reads:   <session_path>/manual_qc_results/dff.h5  (from step 3)
Writes:  <session_path>/manual_qc_results/
             spikes.h5        key 'spikes'       — binary event matrix
             denoised_dff.h5  key 'denoised_dff' — Gaussian-smoothed ΔF/F
             events.h5        key 'event_rates'  — per-neuron mean rate (Hz)
                              group 'per_neuron/roi_N/times'
                              group 'per_neuron/roi_N/amplitudes'

Usage (single session):
    python pipeline/step4_events.py <session_path>

Called by cluster/submit_step34.sh for SLURM array jobs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import config
from modules.EventDetection import detect, save


def main(session_path):
    print('=' * 60)
    print(f'Step 4: Event Detection — {os.path.basename(session_path)}')
    print('=' * 60)

    dff_path = os.path.join(session_path, 'manual_qc_results', 'dff.h5')
    if not os.path.isfile(dff_path):
        print(f'[ERROR] dff.h5 not found. Run step 3 first.\n  {dff_path}')
        sys.exit(1)

    with h5py.File(dff_path, 'r') as f:
        dff = np.array(f['dff'])
    print(f'Loaded dff: {dff.shape}  ({dff.shape[0]} neurons, {dff.shape[1]} frames)')

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

    print(f'Step 4 complete: {session_path}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python pipeline/step4_events.py <session_path>')
        sys.exit(1)
    main(sys.argv[1])
