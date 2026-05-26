#!/usr/bin/env python3
"""
EventDetection.py — Complex-spike calcium event detection.

Designed for GCaMP8s recordings from Purkinje cell dendrites (L7-Cre),
30 Hz imaging, expected firing rate ~1–2 Hz.

Algorithm:
  1. Gaussian-smooth ΔF/F (σ ~ 50 ms) to reduce shot noise
  2. Estimate per-neuron noise floor from the lower half of the ΔF/F
     distribution (robust to positive-skew events)
  3. scipy.signal.find_peaks with height, prominence, and distance thresholds

Saves (all to <session_path>/manual_qc_results/):
  spikes.h5        key 'spikes'       — (n, T) binary event matrix  [backward compat]
  denoised_dff.h5  key 'denoised_dff' — (n, T) smoothed ΔF/F        [backward compat]
  events.h5                           — detailed per-neuron info     [new]
"""

import os
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# ─────────────────────────────────────────────────────────────────────────────

def _noise_floor(trace):
    """
    Robust noise estimate: std of the lower half of the distribution × √2.
    Using only values below the median avoids contamination from positive
    calcium transients, giving a reliable baseline noise level.
    """
    med   = np.median(trace)
    lower = trace[trace < med]
    if len(lower) < 10:
        return max(float(np.std(trace)) * 0.5, 1e-10)
    return max(float(np.std(lower)) * np.sqrt(2.0), 1e-10)


def detect(dff, fs=30, smooth_sigma=1.5, threshold_factor=2.5,
           min_isi_s=0.5, min_prominence_factor=2.0):
    """
    Detect complex-spike calcium events in GCaMP8s Purkinje dendrite data.

    Parameters
    ----------
    dff                   : ndarray (n_neurons, n_frames)  ΔF/F
    fs                    : float   imaging frame rate in Hz
    smooth_sigma          : float   Gaussian smoothing σ in frames
    threshold_factor      : float   threshold = factor × noise floor
    min_isi_s             : float   minimum inter-event interval (s)
    min_prominence_factor : float   minimum peak prominence in noise units

    Returns
    -------
    event_binary    : ndarray (n_neurons, n_frames) float32 — 1 at event frames
    event_times     : list of int64 arrays — event frame indices per neuron
    event_amplitudes: list of float64 arrays — smoothed-DFF amplitude at each event
    event_rates     : ndarray (n_neurons,) float64 — mean event rate in Hz
    smoothed_dff    : ndarray (n_neurons, n_frames) float64 — Gaussian-smoothed ΔF/F
    """
    n_neurons, n_frames = dff.shape
    min_isi_frames = max(1, int(min_isi_s * fs))

    smoothed_dff = gaussian_filter1d(dff.astype(np.float64), sigma=smooth_sigma, axis=1)

    event_binary     = np.zeros((n_neurons, n_frames), dtype=np.float32)
    event_times      = []
    event_amplitudes = []
    event_rates      = np.zeros(n_neurons, dtype=np.float64)

    for i in range(n_neurons):
        trace      = smoothed_dff[i]
        noise      = _noise_floor(trace)
        threshold  = threshold_factor * noise
        prominence = min_prominence_factor * noise

        peaks, _ = find_peaks(
            trace,
            height=threshold,
            distance=min_isi_frames,
            prominence=prominence,
        )

        event_binary[i, peaks] = 1.0
        event_times.append(peaks.astype(np.int64))
        event_amplitudes.append(
            trace[peaks] if len(peaks) > 0 else np.array([], dtype=np.float64)
        )
        event_rates[i] = len(peaks) / (n_frames / fs)

    return event_binary, event_times, event_amplitudes, event_rates, smoothed_dff


def save(session_path, event_binary, event_times, event_amplitudes,
         event_rates, smoothed_dff):
    """
    Save event detection outputs to <session_path>/manual_qc_results/.

    File layout (backward compatible with old pipeline):
      spikes.h5        key 'spikes'       — binary event matrix
      denoised_dff.h5  key 'denoised_dff' — smoothed ΔF/F
      events.h5        key 'event_rates'  — mean rate per neuron (Hz)
                       group 'per_neuron/roi_N/times'      — event frames
                       group 'per_neuron/roi_N/amplitudes' — peak ΔF/F
    """
    out_dir = os.path.join(session_path, 'manual_qc_results')
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(os.path.join(out_dir, 'spikes.h5'), 'w') as f:
        f['spikes'] = event_binary

    with h5py.File(os.path.join(out_dir, 'denoised_dff.h5'), 'w') as f:
        f['denoised_dff'] = smoothed_dff

    with h5py.File(os.path.join(out_dir, 'events.h5'), 'w') as f:
        f['event_rates'] = event_rates
        grp = f.create_group('per_neuron')
        for i, (times, amps) in enumerate(zip(event_times, event_amplitudes)):
            grp.create_dataset(f'roi_{i}/times',      data=times)
            grp.create_dataset(f'roi_{i}/amplitudes', data=amps)

    n = len(event_rates)
    active = int((event_rates > 0).sum())
    print(f'Events saved → {out_dir}')
    print(f'  neurons: {n}  |  active: {active}  |  '
          f'mean rate: {event_rates.mean():.3f} Hz  |  '
          f'range: {event_rates.min():.3f}–{event_rates.max():.3f} Hz')
