#!/usr/bin/env python3
"""
Sanity check — generates a one-page-per-session PDF report.

Pages:
  1. Mean FOV: auto-QC ROIs  vs  manual-QC ROIs  (filled random colours per ROI)
  2. Event rate distribution  +  average Ca²⁺ transient shape (side by side)
  3. 5 sample DFF traces with detected event markers  (random 1-min window)

Usage (single session):
    python pipeline/sanity_check.py <session_path>

Called by cluster/submit_sanity_check.sh for SLURM jobs.

Output:
    <session_path>/sanity_check.pdf
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')           # no display needed — saves directly to PDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import config


# ─────────────────────────────────────────────────────────────────────────────
# Transient window:  ~367 ms before peak, 900 ms after peak  (at 30 Hz)
# ─────────────────────────────────────────────────────────────────────────────
TRANSIENT_PRE  = int(round(0.300 * config.FS)) + 2   # 11 frames ≈ 367 ms
TRANSIENT_POST = int(round(0.900 * config.FS))        # 27 frames = 900 ms


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_ops(session_path):
    ops = np.load(
        os.path.join(session_path, 'suite2p', 'plane0', 'ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = session_path
    return ops


def get_fov_and_crop(ops):
    """Return cropped mean image and crop indices."""
    x1, x2 = ops['xrange'][0], ops['xrange'][1]
    y1, y2 = ops['yrange'][0], ops['yrange'][1]
    mean_img = ops['meanImg'][y1:y2, x1:x2]
    return mean_img, y1, y2, x1, x2


def roi_colored_overlay(masks, y1, y2, x1, x2, alpha=0.45):
    """
    Fill each ROI with a distinct random colour.
    Uses evenly-spaced HSV hues shuffled so adjacent ROIs contrast well.
    """
    m      = masks[y1:y2, x1:x2].astype(np.int32)
    n_rois = int(m.max())
    if n_rois == 0:
        return np.zeros((*m.shape, 4), dtype=np.float32)

    rng  = np.random.default_rng(42)
    hues = rng.permutation(np.linspace(0, 1, n_rois, endpoint=False))
    rgb  = plt.cm.hsv(hues)[:, :3]          # shape (n_rois, 3)

    overlay        = np.zeros((*m.shape, 4), dtype=np.float32)
    valid          = m > 0
    overlay[valid, :3] = rgb[m[valid] - 1]  # masks are 1-indexed
    overlay[valid,  3] = alpha
    return overlay


def compute_avg_transient(session_path,
                           pre=TRANSIENT_PRE, post=TRANSIENT_POST):
    """
    Average calcium transient shape across all neurons and all events.

    Steps:
      1. For each neuron, extract a snippet of raw ΔF/F centred on every
         detected event → average snippets → one waveform per neuron.
      2. Average those per-neuron waveforms → grand mean.
      3. SEM across neurons.

    Returns (t_ms, grand_mean, grand_sem) or (None, None, None) if not enough data.
    """
    with h5py.File(os.path.join(session_path, 'manual_qc_results',
                                'dff.h5'), 'r') as f:
        dff = np.array(f['dff'])

    with h5py.File(os.path.join(session_path, 'manual_qc_results',
                                'events.h5'), 'r') as f:
        grp         = f['per_neuron']
        event_times = [np.array(grp[f'roi_{i}/times'])
                       for i in range(dff.shape[0])]

    n_frames    = dff.shape[1]
    neuron_data = []   # list of (per_neuron_mean, per_neuron_sem)

    for roi, times in enumerate(event_times):
        if len(times) < 2:
            continue
        # z-score full trace so amplitudes are comparable across neurons
        trace = dff[roi].astype(np.float64)
        sd    = trace.std()
        if sd < 1e-10:
            continue
        z_trace = (trace - trace.mean()) / sd

        snippets = []
        for t in times.astype(int):
            t0, t1 = t - pre, t + post + 1
            if t0 >= 0 and t1 <= n_frames:
                snippets.append(z_trace[t0:t1])
        if len(snippets) >= 2:
            arr      = np.array(snippets)
            n_mean   = arr.mean(axis=0)
            n_sem    = arr.std(ddof=1, axis=0) / np.sqrt(len(snippets))
            neuron_data.append((n_mean, n_sem))

    if len(neuron_data) < 2:
        return None, None, None, None

    # ── outlier rejection ────────────────────────────────────────────────────
    # Exclude neurons whose peak transient amplitude is a MAD outlier.
    # Threshold = median + 5 × (1.4826 × MAD)  ≈  median + 5 σ_robust
    peaks     = np.array([np.max(np.abs(m)) for m, _ in neuron_data])
    med_p     = np.median(peaks)
    mad_p     = np.median(np.abs(peaks - med_p)) * 1.4826
    threshold = med_p + 5 * mad_p
    n_before  = len(neuron_data)
    neuron_data = [(m, s) for (m, s), pk in zip(neuron_data, peaks)
                   if pk <= threshold]
    n_removed = n_before - len(neuron_data)
    if n_removed > 0:
        print(f'    [transient] removed {n_removed} outlier neuron(s) '
              f'(peak > {threshold:.2f} z)')

    if len(neuron_data) < 2:
        return None, None, None, None

    means      = np.array([m for m, _ in neuron_data])
    grand_mean = means.mean(axis=0)
    grand_sem  = means.std(ddof=1, axis=0) / np.sqrt(len(means))
    t_ms       = (np.arange(pre + post + 1) - pre) / config.FS * 1000

    return t_ms, neuron_data, grand_mean, grand_sem


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — FOV comparison
# ─────────────────────────────────────────────────────────────────────────────

def page_fov(ops, session_path):
    mean_img, y1, y2, x1, x2 = get_fov_and_crop(ops)

    auto_masks   = np.load(os.path.join(session_path, 'qc_results',
                                        'masks.npy'), allow_pickle=True)
    manual_masks = np.load(os.path.join(session_path, 'manual_qc_results',
                                        'masks.npy'), allow_pickle=True)

    n_auto   = int(np.max(auto_masks))
    n_manual = int(np.max(manual_masks))
    removed  = n_auto - n_manual

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'{os.path.basename(session_path)} — FOV\n'
        f'Removed by manual QC: {removed} ROIs  '
        f'({100*removed/max(n_auto,1):.1f}%)',
        fontsize=11, fontweight='bold'
    )

    panels = [
        (auto_masks,   f'After auto QC  — {n_auto} ROIs'),
        (manual_masks, f'After manual QC — {n_manual} ROIs'),
    ]
    for ax, (masks, title) in zip(axes, panels):
        ax.imshow(mean_img, cmap='gray', aspect='equal', interpolation='nearest')
        ov = roi_colored_overlay(masks, y1, y2, x1, x2)
        ax.imshow(ov, aspect='equal', interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Event rate distribution  +  average Ca²⁺ transient
# ─────────────────────────────────────────────────────────────────────────────

def page_rates(session_path):
    with h5py.File(os.path.join(session_path, 'manual_qc_results',
                                'events.h5'), 'r') as f:
        rates = np.array(f['event_rates'])

    n_total  = len(rates)
    n_active = int((rates > 0).sum())
    n_in_cs  = int(((rates >= 1.0) & (rates <= 2.0)).sum())

    t_ms, neuron_data, grand_mean, grand_sem = compute_avg_transient(session_path)

    fig, (ax_hist, ax_trans) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f'{os.path.basename(session_path)}',
        fontsize=11, fontweight='bold'
    )

    # ── left: histogram ───────────────────────────────────────────────────────
    ax_hist.hist(rates, bins=40, color='#3498db', edgecolor='white', linewidth=0.4)
    ax_hist.axvspan(1.0, 2.0, alpha=0.15, color='#2ecc71',
                    label='Expected CS range (1–2 Hz)')
    ax_hist.axvline(np.median(rates), color='#e74c3c', lw=2, linestyle='--',
                    label=f'Median: {np.median(rates):.3f} Hz')
    ax_hist.axvline(np.mean(rates),   color='#f39c12', lw=2, linestyle='--',
                    label=f'Mean:   {np.mean(rates):.3f} Hz')
    ax_hist.set_xlabel('Event rate (Hz)', fontsize=11)
    ax_hist.set_ylabel('Number of neurons', fontsize=11)
    ax_hist.set_title(
        f'Event rate distribution\n'
        f'{n_active}/{n_total} active  |  {n_in_cs} in 1–2 Hz',
        fontsize=10
    )
    ax_hist.legend(fontsize=9)
    ax_hist.spines[['top', 'right']].set_visible(False)

    # ── right: average transient ──────────────────────────────────────────────
    if grand_mean is not None:
        n_neurons = len(neuron_data)

        # per-neuron: thin dim lines with individual SEM, random HSV colours
        rng_c = np.random.default_rng(7)
        hues  = rng_c.permutation(np.linspace(0, 1, n_neurons, endpoint=False))
        colors = plt.cm.hsv(hues)

        for (n_mean, n_sem), c in zip(neuron_data, colors):
            ax_trans.fill_between(t_ms, n_mean - n_sem, n_mean + n_sem,
                                  color=c, alpha=0.06)
            ax_trans.plot(t_ms, n_mean, color=c, lw=0.5, alpha=0.35)

        # grand mean in black on top
        ax_trans.fill_between(t_ms,
                              grand_mean - grand_sem,
                              grand_mean + grand_sem,
                              color='black', alpha=0.18, label='Grand ±SEM')
        ax_trans.plot(t_ms, grand_mean,
                      color='black', lw=2, label=f'Grand mean (n={n_neurons})')
        ax_trans.axvline(0, color='#e74c3c', lw=1.2, linestyle='--',
                         alpha=0.8, label='Peak (t = 0)')
        ax_trans.axhline(0, color='#95a5a6', lw=0.6)
        ax_trans.set_xlabel('Time from peak (ms)', fontsize=11)
        ax_trans.set_ylabel('z-scored ΔF/F', fontsize=11)
        ax_trans.set_title(
            f'Average Ca²⁺ transient  —  '
            f'−{TRANSIENT_PRE/config.FS*1000:.0f} to '
            f'+{TRANSIENT_POST/config.FS*1000:.0f} ms',
            fontsize=10
        )
        ax_trans.legend(fontsize=9)
        ax_trans.spines[['top', 'right']].set_visible(False)
    else:
        ax_trans.text(0.5, 0.5, 'Not enough events\nto compute average',
                      transform=ax_trans.transAxes,
                      ha='center', va='center', fontsize=11, color='#7f8c8d')
        ax_trans.axis('off')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Sample DFF traces with event markers
# ─────────────────────────────────────────────────────────────────────────────

def page_traces(session_path, n_samples=5, seed=42):
    rng = np.random.default_rng(seed)

    with h5py.File(os.path.join(session_path, 'manual_qc_results',
                                'dff.h5'), 'r') as f:
        dff = np.array(f['dff'])

    with h5py.File(os.path.join(session_path, 'manual_qc_results',
                                'denoised_dff.h5'), 'r') as f:
        dff_smooth = np.array(f['denoised_dff'])

    with h5py.File(os.path.join(session_path, 'manual_qc_results',
                                'events.h5'), 'r') as f:
        rates       = np.array(f['event_rates'])
        grp         = f['per_neuron']
        event_times = [np.array(grp[f'roi_{i}/times'])
                       for i in range(dff.shape[0])]

    n_neurons, n_frames = dff.shape

    active  = np.where(rates > 0)[0]
    pool    = active if len(active) >= n_samples else np.arange(n_neurons)
    neurons = rng.choice(pool, size=min(n_samples, len(pool)), replace=False)
    neurons = sorted(neurons)

    # random 1-min window
    win       = int(60 * config.FS)
    max_start = max(0, n_frames - win)
    start     = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    end       = start + win
    t         = np.arange(win) / config.FS

    fig, axes = plt.subplots(len(neurons), 1,
                             figsize=(14, 2.8 * len(neurons)),
                             sharex=True)
    if len(neurons) == 1:
        axes = [axes]

    fig.suptitle(
        f'Sample DFF traces with detected events — {os.path.basename(session_path)}\n'
        f'Window: {start/config.FS:.0f} – {end/config.FS:.0f} s  (random 60 s)\n'
        f'Grey = raw ΔF/F   |   Blue = smoothed ΔF/F (used for detection)   |   '
        f'● = detected events',
        fontsize=10, fontweight='bold'
    )

    for ax, roi in zip(axes, neurons):
        raw    = dff[roi,        start:end]
        smooth = dff_smooth[roi, start:end]

        ax.plot(t, raw,    lw=0.6, color='#aab4be', alpha=0.7, label='raw ΔF/F')
        ax.plot(t, smooth, lw=1.0, color='#2980b9',             label='smoothed ΔF/F')

        ev        = event_times[roi]
        ev_in_win = ev[(ev >= start) & (ev < end)]
        t_ev      = (ev_in_win - start) / config.FS

        if len(t_ev) > 0:
            ev_amp = smooth[ev_in_win - start]
            y_min  = raw.min() - 0.05 * (raw.max() - raw.min())
            ax.vlines(t_ev, ymin=y_min, ymax=ev_amp,
                      color='#e74c3c', lw=1.0, alpha=0.5)
            ax.scatter(t_ev, ev_amp, color='#e74c3c', s=18, zorder=5,
                       label=f'{len(t_ev)} event(s)')
        else:
            ax.text(0.99, 0.92, 'no events in window',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=7, color='#7f8c8d', style='italic')

        ax.set_ylabel('ΔF/F', fontsize=8)
        ax.set_title(
            f'ROI #{roi}   rate: {rates[roi]:.2f} Hz',
            fontsize=8, loc='left', pad=2
        )
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.6)

    axes[-1].set_xlabel('Time within window (s)', fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(session_path):
    print('=' * 60)
    print(f'Sanity check: {os.path.basename(session_path)}')
    print('=' * 60)

    ops      = read_ops(session_path)
    out_path = os.path.join(session_path, 'sanity_check.pdf')

    with PdfPages(out_path) as pdf:
        print('  Page 1: FOV comparison …')
        fig = page_fov(ops, session_path)
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close(fig)

        print('  Page 2: Event rate distribution + average transient …')
        fig = page_rates(session_path)
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close(fig)

        print('  Page 3: Sample traces …')
        fig = page_traces(session_path)
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close(fig)

    print(f'Saved → {out_path}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python pipeline/sanity_check.py <session_path>')
        sys.exit(1)
    main(sys.argv[1])
