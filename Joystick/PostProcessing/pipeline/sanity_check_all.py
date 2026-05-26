#!/usr/bin/env python3
"""
Multi-session sanity check — one summary PDF for all sessions of a mouse.

Layout: one row-block per session, three columns
  Col 0 (narrow) : mean FOV with ROIs filled in random distinct colours + ROI count
  Col 1 (narrow) : top  = average Ca²⁺ transient shape (mean ± SEM)
                   bottom = event rate histogram
  Col 2 (wide)   : 3 sample ΔF/F traces with detected event markers (middle 30 s)

Output:
    <BASE_PATH>/<MOUSE_FOLDER>/Figures/sanity_check_all_<MOUSE_FOLDER>.pdf

Usage:
    python pipeline/sanity_check_all.py <mouse_folder> [date1 date2 ...]

    If no dates are provided, all sessions that have manual_qc_results/ are
    included automatically.

Called by cluster/submit_sanity_check_all.sh
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config

N_TRACE  = 3     # sample neurons per session
ROW_H    = 1.8   # figure height (inches) per trace row → session block = N_TRACE × ROW_H
FIG_W    = 22    # total figure width in inches
SEED     = 42
WIN_S    = 30    # seconds of DFF trace to show

# Transient window — same as sanity_check.py
# PRE=3, POST=15 → −100 ms to +500 ms at 30 Hz
TRANSIENT_PRE  = 3
TRANSIENT_POST = 15


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
    x1, x2 = ops['xrange'][0], ops['xrange'][1]
    y1, y2 = ops['yrange'][0], ops['yrange'][1]
    mean_img = ops['meanImg'][y1:y2, x1:x2]
    return mean_img, y1, y2, x1, x2


def roi_colored_overlay(masks, y1, y2, x1, x2, alpha=0.45):
    """Fill each ROI with a distinct random colour (evenly-spaced HSV hues)."""
    m      = masks[y1:y2, x1:x2].astype(np.int32)
    n_rois = int(m.max())
    if n_rois == 0:
        return np.zeros((*m.shape, 4), dtype=np.float32)

    rng  = np.random.default_rng(42)
    hues = rng.permutation(np.linspace(0, 1, n_rois, endpoint=False))
    rgb  = plt.cm.hsv(hues)[:, :3]

    overlay        = np.zeros((*m.shape, 4), dtype=np.float32)
    valid          = m > 0
    overlay[valid, :3] = rgb[m[valid] - 1]   # 1-indexed
    overlay[valid,  3] = alpha
    return overlay


def compute_avg_transient(session_path,
                           pre=TRANSIENT_PRE, post=TRANSIENT_POST):
    """
    Per-neuron average snippet → grand mean ± SEM across neurons.
    Returns (t_ms, grand_mean, grand_sem) or (None, None, None).
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
    neuron_avgs = []

    for roi, times in enumerate(event_times):
        if len(times) < 2:
            continue
        snippets = []
        for t in times.astype(int):
            t0, t1 = t - pre, t + post + 1
            if t0 >= 0 and t1 <= n_frames:
                snippets.append(dff[roi, t0:t1])
        if len(snippets) >= 2:
            neuron_avgs.append(np.mean(snippets, axis=0))

    if len(neuron_avgs) < 2:
        return None, None, None

    grand_mean = np.mean(neuron_avgs, axis=0)
    grand_sem  = (np.std(neuron_avgs, ddof=1, axis=0)
                  / np.sqrt(len(neuron_avgs)))
    t_ms = (np.arange(pre + post + 1) - pre) / config.FS * 1000

    return t_ms, grand_mean, grand_sem


def discover_sessions(base_path, mouse_folder, dates=None):
    """Return list of session paths that have manual_qc_results/ ready."""
    mouse_dir    = os.path.join(base_path, mouse_folder)
    mouse_prefix = mouse_folder.split('_')[0]

    if dates:
        paths = [os.path.join(mouse_dir, f'{mouse_prefix}_{d}') for d in dates]
    else:
        paths = sorted([
            os.path.join(mouse_dir, d)
            for d in os.listdir(mouse_dir)
            if d.startswith(mouse_prefix) and
               os.path.isdir(os.path.join(mouse_dir, d))
        ])

    ready, skipped = [], []
    for p in paths:
        if os.path.isdir(os.path.join(p, 'manual_qc_results')):
            ready.append(p)
        else:
            skipped.append(os.path.basename(p))

    if skipped:
        print('WARNING — manual_qc_results/ not found (skipping):')
        for s in skipped:
            print(f'  {s}')
    return ready


# ─────────────────────────────────────────────────────────────────────────────
# Per-panel draw functions
# ─────────────────────────────────────────────────────────────────────────────

def draw_fov(ax, session_path, ops):
    mean_img, y1, y2, x1, x2 = get_fov_and_crop(ops)
    masks  = np.load(os.path.join(session_path, 'manual_qc_results',
                                  'masks.npy'), allow_pickle=True)
    n_rois = int(np.max(masks))
    ov     = roi_colored_overlay(masks, y1, y2, x1, x2)

    ax.imshow(mean_img, cmap='gray', aspect='auto', interpolation='nearest')
    ax.imshow(ov, aspect='auto', interpolation='nearest')
    ax.set_title(
        f'{os.path.basename(session_path)}\n{n_rois} ROIs',
        fontsize=7, fontweight='bold', pad=2
    )
    ax.axis('off')


def draw_transient(ax, session_path):
    t_ms, grand_mean, grand_sem = compute_avg_transient(session_path)

    if grand_mean is None:
        ax.text(0.5, 0.5, 'not enough events',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=6, color='#7f8c8d', style='italic')
        ax.axis('off')
        return

    ax.fill_between(t_ms,
                    grand_mean - grand_sem,
                    grand_mean + grand_sem,
                    color='#2980b9', alpha=0.25)
    ax.plot(t_ms, grand_mean, color='#2980b9', lw=1.2)
    ax.axvline(0, color='#e74c3c', lw=0.8, linestyle='--', alpha=0.7)
    ax.axhline(0, color='#95a5a6', lw=0.4)
    ax.set_xlabel('Time from peak (ms)', fontsize=5)
    ax.set_ylabel('ΔF/F', fontsize=5)
    ax.set_title('Avg Ca²⁺ transient ±SEM', fontsize=6, pad=2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=5)


def draw_hist(ax, session_path):
    with h5py.File(os.path.join(session_path, 'manual_qc_results',
                                'events.h5'), 'r') as f:
        rates = np.array(f['event_rates'])

    n_active = int((rates > 0).sum())
    n_in_cs  = int(((rates >= 1.0) & (rates <= 2.0)).sum())

    ax.hist(rates, bins=30, color='#3498db', edgecolor='white', linewidth=0.3)
    ax.axvspan(1.0, 2.0, alpha=0.18, color='#2ecc71', label='1–2 Hz')
    ax.axvline(np.median(rates), color='#e74c3c', lw=1.2, linestyle='--',
               label=f'med {np.median(rates):.2f}')
    ax.set_xlabel('Rate (Hz)', fontsize=5)
    ax.set_ylabel('# neurons',  fontsize=5)
    ax.set_title(
        f'{n_active} active  |  {n_in_cs} in 1–2 Hz',
        fontsize=6, pad=2
    )
    ax.legend(fontsize=5, loc='upper right', framealpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=5)


def draw_traces(axes, session_path):
    """Draw N_TRACE sample neurons into a list of axes."""
    rng = np.random.default_rng(SEED)

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

    # middle window
    win   = int(WIN_S * config.FS)
    mid   = n_frames // 2
    start = max(0, mid - win // 2)
    end   = min(n_frames, start + win)
    t     = np.arange(end - start) / config.FS

    active  = np.where(rates > 0)[0]
    pool    = active if len(active) >= N_TRACE else np.arange(n_neurons)
    n_pick  = min(N_TRACE, len(pool))
    neurons = sorted(rng.choice(pool, size=n_pick, replace=False))

    for idx, (ax, roi) in enumerate(zip(axes, neurons)):
        raw    = dff[roi,        start:end]
        smooth = dff_smooth[roi, start:end]

        ax.plot(t, raw,    lw=0.5, color='#aab4be', alpha=0.7)
        ax.plot(t, smooth, lw=0.8, color='#2980b9')

        ev        = event_times[roi]
        ev_in_win = ev[(ev >= start) & (ev < end)]
        t_ev      = (ev_in_win - start) / config.FS

        if len(t_ev) > 0:
            ev_amp = smooth[ev_in_win - start]
            y_min  = raw.min() - 0.05 * (raw.max() - raw.min())
            ax.vlines(t_ev, ymin=y_min, ymax=ev_amp,
                      color='#e74c3c', lw=0.6, alpha=0.4)
            ax.scatter(t_ev, ev_amp, color='#e74c3c', s=8, zorder=5)
        else:
            ax.text(0.99, 0.88, 'no events', transform=ax.transAxes,
                    ha='right', va='top', fontsize=5,
                    color='#7f8c8d', style='italic')

        ax.set_ylabel('ΔF/F', fontsize=5)
        ax.set_title(f'ROI #{roi}  {rates[roi]:.2f} Hz',
                     fontsize=6, loc='left', pad=1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=5)

        if idx == len(axes) - 1:
            ax.set_xlabel(f'Middle {WIN_S} s (s)', fontsize=5)
        else:
            ax.set_xticklabels([])

    for ax in axes[len(neurons):]:
        ax.set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Draw one session row-block
# ─────────────────────────────────────────────────────────────────────────────

def draw_session(fig, gs, row_start, session_path):
    try:
        ops = read_ops(session_path)
    except Exception as e:
        print(f'  [SKIP] {os.path.basename(session_path)}: {e}')
        return

    r0, r1 = row_start, row_start + N_TRACE

    # col 0: FOV (spans all N_TRACE rows)
    ax_fov = fig.add_subplot(gs[r0:r1, 0])

    # col 1: nested gridspec — top = transient, bottom = histogram
    gs_mid   = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[r0:r1, 1],
        hspace=0.55, height_ratios=[1, 1]
    )
    ax_trans = fig.add_subplot(gs_mid[0])
    ax_hist  = fig.add_subplot(gs_mid[1])

    # col 2: one axis per trace neuron
    trace_axes = [fig.add_subplot(gs[r0 + j, 2]) for j in range(N_TRACE)]

    try:
        draw_fov(ax_fov, session_path, ops)
    except Exception as e:
        ax_fov.set_title(f'{os.path.basename(session_path)}\nFOV error: {e}',
                         fontsize=6)
        ax_fov.axis('off')

    try:
        draw_transient(ax_trans, session_path)
    except Exception as e:
        ax_trans.set_title(f'Transient error: {e}', fontsize=5)

    try:
        draw_hist(ax_hist, session_path)
    except Exception as e:
        ax_hist.set_title(f'Hist error: {e}', fontsize=5)

    try:
        draw_traces(trace_axes, session_path)
    except Exception as e:
        trace_axes[0].set_title(f'Trace error: {e}', fontsize=5)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(mouse_folder, dates=None):
    print('=' * 60)
    print(f'Multi-session sanity check: {mouse_folder}')
    print('=' * 60)

    sessions = discover_sessions(config.BASE_PATH, mouse_folder, dates)
    if not sessions:
        print('No sessions with manual_qc_results/ found. Exiting.')
        return

    print(f'Sessions to plot: {len(sessions)}')
    for s in sessions:
        print(f'  {os.path.basename(s)}')

    n_sessions = len(sessions)
    total_rows = n_sessions * N_TRACE
    fig_height = total_rows * ROW_H

    fig = plt.figure(figsize=(FIG_W, fig_height))
    gs  = gridspec.GridSpec(
        total_rows, 3,
        figure=fig,
        width_ratios=[1, 1, 3],
        hspace=0.55,
        wspace=0.30,
        left=0.03, right=0.98,
        top=0.99,  bottom=0.02,
    )

    for i, sess_path in enumerate(sessions):
        print(f'  Plotting {i+1}/{n_sessions}: {os.path.basename(sess_path)} …')
        draw_session(fig, gs, i * N_TRACE, sess_path)

    figures_dir = os.path.join(config.BASE_PATH, mouse_folder, 'Figures')
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir,
                            f'sanity_check_all_{mouse_folder}.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python pipeline/sanity_check_all.py <mouse_folder> '
              '[date1 date2 ...]')
        sys.exit(1)
    mouse_folder = sys.argv[1]
    dates        = sys.argv[2:] if len(sys.argv) > 2 else None
    main(mouse_folder, dates)
