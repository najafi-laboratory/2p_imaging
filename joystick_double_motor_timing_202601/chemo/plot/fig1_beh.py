#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

from modules.Alignment import run_get_js_rot
from utils import get_mean_sem
from utils import get_frame_idx_from_time
from utils import add_legend
from utils import hide_all_axis
from utils import plot_mean_sem

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# outcome histogram.
def plot_outcome_hist(ax, list_sess_time, list_trial_labels):
    width = 0.2
    eval_states = [
        'reward',
        'early_2nd_press',
        'no_2nd_press',
        'assist']
    colors = ['dimgrey', 'cornflowerblue', 'coral', 'mediumseagreen']
    # read data.
    list_outcome = [np.array(tl['outcome']) for tl in list_trial_labels]
    list_trial_type = [np.array(tl['trial_type']) for tl in list_trial_labels]
    list_chemo_type = [np.array(tl['chemo_type']) for tl in list_trial_labels]
    # count labels.
    counts_control_1 = np.array([[np.nansum(o[(t==1)*(c==0)] == s) for o, t, c in zip(list_outcome, list_trial_type, list_chemo_type)] for s in eval_states])
    counts_control_2 = np.array([[np.nansum(o[(t==2)*(c==0)] == s) for o, t, c in zip(list_outcome, list_trial_type, list_chemo_type)] for s in eval_states])
    counts_chemo_1   = np.array([[np.nansum(o[(t==1)*(c==1)] == s) for o, t, c in zip(list_outcome, list_trial_type, list_chemo_type)] for s in eval_states])
    counts_chemo_2   = np.array([[np.nansum(o[(t==2)*(c==1)] == s) for o, t, c in zip(list_outcome, list_trial_type, list_chemo_type)] for s in eval_states])
    counts_control_1 = counts_control_1 / (np.nansum(counts_control_1, axis=0)+1e-5)
    counts_control_2 = counts_control_2 / (np.nansum(counts_control_2, axis=0)+1e-5)
    counts_chemo_1   = counts_chemo_1 / (np.nansum(counts_chemo_1, axis=0)+1e-5)
    counts_chemo_2   = counts_chemo_2 / (np.nansum(counts_chemo_2, axis=0)+1e-5)
    # define layouts.
    #fig, ax = plt.subplots(1, 1, figsize=(6, 12))
    ax0 = ax.inset_axes([0, 0, 1, 0.4], transform=ax.transAxes)
    ax1 = ax.inset_axes([0, 0.5, 1, 0.5], transform=ax.transAxes)
    # plot histogram.
    bottom_control_1 = 0
    bottom_control_2 = 0
    bottom_chemo_1   = 0
    bottom_chemo_2   = 0
    for si in range(len(eval_states)):
        ax0.bar(
            0-width/1.5,
            counts_control_1[-si-1,:],
            bottom=bottom_control_1,
            width=width,
            color=colors[si])
        ax0.bar(
            0+width/1.5,
            counts_control_2[-si-1,:],
            bottom=bottom_control_2,
            width=width,
            color=colors[si])
        ax0.bar(
            1-width/1.5,
            counts_chemo_1[-si-1,:],
            bottom=bottom_chemo_1,
            width=width,
            color=colors[si])
        ax0.bar(
            1+width/1.5,
            counts_chemo_2[-si-1,:],
            bottom=bottom_chemo_2,
            width=width,
            color=colors[si])
        bottom_control_1 += counts_control_1[-si-1,:]
        bottom_control_2 += counts_control_2[-si-1,:]
        bottom_chemo_1   += counts_chemo_1[-si-1,:]
        bottom_chemo_2   += counts_chemo_2[-si-1,:]
    # adjust layouts.
    ax0.set_title('Outcome fractions\nshort|long')
    ax0.tick_params(tick1On=False)
    ax0.spines['left'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.yaxis.grid(True)
    ax0.set_xlabel('Training session')
    ax0.set_ylabel('Fraction of trials')
    ax0.set_xticks([0,1])
    ax0.set_xticklabels(['control', 'chemo'])
    ax0.set_xlim([-0.5,1.5])
    add_legend(ax1, colors[::-1], eval_states, 'lower right')
    hide_all_axis(ax)
    hide_all_axis(ax1)

# outcome histogram.
def plot_outcome_box(ax, list_sess_time, list_trial_labels):
    def plot_fraction(ax, data, pos, color):
        data = data[data > 0]
        if len(data) == 0:
            return
        m, s = get_mean_sem(data.reshape(-1, 1))
        ax.errorbar(
            pos, m, s,
            color=color,
            capsize=2, marker='o', linestyle='none',
            markeredgecolor='white', markeredgewidth=0.1)
    offset = 0.1
    eval_states = [
        'reward',
        'early_2nd_press',
        'no_2nd_press']
    colors = ['mediumseagreen', 'coral', 'cornflowerblue']
    chemo_labels = [('control', 0), ('chemo', 1)]
    trial_type_labels = [('Short', 1), ('Long', 2)]
    # read data.
    list_outcome = [np.array(tl['outcome']) for tl in list_trial_labels]
    list_trial_type = [np.array(tl['trial_type']) for tl in list_trial_labels]
    list_chemo_type = [np.array(tl['chemo_type']) for tl in list_trial_labels]
    # count labels.
    outcome_fractions = {}
    for trial_name, trial_type in trial_type_labels:
        for chemo_name, chemo_type in chemo_labels:
            counts = np.array([[
                np.nansum(o[(t == trial_type) * (c == chemo_type)] == state)
                for o, t, c in zip(list_outcome, list_trial_type, list_chemo_type)]
                for state in eval_states])
            outcome_fractions[(trial_name, chemo_name)] = counts / (np.nansum(counts, axis=0) + 1e-5)
    # define layouts.
    ax0 = ax.inset_axes([0, 0, 0.4, 0.8], transform=ax.transAxes)
    ax1 = ax.inset_axes([0.6, 0, 0.4, 0.8], transform=ax.transAxes)
    # plot errorbars.
    for axi, (trial_name, _) in zip([ax0, ax1], trial_type_labels):
        for ci, (chemo_name, _) in enumerate(chemo_labels):
            counts = outcome_fractions[(trial_name, chemo_name)]
            for si, color in enumerate(colors):
                plot_fraction(axi, counts[si, :], ci + si*offset, color)
    # adjust layouts.
    ax.set_title('Outcome fractions')
    for axi, (trial_name, _) in zip([ax0, ax1], trial_type_labels):
        axi.set_title(trial_name)
        axi.tick_params(axis='x', tick1On=False)
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.set_xticks([0,1])
        axi.set_xticklabels(['control', 'chemo'])
        axi.set_xlim([-0.5,1.5])
    hide_all_axis(ax)

# press timing error.
def plot_press2_time_err(ax, list_sess_time, list_trial_labels):
    xlim = [-1000,1500]
    colors = ['dimgrey', 'orangered']
    # read data.
    list_trial_type = [np.array(tl['trial_type']) for tl in list_trial_labels]
    list_delay = [np.array(tl['delay']) for tl in list_trial_labels]
    list_state_delay = [np.stack(tl['state_delay']) for tl in list_trial_labels]
    list_state_press2 = [np.stack(tl['state_press2']) for tl in list_trial_labels]
    list_chemo_type = [np.array(tl['chemo_type']) for tl in list_trial_labels]
    # get press delay error.
    d = np.linspace(np.min(xlim), np.max(xlim), 5000)
    press2 = [(lsp[:,0]-lsd[:,0])-ld for lsp, lsd, ld in zip(list_state_press2, list_state_delay, list_delay)]
    press2_control_1 = [p[(t==1)*(c==0)] for p, t, c in zip(press2, list_trial_type, list_chemo_type)]
    press2_control_2 = [p[(t==2)*(c==0)] for p, t, c in zip(press2, list_trial_type, list_chemo_type)]
    press2_chemo_1   = [p[(t==1)*(c==1)] for p, t, c in zip(press2, list_trial_type, list_chemo_type)]
    press2_chemo_2   = [p[(t==2)*(c==1)] for p, t, c in zip(press2, list_trial_type, list_chemo_type)]
    press2_control_1 = np.concatenate([p[~np.isnan(p)] for p in press2_control_1])
    press2_control_2 = np.concatenate([p[~np.isnan(p)] for p in press2_control_2])
    press2_chemo_1   = np.concatenate([p[~np.isnan(p)] for p in press2_chemo_1])
    press2_chemo_2   = np.concatenate([p[~np.isnan(p)] for p in press2_chemo_2])
    press2_control_1 = gaussian_kde(press2_control_1, bw_method=0.05)(d)
    press2_control_2 = gaussian_kde(press2_control_2, bw_method=0.05)(d)
    press2_chemo_1   = gaussian_kde(press2_chemo_1, bw_method=0.05)(d)
    press2_chemo_2   = gaussian_kde(press2_chemo_2, bw_method=0.05)(d)
    # define layouts.
    #fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax0 = ax.inset_axes([0.1, 0.1, 0.3, 0.8], transform=ax.transAxes)
    ax1 = ax.inset_axes([0.5, 0.1, 0.3, 0.8], transform=ax.transAxes)
    ax2 = ax.inset_axes([0.9, 0.1, 0.1, 0.8], transform=ax.transAxes)
    # plot distribution.
    ax0.plot(d, press2_control_1, color=colors[0])
    ax1.plot(d, press2_control_2, color=colors[0])
    ax0.plot(d, press2_chemo_1,   color=colors[1])
    ax1.plot(d, press2_chemo_2,   color=colors[1])
    # adjust layouts.
    ax.set_title('Press2 and target delay difference across sessions')
    ax.set_xlabel('Press2 - delay (ms)')
    for axi in [ax0,ax1]:
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.axvline(0, color='black', lw=1, linestyle='--')
    ax0.text(
        xlim[0]/2, np.nanmax(press2_control_1)*0.9, 'early',
        ha='center', va='center')
    ax0.text(
        xlim[1]/2, np.nanmax(press2_control_1)*0.9, 'late',
        ha='center', va='center')
    ax1.text(
        xlim[0]/2, np.nanmax(press2_control_2)*0.9, 'early',
        ha='center', va='center')
    ax1.text(
        xlim[1]/2, np.nanmax(press2_control_2)*0.9, 'late',
        ha='center', va='center')
    ax0.set_title('Short trials')
    ax1.set_title('Long trials')
    ax0.set_ylabel('Density')
    ax1.set_yticklabels([])
    add_legend(ax2, colors, ['control', 'chemo'], 'upper right')
    hide_all_axis(ax)
    hide_all_axis(ax2)

# press timing.
def plot_press2_time(ax, list_sess_time, list_trial_labels):
    xlim = [-200,1500]
    colors = ['dimgrey', 'orangered']
    # read data.
    list_trial_type = [np.array(tl['trial_type']) for tl in list_trial_labels]
    list_delay = [np.array(tl['delay']) for tl in list_trial_labels]
    list_state_delay = [np.stack(tl['state_delay']) for tl in list_trial_labels]
    list_state_press2 = [np.stack(tl['state_press2']) for tl in list_trial_labels]
    list_chemo_type = [np.array(tl['chemo_type']) for tl in list_trial_labels]
    # get press delay.
    delay1 = np.nanmean(np.concatenate([ld[t==1] for ld, t in zip(list_delay, list_trial_type)]))
    delay2 = np.nanmean(np.concatenate([ld[t==2] for ld, t in zip(list_delay, list_trial_type)]))
    d = np.linspace(np.min(xlim), np.max(xlim), 5000)
    press2 = [lsp[:,0]-lsd[:,0] for lsp, lsd in zip(list_state_press2, list_state_delay)]
    press2_control_1 = [p[(t==1)*(c==0)] for p, t, c in zip(press2, list_trial_type, list_chemo_type)]
    press2_control_2 = [p[(t==2)*(c==0)] for p, t, c in zip(press2, list_trial_type, list_chemo_type)]
    press2_chemo_1   = [p[(t==1)*(c==1)] for p, t, c in zip(press2, list_trial_type, list_chemo_type)]
    press2_chemo_2   = [p[(t==2)*(c==1)] for p, t, c in zip(press2, list_trial_type, list_chemo_type)]
    press2_control_1 = np.concatenate([p[~np.isnan(p)] for p in press2_control_1])
    press2_control_2 = np.concatenate([p[~np.isnan(p)] for p in press2_control_2])
    press2_chemo_1   = np.concatenate([p[~np.isnan(p)] for p in press2_chemo_1])
    press2_chemo_2   = np.concatenate([p[~np.isnan(p)] for p in press2_chemo_2])
    press2_control_1 = gaussian_kde(press2_control_1, bw_method=0.05)(d)
    press2_control_2 = gaussian_kde(press2_control_2, bw_method=0.05)(d)
    press2_chemo_1   = gaussian_kde(press2_chemo_1, bw_method=0.05)(d)
    press2_chemo_2   = gaussian_kde(press2_chemo_2, bw_method=0.05)(d)
    # define layouts.
    #fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax0 = ax.inset_axes([0.1, 0.1, 0.3, 0.8], transform=ax.transAxes)
    ax1 = ax.inset_axes([0.5, 0.1, 0.3, 0.8], transform=ax.transAxes)
    ax2 = ax.inset_axes([0.9, 0.1, 0.1, 0.8], transform=ax.transAxes)
    # plot distribution.
    ax0.plot(d, press2_control_1, color=colors[0])
    ax1.plot(d, press2_control_2, color=colors[0])
    ax0.plot(d, press2_chemo_1,   color=colors[1])
    ax1.plot(d, press2_chemo_2,   color=colors[1])
    # adjust layouts.
    ax.set_title('Press2 onset timing comparison')
    ax.set_xlabel('Time since delay start (ms)')
    for axi in [ax0,ax1]:
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
    ax0.axvline(delay1, color='black', lw=1, linestyle='--')
    ax1.axvline(delay2, color='black', lw=1, linestyle='--')
    ax0.set_title('Short trials')
    ax1.set_title('Long trials')
    ax0.set_ylabel('Density')
    ax1.set_yticklabels([])
    add_legend(ax2, colors, ['control', 'chemo'], 'upper right')
    hide_all_axis(ax)
    hide_all_axis(ax2)

# joystick deflection for outcomes.
def plot_js_rot_outcome(ax, list_sess_time, list_trial_labels, outcome):
    xlim = [-2000,2000]
    colors = ['dimgrey', 'orangered']
    # read data.
    list_outcome = [np.array(tl['outcome']) for tl in list_trial_labels]
    list_trial_type = [np.array(tl['trial_type']) for tl in list_trial_labels]
    list_chemo_type = [np.array(tl['chemo_type']) for tl in list_trial_labels]
    js_alignment_press2 = run_get_js_rot(list_trial_labels, 'state_press2')
    press2_control_1 = np.concatenate([
        jr[np.isin(o, [outcome])*np.isin(tt, [1])*(c==0),:]
        for jr, o, tt, c in zip(js_alignment_press2['list_js_rot'], list_outcome, list_trial_type, list_chemo_type)], axis=0)
    press2_control_2 = np.concatenate([
        jr[np.isin(o, [outcome])*np.isin(tt, [2])*(c==0),:]
        for jr, o, tt,c in zip(js_alignment_press2['list_js_rot'], list_outcome, list_trial_type, list_chemo_type)], axis=0)
    press2_chemo_1 = np.concatenate([
        jr[np.isin(o, [outcome])*np.isin(tt, [1])*(c==1),:]
        for jr, o, tt, c in zip(js_alignment_press2['list_js_rot'], list_outcome, list_trial_type, list_chemo_type)], axis=0)
    press2_chemo_2 = np.concatenate([
        jr[np.isin(o, [outcome])*np.isin(tt, [2])*(c==1),:]
        for jr, o, tt,c in zip(js_alignment_press2['list_js_rot'], list_outcome, list_trial_type, list_chemo_type)], axis=0)
    # cut data within range.
    l_idx, r_idx = get_frame_idx_from_time(js_alignment_press2['js_time'], 0, xlim[0], xlim[1])
    js_time = js_alignment_press2['js_time'][l_idx:r_idx]
    press2_control_1 = get_mean_sem(press2_control_1[:,l_idx:r_idx])
    press2_control_2 = get_mean_sem(press2_control_2[:,l_idx:r_idx])
    press2_chemo_1  = get_mean_sem(press2_chemo_1[:,l_idx:r_idx])
    press2_chemo_2  = get_mean_sem(press2_chemo_2[:,l_idx:r_idx])
    # define layouts.
    #fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax0 = ax.inset_axes([0.1, 0.1, 0.3, 0.8], transform=ax.transAxes)
    ax1 = ax.inset_axes([0.5, 0.1, 0.3, 0.8], transform=ax.transAxes)
    ax2 = ax.inset_axes([0.9, 0.1, 0.1, 0.8], transform=ax.transAxes)
    # plot lines.
    plot_mean_sem(ax0, js_time, press2_control_1[0], press2_control_1[1], colors[0])
    plot_mean_sem(ax1, js_time, press2_control_2[0], press2_control_2[1], colors[0])
    plot_mean_sem(ax0, js_time, press2_chemo_1[0],   press2_chemo_1[1],   colors[1])
    plot_mean_sem(ax1, js_time, press2_chemo_2[0],   press2_chemo_2[1],   colors[1])
    # adjust layouts.
    for axi in [ax0,ax1]:
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.set_xlim(xlim)
        axi.set_ylim([0,4])
        axi.xaxis.set_major_locator(mtick.MaxNLocator(nbins=5))
        axi.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        axi.yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axi.axvline(0, color='black', lw=1, linestyle='--')
    ax.set_title(f'Press2 aligned at {outcome}')
    ax.set_xlabel('Time since press2 (ms)')
    ax0.set_ylabel('Joystick deflection (deg)')
    ax0.set_title('Short trials')
    ax1.set_title('Long trials')
    add_legend(ax2, colors, ['control', 'chemo'], 'upper right')
    hide_all_axis(ax)
    hide_all_axis(ax2)

# plot all figures.
def run(subject_name, list_sess_time, list_trial_labels):
    print(f'Plotting results for subject {subject_name}')
    size_scale = 3
    n_row = 4
    n_col = 4
    fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
    gs = GridSpec(n_row, n_col, figure=fig)
    plot_outcome_hist(plt.subplot(gs[0:2, 0]), list_sess_time, list_trial_labels)
    plot_outcome_box(plt.subplot(gs[2, 0]), list_sess_time, list_trial_labels)
    plot_press2_time_err(plt.subplot(gs[0, 1:4]), list_sess_time, list_trial_labels)
    plot_press2_time(plt.subplot(gs[1, 1:4]), list_sess_time, list_trial_labels)
    plot_js_rot_outcome(plt.subplot(gs[2, 1:4]), list_sess_time, list_trial_labels, 'reward')
    plot_js_rot_outcome(plt.subplot(gs[3, 1:4]), list_sess_time, list_trial_labels, 'early_2nd_press')
    plt.suptitle(subject_name)
    fig.set_size_inches(n_col*size_scale, n_row*size_scale)
    fig.savefig(os.path.join('results',  subject_name+'.svg'), dpi=300, format='svg')
    plt.close(fig)
