#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import gaussian_kde

from modules.Alignment import run_get_js_rot
from utils import get_mean_sem
from utils import get_frame_idx_from_time
from utils import get_cmap_color
from utils import add_legend
from utils import hide_all_axis
from utils import plot_mean_sem
from utils import plot_mean_sem_acatter_box

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# outcome histogram.
def plot_outcome_hist(ax, list_sess_time, list_trial_labels):
    eval_states = [
        'reward',
        'no_2nd_press',
        'early_2nd_press',
        'assist']
    colors = ['dimgrey', 'cornflowerblue', 'coral', 'mediumseagreen']
    # read data.
    list_outcome = [np.array(tl['outcome']) for tl in list_trial_labels]
    list_trial_type = [np.array(tl['trial_type']) for tl in list_trial_labels]
    # count labels.
    counts_1 = np.array([[np.nansum(o[t==1] == s) for o, t in zip(list_outcome, list_trial_type)] for s in eval_states])
    counts_2 = np.array([[np.nansum(o[t==2] == s) for o, t in zip(list_outcome, list_trial_type)] for s in eval_states])
    counts_1 = counts_1 / np.nansum(counts_1, axis=0)
    counts_2 = counts_2 / np.nansum(counts_2, axis=0)
    # define layouts.
    #fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax0 = ax.inset_axes([0, 0, 0.7, 1], transform=ax.transAxes)
    ax1 = ax.inset_axes([0.8, 0, 0.2, 1], transform=ax.transAxes)
    # plot histogram.
    bottom_1 = np.zeros(len(list_sess_time))
    bottom_2 = np.zeros(len(list_sess_time))
    for si in range(len(eval_states)):
        ax0.bar(
            np.arange(len(list_sess_time))+1-0.15,
            counts_1[-si-1,:],
            bottom=bottom_1,
            width=0.25,
            color=colors[si])
        ax0.bar(
            np.arange(len(list_sess_time))+1+0.15,
            counts_2[-si-1,:],
            bottom=bottom_2,
            width=0.25,
            color=colors[si])
        bottom_1 += counts_1[-si-1,:]
        bottom_2 += counts_2[-si-1,:]
    # adjust layouts.
    ax0.set_title('Outcome percentage\nshort|long')
    ax0.tick_params(tick1On=False)
    ax0.spines['left'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.yaxis.grid(True)
    ax0.set_xlabel('Training session')
    ax0.set_ylabel('Fraction of trials')
    ax0.set_xticks(np.arange(len(list_sess_time))+1)
    ax0.set_xticklabels(list_sess_time, rotation='vertical')
    add_legend(ax1, colors[::-1], eval_states, 'upper right')
    hide_all_axis(ax)
    hide_all_axis(ax1)

# delay progression.
def plot_delay_dist(ax, list_sess_time, list_trial_labels):
    #fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # read data.
    list_delay = [np.array(tl['delay']) for tl in list_trial_labels]
    # plot lines.
    ax.vlines(
        x=np.arange(len(list_sess_time))+1,
        ymin=[np.percentile(d,25) for d in list_delay],
        ymax=[np.percentile(d,75) for d in list_delay],
        color='dimgrey')
    ax.scatter(
        x=np.tile(np.arange(len(list_sess_time))+1, 2),
        y=[np.percentile(d,25) for d in list_delay] + [np.percentile(d,75) for d in list_delay],
        color='dimgrey', s=10)
    # adjust layouts.
    ax.set_title('2nd press delay across sessions')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Training session')
    ax.set_ylabel('2nd press delay (ms)')
    ax.set_xlim([0, len(list_sess_time)+1])
    ax.set_ylim([300,800])
    ax.set_xticks(np.arange(len(list_sess_time))+1)
    ax.set_xticklabels(list_sess_time, rotation='vertical')
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=5))
    
# press timing error.
def plot_press_time_err(ax, list_sess_time, list_trial_labels):
    xlim = [-1000,1500]
    colors = get_cmap_color(len(list_trial_labels), cmap=plt.cm.bone_r)
    # read data.
    list_trial_type = [np.array(tl['trial_type']) for tl in list_trial_labels]
    list_delay = [np.array(tl['delay']) for tl in list_trial_labels]
    list_state_delay = [np.stack(tl['state_delay']) for tl in list_trial_labels]
    list_state_press2 = [np.stack(tl['state_press2']) for tl in list_trial_labels]
    # get press delay error.
    d = np.linspace(np.min(xlim), np.max(xlim), 5000)
    press_err = [(lsp[:,0]-lsd[:,0])-ld for lsp, lsd, ld in zip(list_state_press2, list_state_delay, list_delay)]
    press_err_1 = [e[t==1] for e, t in zip(press_err, list_trial_type)]
    press_err_2 = [e[t==2] for e, t in zip(press_err, list_trial_type)]
    press_err_1 = [e[~np.isnan(e)] for e in press_err_1]
    press_err_2 = [e[~np.isnan(e)] for e in press_err_2]
    press_err_1 = [gaussian_kde(e, bw_method=0.1)(d) for e in press_err_1]
    press_err_2 = [gaussian_kde(e, bw_method=0.1)(d) for e in press_err_2]
    # define layouts.
    #fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax0 = ax.inset_axes([0.1, 0.1, 0.3, 0.8], transform=ax.transAxes)
    ax1 = ax.inset_axes([0.5, 0.1, 0.3, 0.8], transform=ax.transAxes)
    ax2 = ax.inset_axes([0.9, 0.1, 0.1, 0.8], transform=ax.transAxes)
    # plot distribution.
    for si, _ in enumerate(list_trial_labels):
        ax0.plot(d, press_err_1[si], color=colors[si])
        ax1.plot(d, press_err_2[si], color=colors[si])
    # adjust layouts.
    for axi in [ax0,ax1]:
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.axvline(0, color='black', lw=1, linestyle='--')
    ax0.text(
        xlim[0]/2, np.nanmax(press_err_1)*0.9, 'early',
        ha='center', va='center')
    ax0.text(
        xlim[1]/2, np.nanmax(press_err_1)*0.9, 'late',
        ha='center', va='center')
    ax1.text(
        xlim[0]/2, np.nanmax(press_err_2)*0.9, 'early',
        ha='center', va='center')
    ax1.text(
        xlim[1]/2, np.nanmax(press_err_2)*0.9, 'late',
        ha='center', va='center')
    ax.set_title('Press2 and target delay difference across sessions')
    ax0.set_title('Short trials')
    ax1.set_title('Long trials')
    ax0.set_xlabel('Press2 - delay (ms)')
    ax1.set_xlabel('Press2 - delay (ms)')
    ax0.set_ylabel('Density')
    ax1.set_yticklabels([])
    add_legend(ax2, colors, list_sess_time, 'upper right')
    hide_all_axis(ax)
    hide_all_axis(ax2)
    
# assist delay change results.
def plot_assist_delay(ax, list_sess_time, list_trial_labels):
    ylim = [-1000,1000]
    color0 = 'dimgrey'
    color1 = 'royalblue'
    # read data.
    list_trial_type = [np.array(tl['trial_type']) for tl in list_trial_labels]
    list_assist_trial = [np.array(tl['assist_trial']) for tl in list_trial_labels]
    list_delay = [np.array(tl['delay']) for tl in list_trial_labels]
    list_state_delay = [np.stack(tl['state_delay']) for tl in list_trial_labels]
    list_state_press2 = [np.stack(tl['state_press2']) for tl in list_trial_labels]
    # get assist index.
    list_assist_trial_before = [np.concatenate((at[1:], [0])) for at in list_assist_trial]
    list_assist_trial_after  = [np.concatenate(([0], at[:-1])) for at in list_assist_trial]
    # get press delay error.
    press_err = [(lsp[:,0]-lsd[:,0])-ld for lsp, lsd, ld in zip(list_state_press2, list_state_delay, list_delay)]
    press_err_1_before = [e[(t==1)*(at==1)] for e, t, at in zip(press_err, list_trial_type, list_assist_trial_before)]
    press_err_1_after  = [e[(t==1)*(at==1)] for e, t, at in zip(press_err, list_trial_type, list_assist_trial_after)]
    press_err_2_before = [e[(t==2)*(at==1)] for e, t, at in zip(press_err, list_trial_type, list_assist_trial_before)]
    press_err_2_after  = [e[(t==2)*(at==1)] for e, t, at in zip(press_err, list_trial_type, list_assist_trial_after)]
    press_err_1_before = np.concatenate(press_err_1_before)
    press_err_1_after  = np.concatenate(press_err_1_after)
    press_err_2_before = np.concatenate(press_err_2_before)
    press_err_2_after  = np.concatenate(press_err_2_after)
    # define layouts.
    #fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax0 = ax.inset_axes([0, 0, 0.8, 1], transform=ax.transAxes)
    ax1 = ax.inset_axes([0.8, 0, 0.2, 1], transform=ax.transAxes)
    plot_mean_sem_acatter_box(ax0, press_err_1_before, -2, color0, 0.25)
    plot_mean_sem_acatter_box(ax0, press_err_1_after,  -1, color1, 0.25)
    plot_mean_sem_acatter_box(ax0, press_err_2_before,  1, color0, 0.25)
    plot_mean_sem_acatter_box(ax0, press_err_2_after,   2, color1, 0.25)
    # adjust layouts.
    ax0.axhline(0, color='black', lw=1, linestyle='--')
    ax0.set_title('Press2 and target delay difference before and after assist')
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.set_xlim([-3.5,3.5])
    ax0.set_ylim(ylim)
    ax0.set_xticks([-1.5,1.5])
    ax0.set_xticklabels(['Short','Long'])
    ax0.set_xlabel('Trial type')
    ax0.set_ylabel('Press2 - delay (ms)')
    ax0.yaxis.set_major_locator(mtick.MaxNLocator(nbins=5))
    ax0.text(
        -3, ylim[0]*0.8, 'early',
        ha='center', va='center', rotation='vertical')
    ax0.text(
        -3, ylim[1]*0.8, 'late',
        ha='center', va='center', rotation='vertical')
    add_legend(ax1, [color0, color1], ['before', 'after'], 'upper right')
    hide_all_axis(ax)
    hide_all_axis(ax1)

# joystick deflection for states.
def plot_js_rot_state(ax, list_sess_time, list_trial_labels):
    xlim = [-2000,2000]
    colors = get_cmap_color(len(list_trial_labels), cmap=plt.cm.bone_r)
    # read data.
    list_outcome = [np.array(tl['outcome']) for tl in list_trial_labels]
    list_trial_type = [np.array(tl['trial_type']) for tl in list_trial_labels]
    js_alignment_press2 = run_get_js_rot(list_trial_labels, 'state_press2')
    press2_reward_1 = [jr[np.isin(o, ['reward'])*np.isin(tt, [1]),:]          for jr, o, tt in zip(js_alignment_press2['list_js_rot'], list_outcome, list_trial_type)]
    press2_reward_2 = [jr[np.isin(o, ['reward'])*np.isin(tt, [2]),:]          for jr, o, tt in zip(js_alignment_press2['list_js_rot'], list_outcome, list_trial_type)]
    press2_early_1  = [jr[np.isin(o, ['early_2nd_press'])*np.isin(tt, [1]),:] for jr, o, tt in zip(js_alignment_press2['list_js_rot'], list_outcome, list_trial_type)]
    press2_early_2  = [jr[np.isin(o, ['early_2nd_press'])*np.isin(tt, [2]),:] for jr, o, tt in zip(js_alignment_press2['list_js_rot'], list_outcome, list_trial_type)]
    # cut data within range.
    l_idx, r_idx = get_frame_idx_from_time(js_alignment_press2['js_time'], 0, xlim[0], xlim[1])
    js_time = js_alignment_press2['js_time'][l_idx:r_idx]
    press2_reward_1 = [get_mean_sem(p[:,l_idx:r_idx]) for p in press2_reward_1]
    press2_reward_2 = [get_mean_sem(p[:,l_idx:r_idx]) for p in press2_reward_2]
    press2_early_1  = [get_mean_sem(p[:,l_idx:r_idx]) for p in press2_early_1]
    press2_early_2  = [get_mean_sem(p[:,l_idx:r_idx]) for p in press2_early_2]
    # define layouts.
    #fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    axs0 = ax.inset_axes([0, 0, 0.4, 1], transform=ax.transAxes)
    axs1 = ax.inset_axes([0.4, 0, 0.4, 1], transform=ax.transAxes)
    axs2 = ax.inset_axes([0.9, 0, 0.1, 1], transform=ax.transAxes)
    ax0 = axs0.inset_axes([0.2, 0.1, 0.8, 0.3], transform=axs0.transAxes)
    ax1 = axs0.inset_axes([0.2, 0.6, 0.8, 0.3], transform=axs0.transAxes)
    ax2 = axs1.inset_axes([0.2, 0.1, 0.8, 0.3], transform=axs1.transAxes)
    ax3 = axs1.inset_axes([0.2, 0.6, 0.8, 0.3], transform=axs1.transAxes)
    # plot lines.
    for si, _ in enumerate(list_trial_labels):
        plot_mean_sem(ax0, js_time, press2_reward_1[si][0], press2_reward_1[si][1], colors[si])
        plot_mean_sem(ax1, js_time, press2_early_1[si][0],  press2_early_1[si][1],  colors[si])
        plot_mean_sem(ax2, js_time, press2_reward_2[si][0], press2_reward_2[si][1], colors[si])
        plot_mean_sem(ax3, js_time, press2_early_2[si][0],  press2_early_2[si][1],  colors[si])
    # adjust layouts.
    for axi in [ax0,ax1,ax2,ax3]:
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.set_xlim(xlim)
        axi.set_ylim([0,4])
        axi.xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axi.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        axi.yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axi.axvline(0, color='black', lw=1, linestyle='--')
    for axi in [ax1,ax3]:
        axi.set_xticklabels([])
    for axi in [ax2,ax3]:
        axi.set_yticklabels([])
    ax.set_xlabel('Time since press2 (ms)')
    axs0.set_ylabel('Joystick deflection (deg)')
    ax0.set_title('Reward')
    ax1.set_title('Early press2')
    ax2.set_title('Reward')
    ax3.set_title('Early press2')
    axs0.set_title('Short trials')
    axs1.set_title('Long trials')
    add_legend(axs2, colors, list_sess_time, 'upper right')
    hide_all_axis(ax)
    hide_all_axis(axs0)
    hide_all_axis(axs1)
    hide_all_axis(axs2)


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
plot_outcome_hist(ax, list_sess_time, list_trial_labels)

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
plot_press_time_err(ax, list_sess_time, list_trial_labels)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
plot_assist_delay(ax, list_sess_time, list_trial_labels)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
plot_js_rot_state(ax, list_sess_time, list_trial_labels)
