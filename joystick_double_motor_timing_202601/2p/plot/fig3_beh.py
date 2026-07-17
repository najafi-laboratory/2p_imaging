#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from modules.Alignment import run_get_js_rot
from utils import get_mean_sem
from utils import get_frame_idx_from_time
from utils import get_cmap_color
from utils import add_legend
from utils import hide_all_axis
from utils import plot_mean_sem
from utils import plot_half_violin

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# outcome histogram.
def plot_outcome_hist(ax, list_sess_time, list_trial_labels):
    eval_states = [
        'reward',
        'no_1st_press',
        'no_2nd_press',
        'early_2nd_press',
        'late_2nd_press',
        'other']
    colors = get_cmap_color(len(eval_states), cmap=plt.cm.rainbow)
    # read data.
    list_outcome = [np.array(tl['outcome']) for tl in list_trial_labels]
    # count labels.
    counts = np.array([[np.mean(outcomes == s) for outcomes in list_outcome] for s in eval_states])
    # define layouts.
    ax0 = ax.inset_axes([0, 0, 0.7, 1], transform=ax.transAxes)
    ax1 = ax.inset_axes([0.8, 0, 0.2, 1], transform=ax.transAxes)
    # plot histogram.
    bottom = np.zeros(len(list_sess_time))
    for si in range(len(eval_states)):
        ax0.bar(
            np.arange(len(list_sess_time))+1,
            counts[-si-1,:],
            bottom=bottom,
            width=0.5,
            color=colors[si])
        bottom += counts[-si-1,:]
    # adjust layouts.
    ax.set_title('Outcome percentage')
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
    ax.set_xticks(np.arange(len(list_sess_time))+1)
    ax.set_xticklabels(list_sess_time, rotation='vertical')
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=5))
    
# press timing error.
def plot_press_time_err(ax, list_sess_time, list_trial_labels):
    # read data.
    list_outcome = [np.array(tl['outcome']) for tl in list_trial_labels]
    list_delay = [np.array(tl['delay']) for tl in list_trial_labels]
    list_state_delay = [np.stack(tl['state_delay']) for tl in list_trial_labels]
    list_state_press2 = [np.stack(tl['state_press2']) for tl in list_trial_labels]
    # get press delay error for outcomes.
    press_err = [(lsp[:,0]-lsd[:,0])-ld for lsp, lsd, ld in zip(list_state_press2, list_state_delay, list_delay)]
    press_err_reward = [e[np.isin(o,['reward'])]          for e, o in zip(press_err, list_outcome)]
    press_err_early  = [e[np.isin(o,['early_2nd_press'])] for e, o in zip(press_err, list_outcome)]
    press_err_late   = [e[np.isin(o,['late_2nd_press'])]  for e, o in zip(press_err, list_outcome)]
    # define layouts.
    ax0 = ax.inset_axes([0,    0, 0.3, 0.8], transform=ax.transAxes)
    ax1 = ax.inset_axes([0.35, 0, 0.3, 0.8], transform=ax.transAxes)
    ax2 = ax.inset_axes([0.7,  0, 0.3, 0.8], transform=ax.transAxes)
    # plot distribution.
    for si,(e1,e2,e3) in enumerate(zip(press_err_reward, press_err_early, press_err_late)):
        if len(e1)>0:
            plot_half_violin(ax0, e1, si, 'dimgrey', 'right')
        if len(e2)>0:
            plot_half_violin(ax1, e2, si, 'dimgrey', 'right')
        if len(e3)>0:
            plot_half_violin(ax2, e3, si, 'dimgrey', 'right')
    # adjust layouts.
    for axi in [ax0,ax1,ax2]:
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.set_xlim([-0.5,len(list_sess_time)+0.5])
        axi.set_ylim([-1000,6000])
        axi.set_xticks(np.arange(len(list_sess_time)))
        axi.set_xticklabels(list_sess_time, rotation='vertical')
    ax.set_title('Press and target delay difference across sessions')
    ax0.set_title('Reward trials')
    ax1.set_title('Early press2 trials')
    ax2.set_title('Late press2 trials')
    ax1.set_xlabel('Training session')
    ax0.set_ylabel('Press2 - delay (ms)')
    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    hide_all_axis(ax)

# joystick deflection for states.
def plot_js_rot_state(ax, list_trial_labels):
    xlim = [-2000,2000]
    colors = get_cmap_color(len(list_trial_labels), cmap=plt.cm.YlGnBu)
    # read data.
    list_outcome = [np.array(tl['outcome']) for tl in list_trial_labels]
    js_alignment_press2 = run_get_js_rot(list_trial_labels, 'state_press2')
    press2_reward = [jr[np.isin(o, ['reward']),:]          for jr, o in zip(js_alignment_press2['list_js_rot'], list_outcome)]
    press2_early  = [jr[np.isin(o, ['early_2nd_press']),:] for jr, o in zip(js_alignment_press2['list_js_rot'], list_outcome)]
    press2_late   = [jr[np.isin(o, ['late_2nd_press']),:]  for jr, o in zip(js_alignment_press2['list_js_rot'], list_outcome)]
    # cut data within range.
    l_idx, r_idx = get_frame_idx_from_time(js_alignment_press2['js_time'], 0, xlim[0], xlim[1])
    js_time = js_alignment_press2['js_time'][l_idx:r_idx]
    press2_reward = [get_mean_sem(p[:,l_idx:r_idx]) for p in press2_reward]
    press2_early  = [get_mean_sem(p[:,l_idx:r_idx]) for p in press2_early]
    press2_late   = [get_mean_sem(p[:,l_idx:r_idx]) for p in press2_late]
    # define layouts.
    ax0 = ax.inset_axes([0, 0.7,  1, 0.25], transform=ax.transAxes)
    ax1 = ax.inset_axes([0, 0.35, 1, 0.25], transform=ax.transAxes)
    ax2 = ax.inset_axes([0, 0,    1, 0.25], transform=ax.transAxes)
    # plot lines.
    for si, _ in enumerate(list_trial_labels):
        plot_mean_sem(ax0, js_time, press2_reward[si][0], press2_reward[si][1], colors[si])
        plot_mean_sem(ax1, js_time, press2_early[si][0],  press2_early[si][1],  colors[si])
        plot_mean_sem(ax2, js_time, press2_late[si][0],   press2_late[si][1],   colors[si])
    # adjust layouts.
    for axi in [ax0,ax1,ax2]:
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.set_xlim(xlim)
        axi.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        axi.yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
    ax0.set_title('Reward trials')
    ax1.set_title('Early press2 trials')
    ax2.set_title('Late press2 trials')
    hide_all_axis(ax)





