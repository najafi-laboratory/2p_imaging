#!/usr/bin/env python3

import numpy as np

from utils import get_mean_sem, plot_mean_sem, adjust_layout_neu, add_legend


LIST_STATES = [
    'state_vis1', 'state_press1', 'state_press2', 'state_reward',
]
OUTCOMES = [
    'reward',
    'early_2nd_press',
    #'late_2nd_press'
    ]
COLORS = [
    'mediumseagreen',
    'dodgerblue',
    #'lightcoral'
    ]


def plot_state_alignments(axs, alignment, list_labels, list_trial_labels,
                          cate_list, label_names, trial_type, states=None,
                          list_lick_labels=None):
    states = LIST_STATES if states is None else states
    xlim = [-2000, 2000]
    row_labels = ['Joystick deflection (deg)'] + [
        ' + '.join(label_names[str(c)] for c in cate) + ' dF/F'
        for cate in cate_list]
    for ci, state in enumerate(states):
        aligned = alignment[state]
        session_labels = list_lick_labels if state == 'state_lick' else list_trial_labels
        time = aligned['neu_time']
        window = np.isfinite(time) & (time >= xlim[0]) & (time <= xlim[1])
        for ri, ylabel in enumerate(row_labels):
            ax = axs[ri, ci]
            adjust_layout_neu(ax)
            ax.set_xlim(xlim)
            ax.axvline(0, color='black', lw=1, linestyle='--')
            if ci == 0:
                ax.set_ylabel(ylabel)
            if ri == 0:
                ax.set_title(state)
            if ri == len(row_labels)-1:
                ax.set_xlabel('Time since lick (ms)' if state == 'state_lick'
                              else 'Time since state start (ms)')
            if not np.any(window):
                continue
            for outcome, color in zip(OUTCOMES, COLORS):
                samples = []
                for si, trial_labels in enumerate(session_labels):
                    trials = ((np.asarray(trial_labels['trial_type']) == trial_type)
                              & (aligned['list_outcome'][si] == outcome))
                    if not np.any(trials):
                        continue
                    if ri == 0:
                        data = aligned['list_js_rot'][si][trials][:, window]
                    else:
                        neurons = np.isin(list_labels[si], cate_list[ri-1])
                        if not np.any(neurons):
                            continue
                        data = aligned['list_neu_seq'][si][np.ix_(trials, neurons, window)]
                        data = data.reshape(-1, window.sum())
                    samples.append(data)
                if samples:
                    data = np.concatenate(samples, axis=0)
                    if np.any(np.isfinite(data)):
                        mean, sem = get_mean_sem(data)
                        plot_mean_sem(ax, time[window], mean, sem, color)
    add_legend(axs[0, -1], COLORS, OUTCOMES, None, None, None, 'upper right')
