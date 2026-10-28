#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import os
import matplotlib.pyplot as plt
import numpy as np

from utils import get_figure_layout

from modules.ReadResults import read_all
from modules.Alignment import run_sess_alignment, run_lick_alignment
from plot.fig4_alignments import LIST_STATES, plot_state_alignments


def run(session_config_list, smooth, cate_list):
    size_scale = 4.2
    list_labels, _, list_neural_trials = read_all(session_config_list, smooth)
    alignment = run_sess_alignment(list_neural_trials)
    lick_alignment, lick_labels = run_lick_alignment(list_neural_trials)
    alignment.update(lick_alignment)
    states = LIST_STATES + ['state_lick']
    list_trial_labels = [nt['trial_labels'] for nt in list_neural_trials]
    output_dir = os.path.join('results', 'temp_'+session_config_list['subject_name'])
    os.makedirs(output_dir, exist_ok=True)

    def plot_alignments(trial_type, trial_name):
        title = trial_name + ' trial state alignments'
        print('-----------------------------------------------')
        print(title)
        filename = 'state_alignments_' + trial_name
        n_row, n_col = 1 + len(cate_list), len(states)
        fig, gs = get_figure_layout(n_row, n_col, size_scale, 'tight')
        axs = np.empty((n_row, n_col), dtype=object)
        for ri in range(n_row):
            for ci in range(n_col):
                parent = fig.add_subplot(gs[ri, ci])
                parent.axis('off')
                axs[ri, ci] = parent.inset_axes([0.16, 0.20, 0.76, 0.72])
        try:
            plot_state_alignments(
                axs, alignment, list_labels, list_trial_labels,
                cate_list, session_config_list['label_names'], trial_type, states,
                list_lick_labels=lick_labels)
            fig.suptitle(title)
            fig.savefig(os.path.join(output_dir, filename+'.svg'), dpi=300, format='svg')
        finally:
            plt.close(fig)
        return [filename, n_row, n_col, title]

    fig_all = [
        plot_alignments(1, 'short'),
        plot_alignments(2, 'long'),
    ]
    return fig_all
