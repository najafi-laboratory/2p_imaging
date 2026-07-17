#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import os
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from modeling.generative import get_target_dff_time
from modeling.generative import get_factor_all
from modeling.generative import run_glm_multi_sess
from modeling.generative import get_coding_score_fraction
from modeling.generative import get_coding_score_dropout


# fit model.
def main():
    kernel_win = [-3000, 3000]
    # prepare glm input.
    list_glm_time, list_glm_dff = get_target_dff_time(list_neural_trials)
    model = get_factor_all(list_neural_trials)
    list_factor_names, list_all_factor_in = model.run()
    # fit full glm model.
    print('Fitting GLM full model')
    kernel_time, kernel_all, exp_var_all, reconst_all = run_glm_multi_sess(list_glm_dff, list_all_factor_in, list_glm_time, kernel_win)
    results_all = {
        'kernel_time': kernel_time,
        'kernel_all': kernel_all,
        'exp_var_all': exp_var_all,
        'reconst_all': reconst_all,}
    # fit single glm model.
    print('Fitting GLM single factor model')
    results_single = {
        'kernel_time': [],
        'kernel_all': [],
        'exp_var_all': [],
        'reconst_all': []}
    for fi, lfn in enumerate(list_factor_names):
        print(f'Fitting GLM with only {lfn}, {fi+1}/{len(list_factor_names)}')
        list_factor_in = [[f[fi]] for f in list_all_factor_in]
        kernel_time, kernel_all, exp_var_all, reconst_all = run_glm_multi_sess(list_glm_dff, list_factor_in, list_glm_time, kernel_win)
        results_single['kernel_time'].append(kernel_time)
        results_single['kernel_all'].append(kernel_all[:,0,:])
        results_single['exp_var_all'].append(exp_var_all)
        results_single['reconst_all'].append(reconst_all)
    # fit dropout glm model.
    print('Fitting GLM factor dropout model')
    results_dropout = {
        'kernel_time': [],
        'kernel_all': [],
        'exp_var_all': [],
        'reconst_all': []}
    for fi, lfn in enumerate(list_factor_names):
        print(f'Fitting GLM by dropout {lfn}, {fi+1}/{len(list_factor_names)}')
        list_factor_in = [f[:fi]+f[fi+1:] for f in list_all_factor_in]
        kernel_time, kernel_all, exp_var_all, reconst_all = run_glm_multi_sess(list_glm_dff, list_factor_in, list_glm_time, kernel_win)
        results_dropout['kernel_time'].append(kernel_time)
        results_dropout['kernel_all'].append(kernel_all)
        results_dropout['exp_var_all'].append(exp_var_all)
        results_single['reconst_all'].append(reconst_all)
    # evaluation.
    score_fraction = get_coding_score_fraction(results_all['exp_var_all'], results_single['exp_var_all'])
    score_dropout = get_coding_score_dropout(results_all['exp_var_all'], results_dropout['exp_var_all'])
    return [list_factor_names,
            results_all, results_single, results_dropout,
            score_fraction, score_dropout]
[list_factor_names,
 results_all, results_single, results_dropout,
 score_fraction, score_dropout] = main()



# visualize factor from dropout model.
fig, axs = plt.subplots(3, len(list_factor_names), figsize=(4*len(list_factor_names), 8))
for fi in range(len(list_factor_names)):
    try:
        idx = np.argsort(score_dropout[:,fi])[::-1][:int(len(score_dropout[:,fi])*1)]
        m, s = get_mean_sem(results_all['kernel_all'][idx,fi,:])
        plot_mean_sem(axs[0,fi], results_all['kernel_time'], m, s, 'dimgrey')
        axs[0,fi].set_title(list_factor_names[fi])
        axs[0,fi].spines['right'].set_visible(False)
        axs[0,fi].spines['top'].set_visible(False)
        axs[0,fi].axvline(0, color='black', lw=1, linestyle='--')
        axs[0,fi].xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axs[0,fi].yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axs[1,fi].hist(score_dropout[:,fi], color='dimgrey', bins=200)
        axs[1,fi].spines['right'].set_visible(False)
        axs[1,fi].spines['top'].set_visible(False)
        axs[1,fi].set_xlim([-0.1,1])
        axs[1,fi].xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axs[1,fi].yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        plot_heatmap_neuron(
            axs[2,fi], None, results_all['kernel_all'][idx,fi,:], results_all['kernel_time'],
            results_all['kernel_all'][idx,fi,:], norm_mode='minmax')
    except: traceback.print_exc()
# visualize factor from single model.
fig, axs = plt.subplots(3, len(list_factor_names), figsize=(4*len(list_factor_names), 8))
for fi in range(len(list_factor_names)):
    try:
        idx = np.argsort(score_fraction[:,fi])[::-1][:int(len(score_fraction[:,fi])*1)]
        m, s = get_mean_sem(results_single['kernel_all'][fi][idx,:])
        plot_mean_sem(axs[0,fi], results_all['kernel_time'], m, s, 'dimgrey')
        axs[0,fi].set_title(list_factor_names[fi])
        axs[0,fi].spines['right'].set_visible(False)
        axs[0,fi].spines['top'].set_visible(False)
        axs[0,fi].axvline(0, color='black', lw=1, linestyle='--')
        axs[0,fi].xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axs[0,fi].yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axs[1,fi].hist(score_fraction[:,fi], color='dimgrey', bins=200)
        axs[1,fi].spines['right'].set_visible(False)
        axs[1,fi].spines['top'].set_visible(False)
        axs[1,fi].set_xlim([-0.1,1])
        axs[1,fi].xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axs[1,fi].yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        plot_heatmap_neuron(
            axs[2,fi], None, results_single['kernel_all'][fi][idx,:], results_all['kernel_time'],
            results_single['kernel_all'][fi][idx,:], norm_mode='minmax')
    except: traceback.print_exc()
# visualize reconstruction with all model.
si = 0
ni = 0
reconst = np.nanmean(results_all['reconst_all'][si], axis=1)
dff = list_neural_trials[si]['dff']
time = list_neural_trials[si]['time']
state_vis1 = np.array(list_neural_trials[si]['trial_labels']['state_vis1'])
state_press1 = np.array(list_neural_trials[si]['trial_labels']['state_press1'])
state_press2 = np.array(list_neural_trials[si]['trial_labels']['state_press2'])
state_reward = np.array(list_neural_trials[si]['trial_labels']['state_reward'])
fig, ax = plt.subplots(1, 1, figsize=(24, 6), layout='tight')
dff = np.apply_along_axis(
    savgol_filter, 1, dff,
    window_length=9,
    polyorder=3)
ax.plot(time, dff[ni,:], color='black')
ax.plot(time, (reconst[ni,:]-np.nanmean(reconst[ni,:]))/np.nanstd(reconst[ni,:]), color='crimson')
for s in state_vis1:
    if not np.isnan(s[0]):
        ax.fill_between(s, -5, 5, color='dimgrey', edgecolor='none', alpha=0.25, step='mid')
for s in state_press1:
    if not np.isnan(s[0]):
        ax.vlines(s[0], -5, 5, color='dodgerblue', lw=1, linestyle='--') 
for s in state_press2:
    if not np.isnan(s[0]):
        ax.vlines(s[0], -5, 5, color='royalblue', lw=1, linestyle='--')
for s in state_reward:
    if not np.isnan(s[0]):
        ax.fill_between(s, -5, 5, color='mediumseagreen', edgecolor='none', alpha=0.25, step='mid')
add_legend(ax, ['black','crimson', 'dimgrey', 'dodgerblue', 'royalblue', 'mediumseagreen'],
           ['DF/F','Model', 'Visual cue', 'Press 1', 'Press 2', 'Reward'], None, None, None, 'upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('dF/F (z-scored)')
ax.set_xlabel('Time (ms)')
ax.set_title('Explained variance=0.046')

