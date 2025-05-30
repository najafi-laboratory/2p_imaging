#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import gc
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from modules.ReadResults import filter_session_config_list
from modules.ReadResults import read_all
from plot.misc import plot_significance
from plot.fig3_intervals import plot_standard_type
from plot.fig3_intervals import plot_fix_jitter_type
from plot.fig3_intervals import plot_oddball_type
from plot.fig3_intervals import plot_random_type
from plot.fig3_intervals import plot_standard_isi_distribution
from plot.fig3_intervals import plot_jitter_isi_distribution
from plot.fig3_intervals import plot_oddball_isi_distribution
from plot.fig3_intervals import plot_random_isi_distribution
from plot.fig3_intervals import plot_stim_type
from plot.fig3_intervals import plot_stim_label
from plot.fig3_intervals import plot_trial_legend
from plot.fig5_1451ShortLong import plotter_main

def run(session_config_list, smooth):
    size_scale = 3
    target_sess = 'short_long'
    idx_target_sess = np.array(list(session_config_list['list_session_name'].values())) == target_sess
    print('Found {} {} sessions'.format(np.sum(idx_target_sess), target_sess))
    if (np.sum(idx_target_sess)==0) or (np.sum(idx_target_sess)==1 and not idx_target_sess[0]):
        return []
    else:
        print('Reading saved results')
        sub_session_config_list = filter_session_config_list(session_config_list, target_sess)
        [list_labels, _, list_neural_trials, _, list_significance
         ] = read_all(sub_session_config_list, smooth)
        print('Read {} session results'.format(np.sum(idx_target_sess)))
        print('Initiating alignment results')
        plotter = plotter_main(
            list_neural_trials, list_labels, list_significance,
            session_config_list['label_names'], 'temp_'+session_config_list['subject_name'])
        def plot_sess_significance():
            title = 'significance'
            print('-----------------------------------------------')
            print(title)
            filename = '1451ShortLong01_significance'
            n_row = 1
            n_col = 1
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            sign_ax = plt.subplot(gs[0, 0])
            plot_significance(sign_ax, list_significance, list_labels)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_intervals():
            title = 'interval distribution'
            print('-----------------------------------------------')
            print(title)
            filename = '1451ShortLong02_interval_distribution'
            n_row = 2
            n_col = 4
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            type_ax01 = plt.subplot(gs[0, 0])
            type_ax02 = plt.subplot(gs[0, 1])
            type_ax03 = plt.subplot(gs[0, 2])
            type_ax04 = plt.subplot(gs[0, 3])
            plot_standard_type(type_ax01, list_neural_trials)
            plot_fix_jitter_type(type_ax02, list_neural_trials)
            plot_oddball_type(type_ax03, list_neural_trials)
            plot_random_type(type_ax04, list_neural_trials)
            isi_ax01 = plt.subplot(gs[1, 0])
            isi_ax02 = plt.subplot(gs[1, 1])
            isi_ax03 = plt.subplot(gs[1, 2])
            isi_ax04 = plt.subplot(gs[1, 3])
            plot_standard_isi_distribution(isi_ax01, list_neural_trials)
            plot_jitter_isi_distribution(isi_ax02, list_neural_trials)
            plot_oddball_isi_distribution(isi_ax03, list_neural_trials)
            plot_random_isi_distribution(isi_ax04, list_neural_trials)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_trial():
            title = 'interval trial structure'
            print('-----------------------------------------------')
            print(title)
            filename = '1451ShortLong03_trial_structure'
            n_row = 1
            n_col = 6
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            trial_ax01 = plt.subplot(gs[0, 0])
            trial_ax02 = plt.subplot(gs[0, 1:5])
            trial_ax03 = plt.subplot(gs[0, 5])
            plot_stim_type(trial_ax01, list_neural_trials)
            plot_stim_label(trial_ax02, list_neural_trials)
            plot_trial_legend(trial_ax03)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_cluster_all():
            title = 'all cluster neural traces'
            print('-----------------------------------------------')
            print(title)
            filename = '1451ShortLong04_cluster_all'
            n_row = 16
            n_col = 10
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,4,8,12]:
                a = [plt.subplot(gs[s:s+4, i]) for i in range(10)]
                axs_all.append(a)
            plotter.cluster_all(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_cluster_heatmap_all():
            title = 'all cluster neural traces'
            print('-----------------------------------------------')
            print(title)
            filename = '1451ShortLong05_cluster_heatmap_all'
            n_row = 12
            n_col = 5
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,3,6,9]:
                a = [plt.subplot(gs[s:s+2, 0:2])]
                a+= [plt.subplot(gs[s+2, 0:2])]
                a+= [plt.subplot(gs[s:s+3, i]) for i in [2,3,4]]
                axs_all.append(a)
            plotter.cluster_heatmap_all(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_cluster_block_adapt_individual():
            title = 'individual cluster on block transition'
            print('-----------------------------------------------')
            print(title)
            filename = '1451ShortLong06_cluster_block_adapt_individual'
            n_row = 40
            n_col = plotter.n_clusters
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,10,20,30]:
                a = [[plt.subplot(gs[s+0, i]) for i in range(plotter.n_clusters)]]
                a+= [[plt.subplot(gs[s+1, i]) for i in range(plotter.n_clusters)]]
                a+= [[plt.subplot(gs[s+2, i]) for i in range(plotter.n_clusters)]]
                a+= [[plt.subplot(gs[s+3, i]) for i in range(plotter.n_clusters)]]
                a+= [[[plt.subplot(gs[s+4, i]) for i in range(plotter.n_clusters)],
                      [plt.subplot(gs[s+5, i]) for i in range(plotter.n_clusters)]]]
                a+= [[plt.subplot(gs[s+6, i]) for i in range(plotter.n_clusters)]]
                a+= [[[plt.subplot(gs[s+7, i]) for i in range(plotter.n_clusters)],
                      [plt.subplot(gs[s+8, i]) for i in range(plotter.n_clusters)]]]
                a+= [plt.subplot(gs[s+9, 0])]
                axs_all.append(a)
            plotter.cluster_block_adapt_individual(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_cluster_epoch_adapt_individual():
            title = 'individual cluster on epoch adaptation'
            print('-----------------------------------------------')
            print(title)
            filename = '1451ShortLong06_cluster_epoch_adapt_individual'
            n_row = 28
            n_col = plotter.n_clusters
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,7,14,21]:
                a = [[[plt.subplot(gs[s+0, i]) for i in range(plotter.n_clusters)],
                      [plt.subplot(gs[s+1, i]) for i in range(plotter.n_clusters)],
                      [plt.subplot(gs[s+2, i]) for i in range(plotter.n_clusters)]]]
                a+= [[[plt.subplot(gs[s+3, i]) for i in range(plotter.n_clusters)],
                      [plt.subplot(gs[s+4, i]) for i in range(plotter.n_clusters)],
                      [plt.subplot(gs[s+5, i]) for i in range(plotter.n_clusters)]]]
                a+= [plt.subplot(gs[s+6, 0])]
                axs_all.append(a)
            plotter.cluster_epoch_adapt_individual(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_latent_all():
            title = 'latent dynamics for adaptation'
            print('-----------------------------------------------')
            print(title)
            filename = '1451ShortLong08_latent_all'
            n_row = 12
            n_col = 8
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,3,6,9]:
                a = [[plt.subplot(gs[s+0:s+2, i:i+2], projection='3d'), plt.subplot(gs[s+2, i:i+2])] for i in [0,2,4,6]]
                axs_all.append(a)
            plotter.latent_all(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        
        fig_all = [
            #plot_sess_significance(),
            plot_intervals(),
            plot_trial(),
            plot_cluster_all(),
            plot_cluster_heatmap_all(),
            plot_cluster_block_adapt_individual(),
            plot_cluster_epoch_adapt_individual(),
            plot_latent_all(),
            ]
        print('Clearing memory usage')
        del list_labels
        del list_neural_trials
        del list_significance
        del plotter
        gc.collect()
        return fig_all