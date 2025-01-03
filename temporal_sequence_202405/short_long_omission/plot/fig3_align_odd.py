#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from modules.Alignment import run_get_stim_response
from modeling.clustering import clustering_neu_response_mode
from modeling.clustering import get_sorted_corr_mat
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_cross_corr
from plot.utils import get_mean_sem
from plot.utils import get_frame_idx_from_time
from plot.utils import get_multi_sess_neu_trial_average
from plot.utils import get_roi_label_color
from plot.utils import get_epoch_idx
from plot.utils import get_odd_stim_prepost_idx
from plot.utils import get_expect_time
from plot.utils import adjust_layout_neu
from plot.utils import add_legend
from plot.utils import utils

class plotter_utils(utils):

    def __init__(
            self,
            list_neural_trials, list_labels, list_significance
            ):
        super().__init__()
        timescale = 1.0
        self.n_sess = len(list_neural_trials)
        self.l_frames = int(100*timescale)
        self.r_frames = int(200*timescale)
        self.cut_frames = int(80*timescale)
        self.list_stim_labels = [
            nt['stim_labels'][1:-1,:] for nt in list_neural_trials]
        self.list_labels = list_labels
        self.alignment = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='none')
        self.alignment_local = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='local')
        self.expect = np.array([
            np.mean([get_expect_time(sl)[0] for sl in self.list_stim_labels]),
            np.mean([get_expect_time(sl)[1] for sl in self.list_stim_labels])])
        self.epoch_early = [get_epoch_idx(sl)[0] for sl in self.list_stim_labels]
        self.epoch_late  = [get_epoch_idx(sl)[1] for sl in self.list_stim_labels]
        self.list_odd_idx = [
            get_odd_stim_prepost_idx(sl) for sl in self.list_stim_labels]
        self.list_significance = list_significance

    def plot_pre_odd_isi_distribution(self, ax):
        isi_short = [self.alignment['list_pre_isi'][n][self.list_odd_idx[n][0]] for n in range(self.n_sess)]
        isi_long  = [self.alignment['list_pre_isi'][n][self.list_odd_idx[n][1]] for n in range(self.n_sess)]
        isi_short = np.concatenate(isi_short)
        isi_long  = np.concatenate(isi_long)
        isi_max = np.nanmax(np.concatenate([isi_short, isi_long]))
        ax.hist(isi_short,
            bins=100, range=[0, isi_max], align='left',
            color='#9DB4CE', density=True)
        ax.hist(isi_long,
            bins=100, range=[0, isi_max], align='right',
            color='dodgerblue', density=True)
        ax.set_title('pre oddball isi distribution')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('percentage')
        ax.set_xlim([0, isi_max])
        ax.set_xticks(500*np.arange(0, isi_max/500+1).astype('int32'))
        ax.set_xticklabels(
            500*np.arange(0, isi_max/500+1).astype('int32'),
            rotation='vertical')
        ax.plot([], color='#9DB4CE', label='short')
        ax.plot([], color='dodgerblue', label='long')
        ax.legend(loc='upper left')
        
    def plot_odd_normal(self, ax, normal, fix_jitter, cate=None, roi_id=None):
        if cate != None:
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_oddball'],:]
                for i in range(self.n_sess)]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        neu_mean = []
        neu_sem = []
        # collect data.
        if 0 in normal:
            neu_short, _, stim_seq_short, stim_value_short, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'], self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[[-1], [0], [fix_jitter], None, [0]])
            mean_short, sem_short = get_mean_sem(neu_short)
            neu_mean.append(mean_short)
            neu_sem.append(sem_short)
        if 1 in normal:
            neu_long, _, stim_seq_long, stim_value_long, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'], self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
                trial_idx=[l[1] for l in self.list_odd_idx],
                trial_param=[[-1], [1], [fix_jitter], None, [0]])
            mean_long, sem_long = get_mean_sem(neu_long)
            neu_mean.append(mean_long)
            neu_sem.append(sem_long)
        # find bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot voltages.
        if 0 in normal:
            for i in range(2):
                ax.fill_between(
                    stim_seq_short[i+1,:]-self.expect[0]-stim_seq_short[1,1],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color1, alpha=0.15, step='mid')
            self.plot_vol(
                ax, self.alignment['stim_time']-self.expect[0]-stim_seq_short[1,1],
                stim_value_short.reshape(1,-1), color1, upper, lower)
        if 1 in normal:
            for i in range(2):
                ax.fill_between(
                    stim_seq_long[i+1,:]-self.expect[1]-stim_seq_long[1,1],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color2, alpha=0.15, step='mid')
            self.plot_vol(
                ax, self.alignment['stim_time']-self.expect[1]-stim_seq_long[1,1],
                stim_value_long.reshape(1,-1), color2, upper, lower)
        ax.axvline(0, color='gold', lw=1, linestyle='--')
        # plot neural traces.
        if 0 in normal:
            self.plot_mean_sem(
                ax, self.alignment['neu_time']-self.expect[0]-stim_seq_short[1,1],
                mean_short, sem_short, color1, 'short')
        if 1 in normal:
            self.plot_mean_sem(
                ax, self.alignment['neu_time']-self.expect[1]-stim_seq_long[1,1],
                mean_long, sem_long, color2, 'long')
        if not np.isnan(upper+lower):
            adjust_layout_neu(ax)
            ax.set_xlim([np.nanmin(self.alignment['neu_time']), np.nanmax(self.alignment['neu_time'])])
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.set_xlabel('time since omission (ms)')
            ax.legend(loc='upper left')

    def plot_odd_global(self, ax, normal, cate=None, roi_id=None):
        if cate != None:
            color0, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            color0, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
        colors = [color0, [color1, color2][normal]]
        # collect single trial data.
        neu_seq, stim_seq, stim_value, pre_isi = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'], self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=[l[normal] for l in self.list_odd_idx],
            trial_param=[[-1], [normal], [1], None, [0]],
            mean_sem=False)
        # compute bin indices.
        bin_idx_neg = [isi<self.expect[normal] for isi in pre_isi]
        bin_idx_pos = [isi>self.expect[normal] for isi in pre_isi]
        # collect data.
        neu_mean = []
        neu_sem = []
        stim_mean = []
        for bin_idx in [bin_idx_neg, bin_idx_pos]:
            neu = np.concatenate(
                [np.nanmean(neu_seq[n][bin_idx[n],:,:], axis=0)
                 for n in range(self.n_sess)], axis=0)
            m, s = get_mean_sem(neu)
            neu_mean.append(m)
            neu_sem.append(s)
            stim_mean.append(np.concatenate([stim_value[n][bin_idx[n],:] for n in range(self.n_sess)]))
        # compute bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot voltages.
        stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
        for i in range(2):
            ax.fill_between(
                stim_seq[i+1,:]-self.expect[normal]-stim_seq[1,1],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=colors[i], alpha=0.15, step='mid')
        ax.axvline(0, color='gold', lw=1, linestyle='--')
        for i in range(2):
            self.plot_vol(
                ax, self.alignment['stim_time']-self.expect[normal]-stim_seq[1,1],
                stim_mean[i], colors[i], upper, lower)
        # plot neural traces.
        self.plot_mean_sem(
            ax, self.alignment['neu_time']-self.expect[normal]-stim_seq[1,1],
            neu_mean[0], neu_sem[0], colors[0],
            '< {} ms'.format(int(self.expect[normal])))
        self.plot_mean_sem(
            ax, self.alignment['neu_time']-self.expect[normal]-stim_seq[1,1],
            neu_mean[1], neu_sem[1], colors[1],
            '> {} ms'.format(int(self.expect[normal])))
        if not np.isnan(upper+lower):
            adjust_layout_neu(ax)
            ax.set_xlabel('time since omission (ms)')
            ax.set_xlim([np.nanmin(self.alignment['neu_time']), np.nanmax(self.alignment['neu_time'])])
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.legend(loc='upper right')
    
    def plot_odd_overlap(self, ax, cate=None, roi_id=None):
        if cate != None:
            _, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
        neu_mean = []
        neu_sem = []
        stim_mean = []
        # get short trials.
        neu_seq_short, stim_seq_short, stim_value_short, pre_isi_short = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'], self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=[l[0] for l in self.list_odd_idx],
            trial_param=[[-1], [0], [1], None, [0]],
            mean_sem=False)
        bin_idx_short = [isi>self.expect[0] for isi in pre_isi_short]
        neu = np.concatenate(
            [np.nanmean(neu_seq_short[n][bin_idx_short[n],:,:], axis=0)
             for n in range(self.n_sess)], axis=0)
        m, s = get_mean_sem(neu)
        neu_mean.append(m)
        neu_sem.append(s)
        stim_mean.append(np.concatenate([stim_value_short[n][bin_idx_short[n],:] for n in range(self.n_sess)]))
        stim_seq_short = np.nanmean(np.concatenate(stim_seq_short, axis=0), axis=0)
        # get long trials.
        neu_seq_long, stim_seq_long, stim_value_long, pre_isi_long = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'], self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=[l[1] for l in self.list_odd_idx],
            trial_param=[[-1], [1], [1], None, [0]],
            mean_sem=False)
        bin_idx_long = [isi<self.expect[1] for isi in pre_isi_long]
        neu = np.concatenate(
            [np.nanmean(neu_seq_long[n][bin_idx_long[n],:,:], axis=0)
             for n in range(self.n_sess)], axis=0)
        m, s = get_mean_sem(neu)
        neu_mean.append(m)
        neu_sem.append(s)
        stim_mean.append(np.concatenate([stim_value_long[n][bin_idx_long[n],:] for n in range(self.n_sess)]))
        stim_seq_long = np.nanmean(np.concatenate(stim_seq_long, axis=0), axis=0)
        # compute bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot voltages.
        for i in range(2):
            ax.fill_between(
                stim_seq_short[i+1,:]-self.expect[0]-stim_seq_short[1,1],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=color1, alpha=0.15, step='mid')
            ax.fill_between(
                stim_seq_long[i+1,:]-self.expect[1]-stim_seq_long[1,1],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=color2, alpha=0.15, step='mid')
        ax.axvline(0, color='gold', lw=1, linestyle='--')
        self.plot_vol(
            ax, self.alignment['stim_time']-self.expect[0]-stim_seq_short[1,1],
            stim_mean[0], color1, upper, lower)
        self.plot_vol(
            ax, self.alignment['stim_time']-self.expect[1]-stim_seq_long[1,1],
            stim_mean[1], color2, upper, lower)
        # plot neural traces.
        self.plot_mean_sem(
            ax, self.alignment['neu_time']-self.expect[0]-stim_seq_short[1,1],
            neu_mean[0], neu_sem[0], color1, 'short')
        self.plot_mean_sem(
            ax, self.alignment['neu_time']-self.expect[1]-stim_seq_long[1,1],
            neu_mean[1], neu_sem[1], color2, 'long')
        if not np.isnan(upper+lower):
            adjust_layout_neu(ax)
            ax.set_xlabel('time since omission (ms)')
            ax.set_xlim([np.nanmin(self.alignment['neu_time']), np.nanmax(self.alignment['neu_time'])])
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.legend(loc='upper right')

    def plot_odd_local(self, ax, normal, cate=None, roi_id=None):
        idx_cut = np.searchsorted(self.alignment_local['neu_time'], 0)+self.cut_frames
        if cate != None:
            color0, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment_local['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            color0, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [np.expand_dims(self.alignment_local['list_neu_seq'][0][:,roi_id,:], axis=1)]
        colors = [color0, [color1, color2][normal]]
        # collect single trial data.
        neu_seq, stim_seq, stim_value, pre_isi = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment_local['list_stim_seq'],
            self.alignment_local['list_stim_value'], self.alignment_local['list_pre_isi'],
            trial_idx=[l[normal] for l in self.list_odd_idx],
            trial_param=[[-1], [normal], [1], None, [0]],
            mean_sem=False)
        # compute bin indices.
        bin_idx_neg = [isi<self.expect[normal] for isi in pre_isi]
        bin_idx_pos = [isi>self.expect[normal] for isi in pre_isi]
        # collect data.
        neu_mean = []
        neu_sem = []
        stim_mean = []
        for bin_idx in [bin_idx_neg, bin_idx_pos]:
            neu = np.concatenate(
                [np.nanmean(neu_seq[n][bin_idx[n],:,:], axis=0)
                 for n in range(self.n_sess)], axis=0)
            m, s = get_mean_sem(neu[:,:idx_cut])
            neu_mean.append(m)
            neu_sem.append(s)
            stim_mean.append(np.concatenate([stim_value[n][bin_idx[n],:] for n in range(self.n_sess)]))
        # compute bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot voltages.
        self.plot_vol(ax, self.alignment_local['stim_time'], stim_mean[0], colors[0], upper, lower)
        self.plot_vol(ax, self.alignment_local['stim_time'], stim_mean[1], colors[1], upper, lower)
        ax.axvline(0, color='gold', lw=1, linestyle='--')
        # plot neural traces.
        self.plot_mean_sem(
            ax, self.alignment_local['neu_time'][:idx_cut],
            neu_mean[0], neu_sem[0], colors[0],
            '< {} ms'.format(int(self.expect[normal])))
        self.plot_mean_sem(
            ax, self.alignment_local['neu_time'][:idx_cut],
            neu_mean[1], neu_sem[1], colors[1],
            '> {} ms'.format(int(self.expect[normal])))
        if not np.isnan(upper+lower):
            adjust_layout_neu(ax)
            ax.set_xlabel('time since expected stim (ms)')
            ax.set_xlim([self.alignment_local['neu_time'][0], self.alignment_local['neu_time'][idx_cut]])
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.legend(loc='upper right')
    
    def plot_odd_local_prediction(self, ax, normal, cate=None, roi_id=None):
        idx_cut = np.searchsorted(self.alignment_local['neu_time'], 0)+self.cut_frames
        if cate != None:
            _, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment_local['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            _, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [np.expand_dims(self.alignment_local['list_neu_seq'][0][:,roi_id,:], axis=1)]
        # collect single trial data.
        neu_seq, stim_seq, stim_value, pre_isi = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment_local['list_stim_seq'],
            self.alignment_local['list_stim_value'], self.alignment_local['list_pre_isi'],
            trial_idx=[l[normal] for l in self.list_odd_idx],
            trial_param=[[-1], [normal], [1], None, [0]],
            mean_sem=False)
        
    def plot_odd_epoch(self, ax, normal, fix_jitter, cate=None, roi_id=None):
        if cate != None:
            color0, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            color0, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
        colors = [color0, [color1, color2][normal]]
        # collect data.
        neu_early, _, stim_seq_early, stim_value_early, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'], self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=self.epoch_early,
            trial_param=[[-1], [normal], [fix_jitter], None, [0]])
        neu_late, _, stim_seq_late, stim_value_late, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'], self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=self.epoch_late,
            trial_param=[[-1], [normal], [fix_jitter], None, [0]])
        mean_early, sem_early = get_mean_sem(neu_early)
        mean_late,  sem_late  = get_mean_sem(neu_late)
        # compute bounds.
        upper = np.nanmax([mean_early, mean_late]) + np.nanmax([sem_early, sem_late])
        lower = np.nanmin([mean_early, mean_late]) - np.nanmax([sem_early, sem_late])
        # plot voltages.
        stim_seq = (stim_seq_early + stim_seq_late) / 2
        stim_value = (stim_value_early + stim_value_late) / 2
        for i in range(2):
            ax.fill_between(
                stim_seq[i+1,:]-self.expect[normal]-stim_seq[1,1],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=colors[1], alpha=0.15, step='mid')
        self.plot_vol(
            ax, self.alignment['stim_time']-self.expect[normal]-stim_seq[1,1],
            stim_value.reshape(1,-1), colors[1], upper, lower)
        ax.axvline(0, color='gold', lw=1, linestyle='--')
        # plot neural traces.
        self.plot_mean_sem(
            ax, self.alignment['neu_time']-self.expect[normal]-stim_seq[1,1],
            mean_early, sem_early, colors[0], 'ep1')
        self.plot_mean_sem(
            ax, self.alignment['neu_time']-self.expect[normal]-stim_seq[1,1],
            mean_late, sem_late, colors[1],'ep2')
        if not np.isnan(upper+lower):
            adjust_layout_neu(ax)
            ax.set_xlim([np.nanmin(self.alignment['neu_time']), np.nanmax(self.alignment['neu_time'])])
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.set_xlabel('time since omission (ms)')

    def plot_odd_cluster(self, axs, fix_jitter, cate):
        n_clusters = 8
        max_clusters = 16
        win_eval = [-2000,2000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        color0, color1, color2, cmap = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[2,3,4,5], [0], [fix_jitter], None, [0]])
        neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
        # fit model.
        metrics, cluster_id = clustering_neu_response_mode(
             neu, n_clusters, max_clusters, l_idx, r_idx)
        # plot sorted correlation matrix.
        sorted_neu_corr = get_sorted_corr_mat(neu, cluster_id, l_idx, r_idx)
        axs[0].imshow(sorted_neu_corr, cmap=cmap)
        axs[0].set_xlabel('neuron id')
        axs[0].set_ylabel('neuron id')
        axs[0].spines['left'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['bottom'].set_visible(False)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        # plot clustering metrics.
        colors=['#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
        metric_lbl = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'inertia']
        for i in range(4):
            axs[1].plot(metrics['n_clusters'], metrics[metric_lbl[i]], color=colors[i], label=metric_lbl[i])
        axs[1].tick_params(axis='y', tick1On=False)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].set_xlabel('n_clusters')
        axs[1].set_ylabel('normalized value')
        axs[1].set_xlim(2, max_clusters+2)
        add_legend(axs[1], colors, metric_lbl, 'upper right')
        # plot cross cluster correlations.
        cluster_corr = get_cross_corr(neu, n_clusters, cluster_id, l_idx, r_idx)
        mask = np.tril(np.ones_like(cluster_corr, dtype=bool), k=0)
        masked_corr = np.where(mask, cluster_corr, np.nan)
        axs[2].matshow(masked_corr, interpolation='nearest', cmap=cmap)
        axs[2].tick_params(bottom=False, top=False, labelbottom=True, labeltop=False)
        axs[2].tick_params(tick1On=False)
        axs[2].spines['left'].set_visible(False)
        axs[2].spines['right'].set_visible(False)
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['bottom'].set_visible(False)
        axs[2].set_xticks(np.arange(n_clusters))
        axs[2].set_yticks(np.arange(n_clusters))
        axs[2].set_xlabel('cluster id')
        axs[2].set_ylabel('cluster id')
        for i in range(cluster_corr.shape[0]):
            for j in range(cluster_corr.shape[1]):
                if not np.isnan(masked_corr[i, j]):
                    axs[2].text(j, i, f'{masked_corr[i, j]:.2f}',
                                ha='center', va='center', color='grey')
        # plot fraction of each cluster.
        colors = plt.cm.nipy_spectral(np.arange(n_clusters)/n_clusters)
        num = [np.sum(cluster_id==i) for i in np.unique(cluster_id)]
        axs[3].pie(
            num,
            labels=[str(num[i])+' cluster # '+str(i) for i in range(n_clusters)],
            colors=colors,
            autopct='%1.1f%%',
            wedgeprops={'linewidth': 1, 'edgecolor':'white', 'width':0.2})
        # plot within cluster average for normal (short).
        neu_mean, neu_sem = get_mean_sem_cluster(neu, n_clusters, cluster_id, l_idx, r_idx)
        self.plot_cluster_mean_sem(axs[4], neu_mean, neu_sem, stim_seq, color0, colors)
        # plot within cluster average for normal (long).
        neu, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[2,3,4,5], [1], [fix_jitter], None, [0]])
        neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
        neu_mean, neu_sem = get_mean_sem_cluster(neu, n_clusters, cluster_id, l_idx, r_idx)
        self.plot_cluster_mean_sem(axs[5], neu_mean, neu_sem, stim_seq, color0, colors)
        # plot within cluster average for oddball (short).
        neu, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[0] for l in self.list_odd_idx],
            trial_param=[[-1], [0], [fix_jitter], None, [0]])
        neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
        neu_mean, neu_sem = get_mean_sem_cluster(neu, n_clusters, cluster_id, l_idx, r_idx)
        self.plot_cluster_mean_sem(axs[6], neu_mean, neu_sem, stim_seq, color0, colors)
        # plot within cluster average for oddball (long).
        neu, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[1] for l in self.list_odd_idx],
            trial_param=[[-1], [1], [fix_jitter], None, [0]])
        neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
        neu_mean, neu_sem = get_mean_sem_cluster(neu, n_clusters, cluster_id, l_idx, r_idx)
        self.plot_cluster_mean_sem(axs[7], neu_mean, neu_sem, stim_seq, color0, colors)
        
class plotter_VIPTD_G8_align_odd(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)

    def odd_normal_exc(self, axs):
        
        self.plot_odd_normal(axs[0], [0], 0, cate=-1)
        axs[0].set_title('response to oddball \n excitatory (short, fix)')
        
        self.plot_odd_normal(axs[1], [1], 0, cate=-1)
        axs[1].set_title('response to oddball \n excitatory (long, fix)')
        
        self.plot_odd_normal(axs[2], [0,1], 0, cate=-1)
        axs[2].set_title('response to oddball \n excitatory (fix)')
        
        self.plot_odd_normal(axs[3], [0], 1, cate=-1)
        axs[3].set_title('response to oddball \n excitatory (short, jitter)')
        
        self.plot_odd_normal(axs[4], [1], 1, cate=-1)
        axs[4].set_title('response to oddball \n excitatory (long, jitter)')
        
        self.plot_odd_normal(axs[5], [0,1], 1, cate=-1)
        axs[5].set_title('response to oddball \n excitatory (jitter)')

    def odd_normal_inh(self, axs):
        
        self.plot_odd_normal(axs[0], [0], 0, cate=1)
        axs[0].set_title('response to oddball \n inhibitory (short, fix)')
        
        self.plot_odd_normal(axs[1], [1], 0, cate=1)
        axs[1].set_title('response to oddball \n inhibitory (long, fix)')
        
        self.plot_odd_normal(axs[2], [0,1], 0, cate=1)
        axs[2].set_title('response to oddball \n inhibitory (fix)')
        
        self.plot_odd_normal(axs[3], [0], 1, cate=1)
        axs[3].set_title('response to oddball \n inhibitory (short, jitter)')
        
        self.plot_odd_normal(axs[4], [1], 1, cate=1)
        axs[4].set_title('response to oddball \n inhibitory (long, jitter)')
        
        self.plot_odd_normal(axs[5], [0,1], 1, cate=1)
        axs[5].set_title('response to oddball \n inhibitory (jitter)')
    
    def odd_normal_heatmap(self, axs):
        win_sort = [-500, 1500]
        labels = np.concatenate(self.list_labels)
        sig = np.concatenate([self.list_significance[n]['r_oddball'] for n in range(self.n_sess)])
        neu_short_fix, _, stim_seq_short_fix, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=[l[0] for l in self.list_odd_idx],
            trial_param=[[-1], [0], [0], None, [0]])
        neu_long_fix, _, stim_seq_long_fix, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=[l[1] for l in self.list_odd_idx],
            trial_param=[[-1], [1], [0], None, [0]])
        neu_short_jitter, _, stim_seq_short_jitter, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=[l[0] for l in self.list_odd_idx],
            trial_param=[[-1], [0], [1], None, [0]])
        neu_long_jitter, _, stim_seq_long_jitter, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=[l[1] for l in self.list_odd_idx],
            trial_param=[[-1], [1], [1], None, [0]])
        for i in range(8):
            axs[i].set_xlabel('time since omisison (ms)')
        
        if np.sum(~np.isnan(neu_short_fix)) > 0:
            self.plot_heatmap_neuron(
                axs[0], neu_short_fix, self.alignment['neu_time']-self.expect[0]-stim_seq_short_fix[1,1], neu_short_fix,
                win_sort, labels, sig)
            axs[0].set_title('response to oddball \n (short sorted by short, fix)')
        
        if np.sum(~np.isnan(neu_long_fix)) > 0:
            self.plot_heatmap_neuron(
                axs[1], neu_long_fix, self.alignment['neu_time']-self.expect[1]-stim_seq_long_fix[1,1], neu_short_fix,
                win_sort, labels, sig)
            axs[1].set_title('response to oddball \n (long sorted by short, fix)')
        
        if np.sum(~np.isnan(neu_short_fix)) > 0:
            self.plot_heatmap_neuron(
                axs[2], neu_short_fix, self.alignment['neu_time']-self.expect[0]-stim_seq_short_fix[1,1], neu_long_fix,
                win_sort, labels, sig)
            axs[2].set_title('response to oddball \n (short sorted by long, fix)')
        
        if np.sum(~np.isnan(neu_long_fix)) > 0:
            self.plot_heatmap_neuron(
                axs[3], neu_long_fix, self.alignment['neu_time']-self.expect[1]-stim_seq_long_fix[1,1], neu_long_fix,
                win_sort, labels, sig)
            axs[3].set_title('response to oddball \n (long sorted by long, fix)')
        
        if np.sum(~np.isnan(neu_short_jitter)) > 0:
            self.plot_heatmap_neuron(
                axs[4], neu_short_jitter, self.alignment['neu_time']-self.expect[0]-stim_seq_short_jitter[1,1], neu_short_jitter,
                win_sort, labels, sig)
            axs[4].set_title('response to oddball \n (short sorted by short, jitter)')
        
        if np.sum(~np.isnan(neu_long_jitter)) > 0:
            self.plot_heatmap_neuron(
                axs[5], neu_long_jitter, self.alignment['neu_time']-self.expect[1]-stim_seq_long_jitter[1,1], neu_short_jitter,
                win_sort, labels, sig)
            axs[5].set_title('response to oddball \n (long sorted by short, jitter)')
        
        if np.sum(~np.isnan(neu_short_jitter)) > 0:
            self.plot_heatmap_neuron(
                axs[6], neu_short_jitter, self.alignment['neu_time']-self.expect[0]-stim_seq_short_jitter[1,1], neu_long_jitter,
                win_sort, labels, sig)
            axs[6].set_title('response to oddball \n (short sorted by long, jitter)')
        
        if np.sum(~np.isnan(neu_long_jitter)) > 0:
            self.plot_heatmap_neuron(
                axs[7], neu_long_jitter, self.alignment['neu_time']-self.expect[1]-stim_seq_long_jitter[1,1], neu_long_jitter,
                win_sort, labels, sig)
            axs[7].set_title('response to oddball \n (long sorted by long, jitter)')

    def odd_cluster_fix_exc(self, axs):
        
        self.plot_odd_cluster(axs, 0, cate=-1)
        axs[0].set_title('sorted correlation matrix \n fix (normal) excitatory')
        axs[1].set_title('clustering evaluation \n fix (normal) excitatory')
        axs[2].set_title('cross cluster correlation \n fix (normal) excitatory')
        axs[3].set_title('cluster fraction \n fix (normal) excitatory')
        axs[4].set_title('clustering response \n fix (short normal) excitatory')
        axs[5].set_title('clustering response \n fix (long normal) excitatory')
        axs[6].set_title('clustering response \n fix (short oddball) excitatory')
        axs[7].set_title('clustering response \n fix (long oddball) excitatory')
    
    def odd_cluster_jitter_exc(self, axs):
        
        self.plot_odd_cluster(axs, 1, cate=-1)
        axs[0].set_title('sorted correlation matrix \n jitter (normal) excitatory')
        axs[1].set_title('clustering evaluation \n jitter (normal) excitatory')
        axs[2].set_title('cross cluster correlation \n jitter (normal) excitatory')
        axs[3].set_title('cluster fraction \n jitter (normal) excitatory')
        axs[4].set_title('clustering response \n jitter (short normal) excitatory')
        axs[5].set_title('clustering response \n jitter (long normal) excitatory')
        axs[6].set_title('clustering response \n jitter (short oddball) excitatory')
        axs[7].set_title('clustering response \n jitter (long oddball) excitatory')
    
    def odd_cluster_fix_inh(self, axs):
        
        self.plot_odd_cluster(axs, 0, cate=1)
        axs[0].set_title('sorted correlation matrix \n fix (normal) inhibitory')
        axs[1].set_title('clustering evaluation \n fix (normal) inhibitory')
        axs[2].set_title('cross cluster correlation \n fix (normal) inhibitory')
        axs[3].set_title('cluster fraction \n fix (normal) inhibitory')
        axs[4].set_title('clustering response \n fix (short normal) inhibitory')
        axs[5].set_title('clustering response \n fix (long normal) inhibitory')
        axs[6].set_title('clustering response \n fix (short oddball) inhibitory')
        axs[7].set_title('clustering response \n fix (long oddball) inhibitory')
    
    def odd_cluster_jitter_inh(self, axs):
        
        self.plot_odd_cluster(axs, 1, cate=1)
        axs[0].set_title('sorted correlation matrix \n jitter (normal) inhibitory')
        axs[1].set_title('clustering evaluation \n jitter (normal) inhibitory')
        axs[2].set_title('cross cluster correlation \n jitter (normal) inhibitory')
        axs[3].set_title('cluster fraction \n jitter (normal) inhibitory')
        axs[4].set_title('clustering response \n jitter (short normal) inhibitory')
        axs[5].set_title('clustering response \n jitter (long normal) inhibitory')
        axs[6].set_title('clustering response \n jitter (short oddball) inhibitory')
        axs[7].set_title('clustering response \n jitter (long oddball) inhibitory')
        
    def odd_isi_exc(self, axs):
        
        self.plot_odd_global(axs[0], 0, cate=-1)
        axs[0].set_title('response to oddball with preceeding ISI \n excitatory (global, short)')
        
        self.plot_odd_global(axs[1], 1, cate=-1)
        axs[1].set_title('response to oddball with preceeding ISI \n excitatory (global, long)')
        
        self.plot_odd_overlap(axs[2], cate=-1)
        axs[2].set_title('response to oddball with preceeding ISI \n excitatory (global, {}-{} ms overlap)'.format(
            int(self.expect[0]), int(self.expect[1])))
        
        self.plot_odd_local(axs[3], 0, cate=-1)
        axs[3].set_title('response to oddball with preceeding ISI \n excitatory (local, short)')
        
        self.plot_odd_local(axs[4], 1, cate=-1)
        axs[4].set_title('response to oddball with preceeding ISI \n excitatory (local, long)')

    def odd_isi_inh(self, axs):
        
        self.plot_odd_global(axs[0], 0, cate=1)
        axs[0].set_title('response to oddball with preceeding ISI \n inhibitory (global, short)')
        
        self.plot_odd_global(axs[1], 1, cate=1)
        axs[1].set_title('response to oddball with preceeding ISI \n inhibitory (global, long)')
        
        self.plot_odd_overlap(axs[2], cate=1)
        axs[2].set_title('response to oddball with preceeding ISI \n inhibitory (global, {}-{} ms overlap)'.format(
            int(self.expect[0]), int(self.expect[1])))
        
        self.plot_odd_local(axs[3], 0, cate=1)
        axs[3].set_title('response to oddball with preceeding ISI \n inhibitory (local, short)')
        
        self.plot_odd_local(axs[4], 1, cate=1)
        axs[4].set_title('response to oddball with preceeding ISI \n inhibitory (local, long)')

    def odd_epoch_exc(self, axs):
        
        self.plot_odd_epoch(axs[0], 0, 0, cate=-1)
        axs[0].set_title('response to oddball \n excitatory (short, fix)')
        
        self.plot_odd_epoch(axs[1], 1, 0, cate=-1)
        axs[1].set_title('response to oddball \n excitatory (long, fix)')
        
        self.plot_odd_epoch(axs[2], 0, 1, cate=-1)
        axs[2].set_title('response to oddball \n excitatory (short, jitter)')
        
        self.plot_odd_epoch(axs[3], 1, 1, cate=-1)
        axs[3].set_title('response to oddball \n excitatory (long, jitter)')

    def odd_epoch_inh(self, axs):
        
        self.plot_odd_epoch(axs[0], 0, 0, cate=1)
        axs[0].set_title('response to oddball \n inhibitory (short, fix)')
        
        self.plot_odd_epoch(axs[1], 1, 0, cate=1)
        axs[1].set_title('response to oddball \n inhibitory (long, fix)')
        
        self.plot_odd_epoch(axs[2], 0, 1, cate=1)
        axs[2].set_title('response to oddball \n inhibitory (short, jitter)')
        
        self.plot_odd_epoch(axs[3], 1, 1, cate=1)
        axs[3].set_title('response to oddball \n inhibitory (long, jitter)')







