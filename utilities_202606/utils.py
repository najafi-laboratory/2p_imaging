#!/usr/bin/env python3

import os
import time
import functools
import tracemalloc
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import matplotlib.transforms as mtransforms
import rastermap as rm
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.stats import anderson_ksamp
from scipy.stats import cramervonmises_2samp
from scipy.stats import gaussian_kde
from scipy.stats import ks_2samp
from scipy.stats import levene
from scipy.stats import mannwhitneyu
from scipy.stats import permutation_test
from scipy.stats import rankdata
from scipy.stats import t
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from matplotlib.gridspec import GridSpec

#%% general data processing nyaoo~0w0.

# monitor memory usage nyaoo~0w0.
def show_resource_usage(func):
    # wrap a function and print runtime/memory after it finishes nyaoo~0w0.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # start timer and memory tracer nyaoo~0w0.
        print(f'running {func.__name__}')
        tracemalloc.start()
        start = time.perf_counter()
        # run the wrapped function nyaoo~0w0.
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        # collect and print resource summary nyaoo~0w0.
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f'--- current time: {datetime.now().strftime("%Y.%m.%d %H:%M:%S")}')
        print(f'--- time cost: {elapsed:.2f}s')
        print(f'--- current memory: {current/1024/1024:.2f} MB')
        print(f'--- memory peak: {peak/1024/1024:.2f} MB')
        return result
    return wrapper

# remove nan values from one array nyaoo~0w0.
def drop_nan(data):
    # convert to array and keep finite, non-nan values nyaoo~0w0.
    data = np.asarray(data)
    return data[~np.isnan(data)]

# flatten all leading dimensions and keep the last axis as time nyaoo~0w0.
def flatten_time(data):
    # reshape into samples by time for shared summary functions nyaoo~0w0.
    data = np.asarray(data)
    return data.reshape(-1, data.shape[-1])

# z-score each row independently nyaoo~0w0.
def zscore_rows(data):
    # compute row-wise mean and standard deviation nyaoo~0w0.
    data = np.asarray(data, dtype=float)
    m = np.nanmean(data, axis=1, keepdims=True)
    s = np.nanstd(data, axis=1, keepdims=True)
    # avoid division by zero for flat rows nyaoo~0w0.
    s = np.where((s < 1e-8) | np.isnan(s), 1.0, s)
    return (data - m) / s

# rescale voltage recordings nyaoo~0w0.
def rescale(data, upper, lower):
    # normalize to [0, 1], then scale to [lower, upper] nyaoo~0w0.
    data = norm01(data)
    return data * (upper - lower) + lower

# normalization into [0,1] nyaoo~0w0.
def norm01(data):
    # use nan-safe min/max and guard flat inputs nyaoo~0w0.
    data = np.asarray(data, dtype=float)
    d_min = np.nanmin(data)
    d_max = np.nanmax(data)
    return (data - d_min) / (d_max - d_min + 1e-5)

# compute the scale parameters when normalizing data into [0,1] nyaoo~0w0.
def get_norm01_params(data):
    # compute nan-safe range nyaoo~0w0.
    x_min = np.nanmin(data)
    x_max = np.nanmax(data)
    x_range = x_max - x_min + 1e-5
    # return affine parameters and original bounds nyaoo~0w0.
    x_scale = 1 / x_range
    x_offset = -x_min / x_range
    return x_scale, x_offset, x_min, x_max

# bin data and return index nyaoo~0w0.
def get_bin_idx(data, bin_win, bin_num):
    # build equal-width bin edges and centers nyaoo~0w0.
    bins = np.linspace(bin_win[0], bin_win[1], bin_num + 1)
    bin_center = (bins[:-1] + bins[1:]) / 2
    # digitize values into zero-based bins nyaoo~0w0.
    bin_idx = np.digitize(data, bins) - 1
    return bins, bin_center, bin_idx

# compute frame indices around a center time nyaoo~0w0.
def get_frame_idx_from_time(timestamps, c_time, l_time, r_time):
    # find left and right insertion points in sorted timestamps nyaoo~0w0.
    l_idx = np.searchsorted(timestamps, c_time + l_time)
    r_idx = np.searchsorted(timestamps, c_time + r_time)
    return l_idx, r_idx

# compute alignment frame counts and aligned time vector nyaoo~0w0.
def get_frame_window(time_ms, pre_s, post_s):
    # estimate frame interval from timestamps nyaoo~0w0.
    dt = np.nanmedian(np.diff(time_ms))
    l_frames = int(np.round(pre_s * 1000 / dt))
    r_frames = int(np.round(post_s * 1000 / dt))
    # build relative aligned time stamps nyaoo~0w0.
    aligned_time = np.arange(-l_frames, r_frames) * dt
    return l_frames, r_frames, aligned_time

# align one event-centered window nyaoo~0w0.
def align_event(data, time_ms, event_time, pre_s, post_s):
    # compute output shape from the requested window nyaoo~0w0.
    l_frames, r_frames, aligned_time = get_frame_window(time_ms, pre_s, post_s)
    aligned = np.full((data.shape[0], len(aligned_time)), np.nan)
    # locate the event and source slice nyaoo~0w0.
    c_idx = np.searchsorted(time_ms, event_time)
    l_idx = c_idx - l_frames
    r_idx = c_idx + r_frames
    # copy only complete in-bounds windows nyaoo~0w0.
    if l_idx >= 0 and r_idx <= data.shape[1]:
        aligned[:, :] = data[:, l_idx:r_idx]
    return aligned, aligned_time

# align data around multiple event times nyaoo~0w0.
def align_events(data, time_ms, event_times, pre_s, post_s):
    # align each event with the same relative time axis nyaoo~0w0.
    aligned = [align_event(data, time_ms, et, pre_s, post_s)[0] for et in event_times]
    aligned_time = get_frame_window(time_ms, pre_s, post_s)[2]
    return np.stack(aligned), aligned_time

# compute mean and sem for average dF/F within given time window nyaoo~0w0.
def get_mean_sem_win(neu_seq, neu_time, c_time, l_time, r_time, mode, pct=25):
    # cut the requested time window nyaoo~0w0.
    l_idx, r_idx = get_frame_idx_from_time(neu_time, c_time, l_time, r_time)
    neu_win = neu_seq[:, l_idx:r_idx].copy()
    # summarize each row according to the requested mode nyaoo~0w0.
    if mode == 'mean':
        neu = np.nanmean(neu_win, axis=1)
    elif mode == 'lower':
        q = np.nanpercentile(neu_win, pct, axis=1)
        neu = np.nanmean(np.where(neu_win <= q[:, None], neu_win, np.nan), axis=1)
    elif mode == 'higher':
        q = np.nanpercentile(neu_win, 100 - pct, axis=1)
        neu = np.nanmean(np.where(neu_win >= q[:, None], neu_win, np.nan), axis=1)
    else:
        raise ValueError('mode can only be [mean, lower, higher].')
    # compute across-row mean and standard error nyaoo~0w0.
    neu_mean = np.nanmean(neu)
    count = np.sum(~np.isnan(neu))
    neu_sem = np.nanstd(neu) / np.sqrt(max(count, 1))
    return neu, neu_mean, neu_sem

# compute mean and uncertainty nyaoo~0w0.
def get_mean_sem(data, method_m='mean', method_s='standard error', zero_start=False):
    # flatten observations and keep time as the last axis nyaoo~0w0.
    data = flatten_time(data)
    # compute central tendency nyaoo~0w0.
    if method_m == 'mean':
        m = np.nanmean(data, axis=0)
    elif method_m == 'median':
        m = np.nanmedian(data, axis=0)
    else:
        raise ValueError('method_m can only be [mean, median].')
    m = m - m[0] if zero_start else m
    # compute dispersion and valid sample count nyaoo~0w0.
    std = np.nanstd(data, axis=0)
    count = np.sum(~np.isnan(data), axis=0)
    count_safe = np.maximum(count, 1)
    # convert dispersion into requested uncertainty nyaoo~0w0.
    if method_s == 'confidence interval':
        s = t.ppf(0.975, np.maximum(count - 1, 1)) * std / np.sqrt(count_safe)
    elif method_s == 'prediction interval':
        s = t.ppf(0.975, np.maximum(count - 1, 1)) * std * np.sqrt(1 + 1 / count_safe)
    elif method_s == 'standard error':
        s = std / np.sqrt(count_safe)
    elif method_s == 'standard deviation':
        s = std
    else:
        raise ValueError('method_s can only be [confidence interval, prediction interval, standard error, standard deviation].')
    return m, s

# compute peak time within evaluation window nyaoo~0w0.
def get_peak_time(neu, neu_time, win_peak):
    # cut the peak search window nyaoo~0w0.
    l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, win_peak[0], win_peak[1])
    neu_win = neu[l_idx:r_idx]
    # use scipy peak finding and fall back to the maximum sample nyaoo~0w0.
    peaks = find_peaks(neu_win)[0]
    peak_idx = peaks[0] if len(peaks) else np.nanargmax(neu_win)
    return neu_time[l_idx:r_idx][peak_idx]


#%% retreating neural data nyaoo~0w0.

# average across subsampling trials nyaoo~0w0.
def sub_sampling_trial(neu_seq, samping_size=3, sampling_times=252):
    # convert fraction sample size into a trial count nyaoo~0w0.
    n_samples = int(samping_size * neu_seq.shape[0]) + 1 if samping_size < 1 else samping_size
    n_samples = min(int(n_samples), neu_seq.shape[0])
    # average random trial subsets nyaoo~0w0.
    neu_seq_sub = np.zeros([sampling_times, neu_seq.shape[1], neu_seq.shape[2]])
    for ti in range(sampling_times):
        sub_idx = np.random.choice(neu_seq.shape[0], n_samples, replace=False)
        neu_seq_sub[ti, :, :] = np.nanmean(neu_seq[sub_idx, :, :], axis=0)
    return neu_seq_sub

# get index for the first trial of block transition nyaoo~0w0.
def get_block_1st_idx(trial_lbl, prepost='post'):
    # compare each label with its previous or next state nyaoo~0w0.
    if prepost == 'pre':
        diff = np.diff(trial_lbl, prepend=trial_lbl[0])
    elif prepost == 'post':
        diff = np.diff(trial_lbl, prepend=trial_lbl[:2])[:-1]
    else:
        raise ValueError('prepost can only be [pre, post].')
    # split transition starts by target label nyaoo~0w0.
    idx_0 = (diff == -1) & (trial_lbl == 0)
    idx_1 = (diff == 1) & (trial_lbl == 1)
    return idx_0, idx_1

# find index for epoch nyaoo~0w0.
def get_block_epochs_idx(trial_lbl, epoch_len, block_combine=True):
    total_length = len(trial_lbl)
    # identify start and end indices for each continuous block nyaoo~0w0.
    label_changes = np.diff(trial_lbl)
    change_indices = np.where(label_changes != 0)[0]
    block_starts = np.concatenate(([0], change_indices + 1))
    block_ends = np.concatenate((change_indices, [total_length - 1]))
    # collect epoch masks and counts per label nyaoo~0w0.
    blocks_dict = {0: [], 1: []}
    blocks_num_epochs = {0: [], 1: []}
    for start, end in zip(block_starts, block_ends):
        # compute whole epochs inside this block nyaoo~0w0.
        current_label = trial_lbl[start]
        block_length = end - start + 1
        num_epochs = block_length // epoch_len
        if num_epochs == 0:
            continue
        # build one boolean mask per epoch nyaoo~0w0.
        epoch_masks = np.zeros((num_epochs, total_length), dtype=bool)
        for i in range(num_epochs):
            epoch_start = start + i * epoch_len
            epoch_masks[i, epoch_start:epoch_start + epoch_len] = True
        blocks_dict[current_label].append(epoch_masks)
        blocks_num_epochs[current_label].append(num_epochs)
    # trim each label to the same number of epochs across blocks nyaoo~0w0.
    result = []
    for label in [0, 1]:
        if blocks_dict[label]:
            min_epochs = min(blocks_num_epochs[label])
            result_array = np.stack([block[:min_epochs] for block in blocks_dict[label]])
            result_array[0, :, :] = False
            result.append(result_array)
        else:
            result.append(np.empty((0, 0, total_length), dtype=bool))
    # optionally combine repeated blocks into one mask per epoch nyaoo~0w0.
    if block_combine:
        epoch_0s = np.sum(result[0], axis=0).astype(bool)
        epoch_1s = np.sum(result[1], axis=0).astype(bool)
    else:
        epoch_0s = result[0]
        epoch_1s = result[1]
    return epoch_0s, epoch_1s

# find index around transition nyaoo~0w0.
def get_block_transition_idx(trial_lbl, trials_around):
    n_trials = len(trial_lbl)
    # find transition indices nyaoo~0w0.
    diff = np.diff(trial_lbl, prepend=trial_lbl[0])
    trans_0to1 = np.where(diff == 1)[0]
    trans_1to0 = np.where(diff == -1)[0]
    # keep transitions with complete surrounding windows and skip the first block transition nyaoo~0w0.
    valid_0to1 = trans_0to1[(trans_0to1 - trials_around >= 0) & (trans_0to1 + trials_around < n_trials)][1:]
    valid_1to0 = trans_1to0[(trans_1to0 - trials_around >= 0) & (trans_1to0 + trials_around < n_trials)][1:]
    # build boolean masks around each transition nyaoo~0w0.
    trans_0to1 = np.zeros((len(valid_0to1), n_trials), dtype=bool)
    trans_1to0 = np.zeros((len(valid_1to0), n_trials), dtype=bool)
    for i, ti in enumerate(valid_0to1):
        trans_0to1[i, ti - trials_around:ti + trials_around] = True
    for i, ti in enumerate(valid_1to0):
        trans_1to0[i, ti - trials_around:ti + trials_around] = True
    return trans_0to1, trans_1to0

# get index for pre/post special trial nyaoo~0w0.
def get_change_prepost_idx(trial_lbl, target):
    # mark the first trial before target membership changes nyaoo~0w0.
    idx_pre = np.diff(np.isin(trial_lbl, target), append=0)
    idx_pre[idx_pre == -1] = 0
    idx_pre = idx_pre.astype(bool)
    # mark all target trials nyaoo~0w0.
    idx_post = np.isin(trial_lbl, target)
    return idx_pre, idx_post

# get index to split concatenated labels for categories nyaoo~0w0.
def get_split_idx(list_labels, cate):
    # count selected labels in each array nyaoo~0w0.
    split_idx = [np.sum(np.isin(labels, cate)) for labels in list_labels]
    # convert counts into split points for concatenated data nyaoo~0w0.
    split_idx = np.cumsum(split_idx)[:-1]
    return split_idx


#%% detailed processing nyaoo~0w0.

# test if two distributions are separable nyaoo~0w0.
def auc_test(data1, data2):
    n_perm = 1000
    # remove nan values and build binary labels nyaoo~0w0.
    data1 = drop_nan(data1)
    data2 = drop_nan(data2)
    x = np.concatenate([data1, data2])
    y = np.concatenate([np.zeros(len(data1)), np.ones(len(data2))])
    # use abs distance from 0.5 as two-sided statistic nyaoo~0w0.
    auc = roc_auc_score(y, x)
    stat = np.abs(auc - 0.5)
    # permutation p-value nyaoo~0w0.
    cnt = 0
    for _ in range(n_perm):
        yp = np.random.permutation(y)
        if abs(roc_auc_score(yp, x) - 0.5) >= stat:
            cnt += 1
    # return direction-free AUC and p-value nyaoo~0w0.
    auc = np.max([auc, 1 - auc])
    p = (cnt + 1) / (n_perm + 1)
    return auc, p

# compute modulation index for neuron across trial nyaoo~0w0.
def get_modulation_index_neu_seq(neu, neu_time, c_time, win_eval):
    # flatten trials/neurons into rows nyaoo~0w0.
    neu_flat = neu.reshape(-1, neu.shape[-1])
    # estimate low/high reference values from the baseline window nyaoo~0w0.
    v_ref_l = get_mean_sem_win(neu_flat, neu_time, c_time, win_eval[0][0], win_eval[0][1], mode='lower', pct=10)[1]
    v_ref_h = get_mean_sem_win(neu_flat, neu_time, c_time, win_eval[0][0], win_eval[0][1], mode='higher', pct=10)[1]
    # estimate pre and post responses from evaluation windows nyaoo~0w0.
    v_pre = get_mean_sem_win(neu_flat, neu_time, c_time, win_eval[1][0], win_eval[1][1], mode='higher')[0]
    v_post = get_mean_sem_win(neu_flat, neu_time, c_time, win_eval[2][0], win_eval[2][1], mode='higher')[0]
    # normalize the change by the reference response span nyaoo~0w0.
    m = (v_post - v_pre) / (np.abs(v_ref_h - v_ref_l) + 1e-8)
    return m

# get isi based binning average neural response nyaoo~0w0.
def get_isi_bin_neu(neu_seq, stim_seq, camera_pupil, isi, bin_win, bin_num, mean_sem=True):
    # define bins and assign trials to bins for each session nyaoo~0w0.
    bins, bin_center, list_bin_idx = get_bin_idx_list(isi, bin_win, bin_num)
    # preallocate binned outputs nyaoo~0w0.
    bin_neu_seq_trial = []
    bin_neu_seq = []
    bin_neu_mean = np.full((len(bin_center), neu_seq[0].shape[2]), np.nan)
    bin_neu_sem = np.full((len(bin_center), neu_seq[0].shape[2]), np.nan)
    bin_stim_seq = np.full((len(bin_center), stim_seq[0].shape[1], 2), np.nan)
    bin_camera_pupil = []
    for i in range(len(bin_center)):
        # get binned neural traces nyaoo~0w0.
        neu_trial = [n[bi == i, :, :] for n, bi in zip(neu_seq, list_bin_idx)]
        neu_mean_sess = [np.nanmean(n, axis=0) for n in neu_trial if len(n) > 0]
        neu = np.concatenate(neu_mean_sess, axis=0) if len(neu_mean_sess) else np.empty((0, neu_seq[0].shape[2]))
        # get binned stimulus timing and pupil area nyaoo~0w0.
        s_seq = [s[bi == i, :, :] for s, bi in zip(stim_seq, list_bin_idx)]
        s_seq = np.concatenate([s for s in s_seq if len(s) > 0], axis=0) if any(len(s) > 0 for s in s_seq) else np.empty((0, stim_seq[0].shape[1], 2))
        c_pupil = [p[bi == i, :] for p, bi in zip(camera_pupil, list_bin_idx)]
        c_pupil = np.concatenate([p for p in c_pupil if len(p) > 0], axis=0) if any(len(p) > 0 for p in c_pupil) else np.empty((0, camera_pupil[0].shape[1]))
        # collect binned summaries nyaoo~0w0.
        bin_neu_seq_trial.append(neu_trial)
        bin_neu_seq.append(neu)
        if len(neu) > 0:
            bin_neu_mean[i, :] = get_mean_sem(neu)[0]
            bin_neu_sem[i, :] = get_mean_sem(neu)[1]
        if len(s_seq) > 0:
            bin_stim_seq[i, :, :] = np.nanmean(s_seq, axis=0)
        bin_camera_pupil.append(c_pupil)
    return [bins, bin_center.astype('int32'), bin_neu_seq_trial, bin_neu_seq,
            bin_neu_mean, bin_neu_sem, bin_stim_seq, bin_camera_pupil]

# assign a list of arrays into common bins nyaoo~0w0.
def get_bin_idx_list(data_list, bin_win, bin_num):
    # build shared bins nyaoo~0w0.
    bins, bin_center, _ = get_bin_idx(np.array([]), bin_win, bin_num)
    # digitize each array with the same edges nyaoo~0w0.
    list_bin_idx = [np.digitize(data, bins) - 1 for data in data_list]
    return bins, bin_center, list_bin_idx

# stretch data to target time stamps for temporal scaling for 2d data nyaoo~0w0.
def get_temporal_scaling_data(data, t_org, t_target):
    # map target timestamps onto the original time span nyaoo~0w0.
    t_mapped = (t_target - t_target[0]) / (t_target[-1] - t_target[0]) * (t_org[-1] - t_org[0]) + t_org[0]
    # interpolate each row with scipy instead of hand-rolled indexing nyaoo~0w0.
    f = interp1d(t_org, data, axis=1, bounds_error=False, fill_value='extrapolate', assume_sorted=True)
    data_scaled = f(t_mapped)
    return data_scaled

# compute scaled data for single trial response across sessions nyaoo~0w0.
def get_temporal_scaling_trial_multi_sess(neu_seq, stim_seq, neu_time, target_isi):
    # compute target stimulus timing from all sessions nyaoo~0w0.
    stim_seq_target = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
    c_idx = int(stim_seq_target.shape[0] / 2)
    # compute target pre, stimulus, and post windows nyaoo~0w0.
    target_l_pre, target_r_pre = get_frame_idx_from_time(neu_time, 0, -target_isi, stim_seq_target[c_idx, 0])
    target_l_stim, target_r_stim = get_frame_idx_from_time(neu_time, 0, stim_seq_target[c_idx, 0], stim_seq_target[c_idx, 1])
    target_l_post, target_r_post = get_frame_idx_from_time(neu_time, 0, stim_seq_target[c_idx, 1], stim_seq_target[c_idx, 1] + target_isi)
    # compute trial-wise temporal scaling nyaoo~0w0.
    scale_neu_seq = []
    for si in range(len(neu_seq)):
        scale_ns = np.zeros((neu_seq[si].shape[0], neu_seq[si].shape[1], target_r_post - target_l_pre))
        for ti in range(neu_seq[si].shape[0]):
            # find trial-specific source windows nyaoo~0w0.
            l_pre, r_pre = get_frame_idx_from_time(neu_time, 0, stim_seq[si][ti, c_idx - 1, 1], stim_seq[si][ti, c_idx, 0])
            l_stim, r_stim = get_frame_idx_from_time(neu_time, 0, stim_seq[si][ti, c_idx, 0], stim_seq[si][ti, c_idx, 1])
            l_post, r_post = get_frame_idx_from_time(neu_time, 0, stim_seq[si][ti, c_idx, 1], stim_seq[si][ti, c_idx + 1, 0])
            # scale each segment and concatenate back into one trial nyaoo~0w0.
            scale_ns[ti, :, :] = np.concatenate([
                get_temporal_scaling_data(neu_seq[si][ti, :, l_pre:r_pre], neu_time[l_pre:r_pre], neu_time[target_l_pre:target_r_pre]),
                get_temporal_scaling_data(neu_seq[si][ti, :, l_stim:r_stim], neu_time[l_stim:r_stim], neu_time[target_l_stim:target_r_stim]),
                get_temporal_scaling_data(neu_seq[si][ti, :, l_post:r_post], neu_time[l_post:r_post], neu_time[target_l_post:target_r_post]),
                ], axis=1)
        scale_neu_seq.append(scale_ns)
    return scale_neu_seq

# run statistics test and get significance level nyaoo~0w0.
def get_stat_test(data_1, data_2, method):
    p_thres = np.array([5e-2, 5e-4, 5e-6])
    # remove nan values before unpaired tests nyaoo~0w0.
    x = drop_nan(data_1)
    y = drop_nan(data_2)
    # run the requested scipy test nyaoo~0w0.
    if method == 'ttest_ind':
        _, p = ttest_ind(x, y, equal_var=False)
    elif method == 'mannwhitneyu':
        _, p = mannwhitneyu(x, y)
    elif method == 'levene':
        _, p = levene(x, y)
    elif method == 'wilcoxon':
        _, p = wilcoxon(x, y, alternative='two-sided')
    elif method == 'cramervonmises_2samp':
        p = cramervonmises_2samp(x, y).pvalue
    elif method == 'ks_2samp':
        _, p = ks_2samp(x, y, alternative='two-sided', method='auto')
    elif method == 'anderson_ksamp':
        p = anderson_ksamp([x, y]).pvalue
    elif method in ['permutation', 'permution']:
        d = np.asarray(data_1) - np.asarray(data_2)
        d = drop_nan(d)
        p = permutation_test(
            (d,),
            statistic=lambda q: np.mean(q),
            permutation_type='samples',
            n_resamples=252,
            alternative='two-sided'
            ).pvalue
    else:
        raise ValueError('unknown statistics test method.')
    # convert p-value into significance tier nyaoo~0w0.
    r = np.sum(p < p_thres)
    return p, r

# statistics test for response in evaluation windows nyaoo~0w0.
def get_win_mag_quant_stat_test(neu_seq_1, neu_seq_2, neu_time, c_time, win_eval, method):
    baseline_correction = False
    mode = ['lower', 'mean', 'mean', 'mean']
    # get window data for both conditions nyaoo~0w0.
    neu_1 = [get_mean_sem_win(flatten_time(neu_seq_1), neu_time, c_time, win_eval[i][0], win_eval[i][1], mode=mode[i]) for i in range(4)]
    neu_2 = [get_mean_sem_win(flatten_time(neu_seq_2), neu_time, c_time, win_eval[i][0], win_eval[i][1], mode=mode[i]) for i in range(4)]
    # optionally subtract baseline nyaoo~0w0.
    if baseline_correction:
        neu_1 = [neu_1[i][0] - neu_1[0][1] for i in [1, 2, 3]]
        neu_2 = [neu_2[i][0] - neu_2[0][1] for i in [1, 2, 3]]
    else:
        neu_1 = [neu_1[i][0] for i in [1, 2, 3]]
        neu_2 = [neu_2[i][0] for i in [1, 2, 3]]
    # run statistics test for each evaluation window nyaoo~0w0.
    stats = [get_stat_test(n1, n2, method) for n1, n2 in zip(neu_1, neu_2)]
    p = np.array([s[0] for s in stats])
    r = np.array([s[1] for s in stats])
    return p, r

# compute correlation through time between for all neuron pairs across trials nyaoo~0w0.
def get_pair_corr(neu_seq_pair):
    # rank-transform trials for each neuron/time point nyaoo~0w0.
    x = rankdata(neu_seq_pair[0], axis=0)
    y = rankdata(neu_seq_pair[1], axis=0)
    # center and normalize trial vectors nyaoo~0w0.
    x -= np.mean(x, axis=0, keepdims=True)
    y -= np.mean(y, axis=0, keepdims=True)
    x /= np.linalg.norm(x, axis=0, keepdims=True)
    y /= np.linalg.norm(y, axis=0, keepdims=True)
    # compute all pairwise correlations through time nyaoo~0w0.
    corr = np.einsum('ant,bnt->abt', x.transpose(1, 0, 2), y.transpose(1, 0, 2)).reshape(-1, x.shape[2])
    return corr

# average across subsampling neurons nyaoo~0w0.
def sub_sampling_neuron(neu_seq):
    samping_size = 0.2
    sampling_times = 50
    # compute number of sampled neurons nyaoo~0w0.
    n_samples = int(samping_size * neu_seq.shape[0]) + 1
    n_samples = min(n_samples, neu_seq.shape[0])
    # average random neuron subsets nyaoo~0w0.
    neu_seq_sub = np.zeros([sampling_times, neu_seq.shape[1]])
    for qi in range(sampling_times):
        sub_idx = np.random.choice(neu_seq.shape[0], n_samples, replace=False)
        neu_seq_sub[qi, :] = np.nanmean(neu_seq[sub_idx, :], axis=0)
    return neu_seq_sub

# concatenate population activity across sessions nyaoo~0w0.
def get_population_activity(aligned_list):
    # average trials within each session when needed nyaoo~0w0.
    mats = [np.nanmean(a, axis=0) if np.asarray(a).ndim == 3 else np.asarray(a) for a in aligned_list]
    # trim all sessions to the shortest time axis nyaoo~0w0.
    n_time = min(m.shape[-1] for m in mats)
    mats = [m[:, :n_time] for m in mats]
    return np.concatenate(mats, axis=0)

# compute population mean and uncertainty nyaoo~0w0.
def get_population_mean_sem(aligned_list):
    # combine sessions into one unit-by-time matrix nyaoo~0w0.
    population = get_population_activity(aligned_list)
    # compute trace summary across units nyaoo~0w0.
    mean, sem = get_mean_sem(population)
    return {'mean': mean, 'sem': sem, 'n_units': population.shape[0], 'n_time': population.shape[1]}


#%% plotting nyaoo~0w0.
    
# bin rows by averaging adjacent groups me nyaoo~0w0.
def bin_rows(data, max_rows=258):
    # compute row bin size and keep small matrices unchanged nyaoo~0w0.
    data = np.asarray(data)
    bin_size = max(1, int(np.ceil(data.shape[0] / max_rows)))
    if bin_size == 1:
        return data
    # trim to full bins and average inside each bin nyaoo~0w0.
    n_bins = data.shape[0] // bin_size
    data = data[:n_bins * bin_size]
    return np.nanmean(data.reshape(n_bins, bin_size, data.shape[1]), axis=1)

# get color from label nyaoo~0w0.
def get_roi_label_color(labels=None, cate=None, roi_id=None):
    cate = [] if cate is None else cate
    # choose colors from label or category nyaoo~0w0.
    if (roi_id != None and labels[roi_id] == -1) or (cate == [-1]):
        color0, color1, color2 = 'dimgrey', 'deepskyblue', 'royalblue'
        cmap = mcolors.LinearSegmentedColormap.from_list(None, ['lemonchiffon', 'royalblue', 'black'])
    elif (roi_id != None and labels[roi_id] == 1) or (cate == [1]):
        color0, color1, color2 = 'dimgrey', 'chocolate', 'crimson'
        cmap = mcolors.LinearSegmentedColormap.from_list(None, ['lemonchiffon', 'crimson', 'black'])
    elif (roi_id != None and labels[roi_id] == 2) or (cate == [2]):
        color0, color1, color2 = 'dimgrey', 'mediumseagreen', 'forestgreen'
        cmap = mcolors.LinearSegmentedColormap.from_list(None, ['lemonchiffon', 'forestgreen', 'black'])
    else:
        color0, color1, color2 = 'dimgrey', 'hotpink', 'darkviolet'
        cmap = mcolors.LinearSegmentedColormap.from_list(None, ['lemonchiffon', 'violet', 'black'])
    return color0, color1, color2, cmap

# return colors from dim to dark with a base color nyaoo~0w0.
def get_cmap_color(n_colors, base_color=None, cmap=None, return_cmap=False):
    c_margin = 0.05
    # build a colormap from a color list when requested nyaoo~0w0.
    if base_color != None:
        cmap = mcolors.LinearSegmentedColormap.from_list(None, base_color)
    cmap = plt.cm.viridis if cmap is None else cmap
    # sample evenly from the colormap nyaoo~0w0.
    colors = cmap(np.linspace(c_margin, 1 - c_margin, n_colors))
    colors = [mcolors.to_hex(color, keep_alpha=True) for color in colors]
    return (cmap, colors) if return_cmap else colors

# sort each row in a heatmap nyaoo~0w0.
def sort_heatmap_neuron(neu_seq_sort, sort_method):
    win_conv = 9
    n_clusters = 3
    locality = 1
    # smooth values before sorting nyaoo~0w0.
    kernel = np.ones(win_conv) / win_conv
    smoothed = np.array([np.convolve(row, kernel, mode='same') for row in neu_seq_sort])
    # choose sort order nyaoo~0w0.
    if sort_method == 'rastermap':
        model = rm.Rastermap(n_clusters=n_clusters, locality=locality)
        model.fit(smoothed)
        sorted_idx = model.isort
    elif sort_method == 'peak_timing':
        sorted_idx = np.argmax(smoothed, axis=1).argsort()
    elif sort_method == 'trough_timing':
        sorted_idx = np.argmin(smoothed, axis=1).argsort()
    elif sort_method == 'mean':
        sorted_idx = np.nanmean(smoothed, axis=1).argsort()
    elif sort_method == 'shuffle':
        sorted_idx = np.random.permutation(np.arange(smoothed.shape[0]))
    elif sort_method in ['none', None]:
        sorted_idx = np.arange(smoothed.shape[0])
    else:
        raise ValueError('unknown heatmap sort method.')
    return sorted_idx

# alias for reusable heatmap sorting nyaoo~0w0.
def sort_heatmap_rows(data, sort_method='rastermap'):
    # return sorted data and row order nyaoo~0w0.
    sorted_idx = sort_heatmap_neuron(data, sort_method)
    return data[sorted_idx, :], sorted_idx

# apply colormap nyaoo~0w0.
def apply_colormap(data, norm_mode, data_share=None):
    hm_cmap = plt.cm.hot.copy()
    pct = 1
    data = np.asarray(data, dtype=float).copy()
    # return an empty rgb image for empty data nyaoo~0w0.
    if data.size == 0:
        hm_data = np.zeros((data.shape[0], data.shape[1], 3))
        return hm_data, None, hm_cmap
    # binary heatmap nyaoo~0w0.
    if norm_mode == 'binary':
        hm_norm = mcolors.Normalize(vmin=0, vmax=1)
        cs = get_cmap_color(3, cmap=hm_cmap)
        hm_cmap = mcolors.LinearSegmentedColormap.from_list(None, ['#FCFCFC', cs[1]])
    else:
        # choose normalization source nyaoo~0w0.
        scale_data = data_share if norm_mode == 'share' and data_share is not None else data
        if norm_mode in ['none', None, 'share']:
            vmin = np.nanpercentile(scale_data, pct)
            vmax = np.nanpercentile(scale_data, 100 - pct)
            hm_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            data = norm01(np.clip(data, vmin, vmax))
        elif norm_mode == 'minmax':
            data = np.array([norm01(row) for row in data])
            hm_norm = mcolors.Normalize(vmin=0, vmax=1)
        else:
            raise ValueError('norm_mode can only be [binary, none, minmax, share].')
    # convert normalized data into rgb image nyaoo~0w0.
    hm_cmap.set_bad(color='white')
    hm_data = hm_cmap(data)[..., :3]
    return hm_data, hm_norm, hm_cmap

# hide all axis nyaoo~0w0.
def hide_all_axis(ax):
    # remove ticks and spines nyaoo~0w0.
    ax.tick_params(tick1On=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

# generate random rotation matrix for 3d dynamics nyaoo~0w0.
def get_random_rotate_mat_3d():
    # sample a random unit quaternion nyaoo~0w0.
    u1, u2, u3 = np.random.random(3)
    s1, s2 = np.sqrt(1.0 - u1), np.sqrt(u1)
    th1, th2 = 2.0 * np.pi * u2, 2.0 * np.pi * u3
    x = s1 * np.sin(th1)
    y = s1 * np.cos(th1)
    z = s2 * np.sin(th2)
    w = s2 * np.cos(th2)
    # convert quaternion to rotation matrix nyaoo~0w0.
    rotate_mat = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])
    return rotate_mat

# adjust ticks nyaoo~0w0.
def add_ax_ticks(ax, axis, nbins):
    # format millisecond axes as seconds nyaoo~0w0.
    if axis == 'x':
        ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=nbins))
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(
            lambda x, pos: '' if (pos - abs(ax.get_xticks()).argmin()) % 2 else f'{x/1000:.1f}'))
        ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
        ax.tick_params(axis='x', which='minor', labelbottom=False)
    if axis == 'y':
        ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=nbins))
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(
            lambda y, pos: '' if (pos - abs(ax.get_yticks()).argmin()) % 2 else f'{y/1000:.1f}'))
        ax.yaxis.set_minor_locator(mtick.AutoMinorLocator(2))
        ax.tick_params(axis='y', which='minor', labelbottom=False)

# adjust layout for isi example epoch nyaoo~0w0.
def adjust_layout_isi_example_epoch(ax, trial_win, bin_win):
    # set compact axis and scale marker nyaoo~0w0.
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.hlines(bin_win[0], trial_win[0], trial_win[0] + 100, color='black')
    ax.text(trial_win[0] + 50, bin_win[0] - 50, '100 trials', ha='center', va='top', color='black')
    ax.set_ylabel('interval (s)')
    ax.set_xlim(trial_win)
    ax.set_ylim(bin_win)
    ax.set_xticks([])
    ticks = 500 * np.arange((bin_win[0] + 50) / 500, (bin_win[1] - 50) / 500 + 1).astype('int32')
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)

# adjust layout for grand average neural traces nyaoo~0w0.
def adjust_layout_neu(ax):
    # format a standard neural trace axis nyaoo~0w0.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel(r'$\Delta F/F$ (z-scored)')
    add_ax_ticks(ax, 'x', 8)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=1))

# adjust layout for grand average neural traces for clustering nyaoo~0w0.
def adjust_layout_cluster_neu(ax, n_clusters, xlim):
    # format cluster trace axis without y ticks nyaoo~0w0.
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(xlim)
    ax.set_yticks([])
    add_ax_ticks(ax, 'x', 8)

# adjust layout for scatter comparison nyaoo~0w0.
def adjust_layout_scatter(ax, upper, lower):
    # set square-like bounds around paired scatter values nyaoo~0w0.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([lower - 0.1 * (upper - lower), upper + 0.1 * (upper - lower)])
    ax.set_ylim([lower - 0.1 * (upper - lower), upper + 0.1 * (upper - lower)])
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))

# adjust layout for heatmap nyaoo~0w0.
def adjust_layout_heatmap(ax):
    # hide top/right spines and format time ticks nyaoo~0w0.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    add_ax_ticks(ax, 'x', 8)

# adjust layout for 3d latent dynamics nyaoo~0w0.
def adjust_layout_3d_latent(ax):
    # remove pane fills and ticks for 3d trajectory plots nyaoo~0w0.
    ax.grid(False)
    ax.view_init(elev=15, azim=15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

# adjust layout for pupil traces nyaoo~0w0.
def adjust_layout_pupil(ax):
    # format pupil trace axis nyaoo~0w0.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Pupil area (z-scored)')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))

# add legend into subplots nyaoo~0w0.
def add_legend(ax, colors=None, labels=None, n_trials=None, n_neurons=None, n_sessions=None, loc='best', dim=2):
    # create invisible handles that carry labels nyaoo~0w0.
    plot_args = [[], [], []] if dim == 3 else [[], []]
    handles = []
    if colors != None and labels != None:
        handles += [ax.plot(*plot_args, lw=0, color=colors[i], label=labels[i])[0] for i in range(len(labels))]
    if n_trials != None and n_neurons != None:
        handles += [
            ax.plot(*plot_args, lw=0, color='black', label=r'$n_{trial}=$' + str(n_trials))[0],
            ax.plot(*plot_args, lw=0, color='black', label=r'$n_{neuron}=$' + str(n_neurons))[0],
            ax.plot(*plot_args, lw=0, color='black', label=r'$n_{session}=$' + str(n_sessions))[0]]
    # draw legend only when handles exist nyaoo~0w0.
    if len(handles) > 0:
        ax.legend(loc=loc, handles=handles, labelcolor='linecolor', frameon=False, framealpha=0)

# add heatmap colorbar nyaoo~0w0.
def add_heatmap_colorbar(ax, cmap, norm, label, yticklabels=None):
    # draw a compact colorbar inside the provided axis nyaoo~0w0.
    if ax != None:
        hide_all_axis(ax)
        cax = ax.inset_axes([0, 0, 0.3, 0.9], transform=ax.transAxes)
        norm = mcolors.Normalize(vmin=0, vmax=1) if norm == None else norm
        cbar = ax.figure.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
        cbar.outline.set_linewidth(0.05)
        cbar.ax.set_ylabel(label, rotation=90, labelpad=10)
        cbar.ax.tick_params(axis='y', labelsize=7, labelrotation=90)
        for ticklabel in cbar.ax.get_yticklabels():
            ticklabel.set_ha('left')
            ticklabel.set_va('center')
        cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        cbar.ax.yaxis.set_major_locator(
            mtick.FixedLocator([norm.vmin + 0.2 * (norm.vmax - norm.vmin),
                                norm.vmax - 0.2 * (norm.vmax - norm.vmin)]))
        if yticklabels != None:
            cbar.ax.set_yticklabels(yticklabels)


#%% basic plotting utils class nyaoo~0w0.

class utils_basic:

    # initialize plotting defaults nyaoo~0w0.
    def __init__(self):
        # keep shared plotting constants in one place nyaoo~0w0.
        self.min_num_trial = 5
        self.latent_cmap = plt.cm.nipy_spectral
        self.random_bin_cmap = plt.cm.gist_ncar
        self.stat_sym = ['n.s.', '*', '**', '***']
        self.heatmap_sort_frac = 0.5

    # plot mean trace with uncertainty band nyaoo~0w0.
    def plot_mean_sem(self, ax, t, m, s, c, l=None, a=1.0):
        # draw mean and shaded interval nyaoo~0w0.
        ax.plot(t, m, color=c, label=l, alpha=a)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.25, edgecolor='none')
        ax.set_xlim([np.min(t), np.max(t)])

    # plot kernel density curve nyaoo~0w0.
    def plot_density(self, ax, data, xlim, color, bw_method=0.05):
        # evaluate gaussian KDE across the requested range nyaoo~0w0.
        x = np.linspace(np.min(xlim), np.max(xlim), 100)
        d = gaussian_kde(drop_nan(data), bw_method=bw_method)(x)
        ax.plot(x, d, color=color)

    # plot one side of a violin distribution nyaoo~0w0.
    def plot_half_violin(self, ax, data, x, color, side):
        # draw full violin first nyaoo~0w0.
        p = ax.violinplot(drop_nan(data), positions=[x], widths=1, showextrema=False)
        v = p['bodies'][0].get_paths()[0].vertices
        # style and clip vertices to one side nyaoo~0w0.
        p['bodies'][0].set(facecolor=color, alpha=0.3)
        p['bodies'][0].set_edgecolor(color)
        p['bodies'][0].set_linewidth(1)
        m = v[:, 0].mean()
        v[:, 0] = np.minimum(v[:, 0], m) if side.lower().startswith('l') else np.maximum(v[:, 0], m)

    # plot histogram or cumulative distribution nyaoo~0w0.
    def plot_dist(self, ax, data, c, cumulative, xlim=None):
        bins = 25
        # set range nyaoo~0w0.
        data = drop_nan(data)
        if xlim is None:
            xlim = [np.nanmin(data), np.nanmax(data)]
        # convert histogram counts into fractions nyaoo~0w0.
        counts, bin_edges = np.histogram(data, bins=bins, range=xlim)
        total = max(counts.sum(), 1)
        fractions = counts / total
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # optionally convert to cumulative curve nyaoo~0w0.
        if cumulative:
            fractions = np.cumsum(fractions)
            bin_centers = np.concatenate(([xlim[0]], bin_centers, [xlim[1]]))
            fractions = np.concatenate(([0], fractions, [1]))
        ax.plot(bin_centers, fractions, color=c)
        return fractions

    # plot paired scatter comparison nyaoo~0w0.
    def plot_scatter(self, ax, q1, q2, c):
        # subsample longer vector so lengths match nyaoo~0w0.
        if len(q1) > len(q2):
            q1 = q1[np.random.choice(len(q1), size=len(q2), replace=False)]
        if len(q1) < len(q2):
            q2 = q2[np.random.choice(len(q2), size=len(q1), replace=False)]
        # compute bounds and draw scatter/unit line nyaoo~0w0.
        upper = np.nanmax([q1, q2])
        lower = np.nanmin([q1, q2])
        ax.scatter(q1, q2, color=c, s=1, alpha=0.5)
        ax.plot(np.linspace(lower, upper, 1106), np.linspace(lower, upper, 1106), lw=1, linestyle=':', color='#2C2C2C')
        # draw means and significance labels nyaoo~0w0.
        ax.scatter(np.nanmean(q1), np.nanmean(q2), color='black', marker='x', s=10)
        r_m = get_stat_test(q1, q2, 'ttest_ind')[1]
        r_v = get_stat_test(q1, q2, 'levene')[1]
        ax.text(upper - 0.05 * (upper - lower), upper - 0.4 * (upper - lower), self.stat_sym[r_m], ha='center', va='center')
        ax.text(upper - 0.05 * (upper - lower), upper - 0.6 * (upper - lower), self.stat_sym[r_v], ha='center', va='center')
        adjust_layout_scatter(ax, upper, lower)

    # plot neuron-by-time heatmap nyaoo~0w0.
    def plot_heatmap_neuron(self, ax_hm, ax_cb, neu_seq, neu_time, neu_seq_sort,
                            sort_method='rastermap', norm_mode=None, neu_seq_share=None, scale_bar=True, cbar_label=None):
        max_pixel = 258
        if len(neu_seq) > 0:
            # remove pure nan rows and sort neurons nyaoo~0w0.
            neu_idx = np.where(~np.all(np.isnan(neu_seq), axis=1))[0]
            neu_seq = neu_seq[neu_idx, :].copy()
            neu_seq_sort = neu_seq_sort[neu_idx, :].copy()
            sorted_idx = sort_heatmap_neuron(neu_seq_sort, sort_method=sort_method)
            data = neu_seq[sorted_idx, :].copy()
            n_neurons = data.shape[0]
            # downsample rows and normalize colors nyaoo~0w0.
            data = bin_rows(data, max_pixel)
            data_share = np.concatenate(neu_seq_share, axis=0) if neu_seq_share != None else None
            hm_data, hm_norm, hm_cmap = apply_colormap(data, norm_mode, data_share)
            # draw heatmap nyaoo~0w0.
            ax_hm.imshow(hm_data, extent=[neu_time[0], neu_time[-1], 1, hm_data.shape[0]], interpolation='nearest', aspect='auto')
            adjust_layout_heatmap(ax_hm)
            ax_hm.set_ylabel('')
            ax_hm.set_yticks([])
            # add neuron-count scale bar nyaoo~0w0.
            if scale_bar:
                trans = mtransforms.blended_transform_factory(ax_hm.transAxes, ax_hm.transData)
                bar_count = max(1, int(np.floor((n_neurons / 2) / (10 ** np.floor(np.log10(max(n_neurons / 2, 1))))) * (10 ** np.floor(np.log10(max(n_neurons / 2, 1))))))
                scale = hm_data.shape[0] / n_neurons
                y0 = hm_data.shape[0] - bar_count * scale - 1
                y1 = hm_data.shape[0] - 1
                ax_hm.plot([-0.05, -0.05], [y0, y1], color='black', lw=1, transform=trans, clip_on=False)
                ax_hm.text(-0.05, (y0 + y1) / 2, f'{bar_count}', rotation=90, va='center', ha='right', transform=trans)
            add_heatmap_colorbar(ax_cb, hm_cmap, hm_norm, cbar_label or r'$\Delta F/F$')

    # plot trial-by-time heatmap nyaoo~0w0.
    def plot_heatmap_trial(self, ax_hm, ax_cb, neu_seq, neu_time, norm_mode=None, neu_seq_share=None, cbar_label=None):
        n_yticks = 2
        max_pixel = 258
        if len(neu_seq) > 0:
            # remove pure nan rows and downsample for display nyaoo~0w0.
            neu_idx = np.where(~np.all(np.isnan(neu_seq), axis=1))[0]
            neu_seq = neu_seq[neu_idx, :].copy()
            data = bin_rows(neu_seq, max_pixel)
            # prepare shared scale when provided nyaoo~0w0.
            data_share = bin_rows(np.concatenate(neu_seq_share, axis=0), max_pixel) if neu_seq_share != None else None
            hm_data, hm_norm, hm_cmap = apply_colormap(data, norm_mode, data_share)
            # draw heatmap nyaoo~0w0.
            ax_hm.imshow(hm_data, extent=[neu_time[0], neu_time[-1], 1, hm_data.shape[0]], interpolation='nearest', aspect='auto')
            adjust_layout_heatmap(ax_hm)
            ax_hm.set_yticks((((np.arange(n_yticks) + 0.5) / n_yticks) * data.shape[0]).astype('int32'))
            ax_hm.set_yticklabels((((np.arange(n_yticks) + 0.5) / n_yticks) * neu_seq.shape[0]).astype('int32'))
            add_heatmap_colorbar(ax_cb, hm_cmap, hm_norm, cbar_label or r'$\Delta F/F$')

    # plot evaluation window markers nyaoo~0w0.
    def plot_win_mag_quant_win_eval(self, ax, win_eval, color, xlim, baseline=True):
        # draw baseline and evaluation windows as horizontal markers nyaoo~0w0.
        if baseline:
            ax.plot(win_eval[0], [1, 1], color=color, linestyle=':', marker='|')
        for i in range(len(win_eval) - 1):
            ax.plot(win_eval[i + 1], [1, 1], color=color, marker='|')
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(xlim)
        ax.set_ylim([0, 1.1])
        ax.set_yticks([])

    # plot window magnitudes with uncertainty nyaoo~0w0.
    def plot_win_mag_quant(self, ax, neu_seq, neu_time, win_eval, color, c_time, offset):
        mode = ['lower', 'mean', 'mean', 'mean']
        # compute response within each window nyaoo~0w0.
        quant = [get_mean_sem_win(flatten_time(neu_seq), neu_time, c_time, win_eval[i][0], win_eval[i][1], mode=mode[i]) for i in range(4)]
        m = np.array([quant[i][1] for i in range(4)])
        s = np.array([quant[i][2] for i in range(4)])
        # draw baseline-subtracted error bars nyaoo~0w0.
        for i in [1, 2, 3]:
            ax.errorbar(i + offset, m[i] - m[0], s[i], color=color, capsize=2, marker='o',
                        linestyle='none', markeredgecolor='white', markeredgewidth=0.1)
        # adjust layouts nyaoo~0w0.
        ax.tick_params(tick1On=False)
        ax.tick_params(axis='x', labelrotation=90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('evoked magnitude')
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['early', 'late', 'post'])
        ax.set_xlim([0.5, 3.5])

    # plot decoding accuracy across sliding windows nyaoo~0w0.
    def plot_multi_sess_decoding_slide_win(self, ax, eval_time, acc_model, acc_chance, color1, color2):
        # compute mean and sem for model and chance nyaoo~0w0.
        acc_mean_model = np.array([get_mean_sem(a)[0] for a in acc_model]).reshape(-1)
        acc_sem_model = np.array([get_mean_sem(a)[1] for a in acc_model]).reshape(-1)
        acc_mean_chance = np.array([get_mean_sem(a)[0] for a in acc_chance]).reshape(-1)
        acc_sem_chance = np.array([get_mean_sem(a)[1] for a in acc_chance]).reshape(-1)
        # compute y bounds and draw traces nyaoo~0w0.
        upper = np.nanmax([acc_mean_model, acc_mean_chance]) + np.nanmax([acc_sem_model, acc_sem_chance])
        lower = np.nanmin([acc_mean_model, acc_mean_chance]) - np.nanmax([acc_sem_model, acc_sem_chance])
        self.plot_mean_sem(ax, eval_time, acc_mean_model, acc_sem_model, color2, 'model')
        self.plot_mean_sem(ax, eval_time, acc_mean_chance, acc_sem_chance, color1, 'chance')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim([lower - 0.1 * (upper - lower), upper + 0.1 * (upper - lower)])
        add_legend(ax, [color2, color1], ['model', 'chance'], loc='upper left')

    # plot modulation-index distributions nyaoo~0w0.
    def plot_pred_mod_index_dist(self, ax, mi1, mi2, color0, color1, color2):
        # compute distribution means nyaoo~0w0.
        m1, _ = get_mean_sem(mi1.reshape(-1, 1))
        m2, _ = get_mean_sem(mi2.reshape(-1, 1))
        # plot distributions and mean lines nyaoo~0w0.
        f1 = self.plot_dist(ax, mi1, color1, False)
        f2 = self.plot_dist(ax, mi2, color2, False)
        ax.axvline(m1, color=color1, lw=1, linestyle='--')
        ax.axvline(m2, color=color2, lw=1, linestyle='--')
        # compute display bounds and significance nyaoo~0w0.
        yu = np.nanmax([f1, f2])
        yl = np.nanmin([f1, f2])
        auc, p = auc_test(mi1, mi2)
        r_m = np.sum(p < np.array([5e-2, 5e-4, 5e-6]))
        ax.text(0.8, yu + 0.4 * (yu - yl), f'AUC:{auc:.2f}', ha='center', va='center', color=color0, size=9)
        ax.text(0.8, yu + 0.1 * (yu - yl), self.stat_sym[r_m], ha='center', va='center')
        # adjust layouts nyaoo~0w0.
        ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([-1, 1])
        ax.set_ylim([yl, yu + 0.4 * (yu - yl)])

    # plot 3d latent dynamics nyaoo~0w0.
    def plot_3d_latent_dynamics(self, ax, neu_z, stim_seq, neu_time, end_color='black', c_stim=None, cmap=None, add_stim=True, add_mark=None):
        resampling = 5
        cmap = self.latent_cmap if cmap == None else cmap
        # interpolate and smooth dynamics nyaoo~0w0.
        t_new = np.linspace(neu_time[0], neu_time[-1], resampling * len(neu_time))
        f = interp1d(neu_time, neu_z, axis=1, bounds_error=False, fill_value='extrapolate', assume_sorted=True)
        z = f(t_new)
        z = np.apply_along_axis(savgol_filter, 1, z, window_length=resampling * 2 + 1, polyorder=3)
        # plot colored trajectory segments nyaoo~0w0.
        c_neu = get_cmap_color(z.shape[1], cmap=cmap)
        for ti in range(z.shape[1] - 1):
            ax.plot(z[0, ti:ti + 2] - z[0, 0], z[1, ti:ti + 2] - z[1, 0], z[2, ti:ti + 2] - z[2, 0], color=c_neu[ti], lw=0.5)
        # add start/end markers nyaoo~0w0.
        ax.scatter(0, 0, 0, color='black', marker='x', lw=1)
        ax.scatter(z[0, -1] - z[0, 0], z[1, -1] - z[1, 0], z[2, -1] - z[2, 0], color=end_color, marker='o', lw=1)
        # add stimulus markers nyaoo~0w0.
        if add_stim:
            for j in range(stim_seq.shape[0]):
                idx = get_frame_idx_from_time(t_new, 0, stim_seq[j, 0], 0)[0]
                if idx > 0 and idx < len(t_new):
                    ax.scatter(z[0, idx] - z[0, 0], z[1, idx] - z[1, 0], z[2, idx] - z[2, 0], color=c_stim[j], marker='s', lw=0.25)
        # add critical mark points nyaoo~0w0.
        if add_mark != None:
            for tm, c_mark in add_mark:
                idx = get_frame_idx_from_time(t_new, 0, tm, 0)[0]
                if idx > 0 and idx < len(t_new):
                    ax.scatter(z[0, idx] - z[0, 0], z[1, idx] - z[1, 0], z[2, idx] - z[2, 0], color=c_mark, marker='o', lw=2)

    # plot lower triangle distance matrix nyaoo~0w0.
    def plot_dis_mat(self, ax_hm, ax_cb, d, annotate=False):
        # mask upper half nyaoo~0w0.
        mask = np.tril(np.ones_like(d, dtype=bool), k=0)
        masked_mat = np.where(mask, d, np.nan)
        # plot matrix nyaoo~0w0.
        ax_hm.matshow(masked_mat, interpolation='nearest', cmap=plt.cm.hot)
        # annotate values when requested nyaoo~0w0.
        if annotate:
            for i in range(d.shape[0]):
                for j in range(d.shape[1]):
                    if not np.isnan(masked_mat[i, j]):
                        ax_hm.text(j, i, f'{masked_mat[i, j]:.2f}', ha='center', va='center', color='grey')
        # adjust layout nyaoo~0w0.
        ax_hm.tick_params(tick1On=True, bottom=False, top=False, labelbottom=True, labeltop=False)
        ax_hm.spines[:].set_visible(False)
        ax_hm.set_xticks(np.arange(d.shape[0]))
        ax_hm.set_yticks(np.arange(d.shape[0]))
