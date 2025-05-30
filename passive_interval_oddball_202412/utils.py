#!/usr/bin/env python3

import functools
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

from modeling.clustering import get_mean_sem_cluster
from modeling.generative import run_glm_multi_sess

#%% general data processing

# monitor memory usage.
def show_memory_usage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        print(f"{func.__name__} - Current: {current/1024/1024:.6f} MB, Peak: {peak/1024/1024:.6f} MB")
        tracemalloc.stop()
        return result
    return wrapper

# rescale voltage recordings.
def rescale(data, upper, lower):
    data = data.copy()
    data = ( data - np.nanmin(data) ) / (np.nanmax(data) - np.nanmin(data))
    data = data * (upper - lower) + lower
    return data

# normalization into [0,1].
def norm01(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-5)

# compute the scale parameters when normalizing data into [0,1].
def get_norm01_params(data):
    x_scale = 1 / (np.nanmax(data) - np.nanmin(data))
    x_offset = - np.nanmin(data) / (np.nanmax(data) - np.nanmin(data))
    x_min = np.nanmin(data)
    x_max = np.nanmax(data)
    return x_scale, x_offset, x_min, x_max

# bin data and return index.
def get_bin_idx(data, bin_win, bin_num):
    bin_size = (bin_win[1] - bin_win[0]) / bin_num
    bins = np.arange(bin_win[0], bin_win[1] + bin_size, bin_size)
    bin_center = bins[:-1] + bin_size / 2
    bin_idx = np.digitize(data, bins) - 1
    return bins, bin_center, bin_idx

# compute mean and sem for average df/f within given time window.
def get_mean_sem_win(neu_seq, neu_time, c_time, l_time, r_time, mode='mean'):
    pct = 25
    l_idx, r_idx = get_frame_idx_from_time(
        neu_time, c_time, l_time, r_time)
    neu_win = neu_seq[:, l_idx:r_idx].copy()
    neu_win_mean = np.nanmean(neu_win, axis=1)
    # mean values in the window.
    if mode == 'mean':
        neu = neu_win_mean
    # values lower than percentile in the window.
    if mode == 'lower':
        neu = neu_win.reshape(-1)
        neu = neu[neu<np.nanpercentile(neu, pct)]
    # values higher than percentile in the window.
    if mode == 'higher':
        neu = neu_win.reshape(-1)
        neu = neu[neu>np.nanpercentile(neu, pct)]
    # compute mean and sem.
    neu_mean = np.nanmean(neu)
    std = np.nanstd(neu)
    count = np.nansum(~np.isnan(neu_win_mean), axis=0)
    neu_sem = std / np.sqrt(count)
    return neu_mean, neu_sem

# compute mean and sem for 3d array data.
def get_mean_sem(data, zero_start=False):
    m = np.nanmean(data.reshape(-1, data.shape[-1]), axis=0)
    m = m-m[0] if zero_start else m
    std = np.nanstd(data.reshape(-1, data.shape[-1]), axis=0)
    count = np.nansum(~np.isnan(data.reshape(-1, data.shape[-1])), axis=0)
    s = std / np.sqrt(count)
    return m, s

#%% retreating neural data

# find trials based on stim_labels.
def pick_trial(
        stim_labels,
        img_seq_label,
        standard_types,
        fix_jitter_types,
        oddball_types,
        random_types,
        opto_types):
    idx1 = np.isin(stim_labels[:,2], img_seq_label) if img_seq_label    else np.ones_like(stim_labels[:,2])
    idx2 = np.isin(stim_labels[:,3], standard_types)   if standard_types   else np.ones_like(stim_labels[:,3])
    idx3 = np.isin(stim_labels[:,4], fix_jitter_types) if fix_jitter_types else np.ones_like(stim_labels[:,4])
    idx4 = np.isin(stim_labels[:,5], oddball_types) if oddball_types    else np.ones_like(stim_labels[:,5])
    idx5 = np.isin(stim_labels[:,6], random_types)   if random_types     else np.ones_like(stim_labels[:,6])
    idx6 = np.isin(stim_labels[:,7], opto_types)       if opto_types       else np.ones_like(stim_labels[:,7])
    idx = (idx1*idx2*idx3*idx4*idx5*idx6).astype('bool')
    return idx

# for multi session settings find trials based on stim_labels.
def get_multi_sess_neu_trial(
        list_stim_labels,
        neu_cate,
        alignment,
        trial_idx=None,
        trial_param=None,
        mean_sem=True,
        ):
    neu = []
    stim_seq = []
    camera_pupil = []
    pre_isi = []
    post_isi = []
    # use stim_labels to find trials.
    if trial_param != None and trial_idx == None:
        for i in range(len(neu_cate)):
            idx = pick_trial(
                list_stim_labels[i],
                trial_param[0],
                trial_param[1],
                trial_param[2],
                trial_param[3],
                trial_param[4],
                trial_param[5])
            neu.append(neu_cate[i][idx,:,:])
            stim_seq.append(alignment['list_stim_seq'][i][idx,:,:])
            camera_pupil.append(alignment['list_camera_pupil'][i][idx,:])
            pre_isi.append(alignment['list_pre_isi'][i][idx])
            post_isi.append(alignment['list_post_isi'][i][idx])
    # use given idx to find trials.
    if trial_param == None and trial_idx != None:
        for i in range(len(neu_cate)):
            neu.append(neu_cate[i][trial_idx[i],:,:])
            stim_seq.append(alignment['list_stim_seq'][i][trial_idx[i],:,:])
            camera_pupil.append(alignment['list_camera_pupil'][i][trial_idx[i],:])
            pre_isi.append(alignment['list_pre_isi'][i][trial_idx[i]])
            post_isi.append(alignment['list_post_isi'][i][trial_idx[i]])
    # use both.
    if trial_param != None and trial_idx != None:
        for i in range(len(neu_cate)):
            idx = pick_trial(
                list_stim_labels[i],
                trial_param[0],
                trial_param[1],
                trial_param[2],
                trial_param[3],
                trial_param[4],
                trial_param[5])
            neu.append(neu_cate[i][trial_idx[i]*idx,:,:])
            stim_seq.append(alignment['list_stim_seq'][i][trial_idx[i]*idx,:,:])
            camera_pupil.append(alignment['list_camera_pupil'][i][trial_idx[i]*idx,:])
            pre_isi.append(alignment['list_pre_isi'][i][trial_idx[i]*idx])
            post_isi.append(alignment['list_post_isi'][i][trial_idx[i]*idx])
    # get numbers.
    n_trials = np.nansum([n.shape[0] for n in neu])
    n_neurons = np.nansum([n.shape[1] for n in neu])
    # compute trial average and concatenate.
    if mean_sem:
        mean = [np.nanmean(n, axis=0) for n in neu]
        std  = [np.nanstd(n, axis=0) for n in neu]
        mean = np.concatenate(mean, axis=0)
        std  = np.concatenate(std, axis=0)
        stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0),axis=0)
        camera_pupil = np.nanmean(np.concatenate(camera_pupil, axis=0),axis=0)
        return [mean, std, stim_seq, camera_pupil], [n_trials, n_neurons]
    # return single trial response.
    else:
        return [neu, stim_seq, camera_pupil, pre_isi, post_isi], [n_trials, n_neurons]

# find neuron category and trial data.
def get_neu_trial(
        alignment, list_labels, list_significance, list_stim_labels,
        trial_idx=None, trial_param=None, mean_sem=True,
        cate=None, roi_id=None,
        ):
    if cate != None:
        colors = get_roi_label_color(cate=cate)
        neu_labels = np.concatenate([
            list_labels[i][np.in1d(list_labels[i],cate)*list_significance[i]['r_standard']]
            for i in range(len(list_stim_labels))])
        neu_sig = np.concatenate([
            list_significance[i]['r_standard'][np.in1d(list_labels[i],cate)]
            for i in range(len(list_stim_labels))])
        neu_cate = [
            alignment['list_neu_seq'][i][:,np.in1d(list_labels[i],cate)*list_significance[i]['r_standard'],:]
            for i in range(len(list_stim_labels))]
    if roi_id != None:
        colors = get_roi_label_color(list_labels, roi_id=roi_id)
        neu_labels = None
        neu_cate = [np.expand_dims(alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
    neu_trial, neu_num = get_multi_sess_neu_trial(
        list_stim_labels, neu_cate, alignment,
        trial_idx=trial_idx, trial_param=trial_param, mean_sem=mean_sem)
    return colors, neu_trial, [neu_labels, neu_sig], neu_num

# compute indice with givn time window for df/f.
def get_frame_idx_from_time(timestamps, c_time, l_time, r_time):
    l_idx = np.searchsorted(timestamps, c_time+l_time)
    r_idx = np.searchsorted(timestamps, c_time+r_time)
    return l_idx, r_idx

# get subsequence index with given start and end.
def get_sub_time_idx(time, start, end):
    idx = np.where((time >= start) &(time <= end))[0]
    return idx

# find expected standard fix interval.
def get_expect_interval(stim_labels):
    idx_short = (stim_labels[:,2]>1)*(stim_labels[:,3]==0)
    expect_short = stim_labels[1:,0] - stim_labels[:-1,1]
    expect_short = np.median(expect_short[idx_short[:-1]])
    idx_long = (stim_labels[:,2]>1)*(stim_labels[:,3]==1)
    expect_long = stim_labels[1:,0] - stim_labels[:-1,1]
    expect_long = np.median(expect_long[idx_long[:-1]])
    return expect_short, expect_long

# get index for the first trial of block transition.
def get_block_1st_idx(stim_labels, target, prepost='post'):
    if prepost == 'pre':
        diff = np.diff(stim_labels[:,target], prepend=stim_labels[:,target][0])
    if prepost == 'post':
        diff = np.diff(stim_labels[:,target], prepend=stim_labels[:,target][:2])
        diff = diff[:-1]
    idx_0 = (diff==-1) * (stim_labels[:,target]==0)
    idx_1 = (diff==1) * (stim_labels[:,target]==1)
    return idx_0, idx_1

# find index for epoch.
def get_block_epochs_idx(stim_labels, epoch_len, block_combine=True):
    total_length = len(stim_labels)
    # identify start and end indices for each continuous block
    label_changes = np.diff(stim_labels)
    change_indices = np.where(label_changes != 0)[0]
    block_starts = np.concatenate(([0], change_indices + 1))
    block_ends = np.concatenate((change_indices, [total_length - 1]))
    # dictionaries to collect epoch masks and count epochs per block for each label.
    blocks_dict = {0: [], 1: []}
    blocks_num_epochs = {0: [], 1: []}
    # process each block.
    for start, end in zip(block_starts, block_ends):
        current_label = stim_labels[start]
        block_length = end - start + 1
        num_epochs = block_length // epoch_len
        # build epoch masks for the current block.
        epoch_masks = []
        for i in range(num_epochs):
            mask = np.zeros(total_length, dtype=bool)
            epoch_start = start + i * epoch_len
            epoch_end = epoch_start + epoch_len
            mask[epoch_start:epoch_end] = True
            epoch_masks.append(mask)
        # stack masks.
        block_epochs = np.stack(epoch_masks)
        blocks_dict[current_label].append(block_epochs)
        blocks_num_epochs[current_label].append(num_epochs)
    # trim each block to the minimum number of epochs and stack.
    result = []
    for label in [0, 1]:
        if blocks_dict[label]:
            min_epochs = min(blocks_num_epochs[label])
            trimmed_blocks = [block[:min_epochs] for block in blocks_dict[label]]
            result_array = np.stack(trimmed_blocks)
            # exclude the first block.
            result_array[0,:,:] = False
            result.append(result_array)
        else:
            result.append(np.empty((0, 0, total_length), dtype=bool))
    if block_combine:
        epoch_0s = np.sum(result[0], axis=0).astype('bool')
        epoch_1s = np.sum(result[1], axis=0).astype('bool')
    else:
        epoch_0s = result[0]
        epoch_1s = result[1]
    return epoch_0s, epoch_1s

# find index around transition.
def get_block_transition_idx(stim_labels, trials_around):
    n_trials = len(stim_labels)
    # compute the difference to find transitions.
    diff = np.diff(stim_labels, prepend=stim_labels[0])
    trans_0to1 = np.where(diff == 1)[0]
    trans_1to0 = np.where(diff == -1)[0]
    # only keep transitions that allow a full window.
    valid_0to1 = trans_0to1[(trans_0to1-trials_around>=0) & (trans_0to1+trials_around<n_trials)][1:]
    valid_1to0 = trans_1to0[(trans_1to0-trials_around>=0) & (trans_1to0+trials_around<n_trials)][1:]
    # preallocate boolean masks.
    trans_0to1 = np.zeros((len(valid_0to1), n_trials), dtype=bool)
    trans_1to0 = np.zeros((len(valid_1to0), n_trials), dtype=bool)
    # mark the indices corresponding to the window:
    for i, t in enumerate(valid_0to1):
        trans_0to1[i, t-trials_around:t+trials_around] = True
    for i, t in enumerate(valid_1to0):
        trans_1to0[i, t-trials_around:t+trials_around] = True
    return trans_0to1, trans_1to0

# get index for pre/post short/long oddball.
def get_odd_stim_prepost_idx(stim_labels):
    idx_pre_short = (stim_labels[:,2]==-1) * (stim_labels[:,5]==0) * (stim_labels[:,6]==0)
    idx_pre_long  = (stim_labels[:,2]==-1) * (stim_labels[:,5]==1) * (stim_labels[:,6]==0)
    idx_post_short = np.zeros_like(idx_pre_short)
    idx_post_short[1:] = idx_pre_short[:-1]
    idx_post_long = np.zeros_like(idx_pre_long)
    idx_post_long[1:] = idx_pre_long[:-1]
    idx_pre_short[-1] = False
    idx_pre_long[-1] = False
    return idx_pre_short, idx_pre_long, idx_post_short, idx_post_long

# get index for pre/post image change.
def get_change_prepost_idx(stim_labels):
    idx_pre = np.diff(stim_labels[:,2]<-1, append=0)
    idx_pre[idx_pre==-1] = 0
    idx_pre = idx_pre.astype('bool')
    idx_post = stim_labels[:,2]<-1
    return idx_pre, idx_post

# mark stim around image change and oddball as outlier.
def exclude_odd_stim(stim_labels):
    n = 2
    neg = np.where(stim_labels[:,2] < 0)[0]
    stim_labels_mark = stim_labels.copy()
    for i in neg:
        stim_labels_mark[
            np.max([0, i-n]):np.min([stim_labels.shape[0], i+n+1]),2] = -np.abs(
                stim_labels_mark[np.max([0, i-n]):np.min([stim_labels.shape[0], i+n+1]),2])
    return stim_labels_mark

#%% detailed processing

# get isi based binning average neural response.
def get_isi_bin_neu(
        neu_seq, stim_seq, isi,
        bin_win, bin_num,
        mean_sem=True
        ):
    # define bins.
    bin_size = (bin_win[1] - bin_win[0]) / bin_num
    bins = np.arange(bin_win[0], bin_win[1] + bin_size, bin_size)
    bin_center = bins[:-1] + bin_size / 2
    list_bin_idx = [np.digitize(i, bins) - 1 for i in isi]
    # compute binned results.
    bin_neu_seq_trial = []
    bin_neu_seq = []
    bin_neu_mean = np.zeros((len(bin_center), neu_seq[0].shape[2]))
    bin_neu_sem = np.zeros((len(bin_center), neu_seq[0].shape[2]))
    bin_stim_seq = np.zeros((len(bin_center), stim_seq[0].shape[1], 2))
    for i in range(len(bin_center)):
        # get binned neural traces.
        neu_trial = [n[bi==i,:,:] for n, bi in zip(neu_seq, list_bin_idx)]
        neu = [np.nanmean(n, axis=0) for n in neu_trial]
        neu = np.concatenate(neu, axis=0)
        # get binned stimulus timing.
        s_seq = [s[bi==i,:,:] for s, bi in zip(stim_seq, list_bin_idx)]
        s_seq = np.concatenate(s_seq, axis=0)
        s_seq = np.nanmean(s_seq, axis=0)
        # collect results.
        bin_neu_seq_trial.append(neu_trial)
        bin_neu_seq.append(neu)
        bin_neu_mean[i,:] = get_mean_sem(neu)[0]
        bin_neu_sem[i,:] = get_mean_sem(neu)[1]
        bin_stim_seq[i,:,:] = s_seq
    bin_center = bin_center.astype('int32')
    return [bins, bin_center, bin_neu_seq_trial, bin_neu_seq,
            bin_neu_mean, bin_neu_sem, bin_stim_seq]

# compute synchrnization across time.
def get_neu_sync(neu, win_width):
    sync = []
    for t in range(win_width, neu.shape[1]):
        window_data = neu[:, t-win_width:t]
        # normalization.
        norms = np.linalg.norm(window_data, axis=1, keepdims=True)
        normalized_data = window_data / (norms + 1e-10)
        # pairwise absolute cosine distances.
        cosine_distances = pdist(normalized_data, metric='cosine')
        abs_cosine_distances = np.abs(cosine_distances)
        # convert to overall similarity.
        cos_sim = 1 - abs_cosine_distances
        s = (np.nansum(cos_sim) - neu.shape[0]) / (neu.shape[0]**2 - neu.shape[0])
        sync.append(s)
    return sync

# find expected isi.
def get_expect_time(stim_labels):
    idx_short = (stim_labels[:,2]>1)*(stim_labels[:,3]==0)
    expect_short = stim_labels[1:,0] - stim_labels[:-1,1]
    expect_short = np.median(expect_short[idx_short[:-1]])
    idx_long = (stim_labels[:,2]>1)*(stim_labels[:,3]==1)
    expect_long = stim_labels[1:,0] - stim_labels[:-1,1]
    expect_long = np.median(expect_long[idx_long[:-1]])
    return expect_short, expect_long

# compute calcium transient event timing and average.
@show_memory_usage
def get_ca_transient(dff, time_img):
    pct = 95
    win_peak = 25
    timescale = 1.0
    l_frames = int(20*timescale)
    r_frames = int(150*timescale)
    # calculate the area under this window as a threshold.
    thres = np.percentile(dff, pct) * win_peak
    # find the window larger than baseline.
    def detect_spikes_win(dff_traces):
        # use convolution to calculate the sum of sliding windows.
        sliding_window_sum = np.convolve(dff_traces, np.ones(win_peak), mode='same')
        # compare against the threshold.
        ca_tran_win = (sliding_window_sum > thres).astype(int)
        return ca_tran_win
    # remove outliers.
    def win_thres(ca_tran_win):
        k = 0.5
        # reset start and end.
        if ca_tran_win[0] == 1:
            ca_tran_win[:np.where(ca_tran_win==0)[0][0]] = 0
        if ca_tran_win[-1] == 1:
            ca_tran_win[np.where(np.diff(ca_tran_win, append=0)==1)[0][-1]:] = 0
        ca_tran_diff = np.diff(ca_tran_win, append=0)
        # compute window width.
        win_wid = np.where(ca_tran_diff==-1)[0] - np.where(ca_tran_diff==1)[0]
        # find windows with outlier width.
        lower = np.nanmean(win_wid)-k*np.nanstd(win_wid)
        outlier_idx = (win_wid < lower)
        # reset outlier.
        for i in range(len(win_wid)):
            if outlier_idx[i]:
                ca_tran_win[np.where(ca_tran_diff==1)[0][i]:np.where(ca_tran_diff==-1)[0][i]+1] = 0
        return ca_tran_win
    # only keep the start of a transient.
    def get_tran_time(ca_tran_win):
        ca_tran_idx = (np.diff(ca_tran_win, append=0)==1).astype('int32')
        return ca_tran_idx
    # extract dff traces around calcium transient.
    def get_ca_mean(ca_tran):
        dff_ca_neu = []
        dff_ca_time = []
        for i in range(dff.shape[0]):
            ca_event_time = np.where(ca_tran[i,:]==1)[0]
            # find dff.
            cn = [dff[i,t-l_frames:t+r_frames].reshape(1,-1)
                  for t in ca_event_time
                  if t > l_frames and t < len(time_img)-r_frames]
            cn = np.concatenate(cn, axis=0) if len(cn) > 0 else np.array([np.nan])
            # find time.
            ct = [time_img[t-l_frames:t+r_frames].reshape(1,-1)-time_img[t]
                  for t in ca_event_time
                  if t > l_frames and t < len(time_img)-r_frames]
            ct = np.concatenate(ct, axis=0) if len(ct) > 0 else np.array([np.nan])
            ct = np.nanmean(ct, axis=0) if not np.isnan(np.mean(ct)) else np.array([np.nan])
            # collect.
            dff_ca_neu.append(cn)
            dff_ca_time.append(ct)
        dff_ca_time = np.concatenate([t.reshape(1,-1) for t in dff_ca_time], axis=0)
        dff_ca_time = np.nanmean(dff_ca_time, axis=0)
        return dff_ca_neu, dff_ca_time
    # compute transient events.
    ca_tran = np.zeros_like(dff)
    for i in range(dff.shape[0]):
        ca_tran_win = detect_spikes_win(dff[i,:])
        ca_tran_win = win_thres(ca_tran_win)
        ca_tran[i,:] = get_tran_time(ca_tran_win)
    # compute average.
    dff_ca_neu, dff_ca_time = get_ca_mean(ca_tran)
    # get the total number of events for each neurons.
    n_ca = np.nansum(ca_tran, axis=1)
    return n_ca, dff_ca_neu, dff_ca_time

# compute calcium transient event timing and average with multiple sessions.
@show_memory_usage
def get_ca_transient_multi_sess(list_neural_trials):
    list_dff = [nt['dff'] for nt in list_neural_trials] 
    list_time = [nt['time'] for nt in list_neural_trials]
    list_n_ca = []
    list_dff_ca_neu = []
    list_dff_ca_time = []
    for i in range(len(list_dff)):
        n_ca, dff_ca_neu, dff_ca_time = get_ca_transient(
            list_dff[i], list_time[i])
        list_n_ca.append(n_ca)
        list_dff_ca_neu += dff_ca_neu
        list_dff_ca_time.append(dff_ca_time.reshape(1,-1))
    list_n_ca = np.concatenate(list_n_ca)
    list_dff_ca_time = np.nanmean((np.concatenate(list_dff_ca_time, axis=0)), axis=0)
    return list_n_ca, list_dff_ca_neu, list_dff_ca_time

# strech data to target time stamps for temporal scaling for 2d data.
def get_temporal_scaling_data(data, t_org, t_target):
    # map to target time stamps.
    t_mapped = (t_target - t_target[0]) / (t_target[-1] - t_target[0]) * (t_org[-1] - t_org[0]) + t_org[0]
    # find the target inserted index.
    idx_right = np.searchsorted(t_org, t_mapped)
    idx_right = np.clip(idx_right, 1, len(t_org)-1)
    idx_left = idx_right - 1
    # compute the fractional distance between the two original time stamps.
    t_left = t_org[idx_left]
    t_right = t_org[idx_right]
    weights = (t_mapped - t_left) / (t_right - t_left)
    # lienar interpolation for each row.
    left_vals = data[:, idx_left]
    right_vals = data[:, idx_right]
    data_scaled = left_vals + weights * (right_vals - left_vals)
    return data_scaled

# compute scaled data for single trial response across sessions.
def get_temporal_scaling_trial_multi_sess(neu_seq, stim_seq, neu_time, target_isi):
    # compute mean time stamps.
    stim_seq_target = np.nanmean(np.concatenate(stim_seq, axis=0),axis=0)
    c_idx = int(stim_seq_target.shape[0]/2)
    # target isi before.
    target_l_pre, target_r_pre = get_frame_idx_from_time(
        neu_time, 0,
        -target_isi,
        stim_seq_target[c_idx,0])
    # target stimulus.
    target_l_stim, target_r_stim = get_frame_idx_from_time(
        neu_time, 0,
        stim_seq_target[c_idx,0],
        stim_seq_target[c_idx,1])
    # target isi after.
    target_l_post, target_r_post = get_frame_idx_from_time(
        neu_time, 0,
        stim_seq_target[c_idx,1],
        stim_seq_target[c_idx,1]+target_isi)
    # compute trial wise temporal scaling.
    scale_neu_seq = []
    for si in range(len(neu_seq)):
        scale_ns = np.zeros((neu_seq[si].shape[0], neu_seq[si].shape[1], target_r_post-target_l_pre))
        for ti in range(neu_seq[si].shape[0]):
            # trial isi before.
            l_pre, r_pre = get_frame_idx_from_time(
                neu_time, 0,
                stim_seq[si][ti,c_idx-1,1],
                stim_seq[si][ti,c_idx,0])
            # trial stimulus.
            l_stim, r_stim = get_frame_idx_from_time(
                neu_time, 0,
                stim_seq[si][ti,c_idx,0],
                stim_seq[si][ti,c_idx,1])
            # trial isi after.
            l_post, r_post = get_frame_idx_from_time(
                neu_time, 0,
                stim_seq[si][ti,c_idx,1],
                stim_seq[si][ti,c_idx+1,0])
            # compute scaled data.
            scale_ns[ti,:,:] = np.concatenate([
                get_temporal_scaling_data(
                    neu_seq[si][ti,:,l_pre:r_pre],
                    neu_time[l_pre:r_pre],
                    neu_time[target_l_pre:target_r_pre]),
                get_temporal_scaling_data(
                    neu_seq[si][ti,:,l_stim:r_stim],
                    neu_time[l_stim:r_stim],
                    neu_time[target_l_stim:target_r_stim]),
                get_temporal_scaling_data(
                    neu_seq[si][ti,:,l_post:r_post],
                    neu_time[l_post:r_post],
                    neu_time[target_l_post:target_r_post]),
                ], axis=1)
        # collect single session results.
        scale_neu_seq.append(scale_ns)
    return scale_neu_seq

# run wilcoxon signed rank test to compare response across neurons and time.
@show_memory_usage
def run_wilcoxon_trial(neu_seq_trial_1, neu_seq_trial_2, thres):
    p_all = []
    for si in range(len(neu_seq_trial_1)):
        # initialize a matrix for p values.
        p_values = np.zeros((neu_seq_trial_1[si].shape[1], neu_seq_trial_1[si].shape[2]))
        # loop over neurons and time points.
        for n in range(neu_seq_trial_1[si].shape[1]):
            for t in range(neu_seq_trial_1[si].shape[2]):
                # run test.
                _, p = mannwhitneyu(neu_seq_trial_1[si][:, n, t], neu_seq_trial_2[si][:, n, t])
                p_values[n,t] = p
        p_all.append(p_values)
    # combine results across sessions.
    p_all = np.concatenate(p_all, axis=0)
    p_bin = p_all.copy()
    p_bin[p_bin<thres] = 0
    p_bin[p_bin>thres] = 1
    p_bin = 1 - p_bin
    return p_all, p_bin

#%% plotting

# get color from label.
def get_roi_label_color(labels=None, cate=None, roi_id=None):
    if (roi_id != None and labels[roi_id] == -1) or (cate == [-1]):
        color0 = 'dimgrey'
        color1 = 'deepskyblue'
        color2 = 'royalblue'
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'yicong_will_love_you_forever',
            ['lemonchiffon', 'royalblue', 'black'])
    if (roi_id != None and labels[roi_id] == 1) or (cate == [1]):
        color0 = 'dimgrey'
        color1 = 'chocolate'
        color2 = 'crimson'
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'yicong_will_love_you_forever',
            ['lemonchiffon', 'crimson', 'black'])
    if (roi_id != None and labels[roi_id] == 2) or (cate == [2]):
        color0 = 'dimgrey'
        color1 = 'mediumseagreen'
        color2 = 'forestgreen'
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'yicong_will_love_you_forever',
            ['lemonchiffon', 'forestgreen', 'black'])
    if len(cate) > 1:
        color0 = 'dimgrey'
        color1 = 'hotpink'
        color2 = 'darkviolet'
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'yicong_will_love_you_forever',
            ['lemonchiffon', 'violet', 'black'])
    return color0, color1, color2, cmap

# return colors from dim to dark with a base color.
def get_cmap_color(n_colors, base_color=None, cmap=None, return_cmap=False):
    c_margin = 0.2
    if base_color != None:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            None, ['lemonchiffon', base_color, 'black'])
    if cmap != None:
        pass
    colors = cmap(np.linspace(c_margin, 1 - c_margin, n_colors))
    colors = [mcolors.to_hex(color, keep_alpha=True) for color in colors]
    if return_cmap:
        return cmap, colors
    else:
        return colors

def sort_heatmap_neuron(neu_seq_sort_s, neu_time, win_sort):
    win_conv = 9
    factor_mag = 0
    # get data within range.
    l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, win_sort[0], win_sort[1])
    # smooth values.
    smoothed_mean = np.array(
        [np.convolve(row, np.ones(win_conv)/win_conv, mode='same')
         for row in neu_seq_sort_s[:,l_idx:r_idx]])
    # get peak timing.
    peak_time = np.argmax(smoothed_mean, axis=1).reshape(-1)
    # get peak magnitude.
    peak_mag = norm01(np.max(smoothed_mean, axis=1).reshape(-1))
    # scale influence of magnitude with factor.
    peak_mag = peak_mag * factor_mag + (1 - factor_mag)
    # combine and sort.
    sorted_idx = (peak_time*peak_mag).argsort()
    return sorted_idx

# apply colormap.
def apply_colormap(data, hm_cmap, norm_mode, data_share):
    factor = 1
    # no valid data found.
    if len(data) == 0:
        hm_data = np.zeros((0, data.shape[1], 3))
        hm_norm = None
        return hm_data, hm_norm, hm_cmap
    else:
        # remap data range.
        m = np.nanmean(data)
        data = (data - m) * factor + m
        # binary heatmap.
        if norm_mode == 'binary':
            hm_norm = mcolors.Normalize(vmin=0, vmax=1)
            cs = get_cmap_color(3, cmap=hm_cmap)
            hm_cmap = mcolors.LinearSegmentedColormap.from_list(None, ['#FCFCFC', cs[1]])
        else:
            # no normalization.
            if norm_mode == 'none':
                hm_norm = mcolors.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
            # normalized into [0,1].
            elif norm_mode == 'minmax':
                for i in range(data.shape[0]):
                    data[i,:] = norm01(data[i,:])
                hm_norm = mcolors.Normalize(vmin=0, vmax=1)
            # share the global scale.
            elif norm_mode == 'share':
                hm_norm = mcolors.Normalize(vmin=np.nanmin(data_share), vmax=np.nanmax(data_share))
            # handle errors.
            else:
                raise ValueError('norm_mode can only be [binary, none, minmax, share].')
        hm_cmap.set_bad(color='white')
        hm_data = hm_cmap(data)
        hm_data = hm_data[..., :3]
    return hm_data, hm_norm, hm_cmap

# adjust layout for grand average neural traces.
def adjust_layout_neu(ax):
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('df/f (z-scored)')

# adjust layout for grand average neural traces for clustering.
def adjust_layout_cluster_neu(ax, n_clusters, xlim):
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xlim(xlim)
    ax.set_xticks([])
    ax.set_yticks([])
    
# adjust layout for heatmap.
def adjust_layout_heatmap(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# adjust layout for 3d latent dynamics.
def adjust_layout_3d_latent(ax, neu_z, cmap, neu_time, cbar_label):
    scale = 0.9
    ax.grid(False)
    ax.view_init(elev=15, azim=15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim([np.min(neu_z[0,:])*scale, np.max(neu_z[0,:])*scale])
    ax.set_ylim([np.min(neu_z[1,:])*scale, np.max(neu_z[1,:])*scale])
    ax.set_zlim([np.min(neu_z[2,:])*scale, np.max(neu_z[2,:])*scale])
    ax.set_xlabel('latent 1')
    ax.set_ylabel('latent 2')
    ax.set_zlabel('latent 3')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    cbar = ax.figure.colorbar(
        plt.cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0.0,0.2,0.5,0.8,1.0],
        shrink=0.5, aspect=25)
    cbar.outline.set_visible(False)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va='bottom')
    cbar.ax.set_yticklabels(
        ['\u2716',
         str(int(neu_time[int(len(neu_time)*0.2)])),
         str(int(neu_time[int(len(neu_time)*0.5)])),
         str(int(neu_time[int(len(neu_time)*0.8)])),
         '\u25CF'], rotation=-90)

# add legend into subplots.
def add_legend(ax, colors, labels, n_trials, n_neurons, n_sessions, loc, dim=2):
    if dim == 2:
        plot_args = [[],[]]
    if dim == 3:
        plot_args = [[],[],[]]
    handles = []
    if colors != None and labels != None:
        handles += [
            ax.plot(*plot_args, lw=0, color=colors[i], label=labels[i])[0]
            for i in range(len(labels))]
    if n_trials != None and n_neurons != None:
        handles += [
            ax.plot(*plot_args, lw=0, color='black', label=r'$n_{trial}=$'+str(n_trials))[0],
            ax.plot(*plot_args, lw=0, color='black', label=r'$n_{neuron}=$'+str(n_neurons))[0],
            ax.plot(*plot_args, lw=0, color='black', label=r'$n_{session}=$'+str(n_sessions))[0]]
    ax.legend(
        loc=loc,
        handles=handles,
        labelcolor='linecolor',
        frameon=False,
        framealpha=0)
    
# add heatmap colorbar.
def add_heatmap_colorbar(ax, cmap, norm, label):
    if norm != None:
        cbar = ax.figure.colorbar(
            plt.cm.ScalarMappable(
                cmap=cmap,
                norm=norm),
            ax=ax, aspect=100, shrink=0.75, fraction=0.1)
        cbar.ax.set_ylabel(label, rotation=90, labelpad=10)
        for tick in cbar.ax.get_yticklabels():
            tick.set_rotation(90)
        cbar.ax.yaxis.set_label_coords(-1, 1.06)
        cbar.outline.set_linewidth(0.25)


#%% basic plotting utils class

class utils_basic:

    def __init__(self):
        self.min_num_trial = 5
        self.latent_cmap = plt.cm.gnuplot2_r
    
    @show_memory_usage
    def run_glm(self):
        # define kernel window.
        kernel_win = [-500,3000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, kernel_win[0], kernel_win[1])
        kernel_time = self.alignment['neu_time'][l_idx:r_idx]
        l_idx = np.searchsorted(self.alignment['neu_time'], 0) - l_idx
        r_idx = r_idx - np.searchsorted(self.alignment['neu_time'], 0)
        # collect data.
        list_dff = [nt['dff'] for nt in self.list_neural_trials]
        list_neu_time = [nt['time'] for nt in self.list_neural_trials]
        list_input_time  = [nt['vol_time'] for nt in self.list_neural_trials]
        list_input_value = [nt['vol_stim_vis'] for nt in self.list_neural_trials]
        list_stim_labels = [nt['stim_labels'] for nt in self.list_neural_trials]
        # fit glm model.
        kernel_all, exp_var_all = run_glm_multi_sess(
            list_dff, list_neu_time,
            list_input_time, list_input_value, list_stim_labels,
            l_idx, r_idx)
        glm = {
            'kernel_time': kernel_time,
            'kernel_all': kernel_all,
            'exp_var_all': exp_var_all,
            }
        return glm

    def plot_mean_sem(self, ax, t, m, s, c, l, a=1.0):
        ax.plot(t, m, color=c, label=l, alpha=a)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.25, edgecolor='none')
        ax.set_xlim([np.min(t), np.max(t)])

    def plot_vol(self, ax, st, sv, c, u, l):
        v = np.mean(sv, axis=0)
        v = rescale(v, u, l)
        ax.plot(st, v, color=c, lw=0.5, linestyle=':')
    
    def plot_pupil(self, ax, nt, cp, c):
        0
    
    def plot_heatmap_neuron(
            self, ax, neu_seq, neu_time, neu_seq_sort, cmap,
            win_sort, norm_mode=None, neu_seq_share=None,
            cbar_label=None,
            ):
        if len(neu_seq) > 0:
            # exclude pure nan row.
            neu_idx = np.where(~np.all(np.isnan(neu_seq), axis=1))[0]
            neu_seq = neu_seq[neu_idx,:].copy()
            neu_seq_sort = neu_seq_sort[neu_idx,:].copy()
            # sort heatmap.
            sorted_idx = sort_heatmap_neuron(neu_seq_sort, neu_time, win_sort)
            # rearrange the matrix.
            data = neu_seq[sorted_idx,:].copy()
            # compute share scale if give.
            if neu_seq_share != None:
                data_share = np.concatenate(neu_seq_share, axis=0)
            else: 
                data_share = np.nan
            # prepare heatmap.
            hm_data, hm_norm, hm_cmap = apply_colormap(data, cmap, norm_mode, data_share)
            # plot heatmap.
            ax.imshow(
                hm_data,
                extent=[neu_time[0], neu_time[-1], 1, hm_data.shape[0]],
                interpolation='nearest', aspect='auto')
            # adjust layout.
            adjust_layout_heatmap(ax)
            ax.set_ylabel('neuron id (sorted)')
            for tick in ax.get_yticklabels():
                tick.set_rotation(90)
            add_heatmap_colorbar(ax, hm_cmap, hm_norm, cbar_label)

    @show_memory_usage
    def plot_heatmap_neuron_cate(
            self, ax, neu_seq, neu_time, neu_seq_sort,
            win_sort, neu_labels, neu_sig,
            norm_mode=None,
            neu_seq_share=None, neu_labels_share=None, neu_sig_share=None,
            cate=[[-1],[1],[2]]):
        if len(neu_seq) > 0:
            list_cmap = [get_roi_label_color(cate=c)[3] for c in cate]
            # exclude pure nan row.
            neu_idx = np.where(~np.all(np.isnan(neu_seq), axis=1))[0]
            neu_seq = neu_seq[neu_idx,:].copy()
            neu_seq_sort = neu_seq_sort[neu_idx,:].copy()
            neu_labels = neu_labels[neu_idx].copy()
            neu_sig = neu_sig[neu_idx].copy()
            # sort heatmap.
            sorted_idx = sort_heatmap_neuron(neu_seq_sort[neu_sig,:], neu_time, win_sort)
            # rearrange the matrix.
            mean = neu_seq[neu_sig,:][sorted_idx,:].copy()
            # devide data into categories.
            list_data = []
            list_data_share = []
            list_cmap = []
            for ci in range(len(cate)):
                data = mean[np.in1d(neu_labels[neu_sig],cate[ci]),:].copy()
                if neu_seq_share != None:
                    data_share = np.concatenate(
                        [neu_seq_share[ni][np.in1d(neu_labels_share[ni], cate[ci])*neu_sig_share[ni],:]
                         for ni in range(len(neu_seq_share))], axis=0)
                else: 
                    data_share = np.nan
                cmap = get_roi_label_color(cate=cate[ci])[3]
                list_data.append(data)
                list_data_share.append(data_share)
                list_cmap.append(cmap)
            # prepare heatmap.
            list_heatmap = []
            list_norm = []
            for ci in range(len(cate)):
                hm = apply_colormap(list_data[ci], list_cmap[ci], norm_mode, list_data_share[ci])
                list_heatmap.append(hm[0])
                list_norm.append(hm[1])
                list_cmap[ci] = hm[2]
            # plot heatmap.
            neu_h = np.concatenate(list_heatmap, axis=0)
            ax.imshow(
                neu_h,
                extent=[neu_time[0], neu_time[-1], 1, neu_h.shape[0]],
                interpolation='nearest', aspect='auto')
            # adjust layout.
            adjust_layout_heatmap(ax)
            ax.set_ylabel('neuron id (sorted)')
            for tick in ax.get_yticklabels():
                tick.set_rotation(90)
            for ci in range(len(cate)):
                add_heatmap_colorbar(
                    ax, list_cmap[ci], list_norm[ci],
                    self.label_names[str(cate[ci][0])])
    
    def plot_heatmap_trial(
            self, ax,
            neu_seq, neu_time,
            cmap, norm_mode, data_share,
            ):
        neu_h, norm, cmap = apply_colormap(
            neu_seq, cmap, norm_mode=norm_mode, data_share=data_share)
        # plot heatmap.
        ax.imshow(
            neu_h,
            extent=[neu_time[0], neu_time[-1], 1, neu_h.shape[0]],
            interpolation='nearest', aspect='auto')
        # adjust layout.
        adjust_layout_heatmap(ax)
        for tick in ax.get_yticklabels():
            tick.set_rotation(90)
        add_heatmap_colorbar(ax, cmap, norm, 'dF/F')
                
    def plot_win_mag_quant(
            self, ax, neu_seq, neu_time,
            win_base, color, c_time, offset,
            ):
        # compute response within window.
        win_early = [0,250]
        win_late  = [250,500]
        [mean_base, sem_base] = get_mean_sem_win(
            neu_seq.reshape(-1, neu_seq.shape[-1]),
            neu_time, c_time, win_base[0], win_base[1], mode='lower')
        [mean_early, sem_early] = get_mean_sem_win(
            neu_seq.reshape(-1, neu_seq.shape[-1]),
            neu_time, c_time, win_early[0], win_early[1], mode='higher')
        [mean_late, sem_late] = get_mean_sem_win(
            neu_seq.reshape(-1, neu_seq.shape[-1]),
            neu_time, c_time, win_late[0], win_late[1], mode='mean')
        # corrected with baseline.
        mean_early -= mean_base
        mean_late  -= mean_base
        # plot errorbar.
        ax.errorbar(
            0 + offset,
            mean_base, sem_base,
            color=color,
            capsize=2, marker='o', linestyle='none',
            markeredgecolor='white', markeredgewidth=0.1)
        ax.errorbar(
            1 + offset,
            mean_early, sem_early,
            color=color,
            capsize=2, marker='o', linestyle='none',
            markeredgecolor='white', markeredgewidth=0.1)
        ax.errorbar(
            2 + offset,
            mean_late, sem_late,
            color=color,
            capsize=2, marker='o', linestyle='none',
            markeredgecolor='white', markeredgewidth=0.1)
        # adjust layout.
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel(
            'response magnitude df/f (mean$\pm$sem) \n baseline [{},{}] ms'.format(
                win_base[0], win_base[1]))
        ax.set_xticks([0,1,2])
        ax.set_xticklabels([
            'baseline \n [{},{}] ms'.format(win_base[0], win_base[1]),
            'early \n [{},{}] ms'.format(win_early[0], win_early[1]),
            'late \n [{},{}] ms'.format(win_late[0], win_late[1])])
        ax.set_xlim([-0.5, 3.5])

    def plot_multi_sess_decoding_num_neu(
            self, ax,
            sampling_nums, acc_model, acc_chance,
            color1, color2
            ):
        # compute mean and sem.
        acc_mean_model  = np.array([get_mean_sem(a)[0] for a in acc_model]).reshape(-1)
        acc_sem_model   = np.array([get_mean_sem(a)[1] for a in acc_model]).reshape(-1)
        acc_mean_chance = np.array([get_mean_sem(a)[0] for a in acc_chance]).reshape(-1)
        acc_sem_chance  = np.array([get_mean_sem(a)[1] for a in acc_chance]).reshape(-1)
        # plot results.
        self.plot_mean_sem(ax, sampling_nums, acc_mean_model,  acc_sem_model,  color2, 'model')
        self.plot_mean_sem(ax, sampling_nums, acc_mean_chance, acc_sem_chance, color1, 'chance')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('number of sampled neurons')
        add_legend(ax, [color2,color1], ['model','chance'], 'upper left')

    def plot_multi_sess_decoding_slide_win(
            self, ax,
            eval_time, acc_model, acc_chance,
            color1, color2
            ):
        # compute mean and sem.
        acc_mean_model  = np.array([get_mean_sem(a)[0] for a in acc_model]).reshape(-1)
        acc_sem_model   = np.array([get_mean_sem(a)[1] for a in acc_model]).reshape(-1)
        acc_mean_chance = np.array([get_mean_sem(a)[0] for a in acc_chance]).reshape(-1)
        acc_sem_chance  = np.array([get_mean_sem(a)[1] for a in acc_chance]).reshape(-1)
        # compute bounds.
        upper = np.nanmax([acc_mean_model,acc_mean_chance]) + np.nanmax([acc_sem_model,acc_sem_chance])
        lower = np.nanmin([acc_mean_model,acc_mean_chance]) - np.nanmax([acc_sem_model,acc_sem_chance])
        # plot results.
        self.plot_mean_sem(ax, eval_time, acc_mean_model,  acc_sem_model,  color2, 'model')
        self.plot_mean_sem(ax, eval_time, acc_mean_chance, acc_sem_chance, color1, 'chance')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        add_legend(ax, [color2,color1], ['model','chance'], 'upper left')

    def plot_cluster_ca_transient(self, ax, colors, cluster_id, cate):
        n_clusters = len(np.unique(cluster_id))
        lbl = ['cluster #'+str(i) for i in range(n_clusters)]
        # compute calcium transient.
        _, dff_ca_neu, dff_ca_time = get_ca_transient_multi_sess(self.list_neural_trials)
        # get category.
        dff_ca_cate = np.array(dff_ca_neu,dtype='object')[
            np.in1d(np.concatenate(self.list_labels), cate)].copy().tolist()
        # average across trials.
        dff_ca_cate = [get_mean_sem(d)[0].reshape(1,-1) for d in dff_ca_cate]
        dff_ca_cate = np.concatenate(dff_ca_cate, axis=0)
        # collect within cluster average.
        neu_mean = [get_mean_sem(dff_ca_cate[cluster_id==i])[0] for i in range(n_clusters)]
        neu_sem = [get_mean_sem(dff_ca_cate[cluster_id==i])[1] for i in range(n_clusters)]
        # plot results.
        for i in range(n_clusters): 
            self.plot_mean_sem(ax, dff_ca_time, neu_mean[i], neu_sem[i], colors[i], None)
        # adjust layout.
        ax.tick_params(axis='y', tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('time since calcium transient triggered (ms)')
        ax.set_ylabel('df/f (z-scored)')
        ax.set_title('calcium transient average')
        add_legend(ax, colors, lbl, None, None, None, 'upper right')
    
    # plot fraction of each cluster.
    def plot_cluster_fraction(self, ax, colors, cluster_id):
        num = [np.nansum(cluster_id==i) for i in np.unique(cluster_id)]
        ax.pie(
            num,
            labels=[str(num[i])+' cluster # '+str(i) for i in range(len(num))],
            colors=colors,
            autopct='%1.1f%%',
            wedgeprops={'linewidth': 1, 'edgecolor':'white', 'width':0.2})
        ax.set_title('fraction of neurons in each cluster')

    def plot_cluster_cluster_fraction_in_cate(self, ax, colors, cluster_id, neu_labels, label_names):
        bar_width = 0.5
        n_clusters = len(np.unique(cluster_id))
        lbl = ['cluster #'+str(i) for i in range(n_clusters)]
        cate_eval = [int(k) for k in label_names.keys()]
        cate_name = [v for v in label_names.values()]
        # get fraction in each category.
        fraction = np.zeros((n_clusters, len(cate_eval)))
        for i in range(n_clusters):
            for j in range(len(cate_eval)):
                nc = np.nansum((cluster_id==i)*(neu_labels==cate_eval[j]))
                nt = np.nansum(neu_labels==cate_eval[j]) + 1e-5
                fraction[i,j] = nc / nt
        # plot bars.
        bottom = 1-np.cumsum(fraction, axis=0)
        for j in range(len(cate_eval)):
            for i in range(n_clusters):
                if fraction[i,j] != 0:
                    ax.bar(
                        j, fraction[i,j],
                        bottom=bottom[i,j],
                        edgecolor='white',
                        width=bar_width,
                        color=colors[i])
                    ax.text(j, bottom[i,j]+fraction[i,j]/2,
                            f'{fraction[i,j]:.2f}',
                            ha='center', va='center', color='#2C2C2C')
        # adjust layout.
        ax.tick_params(tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(np.arange(len(cate_eval)))
        ax.set_xticklabels(cate_name, rotation='vertical')
        ax.set_yticks([0,0.25,0.5,0.75, 1])
        ax.grid(True, axis='y', linestyle='--')
        ax.set_xlim([-0.5,len(cate_eval)+0.5])
        ax.set_ylim([0,1])
        ax.set_title('fraction of cluster for each cell-type')
        add_legend(ax, colors, lbl, None, None, None, 'upper right')
    
    def plot_cluster_neu_fraction_in_cluster(self, ax, cluster_id, color):
        bar_width = 0.2
        n_clusters = len(np.unique(cluster_id))
        # get fraction in each cluster.
        num = np.array([np.nansum(cluster_id==i) for i in np.unique(cluster_id)])
        fraction = num / np.nansum(num)
        # plot bars.
        ax.axis('off')
        ax = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
        for ci in range(n_clusters):
            ax.barh(
                ci+0.5, fraction[ci],
                left=0,
                edgecolor='white',
                height=bar_width,
                color=color)
            ax.text(0.01, ci+0.7,
                    f'{fraction[ci]:.2f}, n={num[ci]}',
                    ha='left', va='center', color='#2C2C2C')
        # adjust layouts.
        ax.tick_params(tick1On=False)
        ax.tick_params(axis='y', labelrotation=90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0,np.nanmax(fraction)])
        ax.set_ylim([-0.2, n_clusters+0.1])
        ax.set_xlabel('fraction of neurons')

    def plot_cluster_cate_fraction_in_cluster(self, ax, cluster_id, neu_labels, label_names):
        bar_width = 0.2
        n_clusters = len(np.unique(cluster_id))
        cate_eval = [int(k) for k in label_names.keys()]
        cate_color = [get_roi_label_color(cate=[c])[2] for c in cate_eval]
        # get fraction in each category.
        fraction = np.zeros((len(cate_eval), n_clusters))
        for i in range(len(cate_eval)):
            for j in range(n_clusters):
                nc = np.nansum((cluster_id==j)*(neu_labels==cate_eval[i]))
                nt = np.nansum(neu_labels==cate_eval[i]) + 1e-5
                fraction[i,j] = nc / nt
        fraction = fraction / np.nansum(fraction, axis=0)
        # plot bars.
        ax.axis('off')
        ax = ax.inset_axes([0, 0, 0.6, 1], transform=ax.transAxes)
        bottom = 1-np.cumsum(fraction, axis=0)
        for j in range(n_clusters):
            for i in range(len(cate_eval)):
                if fraction[i,j] != 0:
                    ax.barh(
                        j+0.5, fraction[i,j],
                        left=bottom[i,j],
                        edgecolor='white',
                        height=bar_width,
                        color=cate_color[i])
                    ax.text(bottom[i,j]+fraction[i,j]/2, j+0.9,
                            f'{fraction[i,j]:.2f}',
                            ha='center', va='center', rotation=90, color='#2C2C2C')
        # adjust layout.
        ax.tick_params(tick1On=False)
        ax.tick_params(axis='x', labelrotation=90)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([0,0.5,1])
        ax.set_yticks([])
        ax.grid(True, axis='x', linestyle='--')
        ax.set_xlim([-0.1,1.1])
        ax.set_ylim([-0.2, n_clusters+0.1])
        ax.set_xlabel('fraction of cell-type')

    def plot_cluster_mean_sem(
            self, ax, neu_mean, neu_sem, neu_time,
            norm_params, stim_seq, c_stim, c_neu, xlim
            ):
        l_nan_margin = 5
        r_nan_margin = 5
        len_scale_x = 1000
        len_scale_y = 0.5
        # set margin values to nan.
        l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, xlim[0], xlim[1])
        nm = neu_mean.copy()
        ns = neu_sem.copy()
        nm[:,:l_idx+l_nan_margin] = np.nan
        nm[:,r_idx-r_nan_margin-1:] = np.nan
        ns[:,:l_idx+l_nan_margin] = np.nan
        ns[:,r_idx-r_nan_margin-1:] = np.nan
        # plot stimulus.
        if not stim_seq is None:
            for i in range(stim_seq.shape[0]):
                ax.fill_between(
                    stim_seq[i,:],
                    0, nm.shape[0],
                    color=c_stim[i], edgecolor='none', alpha=0.25, step='mid')
        # plot cluster average.
        for i in range(nm.shape[0]):
            a, b, c, d = norm_params[i]
            # add y=0 line.
            ax.hlines(i+b, xlim[0]*0.99, xlim[1]*0.99, linestyle=':', color='#2C2C2C', alpha=0.2)
            # plot neural traces.
            self.plot_mean_sem(
                ax, neu_time,
                (a*nm[i,:]+b)+nm.shape[0]-i-1, np.abs(a)*ns[i,:],
                c_neu[i], None)
            # plot y scalebar.
            y_start = i + 0.5 - len_scale_y/2
            ax.vlines(xlim[0]*0.99, y_start, y_start+len_scale_y, color='#2C2C2C')
            ax.text(xlim[0]*0.99, i+0.5,
                '{:.3f}'.format(len_scale_y*(d-c)),
                va='center', ha='right', rotation=90, color='#2C2C2C')
        # plot x scalebar.
        adjust_layout_cluster_neu(ax, nm.shape[0], xlim)
        ax.hlines(-0.1, xlim[0]*0.99, xlim[0]*0.99+len_scale_x, color='#2C2C2C')
        ax.text(xlim[0]*0.99 + len_scale_x/2, -0.15, '{} ms'.format(int(len_scale_x)),
            va='top', ha='center', color='#2C2C2C')
        ax.set_ylim([-0.2, neu_mean.shape[0]+0.1])
    
    def plot_cluster_metric_quant(self, ax, quant_cluster, metric_target, colors):
        offset = 0.1
        n_clusters = len(quant_cluster)
        n_conditions = len(quant_cluster[0])
        # collect target metrics into n_clusters*[n_conditions*n_neurons]
        metric_cluster = [
            np.concatenate(
                [quant_cluster[ci][di][metric_target].reshape(1,-1)
                 for di in range(n_conditions)], axis=0)
            for ci in range(len(quant_cluster))]
        offsets = (np.arange(len(quant_cluster[0])) - int(len(quant_cluster[0])/2)) * offset
        # plot results for each class.
        for ci in range(n_clusters):
            for di in range(n_conditions):
                m, s = get_mean_sem(metric_cluster[ci][di,:].reshape(-1,1))
                ax.errorbar(
                    m, ci+offsets[di], None, s,
                    color=colors[di],
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=0.1)
        # adjust layout.
        ax.tick_params(axis='y', tick1On=False)
        ax.tick_params(axis='x', labelrotation=90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks(np.arange(n_clusters))
        ax.set_yticklabels([])

    def plot_cluster_interval_norm(
            self, ax, cluster_id,
            bin_neu_seq, bin_stim_seq,
            colors,
            ):
        margin_time = 500
        gap = 0.05
        len_scale_y = 0.5
        # get response within cluster at each bin.
        cluster_bin_neu_mean = [get_mean_sem_cluster(neu, cluster_id)[0] for neu in bin_neu_seq]
        cluster_bin_neu_sem  = [get_mean_sem_cluster(neu, cluster_id)[1] for neu in bin_neu_seq]
        # organize into bin_num*n_clusters*time.
        cluster_bin_neu_mean = [np.expand_dims(neu, axis=0) for neu in cluster_bin_neu_mean]
        cluster_bin_neu_sem  = [np.expand_dims(neu, axis=0) for neu in cluster_bin_neu_sem]
        cluster_bin_neu_mean = np.concatenate(cluster_bin_neu_mean, axis=0)
        cluster_bin_neu_sem  = np.concatenate(cluster_bin_neu_sem, axis=0)
        norm_params = [get_norm01_params(cluster_bin_neu_mean[:,i,:]) for i in range(self.n_clusters)]
        # get line colors for each cluster.
        c_neu = [get_cmap_color(self.bin_num, base_color=c) for c in colors]
        # convert to colors for each bin.
        c_neu = [[c_neu[i][j] for i in range(self.n_clusters)] for j in range(self.bin_num)]
        # only keep 2 stimulus.
        c_idx = int(bin_stim_seq.shape[1]/2)
        ss = bin_stim_seq[:,c_idx-1:c_idx+1,:]
        # plot interval.
        ax.vlines([-gap,0,1,1+gap], 0, self.n_clusters, color='#2C2C2C', linestyle=':')
        for ci in range(self.n_clusters):
            # take binned data for each clsuter.
            a, b, c, d = norm_params[ci]
            nm = cluster_bin_neu_mean[:,ci,:]
            ns = cluster_bin_neu_sem[:,ci,:]
            for bi in range(self.bin_num):
                # plot neural traces before/within/after interval.
                for l_t, r_t, off in zip(
                        [ss[bi,0,0]-margin_time, ss[bi,0,1], 0],
                        [ss[bi,0,1], 0, ss[bi,1,1]+margin_time],
                        [-1-gap, 0, 1+gap],
                        ):
                    # get indice.
                    l_idx, r_idx = get_frame_idx_from_time(
                        self.alignment['neu_time'], 0, l_t, r_t)
                    # take data points.
                    m = nm[bi,l_idx:r_idx]
                    s = ns[bi,l_idx:r_idx]
                    # normalize timestamps.
                    t = norm01(self.alignment['neu_time'][l_idx:r_idx]) + off
                    # plot results.
                    self.plot_mean_sem(
                        ax, t, (a*m+b)+self.n_clusters-ci-1, np.abs(a)*s,
                        c_neu[bi][ci], None)
        # plot y scalebar.
        for ci in range(self.n_clusters):
            a, b, c, d = norm_params[ci]
            y_start = ci + 0.5 - len_scale_y/2
            ax.vlines(-1.1, y_start, y_start+len_scale_y, color='#2C2C2C')
            ax.text(-1.1, ci+0.5,
                '{:.3f}'.format(len_scale_y*(d-c)),
                va='center', ha='right', rotation=90, color='#2C2C2C')
        # adjust layout.
        ax.axis('off')
        ax.set_xlim(-1.5,3)
        ax.set_ylim([-0.2, self.n_clusters+0.1])
        
    def plot_cluster_heatmap(self, ax, neu_seq, neu_time, cluster_id, norm_mode, cmap):
        gap = 5
        win_conv = 5
        win_sort = [-500, 500]
        # create heatmap for all clusters.
        data = []
        yticks = []
        for i in range(self.n_clusters):
            # get data within cluster.
            neu = neu_seq[cluster_id==(self.n_clusters-i-1),:].copy()
            # smooth the values in the sorting window.
            l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, win_sort[0], win_sort[1])
            smoothed_mean = np.array(
                [np.convolve(row, np.ones(win_conv)/win_conv, mode='same')
                 for row in neu[:,l_idx:r_idx]])
            sort_idx_neu = np.argmax(smoothed_mean, axis=1).reshape(-1).argsort()
            # rearrange the matrix.
            neu = neu[sort_idx_neu,:]
            # add into full heatmap.
            data.append(neu)
            data.append(np.nan*np.ones([gap,neu.shape[1]]))
            yticks.append(neu.shape[0]+gap)
        # plot heatmap.
        data = np.concatenate(data,axis=0)
        hm_data, hm_norm, hm_cmap = apply_colormap(data, cmap, norm_mode, data)
        ax.imshow(
            hm_data,
            extent=[neu_time[0], neu_time[-1], 1, hm_data.shape[0]],
            interpolation='nearest', aspect='auto',
            origin='lower')
        # adjust layout.
        adjust_layout_heatmap(ax)
        ax.set_ylabel('neuron id (clustered)')
        ax.set_yticks(np.cumsum(yticks)-gap)
        ax.set_yticklabels([f'#{str(i).zfill(2)}' for i in range(self.n_clusters)])
    
    def plot_dendrogram(self, ax, kernel_all, cmap):
        p=5
        # run hierarchical clustering.
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        model.fit(kernel_all)
        # create the counts of samples under each node.
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        # create linkage matrix.
        linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
        # create colors.
        dists = linkage_matrix[:, 2]
        norm = mcolors.Normalize(vmin=dists.min(), vmax=dists.max())
        color_lookup = {
            i + n_samples: mcolors.rgb2hex(cmap(norm(dist)))
            for i, dist in enumerate(dists)}
        # plot dendrogram.
        dendrogram(
            linkage_matrix,
            truncate_mode='level',
            p=p,
            ax=ax,
            color_threshold=0,
            link_color_func=lambda idx: color_lookup.get(idx, '#000000'))
        # adjust layout.
        ax.tick_params(tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('number of points in node')
        ax.set_yticks([])

    
    def plot_cluster_bin_3d_latents(self, axs, cluster_id, cate=None):
        colors = get_cmap_color(self.n_clusters, cmap=self.cluster_cmap)
        # collect data.
        [_, [neu_seq, stim_seq, stim_value, pre_isi], _, [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[None, None, None, None, None, [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        # bin data based on isi.
        [bins, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
            neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
        # get latent dynamics for each cluster.
        neu_z = []
        for i in range(self.n_clusters):
            # reshape data within cluster.
            neu_x = [np.expand_dims(bns[cluster_id==i,:],2) for bns in bin_neu_seq]
            neu_x = np.concatenate(neu_x, axis=2)
            neu_x = neu_x.reshape(-1, len(self.alignment['neu_time'])*self.bin_num)
            # fit model.
            model = TSNE(n_components=self.d_latent)
            z = model.fit_transform(neu_x.T).T
            z = z.reshape(self.d_latent, len(self.alignment['neu_time']), self.bin_num)
            neu_z.append(z)
        # get line colors for each cluster.
        c_neu = [get_cmap_color(self.bin_num, base_color=c) for c in colors]
        # convert to colors for each bin.
        c_neu = [[c_neu[j][i] for i in range(self.bin_num)] for j in range(self.n_clusters)]
        lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
        # plot latent dynamics.
        for ci in range(self.n_clusters):
            for bi in range(self.bin_num):
                c_idx = int(bin_stim_seq.shape[1]/2)
                l_idx, r_idx = get_frame_idx_from_time(
                    self.alignment['neu_time'], 0, bin_stim_seq[bi,c_idx-1,1], 0)
                cn = [c_neu[ci][bi]]*(r_idx-l_idx)
                nz = neu_z[ci][:,l_idx:r_idx,bi]
                nz -= nz[:,0].reshape(-1,1)
                self.plot_3d_latent_dynamics(axs[ci], nz, None, cn, None, add_stim=False)
            # adjust layout.
            axs[ci].grid(False)
            axs[ci].view_init(elev=30, azim=30)
            axs[ci].set_xticks([])
            axs[ci].set_yticks([])
            axs[ci].set_zticks([])
            axs[ci].set_xlabel('latent 1')
            axs[ci].set_ylabel('latent 2')
            axs[ci].set_zlabel('latent 3')
            axs[ci].xaxis.pane.fill = False
            axs[ci].yaxis.pane.fill = False
            axs[ci].zaxis.pane.fill = False
            add_legend(axs[ci], c_neu[ci], lbl, n_trials, n_neurons, self.n_sess, 'upper right', dim=3)

    def plot_3d_latent_dynamics(self, ax, neu_z, stim_seq, neu_time, c_stim, add_stim=True):
        c_neu = get_cmap_color(neu_z.shape[1], cmap=self.latent_cmap)
        # plot dynamics.
        for t in range(neu_z.shape[1]-1):
            # trajaectory.
            ax.plot(neu_z[0,t:t+2], neu_z[1,t:t+2], neu_z[2,t:t+2], color=c_neu[t])
        # end point.
        ax.scatter(neu_z[0,0], neu_z[1,0], neu_z[2,0], color='black', marker='x')
        ax.scatter(neu_z[0,-1], neu_z[1,-1], neu_z[2,-1], color='black', marker='o')
        # stimulus.
        if add_stim:
            for j in range(stim_seq.shape[0]):
                l_idx, r_idx = get_frame_idx_from_time(
                    neu_time, 0, stim_seq[j,0], stim_seq[j,1])
                ax.plot(neu_z[0,l_idx:r_idx], neu_z[1,l_idx:r_idx], neu_z[2,l_idx:r_idx],
                        lw=3, color=c_stim[j])

    def plot_glm_heatmap(
            self, ax, kernel_time, kernel_all, stim_seq,
            neu_labels, neu_sig,
            ):
        # plot heatmap.
        win_sort = [kernel_time[0], 1000]
        self.plot_heatmap_neuron_cate(
            ax, kernel_all, kernel_time, kernel_all, win_sort, neu_labels, neu_sig)
        ax.set_xlabel('time since stim (ms)')
        # add stimulus line.
        c_idx = int(stim_seq.shape[0]/2)
        xlines = [kernel_time[np.searchsorted(kernel_time, stim_seq[c_idx,0])],
                  kernel_time[np.searchsorted(kernel_time, stim_seq[c_idx,1])]]
        for xl in xlines:
            ax.axvline(xl, color='black', lw=1, linestyle='--')




