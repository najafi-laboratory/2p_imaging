# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:00:46 2025

@author: saminnaji3
"""

#!/usr/bin/env python3

import os
import h5py
import numpy as np

from modules.ReadResults import read_raw_voltages
from modules.ReadResults import read_dff
from modules.ReadResults import read_bpod_mat_data


# detect the rising edge and falling edge of binary series.

def get_trigger_time(
        vol_time,
        vol_bin
        ):
    # find the edge with np.diff and correct it by preappend one 0.
    diff_vol = np.diff(vol_bin, prepend=0)
    idx_up = np.where(diff_vol == 1)[0]
    idx_down = np.where(diff_vol == -1)[0]
    # select the indice for risging and falling.
    # give the edges in ms.
    time_up   = vol_time[idx_up]
    time_down = vol_time[idx_down]
    return time_up, time_down


# correct the fluorescence signal timing.

def correct_time_img_center(time_img):
    # find the frame internal.
    diff_time_img = np.diff(time_img, append=0)
    # correct the last element.
    diff_time_img[-1] = np.mean(diff_time_img[:-1])
    # move the image timing to the center of photon integration interval.
    diff_time_img = diff_time_img / 2
    # correct each individual timing.
    time_neuro = time_img + diff_time_img
    return time_neuro


# align the stimulus sequence with fluorescence signal.

def align_stim(
        vol_time,
        time_neuro,
        vol_stim_vis,
        label_stim,
        ):
    # find the rising and falling time of stimulus.
    stim_time_up, stim_time_down = get_trigger_time(
        vol_time, vol_stim_vis)
    # avoid going up but not down again at the end.
    stim_time_up = stim_time_up[:len(stim_time_down)]
    # assign the start and end time to fluorescence frames.
    stim_start = []
    stim_end = []
    for i in range(len(stim_time_up)):
        # find the nearest frame that stimulus start or end.
        stim_start.append(
            np.argmin(np.abs(time_neuro - stim_time_up[i])))
        stim_end.append(
            np.argmin(np.abs(time_neuro - stim_time_down[i])))
    # reconstruct stimulus sequence.
    stim = np.zeros(len(time_neuro))
    for i in range(len(stim_start)):
        label = label_stim[vol_time==stim_time_up[i]][0]
        stim[stim_start[i]:stim_end[i]] = label
    return stim


# process trial start signal.

def get_trial_start_end(
        vol_time,
        vol_start,
        ):
    time_up, time_down = get_trigger_time(vol_time, vol_start)
    # find the impulse start signal.
    time_start = [time_up[0]]
    for i in range(len(time_up)-1):
        if time_up[i+1] - time_up[i] > 5:
            time_start.append(time_up[i+1])
    start = []
    end = []
    # assume the current trial end at the next start point.
    for i in range(len(time_start)):
        s = time_start[i]
        e = time_start[i+1] if i != len(time_start)-1 else -1
        start.append(s)
        end.append(e)
    return start, end


# trial segmentation.

def trial_split(
        start, end,
        dff, stim, time_neuro,
        label_stim, vol_time,
        ):
    neural_trials = dict()
    pre = 30
    for i in range(len(start)):
        if np.max(time_neuro > start[i]):
            neural_trials[str(i)] = dict()
            #print(i)
            start_idx_dff = np.where(time_neuro > start[i])[0][0]
            end_idx_dff   = np.where(time_neuro < end[i])[0][-1] if end[i] != -1 else -1
            if start_idx_dff > pre-1:
                start_idx_dff = np.where(time_neuro > start[i])[0][0]-pre
                neural_trials[str(i)]['time'] = time_neuro[start_idx_dff:end_idx_dff]
                neural_trials[str(i)]['stim'] = stim[start_idx_dff:end_idx_dff]
                neural_trials[str(i)]['dff'] = dff[:,start_idx_dff:end_idx_dff]
                start_idx_vol = np.where(vol_time > start[i])[0][0]-pre
                end_idx_vol   = np.where(vol_time < end[i])[0][-1] if end[i] != -1 else -1
                neural_trials[str(i)]['vol_stim'] = label_stim[start_idx_vol:end_idx_vol]
                neural_trials[str(i)]['vol_time'] = vol_time[start_idx_vol:end_idx_vol]
                neural_trials[str(i)]['time_start'] = time_neuro[start_idx_dff+pre]
            else:
                neural_trials[str(i)]['time'] = time_neuro[start_idx_dff:end_idx_dff]
                neural_trials[str(i)]['stim'] = stim[start_idx_dff:end_idx_dff]
                neural_trials[str(i)]['dff'] = dff[:,start_idx_dff:end_idx_dff]
                start_idx_vol = np.where(vol_time > start[i])[0][0]
                end_idx_vol   = np.where(vol_time < end[i])[0][-1] if end[i] != -1 else -1
                neural_trials[str(i)]['vol_stim'] = label_stim[start_idx_vol:end_idx_vol]
                neural_trials[str(i)]['vol_time'] = vol_time[start_idx_vol:end_idx_vol]
                neural_trials[str(i)]['time_start'] = time_neuro[start_idx_dff]
    return neural_trials




# add trial information with bpod session data.

def trial_label(
        neural_trials,
        ):
    
    for i in range(len(neural_trials)):
        if any(neural_trials[str(i)]['stim']==1):
            neural_trials[str(i)]['trial_types'] = 1
        else:
            neural_trials[str(i)]['trial_types'] = 0
        
    return neural_trials
    

# save trial neural data.

def save_trials(
        ops,
        neural_trials
        ):
    # file structure:
    # ops['save_path0'] / neural_trials.h5
    # -- trial_id
    # ---- 1
    # ------ time
    # ------ stim
    # ------ dff
    # ---- 2
    # ...
    exclude_start = 1
    exclude_end = 1
    f = h5py.File(
        os.path.join(ops['save_path0'], 'neural_trials.h5'),
        'w')
    grp = f.create_group('trial_id')
    for trial in range(len(neural_trials)):
        if trial > exclude_start and trial < len(neural_trials)-exclude_end:
            trial_group = grp.create_group(str(trial))
            for k in neural_trials[str(trial)].keys():
                trial_group[k] = neural_trials[str(trial)][k]
    f.close()


# main function for trialization

def run(ops):
    print('===============================================')
    print('=============== trial alignment ===============')
    print('===============================================')
    print('Reading dff traces and voltage recordings')
    dff = read_dff(ops)
    dff_z_scored = (dff - dff.mean(axis=1, keepdims=True)) / dff.std(axis=1, keepdims=True)
    [vol_time, vol_start, vol_stim_vis, vol_img, 
     _, _, _, _, vol_led] = read_raw_voltages(ops)
    print('Correcting 2p camera trigger time')
    # signal trigger time stamps.
    time_img, _   = get_trigger_time(vol_time, vol_img)
    # correct imaging timing.
    time_neuro = correct_time_img_center(time_img)
    # stimulus alignment.
    print('Aligning stimulus to 2p frame')
    stim = align_stim(vol_time, time_neuro, vol_led, vol_led)
    # trial segmentation.
    print('Segmenting trials')
    start, end = get_trial_start_end(vol_time, vol_start)
    neural_trials = trial_split(
        start, end,
        dff_z_scored, stim, time_neuro,
        vol_stim_vis, vol_time)
    ####
    neural_trials = trial_label(neural_trials)
    # save the final data.
    print('Saving trial data')
    save_trials(ops, neural_trials)
