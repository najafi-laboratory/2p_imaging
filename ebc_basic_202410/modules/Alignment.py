#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils_2p.alignment import align_neu_seq_utils, pad_seq, trim_seq


# extract response around given stimulus.
def get_stim_response(
        neural_trials, state, trial_types,
        l_frames, r_frames):
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_seq = []

    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        trial_vis = neural_trials[trials][state]
        trial_type = neural_trials[trials]['trial_type']
        if trial_type ==trial_types:
            if not np.isnan(trial_vis[0]):
                idx = np.argmin(np.abs(time - trial_vis[0]))
                if idx > l_frames and idx < len(time)-r_frames:
                    # signal response.
                    f = fluo[:, idx-l_frames : idx+r_frames]
                    f = np.expand_dims(f, axis=0)
                    neu_seq.append(f)
                    # signal time stamps.
                    t = time[idx-l_frames : idx+r_frames] - time[idx]
                    neu_time.append(t)
                    # visual stimulus timestamps.
                    stim_seq.append(trial_vis[1] - trial_vis[0])
                    
                
    if len(neu_seq) > 0:
        neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
        stim_seq = np.median(stim_seq,axis=0)
        
    else:
        neu_seq = np.array([[[np.nan]]])
        neu_time = np.array([np.nan])
        stim_seq = np.array([np.nan, np.nan])
        
    return [neu_seq, neu_time, stim_seq]




# extract spontaneous response during iti.
def get_iti_response(
        neural_trials,
        l_frames, r_frames):
    # initialize list.
    neu_seq  = []
    neu_time = []
    

    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_state = neural_trials[trials]['trial_iti']


        if np.isnan(np.sum(time_state)):
            continue
        for i in range(np.size(time_state)):
            idx = np.argmin(np.abs(time - time_state[i]))
            if idx > l_frames and idx < len(time)-r_frames:
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    
    return [neu_seq, neu_time]
