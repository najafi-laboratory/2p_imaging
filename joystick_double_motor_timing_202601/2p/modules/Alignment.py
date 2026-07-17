#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d

# cut sequence into the same length as the shortest one given pivots.
def trim_seq(data, pivots):
    if len(data[0].shape) == 1:
        len_l_min = np.nanmin(pivots)
        len_r_min = np.nanmin([len(data[i])-p for i,p in enumerate(pivots)])
        trim_data = []
        for i,p in enumerate(pivots):
            if ~np.isnan(p):
                trim_data.append(data[i][int(p-len_l_min):int(p+len_r_min)])
            else:
                trim_data.append(np.array([np.nan]))
    if len(data[0].shape) == 3:
        len_l_min = np.nanmin(pivots)
        len_r_min = np.nanmin([len(data[i][0,0,:])-p for i,p in enumerate(pivots)])
        trim_data = []
        for i,p in enumerate(pivots):
            if ~np.isnan(p):
                trim_data.append(data[i][:, :, int(p-len_l_min):int(p+len_r_min)])
            else:
                trim_data.append(np.array([np.nan]))
    return data

# alignment on task variables.
def get_task_response(neural_trials, target_state):
    l_frames = 3000
    r_frames = 3000
    exclude_start_trials = 0
    exclude_end_trials = 0
    # initialization.
    time = neural_trials['time']
    neu_seq    = []
    neu_time   = []
    state_time = []
    js_rot     = []
    outcome    = []
    # loop over trials.
    for ti in range(len(neural_trials['trial_labels'])):
        t = neural_trials['trial_labels'][target_state][ti][0]
        jr = neural_trials['trial_labels']['js_rot'][ti]
        jt = neural_trials['trial_labels']['js_time'][ti]
        if (not np.isnan(t) and
            ti >= exclude_start_trials and
            ti < len(neural_trials['trial_labels'])-exclude_end_trials
            ):
            # get state start timing.
            idx = np.searchsorted(neural_trials['time'], t)
            if idx > l_frames and idx < len(neural_trials['time'])-r_frames:
                # signal response.
                f = neural_trials['dff'][:, idx-l_frames : idx+r_frames]
                neu_seq.append(f)
                # signal time stamps.
                neu_time.append(neural_trials['time'][idx-l_frames : idx+r_frames] - time[idx])
                # task variables.
                state_time.append(neural_trials['trial_labels'][target_state][ti]-t)
                # joystick trajectory.
                interpolator = interp1d(jt, jr, bounds_error=False)
                js_rot.append(interpolator(neural_trials['time'][idx-l_frames : idx+r_frames]))
                # outcome.
                outcome.append(neural_trials['trial_labels']['outcome'][ti])
            else:
                neu_seq.append(np.full((neural_trials['dff'].shape[0], l_frames+r_frames), np.nan))
                neu_time.append(np.full((l_frames+r_frames), np.nan))
                state_time.append(np.array([np.nan, np.nan]))
                js_rot.append(np.full((l_frames+r_frames), np.nan))
                outcome.append('nan')
        else:
            neu_seq.append(np.full((neural_trials['dff'].shape[0], l_frames+r_frames), np.nan))
            neu_time.append(np.full((l_frames+r_frames), np.nan))
            state_time.append(np.array([np.nan, np.nan]))
            js_rot.append(np.full((l_frames+r_frames), np.nan))
            outcome.append('nan')
    # correct neural data centering at zero.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    js_rot  = trim_seq(js_rot, neu_time_zero)
    # concatenate results.
    neu_seq    = np.stack(neu_seq)
    neu_time   = [nt.reshape(1,-1) for nt in neu_time]
    neu_time   = np.concatenate(neu_time, axis=0)
    state_time = np.stack(state_time)
    js_rot     = np.stack(js_rot)
    outcome    = np.stack(outcome)
    # get mean time stamps.
    neu_time   = np.nanmean(neu_time, axis=0)
    state_time = np.nanmean(state_time, axis=0)
    # combine results.
    return [neu_seq, neu_time, state_time, js_rot, outcome]

# run alignment for all states for all sessions.
def run_sess_alignment(list_neural_trials):
    list_target_state = [
        'state_vis1',
        'state_press1',
        'state_retract1',
        'state_delay',
        'state_vis2',
        'state_press2',
        'state_retract2',
        'state_reward',
        'state_iti'
        ]
    alignment = {}
    # run alignment for each state.
    for ti, target_state in enumerate(list_target_state):
        print(f'Aligning trials for state {ti+1}/{len(list_target_state)} {target_state}')
        list_neu_seq = []
        list_neu_time = []
        list_state_time = []
        list_js_rot = []
        list_outcome = []
        # run alignment for each session.
        for si, neural_trials in enumerate(list_neural_trials):
            [neu_seq, neu_time, state_time, js_rot, outcome
             ] = get_task_response(neural_trials, target_state)
            list_neu_seq.append(neu_seq)
            list_neu_time.append(neu_time)
            list_state_time.append(state_time)
            list_js_rot.append(js_rot)
            list_outcome.append(outcome)
        # combine results.
        neu_time = np.nanmean(np.stack(list_neu_time), axis=0)
        state_time = np.nanmean(np.stack(list_neu_time), axis=0)
        alignment[target_state] = {
            'list_neu_seq': list_neu_seq,
            'neu_time': neu_time,
            'state_time': state_time,
            'list_js_rot': list_js_rot,
            'list_outcome': list_outcome,
            }
    return alignment


