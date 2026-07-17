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

# align joystick trajectory at given state.
def get_js_rot(trial_labels, state):
    l_frames = 500
    r_frames = 500
    interval = 10
    # read data.
    js_rot = np.array(trial_labels['js_rot'])
    js_time = np.array(trial_labels['js_time'])
    time_trial_start = np.array(trial_labels['time_trial_start'])
    time_state = np.stack(trial_labels[state])
    # get common time.
    inter_time = np.arange(-5*l_frames*interval, 5*r_frames*interval, interval)
    # interpolation.
    inter_rot = []
    for (jr, jt) in zip(js_rot, js_time):
        if ~np.isnan(np.sum(jt)):
            interpolator = interp1d(jt, jr, bounds_error=False)
            new_rot = interpolator(inter_time)
            inter_rot.append(new_rot)
        else:
            inter_rot.append(np.full(inter_time.shape, np.nan))
    # get alignments.
    align_rot = []
    align_time = []
    # loop over trials.
    for si,ir in enumerate(inter_rot):
        idx = np.searchsorted(inter_time, time_state[si,0]-time_trial_start[si])
        if idx > l_frames and idx < len(inter_time)-r_frames:
            r = ir[idx-l_frames : idx+r_frames]
            t = inter_time[idx-l_frames : idx+r_frames] - inter_time[idx]
            align_rot.append(r)
            align_time.append(t)
        else:
            align_rot.append(np.full((l_frames+r_frames), np.nan))
            align_time.append(np.full((l_frames+r_frames), np.nan))
    # correct data centering at zero.
    align_time_zero = np.array([np.argmin(np.abs(t)) if ~np.isnan(np.sum(t)) else np.nan for t in align_time])
    align_rot = np.stack(trim_seq(align_rot, align_time_zero))
    align_time = np.nanmean(np.stack(trim_seq(align_time, align_time_zero)),axis=0)
    return [align_rot, align_time]

# run alignment for all sessions
def run_get_js_rot(list_trial_labels, state):
    # run alignment for each session.
    list_js_rot = []
    list_js_time = []
    for si,trial_labels in enumerate(list_trial_labels):
        print(f'Aligning trials for session {si+1}/{len(list_trial_labels)}')
        [align_rot, align_time] = get_js_rot(trial_labels, state)
        list_js_rot.append(align_rot)
        list_js_time.append(align_time)
    # combine time stamps.
    js_time = np.nanmean(np.stack(list_js_time),axis=0)
    # combine results.
    js_alignment = {
        'js_time': js_time,
        'list_js_rot': list_js_rot,
        }
    return js_alignment













