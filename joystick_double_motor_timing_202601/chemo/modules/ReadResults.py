#!/usr/bin/env python3

import gc
import os
import scipy.io as sio
import pandas as pd
import numpy as np
from tqdm import tqdm

session_data_path = './session_data'

# read bpod session data.
def read_bpod_mat_data(subject_name, fname, session_start_time=0):
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], sio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d
    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d
    def _tolist(ndarray):
        elem_list = []
        if ndarray.ndim == 0:
            return ndarray
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    # labeling every trials for a subject
    def states_labeling(trial_states):
        if 'Assist' in trial_states.keys() and not np.isnan(trial_states['Assist'][0]):
            outcome = 'assist'
        elif 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
            outcome = 'reward'
        elif 'DidNotPress1' in trial_states.keys() and not np.isnan(trial_states['DidNotPress1'][0]):
            outcome = 'no_1st_press'
        elif 'DidNotPress2' in trial_states.keys() and not np.isnan(trial_states['DidNotPress2'][0]):
            outcome = 'no_2nd_press'
        elif 'EarlyPress2' in trial_states.keys() and not np.isnan(trial_states['EarlyPress2'][0]):
            outcome = 'early_2nd_press'
        else:
            outcome = 'other'
        return outcome
    # find state timing.
    def get_state(trial_state_dict, target_state, trial_start):
        if target_state in trial_state_dict:
            time_state = 1000*np.array(trial_state_dict[target_state], dtype='float32') + trial_start
        else:
            time_state = np.array([np.nan, np.nan])
        return time_state
    # read raw data.
    raw = sio.loadmat(
        os.path.join(session_data_path, subject_name, fname),
        struct_as_record=False, squeeze_me=True)
    raw = _check_keys(raw)['SessionData']
    trial_labels = dict()
    n_trials = raw['nTrials']
    trial_states = [raw['RawEvents']['Trial'][ti]['States'] for ti in range(n_trials)]
    trial_events = [raw['RawEvents']['Trial'][ti]['Events'] for ti in range(n_trials)]
    # trial start time stamps.
    trial_labels['time_trial_start'] = 1000*np.array(raw['TrialStartTimestamp']).reshape(-1)
    # trial end time stamps.
    trial_labels['time_trial_end'] = 1000*np.array(raw['TrialEndTimestamp']).reshape(-1)
    # correct timestamps starting from session start.
    trial_labels['time_trial_end'] = trial_labels['time_trial_end'] - trial_labels['time_trial_start'][0] + session_start_time
    trial_labels['time_trial_start'] = trial_labels['time_trial_start'] - trial_labels['time_trial_start'][0] + session_start_time
    # session type.
    trial_labels['chemo_type'] = np.array([raw['TrialSettings'][ti]['GUI']['ChemogeneticSession'] for ti in range(n_trials)]).reshape(-1)
    # trial type.
    trial_labels['trial_type'] = np.array(raw['TrialTypes']).reshape(-1)
    trial_labels['probe_trial'] = np.array(raw['ProbeTrial']).reshape(-1)
    if 'AssistTrial' in raw.keys() and len(np.array(raw['AssistTrial']).reshape(-1)) > 1:
        trial_labels['assist_trial'] = np.array(raw['AssistTrial']).reshape(-1)
    else:
        trial_labels['assist_trial'] = np.full(trial_labels['trial_type'].shape[0], 0)
    # press delay.
    trial_labels['delay'] = 1000*np.array(raw['PrePress2Delay']).reshape(-1)
    # trial outcomes.
    trial_labels['outcome'] = np.array([states_labeling(ts) for ts in trial_states], dtype='object')
    # trial state timings.
    trial_labels['state_vis1'] = np.array([
        get_state(trial_states[ti], 'VisualStimulus1', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['state_press1'] = np.array([
        get_state(trial_states[ti], 'Press1', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['state_retract1'] = np.array([
        get_state(trial_states[ti], 'LeverRetract1', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['state_delay'] = np.array([
        get_state(trial_states[ti], 'PrePress2Delay', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    state_press2       = np.stack([get_state(trial_states[ti], 'Press2',      trial_labels['time_trial_start'][ti]) for ti in range(n_trials)])
    state_early_press2 = np.stack([get_state(trial_states[ti], 'EarlyPress2', trial_labels['time_trial_start'][ti]) for ti in range(n_trials)])
    state_late_press2  = np.stack([get_state(trial_states[ti], 'LatePress2',  trial_labels['time_trial_start'][ti]) for ti in range(n_trials)])
    trial_labels['state_press2'] = np.array([
        r for r in np.nanmax(np.stack([state_press2, state_early_press2, state_late_press2]), axis=0)] + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['state_retract2'] = np.array([
        get_state(trial_states[ti], 'LeverRetractFinal', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['state_reward'] = np.array([
        get_state(trial_states[ti], 'Reward', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['state_punish_iti'] = np.array([
        get_state(trial_states[ti], 'Punish_ITI', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['state_iti'] = np.array([
        get_state(trial_states[ti], 'ITI', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    # licking.
    lick = []
    for ti in range(n_trials):
        if 'Port2In' in trial_events[ti].keys():
            lick_all = 1000*np.array(trial_events[ti]['Port2In']).reshape(1,-1)
            lick_label = np.zeros_like(lick_all).reshape(1,-1)
            lick_label[lick_all>1000*np.array(trial_states[ti]['Reward'][0])] = 1
            lick.append(np.concatenate((lick_all, lick_label), axis=0))
        else:
            lick.append(np.array([[np.nan],[np.nan]]))
    trial_labels['lick'] = np.array(lick + ['yicong_forever'], dtype='object')[:-1]
    # joystick deflection.
    js_rot = []
    js_time = []
    for ti in range(n_trials):
        r = np.array(raw['EncoderData'][ti]['Positions'])
        t = 1000*np.array(raw['EncoderData'][ti]['Times']) + session_start_time
        if np.abs(r[0])>0.5 or np.abs(t[0])>1e-5 or len(r)<5:
            r = np.array([np.nan,np.nan])
            t = np.array([np.nan,np.nan])
        js_rot.append(r)
        js_time.append(t)
    trial_labels['js_rot'] = np.array(js_rot + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['js_time'] = np.array(js_time + ['yicong_forever'], dtype='object')[:-1]
    # convert to dataframe.
    trial_labels = pd.DataFrame(trial_labels)
    return trial_labels

# get session results for one subject.
def read_subject(subject_name):
    list_trial_labels = []
    # get file names and sort.
    list_sub_fname = [f for f in os.listdir(os.path.join('session_data', subject_name)) if subject_name in f and '.mat' in f]
    list_sub_fname.sort(key=lambda s: s[-19:-4])
    list_sess_time = [f[-19:-11] for f in list_sub_fname]
    # read files.
    for fname in tqdm(list_sub_fname):
        trial_labels = read_bpod_mat_data(subject_name, fname, session_start_time=0)
        list_trial_labels.append(trial_labels)
        # clear memory usages.
        del trial_labels
        gc.collect()
    # summarize session types.
    session_types = []
    for sess_time, trial_labels in zip(list_sess_time, list_trial_labels):
        chemo_fraction = np.sum(np.array(trial_labels['chemo_type']) == 1) / len(trial_labels)
        if chemo_fraction > 0.8:
            session_type = 'chemo'
        else:
            session_type = 'control'
        session_types.append((session_type, sess_time))
    print('session_type | session')
    print('-------------|--------')
    for session_type, sess_time in session_types:
        print(f'{session_type:<12} | {sess_time}')
    return list_sess_time, list_trial_labels

