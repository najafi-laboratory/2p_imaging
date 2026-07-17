#!/usr/bin/env python3

import gc
import os
import copy
import h5py
import shutil
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from scipy.signal import savgol_filter

# filter configuration list for data reading.
def filter_session_config_list(session_config_list, target_sess):
    sub_session_config_list = copy.deepcopy(session_config_list)
    for si in range(len(sub_session_config_list['list_config'])):
        sub_session_config_list['list_config'][si]['list_session_name'] = {
            k: v for k, v in session_config_list['list_config'][si]['list_session_name'].items()
            if v == target_sess}
    return sub_session_config_list

# create a numpy memmap from an h5py dataset.
def create_memmap(data, dtype, mmap_path):
    memmap_arr = np.array(data, dtype=dtype)
    return memmap_arr

# create folder for h5 data.
def get_memmap_path(ops, h5_file_name):
    mm_folder_name, _ = os.path.splitext(h5_file_name)
    mm_path = os.path.join(ops['save_path0'], 'memmap', mm_folder_name)
    file_path = os.path.join(ops['save_path0'], h5_file_name)
    return mm_path, file_path

# read ops.npy
def read_ops(list_session_data_path):
    list_ops = []
    for session_data_path in list_session_data_path:
        ops = np.load(
            os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
            allow_pickle=True).item()
        ops['save_path0'] = os.path.join(session_data_path)
        list_ops.append(ops)
    return list_ops

# read raw_voltages.h5.
def read_raw_voltages(ops):
    mm_path, file_path = get_memmap_path(ops, 'raw_voltages.h5')
    with h5py.File(file_path, 'r') as f:
        vol_time     = create_memmap(f['raw']['vol_time'],     'float32', os.path.join(mm_path, 'vol_time.mmap'))
        vol_start    = create_memmap(f['raw']['vol_start'],    'int8',    os.path.join(mm_path, 'vol_start.mmap'))
        vol_stim_vis = create_memmap(f['raw']['vol_stim_vis'], 'int8',    os.path.join(mm_path, 'vol_stim_vis.mmap'))
        vol_hifi     = create_memmap(f['raw']['vol_hifi'],     'int8',    os.path.join(mm_path, 'vol_hifi.mmap'))
        vol_img      = create_memmap(f['raw']['vol_img'],      'int8',    os.path.join(mm_path, 'vol_img.mmap'))
        vol_stim_aud = create_memmap(f['raw']['vol_stim_aud'], 'float32', os.path.join(mm_path, 'vol_stim_aud.mmap'))
        vol_flir     = create_memmap(f['raw']['vol_flir'],     'int8',    os.path.join(mm_path, 'vol_flir.mmap'))
        vol_pmt      = create_memmap(f['raw']['vol_pmt'],      'int8',    os.path.join(mm_path, 'vol_pmt.mmap'))
        vol_led      = create_memmap(f['raw']['vol_led'],      'int8',    os.path.join(mm_path, 'vol_led.mmap'))
    return [vol_time, vol_start, vol_stim_vis, vol_img,
            vol_hifi, vol_stim_aud, vol_flir,
            vol_pmt, vol_led]

# read dff traces.
def read_dff(ops):
    mm_path, file_path = get_memmap_path(ops, 'dff.h5')
    with h5py.File(file_path, 'r') as f:
        dff = create_memmap(f['dff'], 'float32', os.path.join(mm_path, 'dff.mmap'))
    return dff

# read camera dlc results.
def read_camera(ops, norm=True):
    window_length = 9
    polyorder = 2
    try:
        h5_file_name = [f for f in os.listdir(ops['save_path0']) if 'camera' in f][0]
        mm_path, file_path = get_memmap_path(ops, h5_file_name)
        with h5py.File(file_path, 'r') as f:
            camera_pupil = np.array(f['camera_dlc']['pupil'], dtype='float32')
            camera_pupil = savgol_filter(camera_pupil, window_length=window_length, polyorder=polyorder)
            if norm:
                camera_pupil = (camera_pupil - np.nanmin(camera_pupil)) / (np.nanmax(camera_pupil) - np.nanmin(camera_pupil) + 1e-5)
    except:
        camera_pupil = np.array([np.nan], dtype='float32')
    return camera_pupil

# read masks.
def read_masks(ops):
    mm_path, file_path = get_memmap_path(ops, 'masks.h5')
    with h5py.File(file_path, 'r') as f:
        labels     = create_memmap(f['labels'],     'int8',    os.path.join(mm_path, 'labels.mmap'))
        masks      = create_memmap(f['masks_func'], 'float32', os.path.join(mm_path, 'masks_func.mmap'))
        mean_func  = create_memmap(f['mean_func'],  'float32', os.path.join(mm_path, 'mean_func.mmap'))
        max_func   = create_memmap(f['max_func'],   'float32', os.path.join(mm_path, 'max_func.mmap'))
        mean_anat  = create_memmap(f['mean_anat'],  'float32', os.path.join(mm_path, 'mean_anat.mmap')) if ops['nchannels'] == 2 else None
        masks_anat = create_memmap(f['masks_anat'], 'float32', os.path.join(mm_path, 'masks_anat.mmap')) if ops['nchannels'] == 2 else None
    return [labels, masks, mean_func, max_func, mean_anat, masks_anat]

# read trial label csv file into dataframe.
def read_trial_label(ops):
    raw_csv = pd.read_csv(os.path.join(ops['save_path0'], 'trial_labels.csv'), index_col=0)
    # recover object numpy array from csv str.
    def object_parse(k, shape):
        arr = np.array(
            [np.fromstring(s.replace('[', '').replace(']', ''), sep=' ').reshape(shape)
             for s in raw_csv[k].to_list()] + ['yicong_forever'],
            dtype='object')[:-1]
        return arr
    # parse all array.
    time_trial_start = raw_csv['time_trial_start'].to_numpy(dtype='float32')
    time_trial_end = raw_csv['time_trial_end'].to_numpy(dtype='float32')
    trial_type = raw_csv['trial_type'].to_numpy(dtype='int8')
    probe_trial = raw_csv['probe_trial'].to_numpy(dtype='int8')
    delay = raw_csv['delay'].to_numpy(dtype='float32')
    outcome = raw_csv['outcome'].to_numpy(dtype='object')
    state_vis1 = object_parse('state_vis1', [-1])
    state_press1 = object_parse('state_press1', [-1])
    state_retract1 = object_parse('state_retract1', [-1])
    state_delay = object_parse('state_delay', [-1])
    state_vis2 = object_parse('state_vis2', [-1])
    state_press2 = object_parse('state_press2', [-1])
    state_retract2 = object_parse('state_retract2', [-1])
    state_reward = object_parse('state_reward', [-1])
    state_iti = object_parse('state_iti', [-1])
    lick = object_parse('lick', [-1])
    js_rot = object_parse('js_rot', [-1])
    js_time = object_parse('js_time', [-1])
    # convert to dataframe.
    trial_labels = pd.DataFrame({
        'time_trial_start': time_trial_start,
        'time_trial_end': time_trial_end,
        'trial_type': trial_type,
        'probe_trial': probe_trial,
        'delay': delay,
        'outcome': outcome,
        'state_vis1': state_vis1,
        'state_press1': state_press1,
        'state_retract1': state_retract1,
        'state_delay': state_delay,
        'state_vis2': state_vis2,
        'state_press2': state_press2,
        'state_retract2': state_retract2,
        'state_reward': state_reward,
        'state_iti': state_iti,
        'lick': lick,
        'js_rot': js_rot,
        'js_time': js_time,
        })
    return trial_labels

# read trailized neural traces with stimulus alignment.
def read_neural_trials(ops, smooth):
    mm_path, file_path = get_memmap_path(ops, 'neural_trials.h5')
    trial_labels = read_trial_label(ops)
    with h5py.File(file_path, 'r') as f:
        neural_trials = dict()
        dff = np.array(f['neural_trials']['dff'])
        if smooth:
            window_length=9
            polyorder=3
            dff = np.apply_along_axis(
                savgol_filter, 1, dff,
                window_length=window_length,
                polyorder=polyorder)
        else: pass
        neural_trials['dff']          = create_memmap(dff,                                'float32', os.path.join(mm_path, 'dff.mmap'))
        neural_trials['time']         = create_memmap(f['neural_trials']['time'],         'float32', os.path.join(mm_path, 'time.mmap'))
        neural_trials['trial_labels'] = trial_labels
        neural_trials['vol_time']     = create_memmap(f['neural_trials']['vol_time'],     'float32', os.path.join(mm_path, 'vol_time.mmap'))
        neural_trials['vol_stim_vis'] = create_memmap(f['neural_trials']['vol_stim_vis'], 'int8',    os.path.join(mm_path, 'vol_stim_vis.mmap'))
        neural_trials['vol_stim_aud'] = create_memmap(f['neural_trials']['vol_stim_aud'], 'float32', os.path.join(mm_path, 'vol_stim_aud.mmap'))
        neural_trials['vol_flir']     = create_memmap(f['neural_trials']['vol_flir'],     'int8',    os.path.join(mm_path, 'vol_flir.mmap'))
        neural_trials['vol_pmt']      = create_memmap(f['neural_trials']['vol_pmt'],      'int8',    os.path.join(mm_path, 'vol_pmt.mmap'))
        neural_trials['vol_led']      = create_memmap(f['neural_trials']['vol_led'],      'int8',    os.path.join(mm_path, 'vol_led.mmap'))
    return neural_trials

# read bpod session data.
def read_bpod_mat_data(ops, session_start_time):
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
        if 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
            outcome = 'reward'
        elif 'DidNotPress1' in trial_states.keys() and not np.isnan(trial_states['DidNotPress1'][0]):
            outcome = 'no_1st_press'
        elif 'DidNotPress2' in trial_states.keys() and not np.isnan(trial_states['DidNotPress2'][0]):
            outcome = 'no_2nd_press'
        elif 'EarlyPress2' in trial_states.keys() and not np.isnan(trial_states['EarlyPress2'][0]):
            outcome = 'early_2nd_press'
        elif 'LatePress2' in trial_states.keys() and not np.isnan(trial_states['LatePress2'][0]):
            outcome = 'late_2nd_press'
        else:
            outcome = 'other'
        return outcome
    # find state timing.
    def get_state(trial_state_dict, target_state, trial_start):
        if target_state in trial_state_dict:
            time_state = 1000*np.array(trial_state_dict[target_state]) + trial_start
        else:
            time_state = np.array([np.nan, np.nan])
        return time_state
    # read raw data.
    raw = sio.loadmat(
        os.path.join(ops['save_path0'], 'bpod_session_data.mat'),
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
    # trial type.
    trial_labels['trial_type'] = np.array(raw['TrialTypes']).reshape(-1)
    trial_labels['probe_trial'] = np.array(raw['ProbeTrial']).reshape(-1)
    # press delay.
    trial_labels['delay'] = 1000*np.array(raw['PrePress2Delay']).reshape(-1)
    # trial outcomes.
    trial_labels['outcome'] = np.array([states_labeling(ts) for ts in trial_states], dtype='object')
    # visual cue 1.
    trial_labels['state_vis1'] = np.array([
        get_state(trial_states[ti], 'VisualStimulus1', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    # press 1.
    trial_labels['state_press1'] = np.array([
        get_state(trial_states[ti], 'Press1', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    # retract 1.
    trial_labels['state_retract1'] = np.array([
        get_state(trial_states[ti], 'LeverRetract1', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    # delay.
    state_delay_st = np.stack([get_state(trial_states[ti], 'PrePress2Delay', trial_labels['time_trial_start'][ti]) for ti in range(n_trials)])
    state_delay_vg = np.stack([get_state(trial_states[ti], 'PreVis2Delay', trial_labels['time_trial_start'][ti]) for ti in range(n_trials)])
    trial_labels['state_delay'] = np.array([
        r for r in np.nanmax(np.stack([state_delay_st, state_delay_vg]), axis=0)] + ['yicong_forever'], dtype='object')[:-1]
    # visual cue 2.
    trial_labels['state_vis2'] = np.array([
        get_state(trial_states[ti], 'VisualStimulus2', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    # press 2.
    state_press2       = np.stack([get_state(trial_states[ti], 'Press2',      trial_labels['time_trial_start'][ti]) for ti in range(n_trials)])
    state_early_press2 = np.stack([get_state(trial_states[ti], 'EarlyPress2', trial_labels['time_trial_start'][ti]) for ti in range(n_trials)])
    state_late_press2  = np.stack([get_state(trial_states[ti], 'LatePress2',  trial_labels['time_trial_start'][ti]) for ti in range(n_trials)])
    trial_labels['state_press2'] = np.array([
        r for r in np.nanmax(np.stack([state_press2, state_early_press2, state_late_press2]), axis=0)] + ['yicong_forever'], dtype='object')[:-1]
    # retract 2.
    trial_labels['state_retract2'] = np.array([
        get_state(trial_states[ti], 'LeverRetractFinal', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    # reward.
    trial_labels['state_reward'] = np.array([
        get_state(trial_states[ti], 'Reward', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    # iti.
    iti_punish = np.stack([get_state(trial_states[ti], 'Punish_ITI', trial_labels['time_trial_start'][ti]) for ti in range(n_trials)])
    iti_reward = np.stack([get_state(trial_states[ti], 'ITI', trial_labels['time_trial_start'][ti]) for ti in range(n_trials)])
    trial_labels['state_iti'] = np.array([
        r for r in np.nanmax(np.stack([iti_punish, iti_reward]), axis=0)] + ['yicong_forever'], dtype='object')[:-1]
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
        t = 1000*np.array(raw['EncoderData'][ti]['Times'])
        if np.abs(r[0])>0.5 or np.abs(t[0])>1e-5 or len(r)<5:
            r = np.array([np.nan,np.nan])
            t = np.array([np.nan,np.nan])
        js_rot.append(r)
        js_time.append(t + trial_labels['time_trial_start'][ti])
    trial_labels['js_rot'] = np.array(js_rot + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['js_time'] = np.array(js_time + ['yicong_forever'], dtype='object')[:-1]
    # convert to dataframe.
    trial_labels = pd.DataFrame(trial_labels)
    return trial_labels

# get session results for one subject.
def read_subject(list_ops, force_label, smooth):
    list_labels = []
    list_masks = []
    list_neural_trials = []
    for ops in tqdm(list_ops):
        if not os.path.exists(os.path.join(ops['save_path0'], 'memmap')):
            os.makedirs(os.path.join(ops['save_path0'], 'memmap'))
        # masks.
        labels, masks, mean_func, max_func, mean_anat, masks_anat = read_masks(ops)
        # trials.
        neural_trials = read_neural_trials(ops, smooth)
        # labels.
        if force_label != None:
            labels = np.ones_like(labels) * force_label
        # append to list.
        list_labels.append(labels)
        list_masks.append(
            [masks,
             mean_func, max_func,
             mean_anat, masks_anat])
        list_neural_trials.append(neural_trials)
        # clear memory usages.
        del labels
        del masks, mean_func, max_func, mean_anat, masks_anat
        del neural_trials
        gc.collect()
    return [list_labels, list_masks, list_neural_trials]

# get session results for all subject.
def read_all(session_config_list, smooth):
    list_labels = []
    list_masks = []
    list_neural_trials = []
    for i in range(len(session_config_list['list_config'])):
        # read ops for each subject.
        print('Reading subject {}/{}'.format(i+1, len(session_config_list['list_config'])))
        list_session_data_path = [
            os.path.join('results', session_config_list['list_config'][i]['session_folder'], n)
            for n in session_config_list['list_config'][i]['list_session_name']]
        list_ops = read_ops(list_session_data_path)
        # read results for each subject.
        labels, masks, neural_trials = read_subject(
             list_ops,
             force_label=session_config_list['list_config'][i]['force_label'],
             smooth=smooth)
        # append to list.
        list_labels += labels
        list_masks += masks
        list_neural_trials += neural_trials
        # clear memory usages.
        del labels
        del masks
        del neural_trials
        gc.collect()
    return [list_labels, list_masks, list_neural_trials]
