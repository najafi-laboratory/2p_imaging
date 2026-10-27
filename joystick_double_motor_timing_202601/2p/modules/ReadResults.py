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
        # Photometry exports contain 1-D dummy arrays, not spatial FOV data.
        if f['masks_func'].ndim != 2:
            return [labels, None, None, None, None, None]
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
    # simplify_cells handles MATLAB structs/cells, including single-trial sessions.
    raw = sio.loadmat(
        os.path.join(ops['save_path0'], 'bpod_session_data.mat'),
        simplify_cells=True)['SessionData']
    n_trials = int(raw['nTrials'])

    def records(value):
        return [value] if isinstance(value, dict) else list(value)

    trials = records(raw['RawEvents']['Trial'])[:n_trials]
    encoders = records(raw['EncoderData'])[:n_trials]
    settings = records(raw.get('TrialSettings', {}))

    def vector(name, default=np.nan):
        values = np.asarray(raw.get(name, np.full(n_trials, default))).reshape(-1)
        return values[:n_trials]

    def state(states, *names):
        # Select one visited interval; never combine endpoints from different states.
        for name in names:
            values = np.asarray(states.get(name, []), dtype=float)
            if values.size:
                intervals = values.reshape(-1, 2)
                valid = intervals[np.isfinite(intervals[:, 0])]
                if len(valid):
                    return valid[0].copy()
        return np.full(2, np.nan)

    starts = vector('TrialStartTimestamp') * 1000
    offset = session_start_time - starts[0]
    labels = {
        'time_trial_start': starts + offset,
        'time_trial_end': vector('TrialEndTimestamp') * 1000 + offset,
        'trial_type': vector('TrialTypes'),
        'probe_trial': vector('ProbeTrialTypes') if 'ProbeTrialTypes' in raw else vector('ProbeTrial', 0),
        'delay': [], 'outcome': [],
    }
    state_names = {
        'state_vis1': ('SensoryCue1', 'VisualStimulus1'),
        'state_press1': ('Press1',),
        'state_retract1': ('LeverRetract1',),
        'state_delay': ('PrePress2Delay', 'PreVis2Delay'),
        'state_vis2': ('VisualStimulus2',),
        'state_press2': ('Press2', 'EarlyPress2', 'Press2Late', 'LatePress2'),
        'state_retract2': ('RewardLeverRetract', 'LeverRetractFinal'),
        'state_reward': ('Reward',),
        'state_iti': ('Punish_ITI', 'ITI'),
    }
    for key in (*state_names, 'lick', 'js_rot', 'js_time'):
        labels[key] = []

    for ti, trial in enumerate(trials):
        states = trial['States']
        events = trial.get('Events', {})
        gui = settings[min(ti, len(settings) - 1)].get('GUI', {})
        modern = 'SensoryCue1' in states
        if 'PrePress2Delay' in raw:
            delay = float(vector('PrePress2Delay')[ti])
        else:
            delay = float(gui['ShortDelay_s' if labels['trial_type'][ti] == 1 else 'LongDelay_s'])
        labels['delay'].append(delay * 1000)
        outcome = 'other'
        for names, value in [
                (('Reward',), 'reward'),
                (('DidNotPress1',), 'no_1st_press'),
                (('DidNotPress2',), 'no_2nd_press'),
                (('EarlyPress2',), 'early_2nd_press'),
                (('Press2Late', 'LatePress2'), 'late_2nd_press')]:
            if np.isfinite(state(states, *names)[0]):
                outcome = value
                break
        labels['outcome'].append(outcome)
        intervals = {key: state(states, *names) for key, names in state_names.items()}
        if modern:
            # Delay and cue 2 are driven by global timer 1, not named states.
            # Assist starts the timer in PrePress2Delay; other trials in WaitForPress2.
            assist = bool(vector('AssistTrial', 0)[ti])
            phase = state(states, 'PrePress2Delay' if assist else 'WaitForPress2')[0]
            cue_start = phase + delay
            stop = state(states, 'Press2', 'DidNotPress2')[0]
            intervals['state_delay'] = np.array([phase, min(cue_start, stop)])
            intervals['state_vis2'] = np.full(2, np.nan)
            if gui.get('TimingMode') == 1 and np.isfinite(cue_start) and cue_start < stop:
                duration = float(gui['SensoryCueDuration_s'])
                intervals['state_vis2'] = np.array([cue_start, min(cue_start + duration, stop)])
        for key, interval in intervals.items():
            labels[key].append(interval * 1000 + labels['time_trial_start'][ti])

        lick_times = np.asarray(events.get('Port2In', [np.nan]), dtype=float).reshape(-1) * 1000
        reward_start = state(states, 'Reward')[0] * 1000
        # Licks and all other event times use the same session clock.
        labels['lick'].append(np.vstack((
            lick_times + labels['time_trial_start'][ti],
            (lick_times > reward_start).astype(float))))
        r = np.asarray(encoders[ti]['Positions'], dtype=float).reshape(-1)
        t = np.asarray(encoders[ti]['Times'], dtype=float).reshape(-1) * 1000
        if len(r) < 2 or len(r) != len(t) or abs(r[0]) > 0.5 or abs(t[0]) > 1e-5:
            r = np.full(2, np.nan)
            t = np.full(2, np.nan)
        labels['js_rot'].append(r)
        labels['js_time'].append(t + labels['time_trial_start'][ti])
    return pd.DataFrame(labels)

# get session results for one subject.
def read_subject(list_ops, force_label, smooth, roi_manual_label_list=None):
    list_labels = []
    list_masks = []
    list_neural_trials = []
    for ops in tqdm(list_ops):
        if not os.path.exists(os.path.join(ops['save_path0'], 'memmap')):
            os.makedirs(os.path.join(ops['save_path0'], 'memmap'))
        # trials.
        neural_trials = read_neural_trials(ops, smooth)
        # FOV data is optional; retain trace labels for sessions without masks.
        if os.path.isfile(os.path.join(ops['save_path0'], 'masks.h5')):
            labels, masks, mean_func, max_func, mean_anat, masks_anat = read_masks(ops)
        else:
            labels = np.full(neural_trials['dff'].shape[0], -1, dtype='int8')
            masks = mean_func = max_func = mean_anat = masks_anat = None
        # labels.
        if force_label != None:
            labels = np.ones_like(labels) * force_label
        if roi_manual_label_list is not None:
            manual_labels = np.load(
                os.path.join(ops['save_path0'], 'roi_manual_labels.npy'),
                allow_pickle=False).reshape(-1)
            if manual_labels.size != labels.size or manual_labels.size != neural_trials['dff'].shape[0]:
                raise ValueError('ROI QC label count does not match masks/traces in {}'.format(ops['save_path0']))
            keep = np.isin(manual_labels, roi_manual_label_list)
            labels = labels.reshape(-1)[keep]
            neural_trials['dff'] = neural_trials['dff'][keep, :]
            # Mask IDs are 1-based and must follow the filtered trace order.
            if masks is not None:
                if (not np.all(np.isfinite(masks)) or
                        np.any(masks < 0) or np.any(masks > keep.size) or
                        np.any(masks != np.floor(masks))):
                    raise ValueError('FOV mask IDs do not match ROI labels in {}'.format(ops['save_path0']))
                mask_ids = np.zeros(keep.size + 1, dtype=masks.dtype)
                mask_ids[np.flatnonzero(keep) + 1] = np.arange(1, np.count_nonzero(keep) + 1)
                masks = mask_ids[masks.astype(int)]
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
             smooth=smooth,
             roi_manual_label_list=session_config_list.get('roi_manual_label_list'))
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
