#!/usr/bin/env python3

import numpy as np


LIST_STATES = [
    'state_vis1', 'state_press1', 'state_delay',
    'state_press2', 'state_reward', 'state_iti',
]


# Align trials into preallocated arrays, preserving trial order for filtering.
def get_task_response(neural_trials, target_state, l_frames=3000, r_frames=3000):
    time = neural_trials['time']
    dff = neural_trials['dff']
    labels = neural_trials['trial_labels']
    states = (np.stack(labels[target_state]).astype(float) if len(labels)
              else np.empty((0, 2)))
    starts = states[:, 0]
    indices = np.searchsorted(time, starts)
    valid = (np.isfinite(starts) & (indices >= l_frames)
             & (indices <= len(time)-r_frames) & (indices < len(time)))
    n_trials, n_frames = len(states), l_frames+r_frames
    neu_seq = np.full((n_trials, dff.shape[0], n_frames), np.nan,
                      dtype=np.result_type(dff.dtype, np.float32))
    js_rot = np.full((n_trials, n_frames), np.nan)
    outcome = np.full(n_trials, 'nan', dtype=object)
    time_sum = np.zeros(n_frames)
    state_time = np.full(states.shape[1], np.nan)
    js_times = labels['js_time'].to_numpy()
    js_values = labels['js_rot'].to_numpy()
    outcomes = labels['outcome'].to_numpy()
    for ti in np.flatnonzero(valid):
        idx = indices[ti]
        window = slice(idx-l_frames, idx+r_frames)
        sample_time = time[window]
        neu_seq[ti] = dff[:, window]
        time_sum += sample_time-time[idx]
        js_rot[ti] = np.interp(sample_time, js_times[ti], js_values[ti],
                               left=np.nan, right=np.nan)
        outcome[ti] = outcomes[ti]
    if np.any(valid):
        neu_time = time_sum / valid.sum()
        state_time = np.nanmean(states[valid]-starts[valid, None], axis=0)
    else:
        neu_time = np.full(n_frames, np.nan)
    return [neu_seq, neu_time, state_time, js_rot, outcome]


# Use a common frame window across sessions, sized to the plotted time range.
def run_sess_alignment(list_neural_trials, window_ms=2000, states=None):
    states = LIST_STATES if states is None else states
    frames = max(
        int(np.ceil(window_ms / np.median(np.diff(nt['time'])))) + 2
        for nt in list_neural_trials)
    alignment = {}
    for ti, state in enumerate(states):
        print(f'Aligning trials for state {ti+1}/{len(states)} {state}')
        responses = [get_task_response(nt, state, frames, frames)
                     for nt in list_neural_trials]
        neu_seq, neu_time, state_time, js_rot, outcome = zip(*responses)
        alignment[state] = {
            'list_neu_seq': list(neu_seq),
            'neu_time': np.nanmean(np.stack(neu_time), axis=0),
            'state_time': np.nanmean(np.stack(state_time), axis=0),
            'list_js_rot': list(js_rot),
            'list_outcome': list(outcome),
        }
    return alignment


def get_lick_times(lick):
    times = np.asarray(lick, dtype=float)
    if times.ndim == 2 and times.shape[0] == 2:
        return times[0]
    times = times.reshape(-1)
    # CSV flattens timestamp/flag rows; older sessions store timestamps only.
    if times.size and times.size % 2 == 0:
        half = times.size // 2
        if np.all(np.isin(times[half:], [0, 1])):
            return times[:half]
    return times


# Expand trial labels once per post-press lick; retain outcome and trial type.
def run_lick_alignment(list_neural_trials, window_ms=2000):
    sessions, list_lick_labels = [], []
    for nt in list_neural_trials:
        labels = nt['trial_labels']
        trial_indices, lick_times = [], []
        for ti in range(len(labels)):
            trial = labels.iloc[ti]
            press2 = trial['state_press2'][0]
            times = get_lick_times(trial['lick'])
            times = times[np.isfinite(times) & (times > press2)
                          & (times <= trial['time_trial_end'])]
            trial_indices.extend([ti] * len(times))
            lick_times.extend(times)
        lick_labels = labels.iloc[trial_indices].reset_index(drop=True).copy()
        lick_labels['state_lick'] = [np.array([t, t]) for t in lick_times]
        sessions.append(dict(nt, trial_labels=lick_labels))
        list_lick_labels.append(lick_labels)
    return run_sess_alignment(sessions, window_ms, ['state_lick']), list_lick_labels
