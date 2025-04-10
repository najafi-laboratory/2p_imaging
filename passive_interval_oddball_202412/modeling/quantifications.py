#!/usr/bin/env python3

import numpy as np
from modeling.utils import get_frame_idx_from_time

'''
ci = 15
nsm, _ = get_mean_sem(neu_seq_long[cluster_id==ci,:])
neu_time = alignment['neu_time']
win_eval = [stim_seq_long[c_idx-1,0], stim_seq_long[c_idx+1,1]]
'''
def get_response_onset(nsm, neu_time, win_eval):
    pct = 25
    window_size = 3
    max_attempts = 4
    threshold_factor = 0.2
    # apply time window to the data.
    win_mask = (neu_time >= win_eval[0]) & (neu_time <= win_eval[1])
    win_nsm = nsm[win_mask]
    wind_nt = neu_time[win_mask]
    # adjust target_index for windowed data.
    stim_onset_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
    # compute the derivative for windowed data.
    derivative = np.gradient(win_nsm, wind_nt)
    # compute initial threshold using pre-stimulus period.
    baseline_derivative = derivative[win_nsm<np.nanpercentile(win_nsm, pct)]
    threshold = abs(np.mean(baseline_derivative)) + abs(np.std(baseline_derivative))
    attempt = 0
    while attempt < max_attempts:
        # detect the ramping onset indices.
        ramping_onset_index = np.where(derivative > threshold)[0]
        if len(ramping_onset_index) > 0:
            # group consecutive indices.
            groups = []
            current_group = [ramping_onset_index[0]]
            # group consecutive indices.
            for i in range(1, len(ramping_onset_index)):
                if ramping_onset_index[i] == ramping_onset_index[i - 1] + 1:
                    current_group.append(ramping_onset_index[i])
                else:
                    # only add groups that meet the minimum window size.
                    if len(current_group) >= window_size:
                        groups.append(current_group)
                    current_group = [ramping_onset_index[i]]
            if len(current_group) >= window_size:
                groups.append(current_group)
            if not groups:
                threshold -= np.std(baseline_derivative) * threshold_factor
                attempt += 1
                continue
            # find the max length among all groups for normalization.
            max_group_length = max([len(group) for group in groups])
            # calculate metrics for remaining groups.
            group_metrics = []
            for group in groups:
                # calculate distance metric.
                dist_to_target = min(abs(np.array(group) - stim_onset_idx))
                distance_score = 1 / (1 + dist_to_target)
                # calculate intensity metric.
                group_derivatives = derivative[group]
                intensity_score = np.mean(group_derivatives) / np.max(derivative)
                # calculate length metric.
                length_score = len(group) / max_group_length
                # combined score.
                combined_score = (0.2* distance_score + 0.4* intensity_score + 0.4 * length_score)
                group_metrics.append({
                    'group': group,
                    'combined_score': combined_score,
                    'distance_score': distance_score,
                    'intensity_score': intensity_score,
                    'length_score': length_score
                })
            # sort groups by combined score.
            group_metrics.sort(key=lambda x: x['combined_score'], reverse=True)
            if group_metrics:
                # convert back to original index space.
                return np.where(win_mask)[0][group_metrics[0]['group'][0]]
        # reduce threshold and try again.
        threshold -= np.std(baseline_derivative) * threshold_factor
        attempt += 1
    return None

def get_response_latency(nsm, neu_time, win_eval):
    stim_onset_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
    response_onset_idx = get_response_onset(nsm, neu_time, win_eval)
    if response_onset_idx is None:
        latency = np.nan
    else:
        latency = neu_time[response_onset_idx] - neu_time[stim_onset_idx]
    return latency

def get_peak_time(nsm, neu_time, win_eval):
    stim_onset_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
    response_onset_idx = get_response_onset(nsm, neu_time, win_eval)
    if response_onset_idx is None:
        peak_time = np.nan
    else:
        start_idx, end_idx = get_frame_idx_from_time(neu_time, 0, win_eval[0], win_eval[1])
        resp_win = nsm[start_idx:end_idx]
        win_resp_onset_idx = response_onset_idx - start_idx
        if win_resp_onset_idx < 0 or win_resp_onset_idx >= len(resp_win):
            peak_time = np.nan
        else:
            post_onset = resp_win[win_resp_onset_idx:]
            peak_offset = int(np.argmax(post_onset))
            full_peak_idx = win_resp_onset_idx + peak_offset
            peak_time = neu_time[start_idx + full_peak_idx] - neu_time[stim_onset_idx]
    return peak_time

def get_peak_amp(nsm, neu_time, win_eval):
    response_onset_idx = get_response_onset(nsm, neu_time, win_eval)
    if response_onset_idx is None:
        peak_amp = np.nan
    else:
        start_idx, end_idx = get_frame_idx_from_time(neu_time, 0, win_eval[0], win_eval[1])
        resp_win = nsm[start_idx:end_idx]
        win_resp_onset_idx = response_onset_idx - start_idx
        if win_resp_onset_idx < 0 or win_resp_onset_idx >= len(resp_win):
            peak_amp = np.nan
        else:
            post_onset = resp_win[win_resp_onset_idx:]
            peak_offset = int(np.argmax(post_onset))
            full_peak_idx = win_resp_onset_idx + peak_offset
            peak_amp = resp_win[full_peak_idx]
    return peak_amp

def get_pre_stim_peak_time(nsm, neu_time, win_eval):
    stim_onset_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
    start_idx, _ = get_frame_idx_from_time(neu_time, 0, win_eval[0], win_eval[1])
    resp_win = nsm[start_idx:]
    win_stim_idx = stim_onset_idx - start_idx
    if win_stim_idx <= 0 or win_stim_idx > len(resp_win):
        pre_peak_time = np.nan
    else:
        pre_win = resp_win[:win_stim_idx]
        pre_peak_idx = int(np.argmax(pre_win))
        pre_peak_time = neu_time[start_idx + pre_peak_idx] - neu_time[stim_onset_idx]
    return pre_peak_time

def get_pre_stim_peak_amp(nsm, neu_time, win_eval):
    stim_onset_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
    start_idx, _ = get_frame_idx_from_time(neu_time, 0, win_eval[0], win_eval[1])
    resp_win = nsm[start_idx:]
    win_stim_idx = stim_onset_idx - start_idx
    if win_stim_idx <= 0 or win_stim_idx > len(resp_win):
        pre_peak_amp = np.nan
    else:
        pre_win = resp_win[:win_stim_idx]
        pre_peak_amp = np.max(pre_win)
    return pre_peak_amp

def get_rise_slope(nsm, neu_time, win_eval):
    response_onset_idx = get_response_onset(nsm, neu_time, win_eval)
    if response_onset_idx is None:
        rise_slope = np.nan
    else:
        start_idx, end_idx = get_frame_idx_from_time(neu_time, 0, win_eval[0], win_eval[1])
        resp_win = nsm[start_idx:end_idx]
        win_resp_onset_idx = response_onset_idx - start_idx
        if win_resp_onset_idx < 0 or win_resp_onset_idx >= len(resp_win):
            rise_slope = np.nan
        else:
            post_onset = resp_win[win_resp_onset_idx:]
            peak_offset = int(np.argmax(post_onset))
            full_peak_idx = win_resp_onset_idx + peak_offset
            if full_peak_idx <= win_resp_onset_idx:
                rise_slope = np.nan
            else:
                onset_value = resp_win[win_resp_onset_idx]
                full_peak_value = resp_win[full_peak_idx]
                rise_time = neu_time[start_idx + full_peak_idx] - neu_time[start_idx + win_resp_onset_idx]
                if rise_time == 0:
                    rise_slope = np.nan
                else:
                    rise_slope = (full_peak_value - onset_value) / rise_time
    return rise_slope

def get_decay_rate(nsm, neu_time, win_eval):
    response_onset_idx = get_response_onset(nsm, neu_time, win_eval)
    if response_onset_idx is None:
        decay_rate = 0
    else:
        start_idx, end_idx = get_frame_idx_from_time(neu_time, 0, win_eval[0], win_eval[1])
        resp_win = nsm[start_idx:end_idx]
        win_resp_onset_idx = response_onset_idx - start_idx
        if win_resp_onset_idx < 0 or win_resp_onset_idx >= len(resp_win):
            decay_rate = 0
        else:
            post_onset = resp_win[win_resp_onset_idx:]
            peak_offset = int(np.argmax(post_onset))
            full_peak_idx = win_resp_onset_idx + peak_offset
            if full_peak_idx >= len(resp_win) - 1:
                decay_rate = 0
            else:
                post_peak = resp_win[full_peak_idx:]
                t_segment = neu_time[start_idx + full_peak_idx:end_idx]
                if len(t_segment) == 0:
                    decay_rate = 0
                else:
                    t_rel = t_segment - t_segment[0]
                    try:
                        coeffs = np.polyfit(t_rel, post_peak, 1)
                        decay_rate = -coeffs[0]
                    except:
                        decay_rate = 0
    return decay_rate

def get_final_amp(nsm, neu_time, win_eval):
    response_onset_idx = get_response_onset(nsm, neu_time, win_eval)
    if response_onset_idx is None:
        final_amp = np.nan
    else:
        start_idx, end_idx = get_frame_idx_from_time(neu_time, 0, win_eval[0], win_eval[1])
        resp_win = nsm[start_idx:end_idx]
        win_resp_onset_idx = response_onset_idx - start_idx
        post_onset = resp_win[win_resp_onset_idx:]
        peak_offset = int(np.argmax(post_onset))
        full_peak_idx = win_resp_onset_idx + peak_offset
        if full_peak_idx < len(resp_win) - 1:
            final_amp = np.mean(resp_win[full_peak_idx:])
        else:
            final_amp = resp_win[full_peak_idx]
    return final_amp

def get_amp_reduction(nsm, neu_time, win_eval):
    peak_amp = get_peak_amp(nsm, neu_time, win_eval)
    final_amp = get_final_amp(nsm, neu_time, win_eval)
    if peak_amp == 0:
        reduction = np.nan
    else:
        reduction = (peak_amp - final_amp) / peak_amp
    return reduction

def get_inh_depth(nsm, neu_time, win_eval):
    stim_onset_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
    response_onset_idx = get_response_onset(nsm, neu_time, win_eval)
    if response_onset_idx is None or response_onset_idx <= stim_onset_idx:
        inh_depth = np.nan
    else:
        stim_resp = nsm[stim_onset_idx:response_onset_idx + 1]
        if len(stim_resp) <= 3:
            inh_depth = np.nan
        else:
            min_val = np.min(stim_resp)
            pre_start, _ = get_frame_idx_from_time(neu_time, neu_time[stim_onset_idx], -500, 0)
            pre_baseline = nsm[pre_start:stim_onset_idx]
            base_mean = np.mean(pre_baseline)
            inh_depth = base_mean - min_val
    return inh_depth

def get_recovery_amp(nsm, neu_time, win_eval):
    stim_onset_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
    response_onset_idx = get_response_onset(nsm, neu_time, win_eval)
    if response_onset_idx is None or response_onset_idx <= stim_onset_idx:
        recovery_amp = np.nan
    else:
        stim_resp = nsm[stim_onset_idx:response_onset_idx + 1]
        if len(stim_resp) <= 3:
            recovery_amp = np.nan
        else:
            min_val = np.min(stim_resp)
            recovery_amp = nsm[response_onset_idx] - min_val
    return recovery_amp

def get_inh_latency(nsm, neu_time, win_eval):
    stim_onset_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
    response_onset_idx = get_response_onset(nsm, neu_time, win_eval)
    if response_onset_idx is None or response_onset_idx <= stim_onset_idx:
        inh_latency = np.nan
    else:
        stim_resp = nsm[stim_onset_idx:response_onset_idx + 1]
        if len(stim_resp) <= 3:
            inh_latency = np.nan
        else:
            min_idx = stim_onset_idx + int(np.argmin(stim_resp))
            inh_latency = neu_time[min_idx] - neu_time[stim_onset_idx]
    return inh_latency

def get_recovery_latency(nsm, neu_time, win_eval):
    stim_onset_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
    response_onset_idx = get_response_onset(nsm, neu_time, win_eval)
    if response_onset_idx is None or response_onset_idx <= stim_onset_idx:
        rec_latency = np.nan
    else:
        stim_resp = nsm[stim_onset_idx:response_onset_idx + 1]
        if len(stim_resp) <= 3:
            rec_latency = np.nan
        else:
            min_idx = stim_onset_idx + int(np.argmin(stim_resp))
            rec_latency = neu_time[response_onset_idx] - neu_time[min_idx]
    return rec_latency

def get_min_value(nsm, neu_time, win_eval):
    stim_onset_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
    response_onset_idx = get_response_onset(nsm, neu_time, win_eval)
    if response_onset_idx is None or response_onset_idx <= stim_onset_idx:
        min_value = np.nan
    else:
        stim_resp = nsm[stim_onset_idx:response_onset_idx + 1]
        if len(stim_resp) <= 3:
            min_value = np.nan
        else:
            min_value = np.min(stim_resp)
    return min_value

def get_min_time(nsm, neu_time, win_eval):
    stim_onset_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
    response_onset_idx = get_response_onset(nsm, neu_time, win_eval)
    if response_onset_idx is None or response_onset_idx <= stim_onset_idx:
        min_time = np.nan
    else:
        stim_resp = nsm[stim_onset_idx:response_onset_idx + 1]
        if len(stim_resp) <= 3:
            min_time = np.nan
        else:
            min_idx = stim_onset_idx + int(np.argmin(stim_resp))
            min_time = neu_time[min_idx] - neu_time[stim_onset_idx]
    return min_time

def _get_ramping_quality_util(nsm, neu_time, win_eval):
    stim_onset_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
    response_onset_idx = get_response_onset(nsm, neu_time, win_eval)
    if response_onset_idx is None:
        return None
    start_idx, end_idx = get_frame_idx_from_time(neu_time, 0, win_eval[0], win_eval[1])
    resp_win = nsm[start_idx:end_idx]
    win_resp_onset_idx = response_onset_idx - start_idx
    post_onset = resp_win[win_resp_onset_idx:]
    peak_offset = int(np.argmax(post_onset))
    full_peak_idx = win_resp_onset_idx + peak_offset
    if full_peak_idx <= win_resp_onset_idx:
        return None
    seg = resp_win[win_resp_onset_idx:full_peak_idx + 1]
    t = np.arange(len(seg))
    slope, intercept = np.polyfit(t, seg, 1)
    line = slope * t + intercept
    ss_tot = np.sum((seg - seg.mean())**2)
    if ss_tot == 0:
        r_squared = np.nan
    else:
        r_squared = 1 - np.sum((seg - line)**2) / ss_tot
    rel_dev = np.mean(np.abs(seg - line)) / (np.max(seg) - np.min(seg)) if (np.max(seg) - np.min(seg)) != 0 else np.nan
    pre_start, _ = get_frame_idx_from_time(neu_time, neu_time[stim_onset_idx], -500, 0)
    pre_baseline = nsm[pre_start:stim_onset_idx]
    pre_mean = np.mean(pre_baseline) if pre_baseline.size else np.nan
    amp_change = (np.mean(seg) - pre_mean) / pre_mean if pre_baseline.size and pre_mean != 0 else np.inf
    ws = max(3, len(seg) // 5)
    roll_means = np.array([np.mean(seg[i:i + ws]) for i in range(len(seg) - ws + 1)])
    monotonic = np.all(np.diff(roll_means) >= -np.std(roll_means) * 0.2)
    return {'r_squared': r_squared, 'relative_deviation': rel_dev, 'amp_change': amp_change, 'is_monotonic': monotonic, 'slope': slope}

def get_ramp_r2(nsm, neu_time, win_eval):
    result = _get_ramping_quality_util(nsm, neu_time, win_eval)
    if result is None:
        r_squared = np.nan
    else:
        r_squared = result['r_squared']
    return r_squared

def get_ramp_relative_dev(nsm, neu_time, win_eval):
    result = _get_ramping_quality_util(nsm, neu_time, win_eval)
    if result is None:
        rel_dev = np.nan
    else:
        rel_dev = result['relative_deviation']
    return rel_dev

def get_ramp_amp_change(nsm, neu_time, win_eval):
    result = _get_ramping_quality_util(nsm, neu_time, win_eval)
    if result is None:
        amp_change = np.nan
    else:
        amp_change = result['amp_change']
    return amp_change

def get_ramp_mono(nsm, neu_time, win_eval):
    result = _get_ramping_quality_util(nsm, neu_time, win_eval)
    if result is None:
        monotonic = False
    else:
        monotonic = result['is_monotonic']
    return monotonic

def get_ramp_slope(nsm, neu_time, win_eval):
    result = _get_ramping_quality_util(nsm, neu_time, win_eval)
    if result is None:
        slope_val = np.nan
    else:
        slope_val = result['slope']
    return slope_val

# compute all metrics for list of all traces.
def get_all_metrics(list_neu_seq_mean, neu_time, list_win_eval):
    list_metrics = []
    for ni in range(len(list_neu_seq_mean)):
        m = {
            'response_latency'   : np.apply_along_axis(get_response_latency,   1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'peak_time'          : np.apply_along_axis(get_peak_time,          1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'peak_amp'           : np.apply_along_axis(get_peak_amp,           1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'pre_stim_peak_time' : np.apply_along_axis(get_pre_stim_peak_time, 1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'pre_stim_peak_amp'  : np.apply_along_axis(get_pre_stim_peak_amp,  1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'rise_slope'         : np.apply_along_axis(get_rise_slope,         1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'decay_rate'         : np.apply_along_axis(get_decay_rate,         1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'final_amp'          : np.apply_along_axis(get_final_amp,          1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'amp_reduction'      : np.apply_along_axis(get_amp_reduction,      1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'inh_depth'          : np.apply_along_axis(get_inh_depth,          1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'recovery_amp'       : np.apply_along_axis(get_recovery_amp,       1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'inh_latency'        : np.apply_along_axis(get_inh_latency,        1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'recovery_latency'   : np.apply_along_axis(get_recovery_latency,   1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'min_value'          : np.apply_along_axis(get_min_value,          1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'min_time'           : np.apply_along_axis(get_min_time,           1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'ramp_r2'            : np.apply_along_axis(get_ramp_r2,            1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'ramp_relative_dev'  : np.apply_along_axis(get_ramp_relative_dev,  1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'ramp_amp_change'    : np.apply_along_axis(get_ramp_amp_change,    1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'ramp_mono'          : np.apply_along_axis(get_ramp_mono,          1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni]),
            'ramp_slope'         : np.apply_along_axis(get_ramp_slope,         1, list_neu_seq_mean[ni], neu_time, list_win_eval[ni])
            }
        list_metrics.append(m)
    return list_metrics

