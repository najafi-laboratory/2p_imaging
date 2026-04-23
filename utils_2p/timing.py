import numpy as np


def get_frame_idx_from_time(timestamps, c_time, l_time, r_time):
    l_idx = np.searchsorted(timestamps, c_time + l_time)
    r_idx = np.searchsorted(timestamps, c_time + r_time)
    return l_idx, r_idx


def get_sub_time_idx(time, start, end):
    return np.where((time >= start) & (time <= end))[0]


def get_trigger_time(vol_time, vol_bin):
    diff_vol = np.diff(vol_bin, prepend=0)
    idx_up = np.where(diff_vol == 1)[0]
    idx_down = np.where(diff_vol == -1)[0]
    time_up = vol_time[idx_up]
    time_down = vol_time[idx_down]
    return time_up, time_down


def correct_time_img_center(time_img):
    diff_time_img = np.diff(time_img, append=0)
    diff_time_img[-1] = np.mean(diff_time_img[:-1])
    diff_time_img = diff_time_img / 2
    time_neuro = time_img + diff_time_img
    return time_neuro
