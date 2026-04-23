#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils_2p.stats import get_mean_sem as shared_get_mean_sem
from utils_2p.stats import get_norm01_params as shared_get_norm01_params
from utils_2p.stats import norm01
from utils_2p.timing import get_frame_idx_from_time

# normalization into gaussian.
def norm_gauss(data):
    return (data - np.nanmean(data)) / (np.nanstd(data) + 1e-5)

# compute the scale parameters when normalizing data into [0,1].
def get_norm01_params(data):
    a, b, _, _ = shared_get_norm01_params(data)
    return a, b

# compute mean and sem for 3d array data
def get_mean_sem(data, zero_start=False):
    return shared_get_mean_sem(data, method_s='standard error', zero_start=zero_start)

# compute mean and sem for average df/f within given time window.
def get_mean_sem_win(neu_seq, neu_time, c_time, l_time, r_time, mode='mean'):
    pct = 25
    l_idx, r_idx = get_frame_idx_from_time(
        neu_time, c_time, l_time, r_time)
    neu_win = neu_seq[:, l_idx:r_idx].copy()
    neu_win_mean = np.nanmean(neu_win, axis=1)
    # mean values in the window.
    if mode == 'mean':
        neu = neu_win_mean
    # values lower than percentile in the window.
    if mode == 'lower':
        neu = neu_win.reshape(-1)
        neu = neu[neu<np.nanpercentile(neu, pct)]
    # values higher than percentile in the window.
    if mode == 'higher':
        neu = neu_win.reshape(-1)
        neu = neu[neu>np.nanpercentile(neu, pct)]
    # compute mean and sem.
    neu_mean = np.nanmean(neu)
    std = np.nanstd(neu)
    count = np.nansum(~np.isnan(neu_win_mean), axis=0)
    neu_sem = std / np.sqrt(count)
    return neu_mean, neu_sem

# get derivative.
def compute_derivative(neu_seq, neu_time, win_eval):
    derivative = np.gradient(neu_seq, neu_time)
    return derivative

# smoothing signal.
def smooth_trace(neu_seq, neu_time, win_eval):
    window_size = 15
    polyorder = 3
    if len(neu_seq) < window_size:
        window_size = len(neu_seq) - 1 if len(neu_seq) % 2 == 0 else len(neu_seq) - 2
    if window_size < polyorder:
        polyorder = window_size - 1
    smoothed = savgol_filter(neu_seq, window_size, polyorder)
    return smoothed
