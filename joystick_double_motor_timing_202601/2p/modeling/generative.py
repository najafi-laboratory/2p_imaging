#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.metrics import explained_variance_score

from utils import get_frame_idx_from_time


#%% utils

# interpolate input trace to the same length as neural response.
def interp_factor_in(input_value, input_time, neu_time):
    model = interp1d(input_time, input_value, bounds_error=False, fill_value=0)
    return model(neu_time)

# linear convolution to obtain neural response from one factor and kernel.
def factor_dff_neu(factor_target, kernel, l_idx, r_idx):
    padded = np.pad(factor_target, (r_idx, l_idx-1), mode='constant')
    return np.convolve(padded, kernel, mode='valid')

# retrieve glm kernels for category.
def get_glm_cate(glm, list_labels, cate):
    idx = np.concatenate([np.in1d(list_labels[i],cate)
           for i in range(len(list_labels))])
    kernel_all = glm['kernel_all'][idx,:]
    return kernel_all

# set factor to 1 when state is presented.
def set_factor(time, trial_factor, state_time):
    if len(state_time) == 1:
        l_idx, _ = get_frame_idx_from_time(time, 0, state_time[0], 0)
        trial_factor[l_idx] = 1
    if len(state_time) == 2:
        l_idx, r_idx = get_frame_idx_from_time(time, 0, state_time[0], state_time[1])
        trial_factor[l_idx:r_idx] = 1
    return trial_factor


#%% construct factors

# process dff data.
def get_target_dff_time(list_neural_trials):
    list_glm_time = []
    list_glm_dff = []
    for neural_trials in list_neural_trials:
        # collect into list.
        list_glm_time.append(neural_trials['time'])
        list_glm_dff.append(neural_trials['dff'])
    return list_glm_time, list_glm_dff

# get all factors.
class get_factor_all:

    def __init__(self, list_neural_trials):
        self.list_neural_trials = list_neural_trials
    
    # visual cues for all trials.
    def factor_vis_all(self, time, trial_labels):
        trial_factor = np.zeros_like(time)
        for s in np.array(trial_labels['state_vis1']): 
            if not np.isnan(np.sum(s)):
                trial_factor = set_factor(time, trial_factor, [s[0], s[1]])
        for s in np.array(trial_labels['state_vis2']): 
            if not np.isnan(np.sum(s)):
                trial_factor = set_factor(time, trial_factor, [s[0], s[1]])
        return trial_factor

    # joystick trajectory.
    def factor_js_all(self, time, trial_labels):
        js_rot = np.concatenate(trial_labels['js_rot'])
        js_time = np.concatenate(trial_labels['js_time'])
        trial_factor = interp_factor_in(js_rot, js_time, time)
        trial_factor = np.nan_to_num(trial_factor, nan=0.0)
        return trial_factor

    # press onset.
    def factor_pre_press(self, time, trial_labels):
        win = 200
        trial_factor = np.zeros_like(time)
        for s in np.array(trial_labels['state_press1']): 
            if not np.isnan(np.sum(s)):
                trial_factor = set_factor(time, trial_factor, [s[0]-win, s[0]])
        for s in np.array(trial_labels['state_press2']): 
            if not np.isnan(np.sum(s)):
                trial_factor = set_factor(time, trial_factor, [s[0]-win, s[0]])
        return trial_factor
    
    # all reward.
    def factor_reward_all(self, time, trial_labels):
        trial_factor = np.zeros_like(time)
        for s in np.array(trial_labels['state_reward']): 
            if not np.isnan(np.sum(s)):
                trial_factor = set_factor(time, trial_factor, [s[0], s[1]])
        return trial_factor

    def run(self, ):
        factor_funcs = [
            self.factor_vis_all,
            self.factor_js_all,
            self.factor_pre_press,
            self.factor_reward_all,
        ]
        list_factor_names = [
            'vis_all',
            'js_all',
            'pre_press',
            'reward_all',
            ]
        list_trial_labels = [nt['trial_labels'] for nt in self.list_neural_trials]
        list_time = [nt['time'] for nt in self.list_neural_trials]
        list_all_factor_in = []
        # loop through sessions.
        for time, trial_labels in zip(list_time, list_trial_labels):
            # get all factors for one session.
            sess_factors = [f(time, trial_labels) for f in factor_funcs]
            list_all_factor_in.append(sess_factors)
        return list_factor_names, list_all_factor_in


#%% fit model

# construct the input design matrix for glm.
def construct_design_matrix(factor_target, l_idx, r_idx):
    total_window = l_idx + r_idx + 2
    padded = np.pad(factor_target, (r_idx+1, l_idx+1), mode='constant')
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=total_window)
    return windows[:len(factor_target), :]

# fit regression model with multiple factors (no cross interactions).
def fit_glm_factor(dff, all_factor_in, time, l_idx, r_idx):
    alpha = 52025
    n_neu, n_time = dff.shape
    n_factors = len(all_factor_in)
    k_len = l_idx + r_idx
    # pre-allocate result arrays.
    kernel = np.nan * np.zeros([n_neu, n_factors, k_len])
    exp_var = np.nan * np.zeros([n_neu])
    reconst = np.nan * np.zeros([n_neu, n_factors, n_time])
    # build design matrix for each factor and concatenate (no interaction terms).
    X_list = [construct_design_matrix(f, l_idx, r_idx) for f in all_factor_in]
    X = np.concatenate(X_list, axis=1)
    # precompute ridge pseudo-inverse.
    TW = (l_idx + r_idx + 2) * n_factors
    A = X.T @ X + alpha * np.eye(TW)
    M = np.linalg.solve(A, X.T)
    # indices to slice betas per factor
    seg = l_idx + r_idx + 2
    slices = [slice(i*seg, (i+1)*seg) for i in range(n_factors)]
    # for each neuron solve and write results.
    for ni in tqdm(range(n_neu), desc='neurons'):
        y = dff[ni, :]
        beta = M @ y
        # collect kernels per factor and reconstruct prediction
        y_hat = np.zeros_like(y)
        for fi in range(n_factors):
            b_seg = beta[slices[fi]]
            k = np.flip(b_seg)[1:-1]
            kernel[ni, fi, :] = k
            reconst[ni, fi, :] = factor_dff_neu(all_factor_in[fi], k, l_idx, r_idx)
            y_hat += reconst[ni, fi, :]
        exp_var[ni] = explained_variance_score(y, y_hat)
    return kernel, exp_var, reconst

# fit glm for multiple sessions.
def run_glm_multi_sess(
        list_dff, list_all_factor_in, list_time, kernel_win):
    # get kernel time.
    l_idx = np.nanmin([get_frame_idx_from_time(time, time[1106], kernel_win[0], kernel_win[1])[0] for time in list_time])
    r_idx = np.nanmax([get_frame_idx_from_time(time, time[1106], kernel_win[0], kernel_win[1])[1] for time in list_time])
    kernel_time = []
    for time in list_time:
        kernel_time.append(time[l_idx:r_idx] - time[1106])
    kernel_time = np.nanmean(np.stack(kernel_time,axis=0),axis=0)
    l_idx = np.searchsorted(kernel_time, 0)
    r_idx = len(kernel_time) - np.searchsorted(kernel_time, 0)
    # run glm on all sessions.
    kernel_all = []
    exp_var_all = []
    reconst_all = []
    for si, (dff, all_factor_in, time) in enumerate(zip(list_dff, list_all_factor_in, list_time)):
        print(f'Fitting GLM for session {si+1}/{len(list_dff)}')
        kernel, exp_var, reconst = fit_glm_factor(dff, all_factor_in, time, l_idx, r_idx)
        kernel_all.append(kernel)
        exp_var_all.append(exp_var)
        reconst_all.append(reconst)
    # concatenate results.
    kernel_all = np.concatenate(kernel_all, axis=0)
    exp_var_all = np.concatenate(exp_var_all)
    return kernel_time, kernel_all, exp_var_all, reconst_all


#%% evaluation

# compute coding score by fraction of explained variance.
def get_coding_score_fraction(exp_var_all, list_exp_var_single):
    bound = [0, 1]
    s = np.stack([ev/exp_var_all for ev in list_exp_var_single], axis=1)
    s[s<bound[0]] = 0
    s[s>bound[1]] = 1
    return s

# compute coding score by dropout single factors
def get_coding_score_dropout(exp_var_all, list_exp_var_dropout):
    bound = [0, 1]
    s = np.stack([1-ev/exp_var_all for ev in list_exp_var_dropout], axis=1)
    s[s<bound[0]] = 0
    s[s>bound[1]] = 1
    return s
