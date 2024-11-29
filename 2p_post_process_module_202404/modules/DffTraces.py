#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from modules import SpikeDeconv

from .SpikeAnalysis import analyze_spike_traces, compute_sta
from .SpikePlotting import plot_baselined_dff_smoothed_sta, plot_for_neuron_with_smoothed_interactive, plot_for_neuron_with_smoothed_interactive_multi_tau, plot_for_neuron_interactive, plot_stas_single_thresh_neuron, plot_stas_multi_thresh_neuron
# compute dff from raw fluorescence signals.
from .Preprocessing import Preprocessor
from .Postprocessing import Postprocessor


def get_dff(
        ops,
        fluo,
        neuropil,
        norm,
):
    # correct with neuropil signals.
    dff = fluo.copy() - ops['neucoeff']*neuropil
    # get baseline.
    f0 = gaussian_filter(dff, [0., ops['sig_baseline']])
    for j in range(dff.shape[0]):
        # baseline subtraction.
        dff[j, :] = (dff[j, :] - f0[j, :]) / f0[j, :]
        if norm:
            # z score.
            dff[j, :] = (dff[j, :] - np.mean(dff[j, :])) / \
                (np.std(dff[j, :]) + 1e-5)
    return dff

# save results


def save(ops, name, data):
    f = h5py.File(os.path.join(ops['save_path0'], 'dff.h5'), 'w')
    f['name'] = data
    f.close()

# main function to compute spikings.


def run(
        ops,
        norm=True,
        plotting_neurons=[5],
        tau=0.4,
        preproc_baseline_method='als',
        preproc_baseline_params={'lam': 1e3,
                                 'p': 0.2,
                                 'n_iter': 10,
                                 'njobs': -1,
                                 'scoring': 'l2'},
        preproc_filter_method='savgol',
        preproc_filter_params={'window_length': 21,
                                'polyorder': 3,
                               'deriv': 0}):

    print('===============================================')
    print('=========== dff trace normalization ===========')
    print('===============================================')
    print('Reading fluorescence signals after quality control')
    fluo = np.load(
        os.path.join(ops['save_path0'], 'qc_results', 'fluo.npy'),
        allow_pickle=True)
    neuropil = np.load(
        os.path.join(ops['save_path0'], 'qc_results', 'neuropil.npy'),
        allow_pickle=True)
    print('Running baseline subtraction and normalization')
    dff = get_dff(ops, fluo, neuropil, norm)

    save(ops, 'dff', dff)

    # PREPROCESSING (baselining, filtering)
    print("Performing desired postprocessing methods")
    preprocessor = Preprocessor(
        dff=dff,
        baseline_method=preproc_baseline_method,
        filtering_method=preproc_filter_method,
        baseline_params=preproc_baseline_params,
        filter_params=preproc_filter_params)
    preprocessor()
    if preprocessor.optimize_baseline:
        print(
            f"Optimized baseline operation to find best (lam, p):{(preprocessor.best_baseline_params['p'], preprocessor.best_baseline_params['lam'])}")

    baselined_dff, filtered_dff = preprocessor.filtered, preprocessor.baselined

    print('tau:', tau)

    # SPIKE COMPUTATIONS
    spikes, uptime = None, None
    if isinstance(tau, float):
        spikes, uptime = SpikeDeconv.run(
            ops=ops, dff=filtered_dff, oasis_tau=tau)
    elif isinstance(tau, list):
        spikes_lst = []
        uptime_lst = []
        for t in tau:
            spikes_t, uptime_t = SpikeDeconv.run(ops=ops, dff=dff, oasis_tau=t)
            spikes_lst.append(spikes_t)
            uptime_lst.append(uptime_t)

        spikes_lst = np.array(spikes_lst)
        uptime_lst = np.array(uptime_lst)
        spikes, uptime = spikes_lst, uptime_lst

    # POSTPROCESSING (normalization, denoising, and STAs)
    postprocessor = Postprocessor(
        corrected_dff=baselined_dff, spikes=spikes)
    postprocessor()

    normalized_spikes, denoised_spikes, stas = postprocessor.normalized_spikes, postprocessor.denoised_spikes, postprocessor.stas
    multi_tau = postprocessor.multi_tau

    if len(plotting_neurons) == 1:
        if isinstance(tau, float):
            plot_baselined_dff_smoothed_sta(timings=uptime, orig_dff=dff, baselined_dff=baselined_dff,
                                            inferred_spikes=normalized_spikes, sta=stas['sta_dff'], tau=tau, neuron=plotting_neurons[0])
        elif isinstance(tau, (list, np.ndarray)):
            print('here')

    save(ops, 'spikes', spikes)
    print("Spike traces saved under name 'spikes'")

    save(ops, 'smoothed', denoised_spikes)
    print("De-noised DFF data saved under name 'smoothed'")
