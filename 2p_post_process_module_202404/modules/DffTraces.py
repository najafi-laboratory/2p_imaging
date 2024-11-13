#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from modules import SpikeDeconv

from .SpikeAnalysis import analyze_spike_traces
from .SpikePlotting import plot_for_neuron_with_smoothed_interactive_multi_tau
# compute dff from raw fluorescence signals.


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
        taus=[0.06, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 5],
        plot_with_smoothed=False,
        plot_with_smoothed_group=True,
        plot_without_smoothed=False):

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
    print("DFF Results saved under name 'dff'")
    # deconvolution code
    if len(taus) > 1:
        spikes_list = []
        convolved_list = []
        thresh_list = []
        # code to perform a parameter search on tau
        tau_spike_dict = {}
        neurons = np.arange(dff.shape[0])
        for tau in taus:
            smoothed, spikes, uptime, threshold_val = SpikeDeconv.run(
                ops,
                dff,
                oasis_tau=tau,
                neurons=neurons,
                plotting_neurons=plotting_neurons,
                plot_with_smoothed=plot_with_smoothed,
                plot_without_smoothed=plot_without_smoothed)
            tau_spike_dict[tau] = spikes

            spikes_list.append(spikes)
            convolved_list.append(smoothed)
            thresh_list.append(threshold_val)

        _, spike_stas, dff_stas = analyze_spike_traces(ops, dff, tau_spike_dict,
                                                       neurons=np.arange(dff.shape[0] // 10))

        # dff_stas = res_df['sta_dff']
        # spike_stas = res_df['sta_spikes']
        # print('here')
        # print(spike_stas)
        # print(len(spike_stas[0]))
        print(thresh_list)
        if plot_with_smoothed_group:
            for neuron in plotting_neurons:
                plot_for_neuron_with_smoothed_interactive_multi_tau(
                    timings=uptime,
                    dff=dff,
                    spikes_list=spikes_list,
                    convolved_spikes_list=convolved_list,
                    sta_list=dff_stas,
                    threshold_val=thresh_list,
                    neuron=neuron,
                    tau_list=taus)

    else:
        # if we just specify one tau value
        tau = taus[0]
        neurons = np.arange(dff.shape[0])
        smoothed, spikes, _, _ = SpikeDeconv.run(
            ops,
            dff,
            oasis_tau=tau,
            neurons=neurons,
            plotting_neurons=plotting_neurons,
            plot_with_smoothed=plot_with_smoothed,
            plot_without_smoothed=plot_without_smoothed)

        save(ops, 'spikes', spikes)
        print("Spike traces saved under name 'spikes'")

        save(ops, 'smoothed', smoothed)
        print("De-noised DFF data saved under name 'smoothed'")
