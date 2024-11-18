from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from suite2p.extraction.dcnv import oasis
import h5py
import os
from scipy.optimize import curve_fit
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from .DenoiseSpikes import denoise
from .SpikePlotting import *


def read_raw_voltages(ops):
    """
    Reads raw voltage data from an HDF5 file.

    This function opens the 'raw_voltages.h5' file located in the directory specified by ops['save_path0'],
    and extracts the voltage time and image data.

    Parameters:
    ops (dict): A dictionary containing operation parameters, including the 'save_path0' key
                which specifies the directory where the HDF5 file is located.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - vol_time (np.array): An array of voltage timestamps.
        - vol_img (np.array): An array of voltage image data.

    Raises:
    FileNotFoundError: If the 'raw_voltages.h5' file is not found in the specified directory.
    KeyError: If the required datasets 'vol_time' or 'vol_img' are not present in the HDF5 file.
    """
    with h5py.File(os.path.join(ops['save_path0'], 'raw_voltages.h5')) as f:
        vol_time = np.array(f['raw']['vol_time'])
        vol_img = np.array(f['raw']['vol_img'])
        f.close()

    return vol_time, vol_img


def get_trigger_time(
        vol_time,
        vol_bin
):
    """
    Finds the trigger times from the voltage data.

    This function calculates the time points when the voltage data transitions from 0 to 1 and from 1 to 0.
    It uses the np.diff function to compute the first difference of the voltage data and then identifies the indices
    where this difference changes from positive to negative (rising edge) and from negative to positive (falling edge).

    Parameters:
    vol_time (np.array): An array of voltage timestamps.
    vol_bin (np.array): An array of binary voltage data.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - time_up (np.array): An array of time points when the voltage transitions from 0 to 1.
        - time_down (np.array): An array of time points when the voltage transitions from 1 to 0.
    """
    # find the edge with np.diff and correct it by preappend one 0.
    diff_vol = np.diff(vol_bin, prepend=0)
    idx_up = np.where(diff_vol == 1)[0]
    idx_down = np.where(diff_vol == -1)[0]
    # select the indice for rising and falling.
    # give the edges in ms.
    time_up = vol_time[idx_up]
    time_down = vol_time[idx_down]
    return time_up, time_down


def read_dff(ops):
    """
    Reads DF/F data from an HDF5 file.

    This function opens the 'dff.h5' file located in the directory specified by ops['save_path0'],
    and extracts the DF/F data.

    Parameters:
    ops (dict): A dictionary containing operation parameters, including the 'save_path0' key
                which specifies the directory where the HDF5 file is located.

    Returns:
    np.array: An array containing the DF/F data.
    """

    f = h5py.File(os.path.join(ops['save_path0'], 'dff.h5'), 'r')
    dff = np.array(f['dff'])
    f.close()
    return dff


def spike_detect(ops, dff, tau=1.25):
    """
    Detects spikes in the DF/F data using the OASIS algorithm.

    Args:
        ops (dict): A dictionary containing operation parameters.
        dff (np.array): DF/F data array.
        tau (float, optional): Tau parameter for the OASIS algorithm. Defaults to 1.25.

    Returns:
        np.array: An array containing the detected spikes.
    """
    spikes = oasis(
        F=dff,
        batch_size=ops['batch_size'],
        tau=tau,
        fs=ops['fs'])

    avg_tau = np.exp(-tau)
    return spikes


def threshold(data, threshold_num_stds, const_threshold=None):
    """Threshold off spikes below a certain hard threshold. This threshold is
    computed as a certain number of standard deviations from the mean.

    Args:
        data (np.array): data to be thresholded
        num_stds (float): sets the threshold at num_stds * std so that only spikes
                          >= num_stds * std are kept.
    Returns:
        np.array: thresholded data
    """
    # center the data
    # data_centered = data - np.mean(data, axis=-1, keepdims=True)
    threshold_val = None

    if const_threshold is None:
        stds = np.std(data, axis=-1, keepdims=True)

        threshold_val = num_stds * stds + np.mean(data, axis=-1, keepdims=True)
    else:
        threshold_val = const_threshold

    above_threshold_mask = data >= threshold_val
    below_threshold_mask = data < threshold_val

    return data * above_threshold_mask, data * below_threshold_mask, threshold_val


# def optimize_denoise_parameters(dff, spikes, neurons, kernel_size_range, std_dev_range):
#     """
#     Finds the optimal kernel size and standard deviation for the denoise function
#     by minimizing the MSE between the smoothed data and the original DF/F data.

#     Args:
#         dff (np.array): DF/F data array.
#         spikes (np.array): Detected spikes array.
#         neurons (np.array): Array of neuron indices to optimize for.
#         kernel_size_range (list): Range of kernel sizes to test.
#         std_dev_range (list): Range of standard deviations to test.

#     Returns:
#         tuple: Optimal kernel size and standard deviation.
#     """
#     best_mse = float("inf")
#     best_params = (None, None)

#     for kernel_size in kernel_size_range:
#         for std_dev in std_dev_range:
#             # Run the denoise function with the current parameters
#             smoothed = denoise(spikes, kernel_size=kernel_size,
#                                std_dev=std_dev, neurons=neurons)

#             # Calculate the MSE for the neurons
#             mse = mean_absolute_error(dff[:, neurons], smoothed[:, neurons])

#             # Update best parameters if the current MAE is lower
#             if mse < best_mse:
#                 best_mse = mse
#                 best_params = (kernel_size, std_dev)

#     return best_params

# Objective function to minimize
def optimize_denoise_parameters(dff, spikes, neurons):
    """
    Optimizes denoising parameters using Bayesian optimization.

    Args:
        dff (np.array): DF/F data array.
        spikes (np.array): Detected spikes array.
        neurons (np.array): Array of neuron indices to optimize for.

    Returns:
        tuple: Optimal kernel size and standard deviation.
    """
    # Define parameter search space
    space = [
        # Adjust range based on expected values
        Integer(10, 300, name='kernel_size'),
        # Adjust range based on expected values
        Real(0.1, 5.0, name='std_dev')
    ]

    # Objective function to minimize MSE between smoothed spikes and dff
    @use_named_args(space)
    def objective(kernel_size, std_dev):
        smoothed = denoise(spikes, kernel_size=kernel_size,
                           std_dev=std_dev, neurons=neurons)
        mse = mean_absolute_error(dff[:, neurons], smoothed[:, neurons])
        return mse

    # Perform Bayesian optimization
    result = gp_minimize(
        func=objective,
        dimensions=space,
        # Number of function calls (iterations of search)
        n_calls=20,
        random_state=0
    )

    # Extract best parameters
    best_kernel_size = result.x[0]
    best_std_dev = result.x[1]
    return best_kernel_size, best_std_dev


def run(
        ops,
        dff,
        oasis_tau=10.0,
        neurons=np.arange(100),
        plotting_neurons=[5, 10, 100],
        threshold_num_stds=3,
        plot_with_smoothed=False,
        plot_without_smoothed=False):

    print('===================================================')
    print('=============== Deconvolving Spikes ===============')
    print('===================================================')

    metrics = read_raw_voltages(ops)
    vol_time = metrics[0]
    vol_img = metrics[1]

    spikes = spike_detect(ops, dff, tau=oasis_tau)
    uptime, _ = get_trigger_time(vol_time, vol_img)

    # Optimize denoising parameters
    # kernel_size_range = range(50, 500, 50)  # Example range for kernel size
    # Example range for standard deviation
    # std_dev_range = range(50, 80, 5)
    # best_kernel_size, best_std_dev = optimize_denoise_parameters(
    #     dff, spikes, neurons)

    # Apply denoise with the best parameters
    # smoothed = denoise(spikes, kernel_size=400,
    #                    std_dev=65, neurons=neurons)
    smoothed = denoise(spikes, window_length=50, polyorder=3)

    thresholded_spikes, below_thresholded_spikes, threshold_val = threshold(
        spikes, threshold_num_stds=None, const_threshold=0.8)

    # Plot for certain neurons
    # if plot_without_smoothed or plot_with_smoothed:
    #     for i in plotting_neurons:
    #         if plot_with_smoothed:
    #             plot_for_neuron_with_smoothed_interactive(timings=uptime, dff=dff, spikes=spikes,
    #                                                       convolved_spikes=smoothed, neuron=i, tau=oasis_tau, threshold_val=threshold_val)
    #         if plot_without_smoothed:
    #             plot_for_neuron_without_smoothed_interactive(
    #                 timings=uptime, dff=dff, spikes=spikes, neuron=i, tau=oasis_tau, threshold_val=threshold_val, thresholded_spikes=thresholded_spikes)

    return smoothed, spikes, uptime, threshold_val, thresholded_spikes, below_thresholded_spikes
