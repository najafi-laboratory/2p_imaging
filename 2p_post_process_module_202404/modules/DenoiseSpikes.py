import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
# Function to create the right-half Gaussian kernel


def right_half_gaussian_kernel(std_dev, kernel_size):
    """
    Create a right-half Gaussian kernel.

    Parameters:
    std_dev : float
        The standard deviation of the Gaussian kernel.
    kernel_size : int
        The size of the Gaussian kernel.

    Returns:
    right_half_gaussian : numpy array
        The right-half of a Gaussian kernel.
    """
    x = np.linspace(
        0, 15*std_dev, kernel_size)  # Range [0, 3*std_dev] for the right-half
    gaussian_full = np.exp(-x**2 / (2 * std_dev**2))
    gaussian_right_half = gaussian_full / \
        gaussian_full.sum()  # Normalize to ensure sum = 1

    return gaussian_right_half


def exponential_kernel(a, b, kernel_size):
    """
    Create an exponential kernel.

    Parameters:
    a : float
        The decay constant of the exponential kernel.
    kernel_size : int
        The size of the exponential kernel.

    Returns:
    exponential_kernel : numpy array
        The exponential kernel.
    """
    x = np.linspace(0, 15 * a, kernel_size)
    exponential_kernel = a * np.exp(-b * x)

    # exponential_kernel = exponential_kernel / exponential_kernel.sum()
    return exponential_kernel


def asymmetric_exp(x, tau_rise, tau_decay):
    y = np.zeros_like(x)
    y[x >= 0] = (1 - np.exp(-x[x >= 0] / tau_rise)) * \
        np.exp(-x[x >= 0] / tau_decay)
    return y


def exponential_moving_average(data, alpha=0.1):
    """
    Smooths data using an exponential moving average.

    Args:
        data (np.array): The input data to be smoothed.
        alpha (float): Smoothing factor, between 0 and 1.
                       Higher values mean less smoothing.

    Returns:
        np.array: Smoothed data.
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


# Usage in your denoising routine:
smoothed_spikes = exponential_moving_average(
    spikes, alpha=0.2)  # adjust alpha as needed


def denoise(spikes, kernel_size=1000, std_dev=1, neurons=None):
    """
    Denoise the spikes using a half-Gaussian kernel.

    Parameters:
    spikes : numpy array
        The spikes to denoise.
    kernel_size : int
        The size of the Gaussian kernel.
    std_dev : float
        The standard deviation of the Gaussian kernel.
    neurons : list, optional
        The indices of the neurons to denoise. If None, all neurons are denoised.

    Returns:
    denoised_spikes : numpy array
        The denoised spikes.
    """
    if neurons is None:
        neurons = range(spikes.shape[0])

    x = np.arange(kernel_size)
    kernel = right_half_gaussian_kernel(
        kernel_size=kernel_size, std_dev=std_dev)

    smoothed = np.zeros_like(spikes)
    for i in neurons:
        smoothed_signal = np.roll(fftconvolve(
            spikes[i], kernel, mode='same'), kernel_size//2)
        smoothed[i] = smoothed_signal
    return smoothed
