import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
# Function to create the right-half Gaussian kernel
from scipy.signal import savgol_filter
import pingouin as pg

from .SpikeAnalysis import compute_sta_parallel


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


def denoise(normalized_spikes, sigma=3):
    return gaussian_filter1d(normalized_spikes, sigma=sigma, axis=-1)


class Postprocessor:
    """Post processing pipeline to perform
            1. Normalization on spike trains
            2. Denoising on spike trains
            3. STA computation on (baselined) DF/F"""

    def __init__(self, corrected_dff, spikes, denoise_sigma=3, sta_pre_spike_window=200, sta_post_spike_window=600):
        self.corrected_dff = corrected_dff
        self.spikes = spikes
        self.denoise_sigma = denoise_sigma
        self.sta_pre_spike_window = sta_pre_spike_window
        self.sta_post_spike_window = sta_post_spike_window

        self.multi_tau = False if len(spikes.shape) == 3 else True

        self.normalized_spikes = None
        self.denoised_spikes = None
        self.stas = None

    def _normalize_spikes(self):
        scale = np.max(self.corrected_dff, axis=-1, keepdims=True) / \
            np.max(self.spikes, axis=-1, keepdims=True)
        self.normalized_spikes = self.spikes * scale

    def _denoise_spikes(self):
        self.denoised_spikes = denoise(
            self.normalized_spikes, self.denoise_sigma)

    def _compute_sta(self):
        self.stas = compute_sta_parallel(
            spikes=self.spikes,
            dff=self.corrected_dff,
            pre_spike_window=self.sta_pre_spike_window,
            post_spike_window=self.sta_post_spike_window)

    def __call__(self):
        # NORMALIZATION
        self._normalize_spikes()

        # DENOISING
        self._denoise_spikes()

        # STA COMPUTATION
        self._compute_sta()
