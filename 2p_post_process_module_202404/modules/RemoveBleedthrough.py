import os
import h5py
import tifffile
import numpy as np
from tqdm import tqdm
from cellpose import models
from cellpose import io
import matplotlib.pyplot as plt
from scipy.stats import linregress
from skimage.segmentation import find_boundaries
from sklearn.linear_model import LinearRegression

from .LabelExcInh import *


def train_reg_model(ops, mean_anat, mean_func, labels):
    # multivariate regression on ROIs

    coords = np.where(labels == 1)

    rows = coords[:, 0]
    cols = coords[:, 1]
    anat_roi_pixels = mean_anat[rows, cols]  # r_i
    func_roi_pixels = mean_func[rows, cols]  # g_i

    anat_roi_pixels = np.array([np.array([p]) for p in anat_roi_pixels])
    func_roi_pixels = np.array([np.array([p]) for p in func_roi_pixels])

    # fit multivariate regression r_i = m_i * g_i + b_i
    model = LinearRegression()

    model.fit(func_roi_pixels, anat_roi_pixels)

    # Get the slopes (coefficients) and intercepts for each variate in r
    slopes = model.coef_.tolist()
    offsets = model.intercept_.tolist()

    # return list of slopes, offsets, coordinates
    return slopes, offsets, coords


def remove_green_bleedthrough(offsets, slopes, mean_func, mean_anat, coordinates):
    # for each coordinate
    # mean_anat_corrected = mean_anat
    mean_anat_corrected = mean_anat
    # mean_anat_corrected(coordinate) -= (slope * mean_func(coordinate) + offset)

    for i, c in enumerate(coordinates):
        mean_anat_corrected[c] = mean_anat_corrected[c] - \
            (slopes[i] * mean_func[c] + offsets[i])

    # return mean_anat_corrected
    return mean_anat_corrected
