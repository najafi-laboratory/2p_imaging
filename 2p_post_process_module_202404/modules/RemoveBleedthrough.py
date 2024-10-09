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
from scipy.optimize import minimize

from .LabelExcInh import *


def objective(params, X, y):
    slope = params[0]
    intercepts = params[1:]

    # Predicted values based on the shared slope and individual intercepts
    predictions = slope * X + intercepts[:, np.newaxis]

    # Residual sum of squares
    residuals = predictions - y
    return np.sum(residuals ** 2)


def train_reg_model(fluo, fluo_chan2):
    # print("mask shape: ", masks_anat.shape)
    # coords = np.argwhere(masks_anat > 0)
    # rows = coords[:, 0]
    # cols = coords[:, 1]
    # func_roi_pixels = fluo[rows, cols]  # g_i
    # anat_roi_pixels = fluo_chan2[rows, cols]  # r_i

    # # Reshape data for regression
    # func_roi_pixels = func_roi_pixels.reshape(-1, 1)
    # anat_roi_pixels = anat_roi_pixels.reshape(-1, 1)

    # model = LinearRegression()
    # model.fit(func_roi_pixels, anat_roi_pixels)

    # slope = model.coef_
    # offset = model.intercept_
    # print("slope shape: ", slope.shape)
    # print("offset shape: ", offset.shape)

    # r_sq = model.score(func_roi_pixels, anat_roi_pixels)

    # print(f"slope: {slope}, offset: {offset}")
    # print(f"R2: {r_sq}")

    # return slope, offset, coords

    # Initial guess: slope = 1, intercepts = 0 for each neuron
    # initial_params = np.zeros(fluo.shape[0] + 1)
    # initial_params[0] = 1  # Initial slope guess

    # # Perform optimization to fit the model
    # result = minimize(objective, initial_params, args=(fluo, fluo_chan2))

    # # Extract the fitted parameters
    # fitted_slope = result.x[0]
    # fitted_intercepts = result.x[1:]

    # print(f"Fitted slope: {fitted_slope}")
    # print(f"Fitted intercepts: {fitted_intercepts}")

    # return fitted_slope
    num_neurons = fluo.shape[0]
    slopes = []
    intercepts = []

    # Fit a separate linear regression for each neuron
    for i in range(num_neurons):
        X = fluo[i, :].reshape(-1, 1)  # Reshape to (sequence_length, 1)
        y = fluo_chan2[i, :]

        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Append the slope and intercept for this neuron
        slopes.append(model.coef_[0])
        intercepts.append(model.intercept_)

    return slopes, intercepts


def Fchan2_corrected_anat(fluo, fluo_chan2, slopes, offsets, masks_anat):
    Fchan2_corrected = fluo_chan2.copy()

    for t in range(fluo.shape[0]):
        Fchan2_corrected[t] = fluo_chan2[t] - \
            (slopes[t] * fluo[t] + offsets[t])

    Fchan2_means = np.mean(Fchan2_corrected, -1)

    return Fchan2_corrected, Fchan2_means


# def Fchan2_corrected_mean(Fchan2_corrected, masks_anat):
#     unique_rois = np.unique(masks_anat)[1:]  # Exclude background (0)
#     mean_anat_corrected = np.zeros(
#         (Fchan2_corrected.shape[0], len(unique_rois)))

#     for i, roi in enumerate(unique_rois):
#         roi_coords = np.argwhere(masks_anat == roi)
#         for t in range(Fchan2_corrected.shape[0]):
#             mean_anat_corrected[i] = np.mean(
#                 Fchan2_corrected[roi_coords[:, 0], roi_coords[:, 1]])

#     return mean_anat_corrected


def update_mean_anat(mean_anat, corrected_means, masks_anat):
    coords = np.argwhere(masks_anat != 0)
    # rows = coords[:, 0]
    # cols = coords[:, 1]

    mean_anat_image_corr = mean_anat.copy()
    for c in coords:
        i = masks_anat[c[0], c[1]] - 1
        mean_anat_image_corr[c[0], c[1]] = corrected_means[i]

    return mean_anat_image_corr
