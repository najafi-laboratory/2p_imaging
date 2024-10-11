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
    fluo_means = np.mean(fluo, -1)
    fluo_chan2_means = np.mean(fluo_chan2, -1)

    # return fitted_slope
    num_neurons = fluo.shape[0]
    # slopes = []
    # intercepts = []

    # Fit a separate linear regression for each neuron
    # for i in range(num_neurons):
    #     X = fluo[i, :].reshape(-1, 1)  # Reshape to (sequence_length, 1)
    #     y = fluo_chan2[i, :]

    #     # Create and fit the linear regression model
    #     model = LinearRegression()
    #     model.fit(X, y)

    #     # Append the slope and intercept for this neuron
    #     slopes.append(model.coef_[0])
    #     intercepts.append(model.intercept_)

    X = fluo_means.reshape(-1, 1)
    y = fluo_chan2_means.reshape(-1, 1)
    model = LinearRegression()

    model.fit(X, y)
    slope = model.coef_
    offset = model.intercept_

    return slope, offset, fluo_means, fluo_chan2_means


def Fchan2_corrected_anat(fluo_means, fluo_chan2_means, slopes, offsets):
    Fchan2_corrected_means = fluo_chan2_means.copy()

    for t in range(len(fluo_means)):
        Fchan2_corrected_means[t] = fluo_chan2_means[t] - \
            (slopes * fluo_means[t] + offsets)

    # Fchan2_means = np.mean(Fchan2_corrected, -1)

    return Fchan2_corrected_means


def update_mean_anat(mean_anat, corrected_means, masks_anat):
    coords = np.argwhere(masks_anat != 0)
    # rows = coords[:, 0]
    # cols = coords[:, 1]

    mean_anat_image_corr = mean_anat.copy()
    for c in coords:
        i = masks_anat[c[0], c[1]] - 1
        mean_anat_image_corr[c[0], c[1]] = corrected_means[i]

    return mean_anat_image_corr
