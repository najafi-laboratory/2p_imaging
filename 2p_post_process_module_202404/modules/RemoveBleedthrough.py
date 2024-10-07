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


from sklearn.linear_model import LinearRegression
import numpy as np


def train_reg_model(mean_anat, mean_func, masks_anat):
    # multivariate regression on ROIs

    coords = np.argwhere(masks_anat == 1)

    rows = coords[:, 0]
    cols = coords[:, 1]
    anat_roi_pixels = mean_anat[rows, cols]  # r_i
    func_roi_pixels = mean_func[rows, cols]  # g_i

    # Initialize lists to store slopes and offsets for each variate
    # slopes = []
    # offsets = []

    # # Perform regression for each variate (i.e., for each coordinate)
    # for i in range(len(anat_roi_pixels)):
    #     anat_variate = anat_roi_pixels[i].reshape(-1, 1)  # Reshape for sklearn
    #     func_variate = func_roi_pixels[i].reshape(-1, 1)

    #     # Create and fit the linear regression model
    #     model = LinearRegression()
    #     model.fit(func_variate, anat_variate)

    #     # Store the slope and intercept for this variate
    #     slopes.append(model.coef_[0][0])  # Extract single value
    #     offsets.append(model.intercept_[0])  # Extract single value

    # print(slopes)
    # print(offsets)

    anat_roi_pixels = anat_roi_pixels.reshape(-1, 1)
    func_roi_pixels = func_roi_pixels.reshape(-1, 1)

    model = LinearRegression()
    model.fit(func_roi_pixels, anat_roi_pixels)

    slope = model.coef_[0][0]
    offset = model.intercept_[0]

    r_sq = model.score(func_roi_pixels, anat_roi_pixels)

    print(f"slope: {slope}, offset: {offset}")
    print(f"R2: {r_sq}")

    # Return list of slopes, offsets, coordinates
    return slope, offset, coords


def remove_green_bleedthrough(offset, slope, mean_func, mean_anat, coordinates):
    # for each coordinate
    # mean_anat_corrected = mean_anat
    mean_anat_corrected = mean_anat.copy()
    # mean_anat_corrected(coordinate) -= (slope * mean_func(coordinate) + offset)

    for c in coordinates:
        mean_anat_corrected[c[0], c[1]] = mean_anat_corrected[c[0],
                                                              c[1]] - (slope * mean_func[c[0], c[1]] + offset)

    # return mean_anat_corrected
    return mean_anat_corrected
