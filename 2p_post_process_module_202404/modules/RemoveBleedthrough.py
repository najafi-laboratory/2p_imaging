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


def train_reg_model(fluo, fluo_chan2, masks_anat):
    print("mask shape: ", masks_anat.shape)
    coords = np.argwhere(masks_anat > 0)
    rows = coords[:, 0]
    cols = coords[:, 1]
    func_roi_pixels = fluo[rows, cols]  # g_i
    anat_roi_pixels = fluo_chan2[rows, cols]  # r_i

    # Reshape data for regression
    func_roi_pixels = func_roi_pixels.reshape(-1, 1)
    anat_roi_pixels = anat_roi_pixels.reshape(-1, 1)

    model = LinearRegression()
    model.fit(func_roi_pixels, anat_roi_pixels)

    slope = model.coef_
    offset = model.intercept_
    print("slope shape: ", slope.shape)
    print("offset shape: ", offset.shape)

    r_sq = model.score(func_roi_pixels, anat_roi_pixels)

    print(f"slope: {slope}, offset: {offset}")
    print(f"R2: {r_sq}")

    return slope, offset, coords


def Fchan2_corrected_anat(fluo, fluo_chan2, slope, offset, masks_anat):
    Fchan2_corrected = fluo_chan2.copy()

    for t in range(fluo.shape[0]):
        Fchan2_corrected[t] = fluo_chan2[t] - \
            (slope * fluo[t] + offset)

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


def update_mean_anat(mean_anat, mean_anat_corrected, masks_anat):
    unique_rois = np.unique(masks_anat)[1:]
    mean_anat_image = mean_anat.copy()

    for i, roi in enumerate(unique_rois):
        roi_coords = np.argwhere(masks_anat == roi)
        mean_value = np.mean(mean_anat_corrected[:, i])
        for coord in roi_coords:
            mean_anat_image[coord[0], coord[1]] = mean_value

    return mean_anat_image
