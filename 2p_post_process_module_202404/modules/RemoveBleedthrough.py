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


def train_reg_model(ops, mean_anat, mean_func):
    # Debugging: Print the ops dictionary
    # print("ops dictionary:", ops)
    # check if slope and offset already computed
    if os.path.exists(os.path.join(ops['save_path0'], 'slope_offset.npy')):
        slope, offset = np.load(os.path.join(
            ops['save_path0'], 'slope_offset.npy'))
    else:
        # compute slope and offset
        # fit func = slope * anat + offset
        print("mean_anat.shape: ", mean_anat.shape)
        print("mean_func.shape: ", mean_func.shape)
        # flatten the arrays
        mean_anat_flat = mean_anat.flatten()
        mean_func_flat = mean_func.flatten().reshape(-1, 1)
        reg_model = LinearRegression().fit(mean_func_flat, mean_anat_flat)
        slope = reg_model.coef_[0]
        offset = reg_model.intercept_
        # save slope and offset
        f = h5py.File(
            os.path.join(ops['save_path0'], 'masks.h5'),
            'w')

        print(f"Slope and offset: {slope}, {offset}")

        f['slope'] = slope
        f['offset'] = offset
        f.close()
    return slope, offset


def remove_green_bleedthrough(offset, slope, mean_func, mean_anat):
    # corrected anat channel = original anat channel - (slope * original anatomical channel)
    mean_anat_new = mean_anat - (slope * mean_func + offset)
    return mean_anat_new
