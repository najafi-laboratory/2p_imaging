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




#def train_reg_model(mean_anat, mean_func, masks_anat):
    # multivariate regression on ROIs

    

    #coords = np.argwhere(masks_anat == 1)

    #rows = coords[:, 0]
    #cols = coords[:, 1]
    #anat_roi_pixels = mean_anat[rows, cols]  # r_i
    #func_roi_pixels = mean_func[rows, cols]  # g_i

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

    #anat_roi_pixels = anat_roi_pixels.reshape(-1, 1)
    #func_roi_pixels = func_roi_pixels.reshape(-1, 1)

    #model = LinearRegression()
    #model.fit(func_roi_pixels, anat_roi_pixels)

    #slope = model.coef_[0][0]
    #offset = model.intercept_[0]

    #r_sq = model.score(func_roi_pixels, anat_roi_pixels)

    #print(f"slope: {slope}, offset: {offset}")
    #print(f"R2: {r_sq}")

    # Return list of slopes, offsets, coordinates
    #return slope, offset, coords


#def remove_green_bleedthrough(offset, slope, mean_func, mean_anat, coordinates):
    # for each coordinate
    # mean_anat_corrected = mean_anat
   # mean_anat_corrected = mean_anat.copy()
    # mean_anat_corrected(coordinate) -= (slope * mean_func(coordinate) + offset)

    #for c in coordinates:
        #mean_anat_corrected[c[0], c[1]] = mean_anat_corrected[c[0],
                                                              #c[1]] - (slope * mean_func[c[0], c[1]] + offset)

    # return mean_anat_corrected
    #return mean_anat_corrected



def train_reg_model(fluo, fluo_chan2, masks_anat):
    coords = np.argwhere(masks_anat > 0)
    rows = coords[:, 0]
    cols = coords[:, 1]
    func_roi_pixels = fluo[:, rows, cols]  # g_i
    anat_roi_pixels = fluo_chan2[:, rows, cols]  # r_i

    # Reshape data for regression
    func_roi_pixels = func_roi_pixels.reshape(-1, 1)
    anat_roi_pixels = anat_roi_pixels.reshape(-1, 1)

    model = LinearRegression()
    model.fit(func_roi_pixels, anat_roi_pixels)

    slope = model.coef_[0][0]
    offset = model.intercept_[0]

    r_sq = model.score(func_roi_pixels, anat_roi_pixels)

    print(f"slope: {slope}, offset: {offset}")
    print(f"R2: {r_sq}")

    return slope, offset, coords

def Fchan2_corrected_anat(fluo, fluo_chan2, slope, offset, masks_anat):
    coords = np.argwhere(masks_anat > 0)
    rows = coords[:, 0]
    cols = coords[:, 1]

    Fchan2_corrected = fluo_chan2.copy()
    for t in range(fluo.shape[0]):
        for c in coords:
            Fchan2_corrected[t, c[0], c[1]] = fluo_chan2[t, c[0], c[1]] - (slope * fluo[t, c[0], c[1]] + offset)

    return Fchan2_corrected

def Fchan2_corrected_mean(Fchan2_corrected, masks_anat):
    unique_rois = np.unique(masks_anat)[1:]  # Exclude background (0)
    mean_anat_corrected = np.zeros((Fchan2_corrected.shape[0], len(unique_rois)))

    for i, roi in enumerate(unique_rois):
        roi_coords = np.argwhere(masks_anat == roi)
        for t in range(Fchan2_corrected.shape[0]):
            mean_anat_corrected[ i] = np.mean(Fchan2_corrected[:, roi_coords[:, 0], roi_coords[:, 1]])

    return mean_anat_corrected

def update_mean_anat(mean_anat, mean_anat_corrected, masks_anat):
    unique_rois = np.unique(masks_anat)[1:]  
    mean_anat_image = mean_anat.copy()

    for i, roi in enumerate(unique_rois):
        roi_coords = np.argwhere(masks_anat == roi)
        mean_value = np.mean(mean_anat_corrected[:, i])
        for coord in roi_coords:
            mean_anat_image[coord[0], coord[1]] = mean_value

    return mean_anat_image

