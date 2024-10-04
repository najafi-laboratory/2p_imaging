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


def fit_reg_for_roi(mean_anat_pixel, mean_func_pixel):
    pass


def train_reg_model(ops, mean_anat, mean_func, labels):
    # multivariate regression on ROIs
    
    # first, identify anat ROIs with labels --> r_i
    # the same pixels in mean_func --> g_i
    
    # get list of pixel coordinates
    # fit multivariate regression r_i = m_i * g_i + b_i

    # return list of slopes, offsets, coordinates
    pass


def remove_green_bleedthrough(offsets, slopes, mean_func, mean_anat, coordinates):
    # for each coordinate
    # mean_anat_corrected = mean_anat
    # mean_anat_corrected(coordinate) = (mean * mean_func(coordinate) + offset)


    # return mean_anat_corrected
    pass
