import os
import h5py
import tifffile
import numpy as np
from tqdm import tqdm
from cellpose import models, io
import matplotlib.pyplot as plt
from scipy.stats import linregress
from skimage.segmentation import find_boundaries
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter


def quadrant_mask(Ly, Lx, ny, nx, sT):
    """
    Creates a Gaussian-filtered mask for a specified quadrant of the image.

    Parameters:
    - Ly (int): Height of the image (number of rows).
    - Lx (int): Width of the image (number of columns).
    - ny (array_like): Array of y-indices defining the rows of the quadrant.
    - nx (array_like): Array of x-indices defining the columns of the quadrant.
    - sT (float): Sigma value for the Gaussian filter.

    Returns:
    - mask (numpy.ndarray): Gaussian-filtered mask of shape (Ly, Lx).
    """
    mask = np.zeros((Ly, Lx), dtype=np.float32)
    mask[np.ix_(ny, nx)] = 1
    mask = gaussian_filter(mask, sT)
    return mask


def correct_bleedthrough(Ly, Lx, nblks, mimg, mimg2):
    """
    Corrects bleedthrough of one channel into another by performing non-rigid regression.

    This function divides the images into blocks and computes local weights to adjust
    for bleedthrough, effectively subtracting the estimated bleedthrough from the
    second channel.

    Parameters:
    - Ly (int): Height of the image (number of rows).
    - Lx (int): Width of the image (number of columns).
    - nblks (int): Number of blocks to divide the image into along each axis.
    - mimg (numpy.ndarray): Mean image of the first channel (e.g., green channel).
    - mimg2 (numpy.ndarray): Mean image of the second channel (e.g., red channel) to be corrected.

    Returns:
    - mimg2_corrected (numpy.ndarray): Corrected mean image of the second channel.
    """
    sT = np.round((Ly + Lx) / (nblks * 2) * 0.25)
    mask = np.zeros((Ly, Lx, nblks, nblks), dtype=np.float32)
    weights = np.zeros((nblks, nblks), dtype=np.float32)
    yb = np.linspace(0, Ly, nblks + 1).astype(int)
    xb = np.linspace(0, Lx, nblks + 1).astype(int)

    for iy in range(nblks):
        for ix in range(nblks):
            ny = np.arange(yb[iy], yb[iy + 1])
            nx = np.arange(xb[ix], xb[ix + 1])
            mask[:, :, iy, ix] = quadrant_mask(Ly, Lx, ny, nx, sT)
            x = mimg[np.ix_(ny, nx)].flatten()
            x2 = mimg2[np.ix_(ny, nx)].flatten()
            a = (x * x2).sum() / (x * x).sum()
            weights[iy, ix] = a

    mask_sum = mask.sum(axis=-1).sum(axis=-1)
    mask /= mask_sum[:, :, np.newaxis, np.newaxis]
    mask *= weights
    mask *= mimg[:, :, np.newaxis, np.newaxis]
    mimg2_corrected = mimg2 - mask.sum(axis=-1).sum(axis=-1)
    mimg2_corrected = np.maximum(0, mimg2_corrected)
    return mimg2_corrected
