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

from .RemoveBleedthrough import *
from .LabelExcInh import *


def prep_img(mean_anat, mean_func=None):
    """
    Prepares an image to visualize anatomical channel data.

    Parameters:
    - mean_anat (numpy array): The mean anatomical image data.
    - mean_func (numpy array, optional): The mean functional image data. Defaults to None.

    Returns:
    - img (numpy array): A composite image for visualization.
    """
    img = np.zeros((mean_anat.shape[0], mean_anat.shape[1], 3), dtype='int32')
    img[:, :, 0] = adjust_contrast(mean_anat)
    if mean_func is not None:
        img[:, :, 1] = adjust_contrast(mean_func)

    return img


def get_labels_before_after(labels, labels_corrected):
    inhibitory_rois_before = np.where(labels == 1)[0] + 1
    inhibitory_rois_after = np.where(labels_corrected == 1)[0] + 1

    return inhibitory_rois_before, inhibitory_rois_after


def adjust_contrast(org_img, lower_percentile=50, upper_percentile=99):
    """
    Adjusts the contrast of an image by clipping and scaling the intensity values.

    Parameters:
    - org_img (numpy array): The original image data.
    - lower_percentile (int, optional): The lower percentile for contrast adjustment. Defaults to 50.
    - upper_percentile (int, optional): The upper percentile for contrast adjustment. Defaults to 99.

    Returns:
    - img (numpy array): The image with adjusted contrast.
    """
    lower = np.percentile(org_img, lower_percentile)
    upper = np.percentile(org_img, upper_percentile)
    img = np.clip((org_img - lower) * 255 / (upper - lower), 0, 255)
    img = img.astype('int32')
    return img

# adjust layout for masks plot.


def adjust_layout(ax):
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def anat(ax, mean_anat, masks, labeled_masks_img, unsure_masks_img, with_mask=True, title='anatomy channel mean image'):
    """
    Plots the anatomy channel mean image with optional mask boundaries.

    Parameters:
    - ax (matplotlib axes): The axes to plot on.
    - mean_anat (numpy array): The mean anatomical image data.
    - masks (numpy array): The masks for ROIs.
    - labeled_masks_img (numpy array): The labeled masks image for visualization.
    - unsure_masks_img (numpy array): The unsure masks image for visualization.
    - with_mask (bool, optional): Whether to include mask boundaries in the plot. Defaults to True.
    - title (str, optional): The title of the plot. Defaults to 'anatomy channel mean image'.
    """
    anat_img = np.zeros(
        (mean_anat.shape[0], mean_anat.shape[1], 3), dtype='int32')
    anat_img[:, :, 0] = adjust_contrast(mean_anat)
    anat_img = adjust_contrast(anat_img)
    if with_mask:
        for mask, color in [(masks, [255, 255, 255]), (labeled_masks_img[:, :, 0], [255, 255, 0]), (unsure_masks_img[:, :, 0], [0, 196, 255])]:
            x_all, y_all = np.where(find_boundaries(mask))
            for x, y in zip(x_all, y_all):
                anat_img[x, y, :] = np.array(color)
    ax.matshow(anat_img)
    adjust_layout(ax)
    ax.set_title(f'{title}')


def roi_comparison_image(ax, mean_anat, masks_anat, labels, labels_corrected, rois, title, mean_func):
    """
    Plots the comparison image for ROI analysis with optional mean function application.

    Parameters:
    - ax (matplotlib axes): The axes to plot on.
    - mean_anat (numpy array): The mean anatomical image data.
    - masks_anat (numpy array): The masks for anatomical ROIs.
    - labels (numpy array): The labels for ROIs before correction.
    - labels_corrected (numpy array): The labels for ROIs after correction.
    - rois (list): The list of ROI IDs to highlight in the plot.
    - title (str): The title of the plot.
    - mean_func (numpy array, optional): The mean functional image data for applying a mean function. Defaults to None.

    This function plots the comparison image for ROI analysis, highlighting the specified ROIs with a white boundary. It also applies a mean function to the image if provided.
    """

    img = prep_img(mean_anat, mean_func)

    # Add a white boundary around each misidentified inhibitory ROI
    for roi_id in rois:
        boundaries = find_boundaries(masks_anat == roi_id)
        x_all, y_all = np.where(boundaries)
        for x, y in zip(x_all, y_all):
            img[x, y, :] = np.array([255, 255, 255])

    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')


def main_channel_comparison_image(
        labels,
        labels_corrected,
        mean_anat,
        mean_anat_corrected,
        masks_anat,
        masks_anat_corrected,
        labeled_masks_img_orig,
        labeled_masks_img_corr,
        unsure_masks_img_orig,
        unsure_masks_img_corr,
        with_mask,
        mean_func):
    """
    Generates a figure with comparisons of the original and corrected anatomical channel images with masks.

    Parameters:
    - mean_anat (numpy array): The mean anatomical image data.
    - masks_anat (numpy array): The masks for anatomical ROIs.
    - labeled_masks_img_orig (numpy array): The labeled masks image for original visualization.
    - unsure_masks_img_orig (numpy array): The unsure masks image for original visualization.
    - with_mask (bool, optional): Whether to include mask boundaries in the plot. Defaults to True.
    - mean_func (numpy array, optional): The mean functional image data for applying a mean function. Defaults to None.

    This function generates a figure with comparisons of the original and corrected anatomical channel images, including plots for misidentified, correct, and missed ROIs.
    """
    fig, ax = plt.subplots(4, 2, figsize=(25, 10))
    anat(ax[0][0], mean_anat, masks_anat, labeled_masks_img_orig,
         unsure_masks_img_orig, with_mask=True, title='Orig. Anat. + Mask')

    anat(ax[0][1], mean_anat_corrected, masks_anat_corrected, labeled_masks_img_corr,
         unsure_masks_img_corr, with_mask=True, title='Corr. Anat. + Mask')

    inhibitory_rois_before, inhibitory_rois_after = get_labels_before_after(
        labels, labels_corrected)

    # 1. Plot the ROIs present before the correction but not after.
    #    I.e., the ROIs that were misidentified.
    misidentified_rois = np.setdiff1d(
        inhibitory_rois_before, inhibitory_rois_after)

    roi_comparison_image(
        ax[1][0], mean_anat,
        masks_anat, labels,
        labels_corrected, rois=misidentified_rois,
        title='Anat ROIs ONLY Before Corr (No Func)',
        mean_func=None)
    roi_comparison_image(
        ax[1][1], mean_anat,
        masks_anat, labels,
        labels_corrected,
        rois=misidentified_rois,
        title='Anat ROIs ONLY Before Corr (With Func)',
        mean_func=mean_func)

    # 2. Plot the ROIs present both before and after the correction.
    #    These are the ROIs that were correctly identified.
    correct_rois = np.intersect1d(
        inhibitory_rois_before, inhibitory_rois_after)

    roi_comparison_image(
        ax[2][0], mean_anat,
        masks_anat, labels,
        labels_corrected, rois=correct_rois,
        title='Anat ROIs Before AND After Corr (No Func)',
        mean_func=None)
    roi_comparison_image(
        ax[2][1], mean_anat,
        masks_anat, labels,
        labels_corrected, rois=correct_rois,
        title='Anat ROIs Before AND After Corr (With Func)',
        mean_func=mean_func)

    # 3. Plot the ROIs present after the correction but not before.
    #    These are the ROIs that were missed originally.
    missed_rois = np.setdiff1d(
        inhibitory_rois_after, inhibitory_rois_before)

    roi_comparison_image(
        ax[3][0], mean_anat,
        masks_anat, labels,
        labels_corrected, rois=missed_rois,
        title='Anat ROIs ONLY After Corr (No Func)',
        mean_func=None)
    roi_comparison_image(
        ax[3][1], mean_anat,
        masks_anat, labels,
        labels_corrected, rois=missed_rois,
        title='Anat ROIs ONLY After Corr (With Func)',
        mean_func=mean_func)

    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig('bleedthrough_channel_comparison.pdf')
    # plt.show()


def plot_anat_intensities(original_anat_channel, corrected_anat_channel):
    """
    Plots the average intensity of original and corrected red channels for each neuron.

    Parameters:
    original_anat_channel (numpy array): The original anatomical channel image.
    corrected_anat_channel (numpy array): The corrected anatomical channel image.
    """

    x = np.arange(original_anat_channel.shape[0])

    y_1 = np.mean(original_anat_channel, -1)
    y_2 = np.mean(corrected_anat_channel, -1)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_1, label='original')
    plt.plot(x, y_2, label='corrected')
    plt.legend()

    plt.show()
