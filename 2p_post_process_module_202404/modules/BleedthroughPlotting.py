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
from scipy.ndimage import label

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

    iter_lst = []

    if masks is not None:
        iter_lst.append((masks, [255, 255, 255]))

    if labeled_masks_img is not None:
        iter_lst.append((labeled_masks_img[:, :, 0], [255, 255, 0]))

    if unsure_masks_img is not None:
        iter_lst.append((unsure_masks_img[:, :, 0], [0, 196, 255]))

    if with_mask:
        for mask, color in iter_lst:
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
        comp_mask_type,
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
    fig, ax = plt.subplots(1, 2, figsize=(25, 10))
    anat(ax[0], mean_anat, masks_anat, labeled_masks_img_orig,
         unsure_masks_img_orig, with_mask=True, title='Orig. Anat. + Mask')

    anat(ax[1], mean_anat_corrected, masks_anat_corrected, labeled_masks_img_corr,
         unsure_masks_img_corr, with_mask=True, title='Corr. Anat. + Mask')

    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(f'{comp_mask_type}_bleedthrough_channel_comparison.pdf')
    # # plt.show()


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


def identify_removed_neurons(masks_anat, masks_anat_corrected):
    # Label each connected component in both masks
    labeled_anat, num_features_anat = label(masks_anat)
    labeled_anat_corrected, num_features_corrected = label(
        masks_anat_corrected)

    # Create a set to track neurons present only in uncorrected mask
    only_in_anat = set()

    # Iterate through each labeled region in the uncorrected mask
    for i in range(1, num_features_anat + 1):
        # Create a mask for the current neuron in uncorrected mask
        neuron_mask = (labeled_anat == i)

        # Check if any pixels overlap in the corrected mask
        overlap = np.any(neuron_mask & (labeled_anat_corrected > 0))

        # If there is no overlap, it means this neuron is missing after correction
        if not overlap:
            only_in_anat.add(i)

    return only_in_anat


def identify_new_neurons(masks_anat, masks_anat_corrected):
    # Label each connected component in both masks
    labeled_anat, num_features_anat = label(masks_anat)
    labeled_anat_corrected, num_features_corrected = label(
        masks_anat_corrected)

    # Create a set to track neurons present only in corrected mask
    only_in_corrected = set()

    # Iterate through each labeled region in the corrected mask
    for i in range(1, num_features_corrected + 1):
        # Create a mask for the current neuron in corrected mask
        neuron_mask = (labeled_anat_corrected == i)

        # Check if any pixels overlap in the uncorrected mask
        overlap = np.any(neuron_mask & (labeled_anat > 0))

        # If there is no overlap, it means this neuron is new after correction
        if not overlap:
            only_in_corrected.add(i)

    return only_in_corrected


def isolate_neurons(mask, neuron_indices):
    # Label each connected component in the mask
    labeled_mask, num_features = label(mask)

    # Create an empty mask to store only the specified neurons
    isolated_neurons = np.zeros_like(mask, dtype=int)

    # Iterate through each specified neuron index
    for i in neuron_indices:
        # Create a mask for the current neuron in the labeled mask
        neuron_mask = (labeled_mask == i)

        # Add this neuron to the isolated_neurons mask
        isolated_neurons[neuron_mask] = i

    return isolated_neurons


def removed_neurons_comparison_image(
        comp_mask_type,
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
    Generates a figure with comparisons of the original and corrected anatomical channel images with masks, including removed neurons.

    Parameters:
    - mean_anat (numpy array): The mean anatomical image data.
    - masks_anat (numpy array): The masks for anatomical ROIs.
    - labeled_masks_img_orig (numpy array): The labeled masks image for original visualization.
    - unsure_masks_img_orig (numpy array): The unsure masks image for original visualization.
    - with_mask (bool, optional): Whether to include mask boundaries in the plot. Defaults to True.
    - mean_func (numpy array, optional): The mean functional image data for applying a mean function. Defaults to None.

    This function generates a figure with comparisons of the original and corrected anatomical channel images, including plots for misidentified, correct, and missed ROIs.
    """
    # Identify removed neurons
    labeled_anat, num_features_anat = label(masks_anat)
    labeled_anat_corrected, num_features_corrected = label(
        masks_anat_corrected)

    removed_neurons = list(identify_removed_neurons(
        masks_anat, masks_anat_corrected))
    # Isolate removed neurons
    removed_neurons_mask = isolate_neurons(masks_anat, removed_neurons)

    # Generate the figure
    fig, ax = plt.subplots(1, 2, figsize=(25, 10))
    anat(ax[0], mean_anat=mean_anat, masks=removed_neurons_mask, labeled_masks_img=unsure_masks_img_corr,
         unsure_masks_img=unsure_masks_img_orig, with_mask=True, title='Orig+removed')

    anat(ax[1], mean_anat_corrected, masks_anat_corrected, labeled_masks_img_corr,
         unsure_masks_img_corr, with_mask=True, title='Corr. Anat. + Mask')
    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(f'removed_neurons_bleedthrough_channel_comparison.pdf')
    # # plt.show()


def new_neurons_comparison_image(
        comp_mask_type,
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
    Generates a figure with comparisons of the original and corrected anatomical channel images with masks, including removed neurons.

    Parameters:
    - mean_anat (numpy array): The mean anatomical image data.
    - masks_anat (numpy array): The masks for anatomical ROIs.
    - labeled_masks_img_orig (numpy array): The labeled masks image for original visualization.
    - unsure_masks_img_orig (numpy array): The unsure masks image for original visualization.
    - with_mask (bool, optional): Whether to include mask boundaries in the plot. Defaults to True.
    - mean_func (numpy array, optional): The mean functional image data for applying a mean function. Defaults to None.

    This function generates a figure with comparisons of the original and corrected anatomical channel images, including plots for misidentified, correct, and missed ROIs.
    """
    # Identify removed neurons
    labeled_anat, num_features_anat = label(masks_anat)
    labeled_anat_corrected, num_features_corrected = label(
        masks_anat_corrected)

    new_neurons = list(identify_new_neurons(
        masks_anat, masks_anat_corrected))
    # Isolate removed neurons
    new_neurons_mask = isolate_neurons(masks_anat, new_neurons)

    # Generate the figure
    fig, ax = plt.subplots(1, 2, figsize=(25, 10))
    anat(ax[0], mean_anat=mean_anat, masks=masks_anat, labeled_masks_img=unsure_masks_img_orig,
         unsure_masks_img=unsure_masks_img_orig, with_mask=True, title='Orig')

    anat(ax[1], mean_anat_corrected, new_neurons_mask, labeled_masks_img_corr,
         unsure_masks_img_corr, with_mask=True, title='Corr. Anat. New')
    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(f'new_neurons_bleedthrough_channel_comparison.pdf')
    # # plt.show()
