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
from scipy.ndimage import label
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from .RemoveBleedthrough import *


def identify_removed_neurons(masks_anat, masks_anat_corrected):
    """
    Identifies neurons present in the original anatomical mask but not in the corrected mask.

    Parameters:
    - masks_anat (numpy.ndarray): Original anatomical masks.
    - masks_anat_corrected (numpy.ndarray): Corrected anatomical masks.

    Returns:
    - only_in_anat (set): Set of neuron IDs present only in the original mask.
    """
    # Label each connected component in both masks
    labeled_anat, num_features_anat = label(masks_anat)
    labeled_anat_corrected, _ = label(masks_anat_corrected)

    only_in_anat = set()

    for i in range(1, num_features_anat + 1):
        neuron_mask = (labeled_anat == i)
        overlap = np.any(neuron_mask & (labeled_anat_corrected > 0))

        if not overlap:
            only_in_anat.add(i)

    return only_in_anat


def identify_new_neurons(masks_anat, masks_anat_corrected):
    """
    Identifies neurons present in the corrected anatomical mask but not in the original mask.

    Parameters:
    - masks_anat (numpy.ndarray): Original anatomical masks.
    - masks_anat_corrected (numpy.ndarray): Corrected anatomical masks.

    Returns:
    - only_in_corrected (set): Set of neuron IDs present only in the corrected mask.
    """
    labeled_anat, _ = label(masks_anat)
    labeled_anat_corrected, num_features_corrected = label(
        masks_anat_corrected)

    only_in_corrected = set()

    for i in range(1, num_features_corrected + 1):
        neuron_mask = (labeled_anat_corrected == i)
        overlap = np.any(neuron_mask & (labeled_anat != 0))

        if not overlap:
            only_in_corrected.add(i)

    return only_in_corrected


def identify_common_neurons(masks_anat, masks_anat_corrected):
    """
    Identifies neurons present in both the original and corrected anatomical masks.

    Parameters:
    - masks_anat (numpy.ndarray): Original anatomical masks.
    - masks_anat_corrected (numpy.ndarray): Corrected anatomical masks.

    Returns:
    - common_neurons (set): Set of neuron IDs present in both masks.
    """
    labeled_anat, num_features_anat = label(masks_anat)
    labeled_anat_corrected, _ = label(masks_anat_corrected)

    common_neurons = set()
    for i in range(1, num_features_anat + 1):
        neuron_mask = (labeled_anat == i)
        overlap = np.any(neuron_mask & (labeled_anat_corrected > 0))

        if overlap:
            common_neurons.add(i)

    return common_neurons


def isolate_neurons(mask, neuron_indices):
    """
    Isolates specific neurons from a labeled mask.

    Parameters:
    - mask (numpy.ndarray): Labeled mask from which to isolate neurons.
    - neuron_indices (list or set): Neuron IDs to isolate.

    Returns:
    - isolated_neurons (numpy.ndarray): Mask containing only the specified neurons.
    """
    labeled_mask, _ = label(mask)

    isolated_neurons = np.zeros_like(mask, dtype=int)

    for i in neuron_indices:
        neuron_mask = (labeled_mask == i)
        isolated_neurons[neuron_mask] = i

    return isolated_neurons


def prep_img(mean_anat, mean_func=None):
    """
    Prepares an RGB image by adjusting the contrast of the anatomical and functional images.

    Parameters:
    - mean_anat (numpy.ndarray): Mean anatomical image data.
    - mean_func (numpy.ndarray, optional): Mean functional image data.

    Returns:
    - img (numpy.ndarray): Composite RGB image.
    """
    img = np.zeros((mean_anat.shape[0], mean_anat.shape[1], 3), dtype='int32')
    img[:, :, 0] = adjust_contrast(mean_anat)
    if mean_func is not None:
        img[:, :, 1] = adjust_contrast(mean_func)

    return img


def adjust_contrast(org_img, lower_percentile=50, upper_percentile=99):
    """
    Adjusts the contrast of an image by clipping and scaling intensity values.

    Parameters:
    - org_img (numpy.ndarray): Original image data.
    - lower_percentile (int, optional): Lower percentile for contrast adjustment.
    - upper_percentile (int, optional): Upper percentile for contrast adjustment.

    Returns:
    - img (numpy.ndarray): Image with adjusted contrast.
    """
    lower = np.percentile(org_img, lower_percentile)
    upper = np.percentile(org_img, upper_percentile)
    # Add a small epsilon to prevent division by zero
    img = np.clip((org_img - lower) * 255 / (upper - lower + 1e-5), 0, 255)
    img = img.astype('int32')
    return img


def adjust_layout(ax):
    """
    Adjusts the layout of a matplotlib Axes object by hiding ticks and spines.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to adjust.
    """
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def display_mean_image(mean_img, title='Anatomical Mean Image'):
    """
    Displays the mean anatomical image in red.

    Parameters:
    - mean_img (numpy.ndarray): The mean anatomical image data.
    - title (str, optional): The title of the plot.

    This function displays the mean anatomical image in red, similar to other images generated in BleedthroughPlotting.py.
    """
    img = prep_img(mean_img)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.savefig('plot_results/mean_anat_image.png')


def anat(ax, mean_anat, masks=None, labeled_masks_img=None, unsure_masks_img=None,
         with_mask=True, title='Anatomy Channel Mean Image'):
    """
    Plots the anatomy channel mean image with optional mask boundaries.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to plot on.
    - mean_anat (numpy.ndarray): The mean anatomical image data.
    - masks (numpy.ndarray, optional): Masks for ROIs.
    - labeled_masks_img (numpy.ndarray, optional): Labeled masks image for visualization.
    - unsure_masks_img (numpy.ndarray, optional): Unsure masks image for visualization.
    - with_mask (bool, optional): Whether to include mask boundaries in the plot.
    - title (str, optional): The title of the plot.
    """
    anat_img = np.zeros(
        (mean_anat.shape[0], mean_anat.shape[1], 3), dtype='int32')
    anat_img[:, :, 0] = adjust_contrast(mean_anat)
    anat_img = adjust_contrast(anat_img)

    iter_lst = []

    if masks is not None:
        iter_lst.append((masks, [255, 255, 255]))  # White

    if labeled_masks_img is not None:
        iter_lst.append((labeled_masks_img[:, :, 0], [255, 255, 0]))  # Yellow

    if unsure_masks_img is not None:
        iter_lst.append((unsure_masks_img[:, :, 0], [
                        0, 196, 255]))  # Light Blue

    if with_mask:
        for mask, color in iter_lst:
            x_all, y_all = np.where(find_boundaries(mask))
            anat_img[x_all, y_all, :] = color

    ax.imshow(anat_img)
    adjust_layout(ax)
    ax.set_title(title)


def roi_comparison_image(ax, mean_anat, masks_anat, labels, labels_corrected,
                         rois, title, mean_func=None):
    """
    Plots a comparison image for ROI analysis, highlighting specified ROIs.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to plot on.
    - mean_anat (numpy.ndarray): The mean anatomical image data.
    - masks_anat (numpy.ndarray): Masks for anatomical ROIs.
    - labels (numpy.ndarray): Labels for ROIs before correction.
    - labels_corrected (numpy.ndarray): Labels for ROIs after correction.
    - rois (list): List of ROI IDs to highlight in the plot.
    - title (str): The title of the plot.
    - mean_func (numpy.ndarray, optional): Mean functional image data.

    This function plots the comparison image for ROI analysis, highlighting the specified ROIs with a white boundary.
    """
    img = prep_img(mean_anat, mean_func)

    # Add a white boundary around each specified ROI
    for roi_id in rois:
        boundaries = find_boundaries(masks_anat == roi_id)
        x_all, y_all = np.where(boundaries)
        img[x_all, y_all, :] = [255, 255, 255]

    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')


def main_channel_comparison_image(comp_mask_type, labels, labels_corrected,
                                  mean_anat, mean_anat_corrected,
                                  masks_anat, masks_anat_corrected,
                                  labeled_masks_img_orig, labeled_masks_img_corr,
                                  unsure_masks_img_orig, unsure_masks_img_corr,
                                  with_mask=True, mean_func=None):
    """
    Generates a figure comparing the original and corrected anatomical images with masks.

    Parameters:
    - comp_mask_type (str): Type of comparison mask.
    - labels (numpy.ndarray): Labels before correction.
    - labels_corrected (numpy.ndarray): Labels after correction.
    - mean_anat (numpy.ndarray): Mean anatomical image before correction.
    - mean_anat_corrected (numpy.ndarray): Mean anatomical image after correction.
    - masks_anat (numpy.ndarray): Anatomical masks before correction.
    - masks_anat_corrected (numpy.ndarray): Anatomical masks after correction.
    - labeled_masks_img_orig (numpy.ndarray): Labeled masks image before correction.
    - labeled_masks_img_corr (numpy.ndarray): Labeled masks image after correction.
    - unsure_masks_img_orig (numpy.ndarray): Unsure masks image before correction.
    - unsure_masks_img_corr (numpy.ndarray): Unsure masks image after correction.
    - with_mask (bool, optional): Whether to include mask boundaries.
    - mean_func (numpy.ndarray, optional): Mean functional image data.

    This function generates a figure comparing the original and corrected anatomical images, including masks.
    """
    fig, ax = plt.subplots(1, 2, figsize=(25, 10))
    anat(ax[0], mean_anat, masks_anat, labeled_masks_img_orig,
         unsure_masks_img_orig, with_mask=True, title='Original Anatomy + Mask')

    anat(ax[1], mean_anat_corrected, masks_anat_corrected, labeled_masks_img_corr,
         unsure_masks_img_corr, with_mask=True, title='Corrected Anatomy + Mask')

    plt.savefig(
        f'plot_results/{comp_mask_type}_bleedthrough_channel_comparison.png')


def removed_neurons_comparison_image(comp_mask_type, labels, labels_corrected,
                                     mean_anat, mean_anat_corrected,
                                     masks_anat, masks_anat_corrected,
                                     labeled_masks_img_orig, labeled_masks_img_corr,
                                     unsure_masks_img_orig, unsure_masks_img_corr,
                                     with_mask=True, mean_func=None):
    """
    Generates a figure comparing original and corrected images, highlighting removed neurons.

    Parameters:
    - comp_mask_type (str): Type of comparison mask.
    - labels (numpy.ndarray): Labels before correction.
    - labels_corrected (numpy.ndarray): Labels after correction.
    - mean_anat (numpy.ndarray): Mean anatomical image before correction.
    - mean_anat_corrected (numpy.ndarray): Mean anatomical image after correction.
    - masks_anat (numpy.ndarray): Anatomical masks before correction.
    - masks_anat_corrected (numpy.ndarray): Anatomical masks after correction.
    - labeled_masks_img_orig (numpy.ndarray): Labeled masks image before correction.
    - labeled_masks_img_corr (numpy.ndarray): Labeled masks image after correction.
    - unsure_masks_img_orig (numpy.ndarray): Unsure masks image before correction.
    - unsure_masks_img_corr (numpy.ndarray): Unsure masks image after correction.
    - with_mask (bool, optional): Whether to include mask boundaries.
    - mean_func (numpy.ndarray, optional): Mean functional image data.

    This function highlights neurons that are present in the original mask but not in the corrected mask.
    """
    # Identify removed neurons
    removed_neurons = list(identify_removed_neurons(
        masks_anat, masks_anat_corrected))
    # Isolate removed neurons
    removed_neurons_mask = isolate_neurons(masks_anat, removed_neurons)

    # Generate the figure
    fig, ax = plt.subplots(1, 2, figsize=(25, 10))
    anat(ax[0], mean_anat=mean_anat, masks=removed_neurons_mask,
         labeled_masks_img=None, unsure_masks_img=None,
         with_mask=True, title='Original + Removed Neurons')

    anat(ax[1], mean_anat_corrected, masks_anat_corrected,
         labeled_masks_img=labeled_masks_img_corr,
         unsure_masks_img=unsure_masks_img_corr,
         with_mask=True, title='Corrected Anatomy + Mask')

    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(
        'plot_results/removed_neurons_bleedthrough_channel_comparison.pdf')


def new_neurons_comparison_image(comp_mask_type, labels, labels_corrected,
                                 mean_anat, mean_anat_corrected,
                                 masks_anat, masks_anat_corrected,
                                 labeled_masks_img_orig, labeled_masks_img_corr,
                                 unsure_masks_img_orig, unsure_masks_img_corr,
                                 with_mask=True, mean_func=None):
    """
    Generates a figure comparing original and corrected images, highlighting new neurons.

    Parameters:
    - comp_mask_type (str): Type of comparison mask.
    - labels (numpy.ndarray): Labels before correction.
    - labels_corrected (numpy.ndarray): Labels after correction.
    - mean_anat (numpy.ndarray): Mean anatomical image before correction.
    - mean_anat_corrected (numpy.ndarray): Mean anatomical image after correction.
    - masks_anat (numpy.ndarray): Anatomical masks before correction.
    - masks_anat_corrected (numpy.ndarray): Anatomical masks after correction.
    - labeled_masks_img_orig (numpy.ndarray): Labeled masks image before correction.
    - labeled_masks_img_corr (numpy.ndarray): Labeled masks image after correction.
    - unsure_masks_img_orig (numpy.ndarray): Unsure masks image before correction.
    - unsure_masks_img_corr (numpy.ndarray): Unsure masks image after correction.
    - with_mask (bool, optional): Whether to include mask boundaries.
    - mean_func (numpy.ndarray, optional): Mean functional image data.

    This function highlights neurons that are present in the corrected mask but not in the original mask.
    """
    # Identify new neurons
    new_neurons = list(identify_new_neurons(masks_anat, masks_anat_corrected))
    # Isolate new neurons
    new_neurons_mask = isolate_neurons(masks_anat_corrected, new_neurons)

    # Generate the figure
    fig, ax = plt.subplots(1, 2, figsize=(25, 10))
    anat(ax[0], mean_anat=mean_anat, masks=None,
         labeled_masks_img=None, unsure_masks_img=None,
         with_mask=True, title='Original Anatomy')

    anat(ax[1], mean_anat_corrected, masks=new_neurons_mask,
         labeled_masks_img=None, unsure_masks_img=None,
         with_mask=True, title='Corrected Anatomy + New Neurons')

    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig('plot_results/new_neurons_bleedthrough_channel_comparison.pdf')


def common_neurons_comparison_image(comp_mask_type, labels, labels_corrected,
                                    mean_anat, mean_anat_corrected,
                                    masks_anat, masks_anat_corrected,
                                    labeled_masks_img_orig, labeled_masks_img_corr,
                                    unsure_masks_img_orig, unsure_masks_img_corr,
                                    with_mask=True, mean_func=None):
    """
    Generates a figure comparing original and corrected images, highlighting common neurons.

    Parameters:
    - comp_mask_type (str): Type of comparison mask.
    - labels (numpy.ndarray): Labels before correction.
    - labels_corrected (numpy.ndarray): Labels after correction.
    - mean_anat (numpy.ndarray): Mean anatomical image before correction.
    - mean_anat_corrected (numpy.ndarray): Mean anatomical image after correction.
    - masks_anat (numpy.ndarray): Anatomical masks before correction.
    - masks_anat_corrected (numpy.ndarray): Anatomical masks after correction.
    - labeled_masks_img_orig (numpy.ndarray): Labeled masks image before correction.
    - labeled_masks_img_corr (numpy.ndarray): Labeled masks image after correction.
    - unsure_masks_img_orig (numpy.ndarray): Unsure masks image before correction.
    - unsure_masks_img_corr (numpy.ndarray): Unsure masks image after correction.
    - with_mask (bool, optional): Whether to include mask boundaries.
    - mean_func (numpy.ndarray, optional): Mean functional image data.

    This function highlights neurons that are present in both the original and corrected masks.
    """
    # Identify common neurons
    common_neurons = list(identify_common_neurons(
        masks_anat, masks_anat_corrected))
    # Isolate common neurons
    common_neurons_mask = isolate_neurons(masks_anat, common_neurons)

    # Generate the figure
    fig, ax = plt.subplots(1, 2, figsize=(25, 10))
    anat(ax[0], mean_anat=mean_anat, masks=common_neurons_mask,
         labeled_masks_img=None, unsure_masks_img=None,
         with_mask=True, title='Original Anatomy + Common Neurons')

    anat(ax[1], mean_anat_corrected, masks=common_neurons_mask,
         labeled_masks_img=None, unsure_masks_img=None,
         with_mask=True, title='Corrected Anatomy + Common Neurons')

    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(
        'plot_results/common_neurons_bleedthrough_channel_comparison.pdf')


def get_excitory_rois(masks_anat, inhibitory_mask, unsure_mask):
    """
    Identifies excitatory ROIs by excluding inhibitory and unsure ROIs from the anatomical masks.

    Parameters:
    - masks_anat (numpy.ndarray): Anatomical masks.
    - inhibitory_mask (numpy.ndarray): Mask of inhibitory neurons.
    - unsure_mask (numpy.ndarray): Mask of unsure neurons.

    Returns:
    - excitory (numpy.ndarray): Mask of excitatory neurons.

    Note:
    - The function name contains a typo ('excitory' instead of 'excitatory').
      Consider renaming the function to 'get_excitatory_rois' for clarity.
    """
    inhibit = inhibitory_mask[:, :, 0]
    unsure = unsure_mask[:, :, 0]

    anat_minus_inhibit = identify_new_neurons(inhibit, masks_anat)
    anat_minus_inhibit = isolate_neurons(masks_anat, list(anat_minus_inhibit))

    excitory = identify_new_neurons(unsure, anat_minus_inhibit)
    excitory = isolate_neurons(masks_anat, list(excitory))

    return excitory


def superimpose(ax, with_mask=True, mean_func=None, max_func=None,
                mean_anat=None, inhibit_mask=None):
    """
    Creates a superimposed image of anatomical and functional data with optional mask overlays.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to plot on.
    - with_mask (bool, optional): Whether to include mask boundaries.
    - mean_func (numpy.ndarray, optional): Mean functional image data.
    - max_func (numpy.ndarray, optional): Max projection of functional data.
    - mean_anat (numpy.ndarray, optional): Mean anatomical image data.
    - inhibit_mask (numpy.ndarray, optional): Mask of inhibitory neurons.

    This function superimposes the mean anatomical and functional images, and overlays inhibitory neuron boundaries.
    """
    if max_func is None:
        f = mean_func
    elif mean_func is None:
        f = max_func
    else:
        raise ValueError('Need to specify either mean_func or max_func.')

    super_img = np.zeros((f.shape[0], f.shape[1], 3), dtype='int32')
    super_img[:, :, 0] = adjust_contrast(mean_anat)
    super_img[:, :, 1] = adjust_contrast(f)
    super_img = adjust_contrast(super_img)
    if with_mask and inhibit_mask is not None:
        x_all, y_all = np.where(find_boundaries(inhibit_mask[:, :, 0]))
        super_img[x_all, y_all, :] = [255, 255, 255]
    ax.imshow(super_img)
    adjust_layout(ax)
    ax.set_title('Channel Images Superimposed')


def shared_masks(ax, masks_anat=None, inhibit_mask=None,
                 unsure_mask=None, labels=None):
    """
    Displays shared masks by overlaying anatomical, inhibitory, and unsure masks.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to plot on.
    - masks_anat (numpy.ndarray, optional): Anatomical masks.
    - inhibit_mask (numpy.ndarray, optional): Mask of inhibitory neurons.
    - unsure_mask (numpy.ndarray, optional): Mask of unsure neurons.
    - labels (numpy.ndarray, optional): Labels for ROIs.

    This function overlays masks of different types in different color channels for visualization.
    """
    label_masks = np.zeros(
        (masks_anat.shape[0], masks_anat.shape[1], 3), dtype='int32')
    if inhibit_mask is not None:
        label_masks[:, :, 0] = inhibit_mask[:, :, 0]
    if masks_anat is not None:
        label_masks[:, :, 1] = masks_anat
    if unsure_mask is not None:
        label_masks[:, :, 2] = unsure_mask[:, :, 0]

    label_masks[label_masks >= 1] = 255
    label_masks = label_masks.astype('int32')
    ax.imshow(label_masks)
    adjust_layout(ax)
    ax.set_title('Channel Masks Superimposed')


def superimposed_plots(with_mask, mean_func, max_func, mean_anat,
                       masks_anat_both, inhibit_mask_both,
                       unsure_mask_both, labels_both):
    """
    Generates superimposed plots of images and masks before and after correction.

    Parameters:
    - with_mask (bool): Whether to include mask boundaries.
    - mean_func (numpy.ndarray): Mean functional image data.
    - max_func (numpy.ndarray): Max projection of functional data.
    - mean_anat (numpy.ndarray): Mean anatomical image data.
    - masks_anat_both (tuple): Tuple containing anatomical masks before and after correction.
    - inhibit_mask_both (tuple): Tuple containing inhibitory masks before and after correction.
    - unsure_mask_both (tuple): Tuple containing unsure masks before and after correction.
    - labels_both (tuple): Tuple containing labels before and after correction.

    This function creates plots comparing the superimposed images and masks before and after correction.
    """
    masks_anat, masks_anat_corrected = masks_anat_both
    inhibit_mask, inhibit_mask_corr = inhibit_mask_both
    unsure_mask, unsure_mask_corr = unsure_mask_both

    labels, labels_corr = labels_both

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

    superimpose(axs[0][0], with_mask=with_mask, mean_func=mean_func,
                mean_anat=mean_anat, inhibit_mask=inhibit_mask)
    shared_masks(ax=axs[0][1], masks_anat=masks_anat,
                 inhibit_mask=inhibit_mask, unsure_mask=unsure_mask, labels=labels)

    superimpose(axs[1][0], with_mask=with_mask, mean_func=mean_func,
                mean_anat=mean_anat, inhibit_mask=inhibit_mask_corr)
    shared_masks(ax=axs[1][1], masks_anat=masks_anat_corrected,
                 inhibit_mask=inhibit_mask_corr, unsure_mask=unsure_mask_corr, labels=labels_corr)

    plt.tight_layout()
    plt.savefig('plot_results/superimposed_plots.png')


def match_neurons(mask1, mask2, overlap_threshold=0.5):
    """
    Matches neurons between two masks based on spatial overlap.

    Parameters:
    - mask1 (numpy.ndarray): Labeled mask before correction.
    - mask2 (numpy.ndarray): Labeled mask after correction.
    - overlap_threshold (float, optional): Minimum overlap ratio to consider neurons as matching.

    Returns:
    - matches (dict): Keys are neuron IDs in mask1, values are matching neuron IDs in mask2.
    - unmatched_mask1 (set): Neuron IDs in mask1 with no match in mask2.
    - unmatched_mask2 (set): Neuron IDs in mask2 with no match in mask1.
    """
    matches = {}
    unmatched_mask1 = set(np.unique(mask1)) - {0}
    unmatched_mask2 = set(np.unique(mask2)) - {0}

    for neuron_id1 in unmatched_mask1.copy():
        neuron_mask1 = (mask1 == neuron_id1).astype(np.int32)

        for neuron_id2 in unmatched_mask2:
            neuron_mask2 = (mask2 == neuron_id2).astype(np.int32)
            intersection = np.logical_and(neuron_mask1, neuron_mask2).sum()
            union = neuron_mask1.sum() + neuron_mask2.sum() - intersection
            overlap_ratio = intersection / union if union != 0 else 0

            if overlap_ratio >= overlap_threshold:
                matches[neuron_id1] = neuron_id2
                unmatched_mask1.remove(neuron_id1)
                unmatched_mask2.remove(neuron_id2)
                break  # Assuming one-to-one matching

    return matches, unmatched_mask1, unmatched_mask2


def identify_neuron_changes(mask1, mask2, overlap_threshold=0.5):
    """
    Identifies common, removed, and new neurons between two masks based on spatial overlap.

    Parameters:
    - mask1 (numpy.ndarray): Labeled mask before correction.
    - mask2 (numpy.ndarray): Labeled mask after correction.
    - overlap_threshold (float, optional): Minimum overlap ratio to consider neurons as matching.

    Returns:
    - common_neurons_mask1 (set): Neuron IDs in mask1 that have matches in mask2.
    - common_neurons_mask2 (set): Neuron IDs in mask2 that have matches in mask1.
    - removed_neurons (set): Neuron IDs in mask1 not present in mask2.
    - new_neurons (set): Neuron IDs in mask2 not present in mask1.
    """
    matches, unmatched_mask1, unmatched_mask2 = match_neurons(
        mask1, mask2, overlap_threshold)
    common_neurons_mask1 = set(matches.keys())
    common_neurons_mask2 = set(matches.values())
    removed_neurons = unmatched_mask1
    new_neurons = unmatched_mask2

    return common_neurons_mask1, common_neurons_mask2, removed_neurons, new_neurons


def three_by_four_comparison(with_mask, mean_func, mean_anat, mean_anat_corrected,
                             masks_anat_both, masks_func, labels_both):
    """
    Generates a 3x4 grid of images comparing the effects of bleedthrough correction on different ROI categories.

    Parameters:
    - with_mask (bool): Whether to include mask boundaries.
    - mean_func (numpy.ndarray): Mean functional image data.
    - mean_anat (numpy.ndarray): Mean anatomical image before correction.
    - mean_anat_corrected (numpy.ndarray): Mean anatomical image after correction.
    - masks_anat_both (tuple): Tuple containing anatomical masks before and after correction.
    - masks_func (numpy.ndarray): Functional masks.
    - labels_both (tuple): Tuple containing labels before and after correction.

    This function creates a grid of images, each row corresponding to a category (inhibitory, excitatory, unsure),
    and overlays common, removed, and new neurons in different colors.
    """
    masks_anat, masks_anat_corrected = masks_anat_both
    labels, labels_corrected = labels_both

    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))

    column_titles = ['Green Channel', 'Original Red Channel',
                     'Corrected Red Channel', 'Superimposed Channel']
    for ax, col in zip(axs[0], column_titles):
        ax.set_title(col, fontsize=14)

    categories = [1, -1, 0]  # inhibitory, excitatory, unsure
    category_names = ['Inhibitory', 'Excitatory', 'Unsure']

    colors = {
        'common': [255, 255, 255],   # White
        'removed': [0, 255, 255],    # Cyan
        'new': [255, 255, 0]         # Yellow
    }

    # Identify neuron changes
    common_neurons_mask1, common_neurons_mask2, removed_neurons, new_neurons = identify_neuron_changes(
        masks_anat, masks_anat_corrected, overlap_threshold=0.5)

    for row_idx, cate in enumerate(categories):
        axs[row_idx][0].set_ylabel(
            f'{category_names[row_idx]} ROIs', fontsize=14)

        idxs_before = set(np.where(labels == cate)[0] + 1)
        idxs_after = set(np.where(labels_corrected == cate)[0] + 1)

        common_cate = common_neurons_mask1.intersection(idxs_before)
        removed_cate = removed_neurons.intersection(idxs_before)
        new_cate = new_neurons.intersection(idxs_after)

        common_mask = isolate_neurons(masks_anat, list(common_cate))
        removed_mask = isolate_neurons(masks_anat, list(removed_cate))
        new_mask = isolate_neurons(masks_anat_corrected, list(new_cate))

        overlay_masks = np.zeros(
            (mean_anat.shape[0], mean_anat.shape[1], 3), dtype='int32')
        overlay_masks[common_mask > 0] = colors['common']
        overlay_masks[removed_mask > 0] = colors['removed']
        overlay_masks[new_mask > 0] = colors['new']

        base_images = [
            prep_img(mean_func),             # Green Channel
            prep_img(mean_anat),             # Original Red Channel
            prep_img(mean_anat_corrected),   # Corrected Red Channel
            prep_img(mean_anat_corrected, mean_func)  # Superimposed Channel
        ]

        for col_idx in range(4):
            base_img = base_images[col_idx].copy()
            overlay_img = overlay_masks.copy()

            alpha = 0.5
            overlay_img_float = overlay_img.astype(np.float32)
            base_img_float = base_img.astype(np.float32)

            overlay_mask = (overlay_img.sum(axis=2) > 0)[:, :, np.newaxis]

            blended_img = np.where(
                overlay_mask,
                (1 - alpha) * base_img_float + alpha * overlay_img_float,
                base_img_float
            )
            blended_img = np.clip(blended_img, 0, 255).astype('uint8')

            axs[row_idx][col_idx].imshow(blended_img)
            axs[row_idx][col_idx].axis('off')

    handles = [mpatches.Patch(color=np.array(colors[key]) / 255, label=key.capitalize())
               for key in colors]
    fig.legend(handles=handles, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig('plot_results/three_by_four.png')
