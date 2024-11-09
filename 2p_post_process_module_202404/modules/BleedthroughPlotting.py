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
from skimage.morphology import dilation, square
from scipy.ndimage import label

from .RemoveBleedthrough import *
from .LabelExcInh import *
from skimage.measure import regionprops


def prep_img(mean_anat, mean_func=None):
    """
    Prepares an image to visualize anatomical channel data.

    Parameters:
    - mean_anat (numpy array): The mean anatomical image data.
    - mean_func (numpy array, optional): The mean functional image data. Defaults to None.

    Returns:
    - img (numpy array): A composite image for visualization.
    """
    img = np.zeros((mean_anat.shape[0], mean_anat.shape[1], 3), dtype='int32') #new numpy array named img is created filled with 0 
    img[:, :, 0] = adjust_contrast(mean_anat) #assigning red channel to the mean anatomical image data, : , then adjust contrast image 
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
        (mean_anat.shape[0], mean_anat.shape[1], 3), dtype='int32') #new numpy array named anat_img is created filled with 0 with shape of mean_anat
    anat_img[:, :, 0] = adjust_contrast(mean_anat) #assigning red channel to the mean anatomical image data, : , then adjust contrast image
    anat_img = adjust_contrast(anat_img) #adjust contrast of the image

    iter_lst = []

    if masks is not None:
        iter_lst.append((masks, [255, 255, 255])) #white 

    if labeled_masks_img is not None:
        iter_lst.append((labeled_masks_img[:, :, 0], [255, 255, 0])) #yellow

    if unsure_masks_img is not None:
        iter_lst.append((unsure_masks_img[:, :, 0], [0, 196, 255])) #blue cyan 

    if with_mask:
        for mask, color in iter_lst:
            x_all, y_all = np.where(find_boundaries(mask)) #find boundaries of the mask return coordinates
            for x, y in zip(x_all, y_all): #zip the coordinates
                anat_img[x, y, :] = np.array(color) #assign the color to the coordinates
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
        labeled_masks_img_orig, # original red cannel labels (anat) - y,w,b 
        labeled_masks_img_corr, # corrected red cannel labels (anat)
        unsure_masks_img_orig, # red blue red channel unsure labels (anat)
        unsure_masks_img_corr, # red blue red channel unsure labels (anat)
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
    
    if mean_func is not None:
        fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(25, 10))

    anat(ax[0], mean_anat, masks_anat, labeled_masks_img_orig,
         unsure_masks_img_orig, with_mask=True, title='Orig. Anat. + Mask')

    anat(ax[1], mean_anat_corrected, masks_anat_corrected, labeled_masks_img_corr,
         unsure_masks_img_corr, with_mask=True, title='Corr. Anat. + Mask')

    # Corrected anatomical image with functional ROIs
    if mean_func is not None:
        func_img = prep_img(mean_anat_corrected, mean_func)
        ax[2].imshow(func_img)
        ax[2].set_title('Corr. Anat. + Func ROIs')
        adjust_layout(ax[2])
    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(f'{comp_mask_type}_bleedthrough_channel_comparison.pdf')
    # # plt.show()
    
def identify_removed_neurons(masks_anat, masks_anat_corrected):
    # Label each connected component in both masks
    labeled_anat, num_features_anat = label(masks_anat) 
    labeled_anat_corrected, num_features_corrected = label(
        masks_anat_corrected) #label willcount individual neurons 

    # Create a set to track neurons present only in uncorrected mask
    only_in_anat = set()

    # Iterate through each labeled region in the uncorrected mask
    for i in range(1, num_features_anat + 1): #neuron wise comparison to create a binary mask. compares each elemnt of labeled anat with i (current neuron label) and  if the i is true then it will be added to the set
        # Create a mask for the current neuron in uncorrected mask
        neuron_mask = (labeled_anat == i) #It contains True values wherever the pixel belongs to the neuron/ROI with label i and False everywhere else.

        # Check if any pixels overlap in the corrected mask
        overlap = np.any(neuron_mask & (labeled_anat_corrected > 0)) #checks if any pixel in the neuron_mask is also present in the corrected mask

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
    labeled_mask, num_features = label(mask) #label each connected neuronthe mask

    # Create an empty mask to store only the specified neurons
    isolated_neurons = np.zeros_like(mask, dtype=int)

    # Iterate through each specified neuron index
    for i in neuron_indices:
        # Create a mask for the current neuron in the labeled mask
        neuron_mask = (labeled_mask == i)

        # Add this neuron to the isolated_neurons mask
        isolated_neurons[neuron_mask] = i

    return isolated_neurons


def identify_common_neurons(masks_anat, masks_anat_corrected):
    # Label each connected component in both masks
    labeled_anat, num_features_anat = label(masks_anat)
    labeled_anat_corrected, num_features_corrected = label(
        masks_anat_corrected)

    # Create a set to track neurons present in both masks
    common_neurons = set()

    # Iterate through each labeled region in the uncorrected mask
    for i in range(1, num_features_anat + 1):
        # Create a mask for the current neuron in uncorrected mask
        neuron_mask = (labeled_anat == i)

        # Check if any pixels overlap in the corrected mask
        overlap = np.any(neuron_mask & (labeled_anat_corrected > 0))

        # If there is an overlap, it means this neuron is present in both masks
        if overlap:
            common_neurons.add(i)

    return common_neurons


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
    
    print(f'Total number of removed neurons: {len(removed_neurons)}')
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
    print(f'Total number of new neurons: {len(new_neurons)})')
    # Isolate removed neurons
    new_neurons_mask = isolate_neurons(masks_anat_corrected, new_neurons)

    # Generate the figure
    fig, ax = plt.subplots(1, 2, figsize=(25, 10))
    anat(ax[0], mean_anat=mean_anat, masks=new_neurons_mask, labeled_masks_img=None,
         unsure_masks_img=None, with_mask=True, title='Orig')

    anat(ax[1], mean_anat_corrected, new_neurons_mask, None,
         None, with_mask=True, title='Corr. Anat. New')
    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(f'new_neurons_bleedthrough_channel_comparison.pdf')
    # # plt.show()


def common_neurons_comparison_image(
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

    common_neurons = list(identify_common_neurons(
        masks_anat, masks_anat_corrected))
    print(f'Total number of common neurons: {len(common_neurons)}')
    # Isolate removed neurons
    common_neurons_mask = isolate_neurons(masks_anat, common_neurons)

    # Generate the figure
    fig, ax = plt.subplots(1, 2, figsize=(25, 10))
    anat(ax[0], mean_anat=mean_anat, masks=common_neurons_mask, labeled_masks_img=None,
         unsure_masks_img=None, with_mask=True, title='Orig')

    anat(ax[1], mean_anat_corrected, common_neurons_mask, None,
         None, with_mask=True, title='Corr. Anat. New')
    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(f'common_neurons_bleedthrough_channel_comparison.pdf')
    # plt.show()


def plot_unsure_rois(mean_anat, mean_anat_corrected, unsure_masks_img_orig, unsure_masks_img_corr,mean_func):
    """
    Plots the unsure ROIs that are removed, new, and common between the original and corrected anatomical channels.

    Parameters:
    - mean_anat: numpy array, the mean anatomical image before correction
    - mean_anat_corrected: numpy array, the mean anatomical image after correction
    - unsure_masks_img_orig: numpy array, binary mask representing the unsure ROIs in the original image
    - unsure_masks_img_corr: numpy array, binary mask representing the unsure ROIs in the corrected image
    """

    # Ensure identification functions return list of IDs
    removed_neurons = identify_removed_neurons(unsure_masks_img_orig, unsure_masks_img_corr)
    new_neurons = identify_new_neurons(unsure_masks_img_orig, unsure_masks_img_corr)
    common_neurons = identify_common_neurons(unsure_masks_img_orig, unsure_masks_img_corr)

    # Ensure masks are binary and correctly represented
    removed_neurons_mask = isolate_neurons(unsure_masks_img_orig, removed_neurons)
    new_neurons_mask = isolate_neurons(unsure_masks_img_corr, new_neurons)
    common_neurons_mask = isolate_neurons(unsure_masks_img_orig, common_neurons)

    # Ensure masks are strictly 2D if they are not already
    removed_neurons_mask = removed_neurons_mask if removed_neurons_mask.ndim == 2 else removed_neurons_mask[:, :, 0]
    new_neurons_mask = new_neurons_mask if new_neurons_mask.ndim == 2 else new_neurons_mask[:, :, 0]
    common_neurons_mask = common_neurons_mask if common_neurons_mask.ndim == 2 else common_neurons_mask[:, :, 0]

    # Prepare the composite images for plotting
    img_anat = prep_img(mean_anat)
    img_anat_corrected = prep_img(mean_anat_corrected)
    img_func = np.zeros((mean_func.shape[0], mean_func.shape[1], 3), dtype='int32')
    img_func[:, :, 1] = adjust_contrast(mean_func)
    
    # Loop over removed neurons and add boundaries
    for neuron_id in removed_neurons:
        boundaries = find_boundaries(removed_neurons_mask == neuron_id)
        if boundaries.ndim != 2:
            raise ValueError("Boundaries must be 2D.")
        x_all, y_all = np.where(boundaries)
        for x, y in zip(x_all, y_all):
            img_anat[x, y, :] = np.array([255, 0, 255])  # Magenta for removed unsure ROIs
            img_anat_corrected[x, y, :] = np.array([255, 0, 255])  # Magenta for removed unsure ROIs
            img_func[x, y, :] = np.array([255, 0, 255])

    # Loop over new neurons and add boundaries
    for neuron_id in new_neurons:
        boundaries = find_boundaries(new_neurons_mask == neuron_id)
        if boundaries.ndim != 2:
            raise ValueError("Boundaries must be 2D.")
        x_all, y_all = np.where(boundaries)
        for x, y in zip(x_all, y_all):
            img_anat[x, y, :] = np.array([0, 0, 255])  # Blue for new unsure ROIs
            img_anat_corrected[x, y, :] = np.array([0, 0, 255])  # Blue for new unsure ROIs
            img_func[x, y, :] = np.array([0, 0, 255])

    # Loop over common neurons and add boundaries
    for neuron_id in common_neurons:
        boundaries = find_boundaries(common_neurons_mask == neuron_id)
        if boundaries.ndim != 2:
            raise ValueError("Boundaries must be 2D.")
        x_all, y_all = np.where(boundaries)
        for x, y in zip(x_all, y_all):
            img_anat[x, y, :] = np.array([255, 165, 0])  # Orange for common unsure ROIs
            img_anat_corrected[x, y, :] = np.array([255, 165, 0])  # Orange for common unsure ROIs
            img_func[x, y, :] = np.array([255, 165, 0])

    # Plotting the results
    fig, ax = plt.subplots(1, 3, figsize=(30, 8))
    ax[0].imshow(img_anat)
    ax[0].set_title('Original Anatomical Image with Removed, New, and Common Unsure ROIs')
    adjust_layout(ax[0])

    ax[1].imshow(img_anat_corrected)
    ax[1].set_title('Corrected Anatomical Image with Removed, New, and Common Unsure ROIs')
    adjust_layout(ax[1])

    ax[2].imshow(img_func)
    ax[2].set_title('Functional Channel with Unsure ROIs')
    adjust_layout(ax[2])
    
    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(f'unsure_rois_combined.pdf')
  
    #plt.show()


def plot_combined_inhibitory_rois(mean_func, mean_anat, mean_anat_corrected, labeled_masks_img_orig, labeled_masks_img_corr):
    """
    Plots removed, new, and common inhibitory ROIs on both anatomical and functional channels.

    Parameters:
    - mean_func: numpy array, mean functional image (green channel).
    - mean_anat: numpy array, mean anatomical image before correction.
    - mean_anat_corrected: numpy array, mean anatomical image after correction.
    - labeled_masks_img_orig: numpy array, inhibitory masks in the original anatomical image.
    - labeled_masks_img_corr: numpy array, inhibitory masks in the corrected anatomical image.
    """
    
    # Identify removed, new, and common inhibitory neurons
    removed_inhibitory = identify_removed_neurons(labeled_masks_img_orig, labeled_masks_img_corr)
    new_inhibitory = identify_new_neurons(labeled_masks_img_orig, labeled_masks_img_corr)
    common_inhibitory = identify_common_neurons(labeled_masks_img_orig, labeled_masks_img_corr)
    print(f"Number of common inhibitory ROIs: {len(common_inhibitory)}")

    # Create masks for removed, new, and common inhibitory ROIs
    labeled_orig_inhibitory, _ = label(labeled_masks_img_orig[:, :, 0])
    labeled_corr_inhibitory, _ = label(labeled_masks_img_corr[:, :, 0])

    removed_mask = isolate_neurons(labeled_orig_inhibitory, removed_inhibitory)
    new_mask = isolate_neurons(labeled_corr_inhibitory, new_inhibitory)
    common_mask = isolate_neurons(labeled_corr_inhibitory, common_inhibitory)

    # Create a figure with three subplots
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))

    # Plot 1: Functional channel with inhibitory ROI labels (Green channel) - Overlays removed, new, and common inhibitory ROIs
    func_img = np.zeros((mean_func.shape[0], mean_func.shape[1], 3), dtype='int32')
    func_img[:, :, 1] = adjust_contrast(mean_func)  # Green channel for functional data

    # Removed inhibitory ROIs - Magenta
    x_all, y_all = np.where(find_boundaries(removed_mask))
    for x, y in zip(x_all, y_all):
        func_img[x, y, :] = np.array([255, 0, 255])  # Magenta for removed inhibitory ROIs

    # New inhibitory ROIs - Orange
    x_all, y_all = np.where(find_boundaries(new_mask))
    for x, y in zip(x_all, y_all):
        func_img[x, y, :] = np.array([255, 165, 0])  # Orange for new inhibitory ROIs

    # Common inhibitory ROIs - Blue
    x_all, y_all = np.where(find_boundaries(common_mask))
    for x, y in zip(x_all, y_all):
        func_img[x, y, :] = np.array([0, 0, 255])  # Blue for common inhibitory ROIs

    ax[0].imshow(func_img)
    ax[0].set_title('Functional Channel (Green) with Removed, New, and Common Inhibitory ROIs')
    ax[0].axis('off')

    # Plot 2: Original anatomical channel with removed, new, and common inhibitory ROIs (Red channel)
    anat_img = np.zeros((mean_anat.shape[0], mean_anat.shape[1], 3), dtype='int32')
    anat_img[:, :, 0] = adjust_contrast(mean_anat)  # Red channel for anatomical data

    # Removed inhibitory ROIs - Magenta
    x_all, y_all = np.where(find_boundaries(removed_mask))
    for x, y in zip(x_all, y_all):
        anat_img[x, y, :] = np.array([255, 0, 255])  # Magenta for removed inhibitory ROIs

    # New inhibitory ROIs - Orange
    x_all, y_all = np.where(find_boundaries(new_mask))
    for x, y in zip(x_all, y_all):
        anat_img[x, y, :] = np.array([255, 165, 0])  # Orange for new inhibitory ROIs

    # Common inhibitory ROIs - Blue
    x_all, y_all = np.where(find_boundaries(common_mask))
    for x, y in zip(x_all, y_all):
        anat_img[x, y, :] = np.array([0, 0, 255])  # Blue for common inhibitory ROIs

    ax[1].imshow(anat_img)
    ax[1].set_title('Original Anatomical Channel with Inhibitory ROI Changes')
    ax[1].axis('off')

    # Plot 3: Corrected anatomical channel with removed, new, and common inhibitory ROIs (Red channel)
    corrected_img = np.zeros((mean_anat_corrected.shape[0], mean_anat_corrected.shape[1], 3), dtype='int32')
    corrected_img[:, :, 0] = adjust_contrast(mean_anat_corrected)  # Red channel for anatomical data

    # Removed inhibitory ROIs - Magenta
    x_all, y_all = np.where(find_boundaries(removed_mask))
    for x, y in zip(x_all, y_all):
        corrected_img[x, y, :] = np.array([255, 0, 255])  # Magenta for removed inhibitory ROIs

    # New inhibitory ROIs - Orange
    x_all, y_all = np.where(find_boundaries(new_mask))
    for x, y in zip(x_all, y_all):
        corrected_img[x, y, :] = np.array([255, 165, 0])  # Orange for new inhibitory ROIs

    # Common inhibitory ROIs - Blue
    x_all, y_all = np.where(find_boundaries(common_mask))
    for x, y in zip(x_all, y_all):
        corrected_img[x, y, :] = np.array([0, 0, 255])  # Blue for common inhibitory ROIs

    ax[2].imshow(corrected_img)
    ax[2].set_title('Corrected Anatomical Channel with Inhibitory ROI Changes')
    ax[2].axis('off')

    # Adjust layout to ensure there is no overlap
    plt.tight_layout()

    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(f'inhibitory_rois_combined_f.pdf')
    # # plt.show()
   
"""
def plot_combined_excitatory_rois(mean_func, mean_anat, mean_anat_corrected, excitatory_masks_img_orig, excitatory_masks_img_corr):
   
    Plots removed, new, and common excitatory ROIs on both anatomical and functional channels.

    Parameters:
    - mean_func: numpy array, mean functional image (green channel).
    - mean_anat: numpy array, mean anatomical image before correction.
    - mean_anat_corrected: numpy array, mean anatomical image after correction.
    - excitatory_masks_img_orig: numpy array, excitatory masks in the original anatomical image.
    - excitatory_masks_img_corr: numpy array, excitatory masks in the corrected anatomical image.
    
    
    # Identify removed, new, and common excitatory neurons
    removed_excitatory = identify_removed_neurons(excitatory_masks_img_orig, excitatory_masks_img_corr)
    new_excitatory = identify_new_neurons(excitatory_masks_img_orig, excitatory_masks_img_corr)
    common_excitatory = identify_common_neurons(excitatory_masks_img_orig, excitatory_masks_img_corr)
    
    print(f"Number of common excitatory ROIs: {len(common_excitatory)}")

    # Create masks for removed, new, and common excitatory ROIs
    labeled_orig_excitatory, _ = label(excitatory_masks_img_orig[:, :, 0])
    labeled_corr_excitatory, _ = label(excitatory_masks_img_corr[:, :, 0])

    removed_mask = isolate_neurons(labeled_orig_excitatory, removed_excitatory)
    new_mask = isolate_neurons(labeled_corr_excitatory, new_excitatory)
    common_mask = isolate_neurons(labeled_corr_excitatory, common_excitatory)

    # Create a figure with three subplots
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))

    # Plot 1: Functional channel with excitatory ROI labels (Green channel) - Overlays removed, new, and common excitatory ROIs
    func_img = np.zeros((mean_func.shape[0], mean_func.shape[1], 3), dtype='int32')
    func_img[:, :, 1] = adjust_contrast(mean_func)  # Green channel for functional data

    # Removed excitatory ROIs - Magenta
    x_all, y_all = np.where(find_boundaries(removed_mask))
    for x, y in zip(x_all, y_all):
        func_img[x, y, :] = np.array([255, 0, 255])  # Magenta for removed excitatory ROIs

    # New excitatory ROIs - Orange
    x_all, y_all = np.where(find_boundaries(new_mask))
    for x, y in zip(x_all, y_all):
        func_img[x, y, :] = np.array([255, 165, 0])  # Orange for new excitatory ROIs

    # Common excitatory ROIs - Blue
    x_all, y_all = np.where(find_boundaries(common_mask))
    for x, y in zip(x_all, y_all):
        func_img[x, y, :] = np.array([0, 0, 255])  # Blue for common excitatory ROIs

    ax[0].imshow(func_img)
    ax[0].set_title('Functional Channel (Green) with Removed, New, and Common Excitatory ROIs')
    ax[0].axis('off')

    # Plot 2: Original anatomical channel with removed, new, and common excitatory ROIs (Red channel)
    anat_img = np.zeros((mean_anat.shape[0], mean_anat.shape[1], 3), dtype='int32')
    anat_img[:, :, 0] = adjust_contrast(mean_anat)  # Red channel for anatomical data

    # Removed excitatory ROIs - Magenta
    x_all, y_all = np.where(find_boundaries(removed_mask))
    for x, y in zip(x_all, y_all):
        anat_img[x, y, :] = np.array([255, 0, 255])  # Magenta for removed excitatory ROIs

    # New excitatory ROIs - Orange
    x_all, y_all = np.where(find_boundaries(new_mask))
    for x, y in zip(x_all, y_all):
        anat_img[x, y, :] = np.array([255, 165, 0])  # Orange for new excitatory ROIs

    # Common excitatory ROIs - Blue
    x_all, y_all = np.where(find_boundaries(common_mask))
    for x, y in zip(x_all, y_all):
        anat_img[x, y, :] = np.array([0, 0, 255])  # Blue for common excitatory ROIs

    ax[1].imshow(anat_img)
    ax[1].set_title('Original Anatomical Channel with Excitatory ROI Changes')
    ax[1].axis('off')

    # Plot 3: Corrected anatomical channel with removed, new, and common excitatory ROIs (Red channel)
    corrected_img = np.zeros((mean_anat_corrected.shape[0], mean_anat_corrected.shape[1], 3), dtype='int32')
    corrected_img[:, :, 0] = adjust_contrast(mean_anat_corrected)  # Red channel for anatomical data

    # Removed excitatory ROIs - Magenta
    x_all, y_all = np.where(find_boundaries(removed_mask))
    for x, y in zip(x_all, y_all):
        corrected_img[x, y, :] = np.array([255, 0, 255])  # Magenta for removed excitatory ROIs

    # New excitatory ROIs - Orange
    x_all, y_all = np.where(find_boundaries(new_mask))
    for x, y in zip(x_all, y_all):
        corrected_img[x, y, :] = np.array([255, 165, 0])  # Orange for new excitatory ROIs

    # Common excitatory ROIs - Blue
    x_all, y_all = np.where(find_boundaries(common_mask))
    for x, y in zip(x_all, y_all):
        corrected_img[x, y, :] = np.array([0, 0, 255])  # Blue for common excitatory ROIs

    ax[2].imshow(corrected_img)
    ax[2].set_title('Corrected Anatomical Channel with Excitatory ROI Changes')
    ax[2].axis('off')

    # Adjust layout to ensure there is no overlap
    plt.tight_layout()

    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig('excitatory_rois_combined_plot_f.pdf')

    # Show the plot (optional, for debugging purposes)
    # plt.show()
"""

