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
    labeled_mask, num_features = label(mask) #label each connected neuron in the mask

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

    #Functional Plotting 

def func(ax, mean_func, masks, labeled_masks_img_func, unsure_masks_img_func, with_mask=True, title='functional channel mean image'):
    """
    Plots the functional channel mean image with optional mask boundaries.

    Parameters:
    - ax (matplotlib axes): The axes to plot on.
    - mean_func (numpy array): The mean functional image data.
    - masks (numpy array): The masks for ROIs.
    - labeled_masks_img (numpy array): The labeled masks image for visualization.
    - unsure_masks_img (numpy array): The unsure masks image for visualization.
    - with_mask (bool, optional): Whether to include mask boundaries in the plot. Defaults to True.
    - title (str, optional): The title of the plot. Defaults to 'functional channel mean image'.
    """
    func_img = np.zeros(
        (mean_func.shape[0], mean_func.shape[1], 3), dtype='int32')
    func_img[:, :, 1] = adjust_contrast(mean_func)  # Green channel for functional data
    func_img  = adjust_contrast(func_img)
    iter_lst = []

    if masks is not None:
        iter_lst.append((masks, [255, 255, 255]))  # White for  masks

    if labeled_masks_img_func is not None:
        iter_lst.append((labeled_masks_img_func[:, :, 0], [255, 255, 0]))  # Yellow for labeled ROIs

    if unsure_masks_img_func is not None:
        iter_lst.append((unsure_masks_img_func[:, :, 0], [0, 196, 255]))  # Blue for unsure labels

    if with_mask:
        for mask, color in iter_lst:
            x_all, y_all = np.where(find_boundaries(mask))
            for x, y in zip(x_all, y_all):
                func_img[x, y, :] = np.array(color)
    ax.matshow(func_img)
    adjust_layout(ax)
    ax.set_title(f'{title}')

def func_channel_roi_image(
        comp_mask_type,
        mean_func,
        masks_func,
        labeled_masks_img_func,
        unsure_masks_img_func,
        with_mask=True):
    """
    Generates a figure with comparisons of the functional channel images with masks.

    Parameters:
    - mean_func (numpy array): The mean functional image data.
    - masks_func (numpy array): The masks for functional ROIs.
    - labeled_masks_img_func (numpy array): The labeled masks image for functional visualization.
    - unsure_masks_img_func (numpy array): The unsure masks image for functional visualization.
    - with_mask (bool, optional): Whether to include mask boundaries in the plot. Defaults to True.

    This function generates a figure with comparisons of the functional channel images, including plots for labeled and unsure ROIs.
    """
    
    fig, ax = plt.subplots(figsize=(12, 10))

    func(ax, mean_func, masks_func, labeled_masks_img_func,
              unsure_masks_img_func, with_mask=True, title='Func. Channel + Label Mask')

    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(f'{comp_mask_type}_functional_channel_comparison.pdf')
    # plt.show()
    
"""
def plot_images(mean_anat,masks_anat,
                labeled_masks_img_func,
                labeled_masks_img_orig,
                labeled_masks_img_corr,
                mean_func,
                with_mask=True):
    
    # Helper function : 
    def plot_with_mask(ax,image,mask,title):
        img = np.zeros((image.shape[0],image.shape[1],3),dtype='int32')
        img[:,:,1] = adjust_contrast(image)
        
        if mask is not None:
            x_all , y_all = np.where(find_boundaries(mask))
            for x,y in zip(x_all,y_all):
                img[x,y,:] = np.array([255,255,0])
        
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
        

    # Plot functional and anatomical channels
    fig,ax = plt.subplots(1,4,figsize=(25,10))

    # 1. Functional channel plol
    plot_with_mask(ax[0],mean_func,labeled_masks_img_func[:,:,1],'Functional channel (yellow)')

    # 2. Anatomical channel plot (uncorrected original)
    plot_with_mask(ax[1],mean_anat,labeled_masks_img_orig[:,:,0],'Anatomical Channel (yellow uncorrected)')
    
    # 3. Anatomical channel plot (corrected original)
    plot_with_mask(ax[1],mean_anat,labeled_masks_img_corr[:,:,0],'Anatomical Channel (yellow corrected)')
    
    # 4. Difference between original and corrected
    yellow_mask_orig = (labeled_masks_img_orig==1).astype(int)
    yellow_mask_corr = (labeled_masks_img_corr==1).astype(int)
    
    diff_corr = np.abs( yellow_mask_orig -  yellow_mask_corr)
    ax[3].imshow(diff_corr,cmap='gray')
    ax[3].set_title('Difference')
    ax[3].axis('off')
    
    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(f'yellow_ROI_comparison.pdf')
"""

def yellow_combined_roi_plot(mean_anat, mean_anat_corrected, mean_func, masks_anat, masks_anat_corrected, labeled_masks_img_orig, labeled_masks_img_corr, unsure_masks_img_orig, unsure_masks_img_corr, labeled_masks_img_func, unsure_masks_img_func):
   
   """
    Generates a combined figure showing three different ROI analysis images side-by-side:
    1. Original anatomical channel with ROI labels.
    2. Corrected red channel with ROI labels.
    3. Green channel with ROI labels.
    

    Parameters:
    - mean_anat (numpy array): The mean anatomical image data.
    - mean_anat_corrected (numpy array): The mean corrected anatomical image data.
    - mean_func (numpy array): The mean functional image data (green channel).
    - masks_anat (numpy array): The masks for anatomical ROIs (original).
    - masks_anat_corrected (numpy array): The masks for anatomical ROIs (corrected).
    - labeled_masks_img_orig (numpy array): The labeled masks image for original visualization.
    - labeled_masks_img_corr (numpy array): The labeled masks image for corrected visualization.
    - unsure_masks_img_orig (numpy array): The unsure masks image for original visualization.
    - unsure_masks_img_corr (numpy array): The unsure masks image for corrected visualization.
    - labeled_masks_img_func (numpy array): The labeled masks image for functional visualization.
    - unsure_masks_img_func (numpy array): The unsure masks image for functional visualization.
    """
   
    # Create a figure with four subplots side-by-side
fig, ax = plt.subplots(1, 4, figsize=(40, 10))

    # Plot 1: Functional channel with yellow ROI labels
    func(ax[0], mean_func= mean_func, masks=None, labeled_masks_img_func=labeled_masks_img_func,
              unsure_masks_img_func=None, with_mask=True, title='Functional Channel (Yellow ROIs)')

    # Plot 2: Original anatomical channel with yellow ROI labels
anat(ax[1], mean_anat=mean_anat, masks=None, labeled_masks_img=labeled_masks_img_orig,
         unsure_masks_img=None, with_mask=True, title='Original Anatomical Channel (Yellow ROIs)')

    # Plot 3: Corrected red anatomical channel with yellow ROI labels
anat(ax[2], mean_anat=mean_anat_corrected, masks=None, labeled_masks_img=labeled_masks_img_corr,
         unsure_masks_img=None, with_mask=True, title='Corrected Anatomical Channel (Yellow ROIs)')

    # Plot 4: Difference between original and corrected anatomical channel with yellow labels
yellow_mask_orig = (labeled_masks_img_orig[:, :, 0] == 1).astype(int)
yellow_mask_corr = (labeled_masks_img_corr[:, :, 0] == 1).astype(int)
diff_mask = np.abs(yellow_mask_orig - yellow_mask_corr)

diff_img = np.zeros((mean_anat.shape[0], mean_anat.shape[1], 3), dtype='int32')
diff_img[:, :, 0] = adjust_contrast(mean_anat)  # Red channel for anatomical data

x_all, y_all = np.where(find_boundaries(diff_mask))
for x, y in zip(x_all, y_all):
 diff_img[x, y, :] = np.array([255, 255, 0])  # Yellow for difference in labels

ax[3].imshow(diff_img)
adjust_layout(ax[3])
ax[3].set_title('Difference (Original - Corrected) Yellow Labels')

    # Adjust layout and save the figure
plt.rcParams['savefig.dpi'] = 1000
plt.tight_layout()
plt.savefig('yellow_ROI_comparison.pdf')
    # plt.show()
    