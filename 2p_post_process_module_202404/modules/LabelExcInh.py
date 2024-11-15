#!/usr/bin/env python3

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
import matplotlib.colors as mcolors

from .RemoveBleedthrough import *
from .BleedthroughPlotting import *


def normz(data):
    """
    Performs Z-score normalization on the input data.

    Parameters:
    - data (numpy.ndarray): Input data to normalize.

    Returns:
    - numpy.ndarray: Z-score normalized data.
    """
    return (data - np.mean(data)) / (np.std(data) + 1e-5)


def run_cellpose(ops, mean_anat, diameter, flow_threshold=0.5): 
    """
    run cellpose on anatomical mean image to segment cells and saves the segmentation results to a directory
    Parameters:
    ops 
    mean_anat: mean image of anatomical channel
    diameter: diameter of cells
    flow_threshold: threshold for cellpose flow
    """
    save_dir = os.path.join(ops['save_path0'], 'cellpose')  #create save directory
    if not os.path.exists(save_dir): #if directory does not exist, create it
        os.makedirs(save_dir)

    tifffile.imwrite(os.path.join(save_dir, 'mean_anat.tif'), mean_anat) #the mean_anat is saved as TIFF file in the save directory

    model = models.Cellpose(model_type="cyto3")
    masks_anat, flows, styles, diams = model.eval(
        mean_anat, diameter=diameter, flow_threshold=flow_threshold
    )

    io.masks_flows_to_seg(
        images=mean_anat,
        masks=masks_anat,
        flows=flows,
        file_names=os.path.join(save_dir, 'mean_anat'),
        diams=diameter,
    )

    return masks_anat


def get_mask(ops, mean_anat_corr=None):
    """
    Reads and extracts masks and images from the ops dictionary.

    Parameters:
    - ops (dict): Dictionary containing operation parameters.
    - mean_anat_corr (numpy.ndarray, optional): Corrected mean anatomical image.

    Returns:
    - If mean_anat_corr is provided:
        - numpy.ndarray: Cropped corrected mean anatomical image.
    - If mean_anat_corr is None:
        - masks_func (numpy.ndarray): Functional masks.
        - mean_func (numpy.ndarray): Mean functional image.
        - max_func (numpy.ndarray): Maximum projection functional image.
        - mean_anat (numpy.ndarray or None): Mean anatomical image if available, else None.
    """
    masks_npy = np.load(
        os.path.join(ops['save_path0'], 'qc_results', 'masks.npy'), allow_pickle=True
    )
    x1, x2 = ops['xrange']
    y1, y2 = ops['yrange']

    masks_func = masks_npy[y1:y2, x1:x2]
    mean_func = ops['meanImg'][y1:y2, x1:x2]
    max_func = ops['max_proj']

    mean_anat = (
        ops['meanImg_chan2'][y1:y2, x1:x2] if ops['nchannels'] == 2 else None
    )
    if mean_anat_corr is not None:
        mean_anat_corr = (
            mean_anat_corr[y1:y2, x1:x2] if ops['nchannels'] == 2 else None
        )
        return mean_anat_corr

    return masks_func, mean_func, max_func, mean_anat


def get_label(masks_func, masks_anat, thres1=0.3, thres2=0.5): # asign labels to rois 
    """
    Compare ROIs from the functional channel with the anatomical channel to assign labels to the ROIs based on the overlap
    based on overlap - the function assigns labels- -1 for excitatory and 1 for inhibitory and 0 for unsure
    """
    print(masks_anat)
    print(masks_func)
    
    anat_roi_ids = np.unique(masks_anat)[1:] #unique ids of the anatomical masks, skipping 1st ID which is background
    masks_3d = np.zeros( 
        (len(anat_roi_ids), masks_anat.shape[0], masks_anat.shape[1])) #neurons active unique values 

    for i, roi_id in enumerate(anat_roi_ids):
        masks_3d[i] = (masks_anat == roi_id).astype(int) #3d aray each slice contains the mask of a single roi , 3d each roi id - make a cop
    masks_3d[masks_3d != 0] = 1 #convert to binary 1- roi, 0- background 

    prob = []
    for i in tqdm(np.unique(masks_func)[1:]): #tqdm create progressive bars shows the progress of the loop - :1 remove background 
        roi_masks = (masks_func == i).astype('int32') # roi mask in functional channel
        roi_masks_tile = np.tile(np.expand_dims(
            roi_masks, 0), (len(anat_roi_ids), 1, 1))  # adds an axis making it 3d allowing for comparison 
        overlap = (roi_masks_tile * masks_3d).reshape(len(anat_roi_ids), -1)  #number of rois x total number of pixels , number rows (no. of rois) -1 - figure out the number of columns
        overlap = np.sum(overlap, axis=1)
        prob.append(np.max(overlap) / np.sum(roi_masks)) #wherever there is overlap total number of outcomes - total unique id , overlap probability. maximum overlap, total number of pixels in func roi (having value 1 ) 

    prob = np.array(prob)
    labels = np.zeros_like(prob)
    labels[prob < thres1] = -1  # Excitatory white 
    labels[prob > thres2] = 1   # Inhibitory yellow     

    
    # calculates labes fro func 

    #mask_anat- mask then func is calculated from this 
    #np.zero get label function 

    return labels





# Save channel images and masks results


def save_masks(ops, masks_func, masks_anat, masks_anat_corrected, mean_func, max_func, mean_anat, mean_anat_corrected, labels):
    with h5py.File(os.path.join(ops['save_path0'], 'masks.h5'), 'w') as f:
        f['labels'] = labels
        f['masks_func'] = masks_func
        f['mean_func'] = mean_func
        f['max_func'] = max_func

        if ops['nchannels'] == 2:
            f['mean_anat'] = mean_anat
            f['masks_anat'] = masks_anat
            f['masks_anat_corrected'] = masks_anat_corrected
            f['meanImg_chan2_corrected'] = mean_anat_corrected


def get_labeled_masks_img(masks, labels, cate, channel_color):
    """
    genberate labeled masks image based on their category
    """
    labeled_masks_img = np.zeros(
        (masks.shape[0], masks.shape[1], 3), dtype='int32') #initialize an array of zeros with dimensions (h,w,3(RGB image)) matches mask 
    neuron_idx = np.where(labels == cate)[0] + 1 #get the index of the neurons tht matches the category , +1 label starts from 1 , list of rois in labels where category = cate 
    for i in neuron_idx:
        neuron_mask = ((masks == i) * 255).astype('int32') #create a binary mask for current neuron where pixels of the neuron marked True, 255 converts binary values (T/F), 8 bit image range 0-255
        if channel_color=='red':
            labeled_masks_img[:, :, 0] += neuron_mask
        elif channel_color=='green':
            labeled_masks_img[:, :, 1] += neuron_mask
        
                                                       # all neurons of specific category will be marked with red, roi ids are added to the red channel
    return labeled_masks_img



def read_fluos(ops):
    """
    Reads fluorescence data from the qc_results directory.

    Parameters:
    - ops (dict): Dictionary containing operation parameters.

    Returns:
    - fluo (numpy.ndarray): Fluorescence data for the first channel.
    - fluo_chan2 (numpy.ndarray): Fluorescence data for the second channel.
    """
    path = ops['save_path0']

    fluo = np.load(
        os.path.join(path, 'qc_results', 'fluo.npy'), allow_pickle=True
    )
    print("fluo shape: ", fluo.shape)

    fluo_chan2 = np.load(
        os.path.join(path, 'qc_results', 'fluo_chan2.npy'), allow_pickle=True
    )
    print("fluo chan2 shape: ", fluo_chan2.shape)

    return fluo, fluo_chan2


def run(ops, diameter):
    """
    Main function to run the two-channel data ROI identification workflow.

    Parameters:
    - ops (dict): Dictionary containing operation parameters.
    - diameter (float): Diameter of cells for Cellpose.

    This function performs the following steps:
    - Reads masks and images.
    - Runs Cellpose on the anatomical channel to detect cells.
    - Computes labels for each ROI based on overlap.
    - Generates images of labeled masks.
    - Corrects the red channel for bleedthrough.
    - Repeats cell detection and labeling on the corrected images.
    - Generates comparison images and saves results.
    """
    print('===============================================')
    print('===== Two channel data ROI identification =====')
    print('===============================================')

    print('Reading masks in functional channel')
    masks_func, mean_func, max_func, mean_anat = get_mask(ops)
#mask_func = roi masks in functional channel, mean_func= mean image in functional channel, mean_anat = mean image in anatomical channel, mask_anat
    if np.max(masks_func) == 0:
        raise ValueError('No masks found.')

    if ops['nchannels'] == 1: 
        print('Single channel recording, skipping ROI labeling')
        labels = -1 * np.ones(int(np.max(masks_func))).astype('int32')
        save_masks(
            ops, masks_func, None, None, mean_func, max_func, None, labels
        )
    else:
        print('Running Cellpose on anatomical channel mean image')
        print(f'Found diameter as {diameter}')

        # UNCORRECTED MASKS AND LABELS

        print('Computing masks for uncorrected red channel')
        masks_anat = run_cellpose(ops, mean_anat, diameter) #red channel
        print(f"mean_anat: {mean_anat}")
        print(f"mean_func: {mean_func}")
        print('Computing labels for each ROI on uncorrected red channel')
        labels = get_label(masks_func, masks_anat)

        print('Computing labeled masks on uncorrected red channel')
        labeled_masks_img_orig = get_labeled_masks_img(masks_anat, labels, 1, 'red')
        print (f'labeled_masks_img_orig: {labeled_masks_img_orig}')
        print('Computing inhibitory masks on uncorrected red channel')
        labeled_masks_img_orig = get_labeled_masks_img(masks_anat, labels, 1)

        print('Computing unsure masks on uncorrected red channel')
        unsure_masks_img_orig = get_labeled_masks_img(masks_anat, labels, 0)

        # Fluorescence data
        fluo, fluo_chan2 = read_fluos(ops)

        print("Fitting linear model to correct green channel bleedthrough")
        # slope, offset, fluo_means, fluo_chan2_means = train_reg_model(
        #     fluo, fluo_chan2)

        print("Correcting red channel")
        # Fchan2_corrected_means = Fchan2_corrected_anat(
        #     fluo_means, fluo_chan2_means, slope, 0)

        # mean_anat_corrected = update_mean_anat(
        #     mean_anat, Fchan2_corrected_means, masks_anat)
        # mean_anat_corrected = mean_anat.copy()

        mean_anat_corrected = correct_bleedthrough(
            ops['Ly'], ops['Lx'], nblks=3, mimg=ops['meanImg'], mimg2=ops['meanImg_chan2'])
        mean_anat_corrected = get_mask(ops, mean_anat_corr=mean_anat_corrected)

        # CORRECTED MASKS AND LABELS

        print('Computing corrected mask')
        masks_anat_corrected = run_cellpose(
            ops, mean_anat_corrected, diameter
        )

        # np.savetxt('masks_anat.csv', masks_anat, delimiter=',')
        # np.savetxt('masks_anat_corr.csv', masks_anat_corrected, delimiter=',')

        print('Computing corrected labels for each ROI')
        labels_corrected = get_label(masks_func, masks_anat_corrected)

        print('Computing inhibitory masks on corrected red channel')
        labeled_masks_img_corr = get_labeled_masks_img(
            masks_anat_corrected, labels_corrected, 1, 'red')

        print('Computing unsure masks on corrected red channel')
        unsure_masks_img_corr = get_labeled_masks_img(
            masks_anat_corrected, labels_corrected, 0, 'red')
  

        print('Computing excitatory masks on corrected red channel')
        excitory_masks_img_corr = get_labeled_masks_img(
            masks_anat_corrected, labels_corrected, -1
        )

        print(
            f"Anat before: {len(np.unique(masks_anat)) - 1}, Anat after: {len(np.unique(masks_anat_corrected)) - 1}")
        
        
         # FUNCTIONAL DATA PROCESSING

        print('Running Cellpose on functional channel mean image')
        masks_func = run_cellpose(ops, mean_func, diameter) #green channel

       # print('Computing labels for each ROI on functional channel')
       # labels_func = get_label(max_func, masks_anat)
       # print(f'Functional ROI labels: {labels_func}')
        #print(f'Unique functional ROI labels: {np.unique(labels_func)}')

        #print('Computing labeled masks on functional channel')
        #labeled_masks_img_func = get_labeled_masks_img(masks_func, labels_func, 1, 'green')
        #print(f'labeled_masks_img_func: {labeled_masks_img_func}')

       # print('Computing unsure masks on functional channel')
       # unsure_masks_img_func = get_labeled_masks_img(masks_func, labels_func, 0, 'green')
       # print(f'Unique unsure masks: {np.unique(unsure_masks_img_func)}')
       # print(f'labeled_masks_img_func: {np.unique(labeled_masks_img_func)}')
        
        # print(
        #     f"Unsure before: {len(np.argwhere(unsure_masks_img_orig != 0))}, Unsure after: {len(np.argwhere(unsure_masks_img_orig != 0))}")

        # main_channel_comparison_image('Unsure', labels, labels_corrected, mean_anat, mean_anat_corrected, masks_anat, masks_anat_corrected,
        #                               labeled_masks_img_orig, labeled_masks_img_corr, unsure_masks_img_orig, unsure_masks_img_corr, True, mean_func)

        # main_channel_comparison_image('Labeled', labels, labels_corrected, mean_anat, mean_anat_corrected, masks_anat, masks_anat_corrected,
        #                               labeled_masks_img_orig, labeled_masks_img_corr, unsure_masks_img_orig, unsure_masks_img_corr, True, mean_func)

        main_channel_comparison_image('Anat', labels, labels_corrected, mean_anat, mean_anat_corrected, masks_anat, masks_anat_corrected,
                                      labeled_masks_img_orig, labeled_masks_img_corr, unsure_masks_img_orig, unsure_masks_img_corr, True, mean_func)

        removed_neurons_comparison_image('Anat', labels, labels_corrected, mean_anat, mean_anat_corrected, masks_anat, masks_anat_corrected,
                                         None, labeled_masks_img_corr, None, None, True, mean_func)

        new_neurons_comparison_image('Anat', labels, labels_corrected, mean_anat, mean_anat_corrected, masks_anat, masks_anat_corrected,
                                     labeled_masks_img_orig, labeled_masks_img_corr, None, None, True, mean_func)
        
        common_neurons_comparison_image('Common', labels, labels_corrected, mean_anat, mean_anat_corrected, masks_anat, masks_anat_corrected,
                                labeled_masks_img_orig, labeled_masks_img_corr, None, None, True, mean_func)
        
       # plot_unsure_rois(mean_anat, mean_anat_corrected, unsure_masks_img_orig, unsure_masks_img_corr, mean_func)
        plot_combined_unsure_neurons(mean_func, mean_anat, mean_anat_corrected, unsure_masks_img_orig, unsure_masks_img_corr)

        plot_combined_inhibitory_rois(mean_func, mean_anat, mean_anat_corrected, labeled_masks_img_orig, labeled_masks_img_corr)

       # plot_combined_excitatory_rois(mean_func, mean_anat, mean_anat_corrected, excitatory_masks_img_orig, excitatory_masks_img_corr)
        
        

        print(
            f'Found {np.sum(labels == 1)} labeled ROIs out of {len(labels)} in total')

        print(
            f'Found {np.sum(labels == 1)} labeled ROIs out of '
            f'{len(labels)} in total'
        )

        save_masks(
            ops,
            masks_func,
            masks_anat,
            masks_anat_corrected,
            mean_func,
            max_func,
            mean_anat,
            mean_anat_corrected,
            labels,
        )

        print('Masks results saved')
