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
    Runs Cellpose for cell detection on the mean anatomical image and saves the results.

    Parameters:
    - ops (dict): Dictionary containing operation parameters.
    - mean_anat (numpy.ndarray): The mean anatomical image.
    - diameter (float): Diameter of cells to be detected.
    - flow_threshold (float, optional): Flow threshold for Cellpose. Defaults to 0.5.

    Returns:
    - numpy.ndarray: Masks of detected cells in the anatomical image.
    """
    save_dir = os.path.join(ops['save_path0'], 'cellpose')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tifffile.imwrite(os.path.join(save_dir, 'mean_anat.tif'), mean_anat)

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


def get_label(masks_func, masks_anat, thres1=0.3, thres2=0.5):
    """
    Computes labels for each ROI based on overlap between functional and anatomical masks.

    Parameters:
    - masks_func (numpy.ndarray): Functional masks.
    - masks_anat (numpy.ndarray): Anatomical masks.
    - thres1 (float, optional): Lower threshold for labeling excitatory neurons.
    - thres2 (float, optional): Upper threshold for labeling inhibitory neurons.

    Returns:
    - numpy.ndarray: Array of labels for each ROI:
        - 1 indicates inhibitory neuron.
        - -1 indicates excitatory neuron.
        - 0 indicates unsure.
    """
    anat_roi_ids = np.unique(masks_anat)[1:]
    masks_3d = np.zeros(
        (len(anat_roi_ids), masks_anat.shape[0], masks_anat.shape[1])
    )

    for i, roi_id in enumerate(anat_roi_ids):
        masks_3d[i] = (masks_anat == roi_id).astype(int)
    masks_3d[masks_3d != 0] = 1

    prob = []
    for i in tqdm(np.unique(masks_func)[1:]):
        roi_masks = (masks_func == i).astype('int32')
        roi_masks_tile = np.tile(
            np.expand_dims(roi_masks, 0), (len(anat_roi_ids), 1, 1)
        )
        overlap = (roi_masks_tile * masks_3d).reshape(len(anat_roi_ids), -1)
        overlap = np.sum(overlap, axis=1)
        prob.append(np.max(overlap) / np.sum(roi_masks))

    prob = np.array(prob)
    labels = np.zeros_like(prob)
    labels[prob < thres1] = -1  # Excitatory
    labels[prob > thres2] = 1   # Inhibitory

    return labels


def save_masks(
    ops,
    masks_func,
    masks_anat,
    masks_anat_corrected,
    mean_func,
    max_func,
    mean_anat,
    mean_anat_corrected,
    labels,
):
    """
    Saves channel images and masks results to an HDF5 file.

    Parameters:
    - ops (dict): Dictionary containing operation parameters.
    - masks_func (numpy.ndarray): Functional masks.
    - masks_anat (numpy.ndarray): Anatomical masks before correction.
    - masks_anat_corrected (numpy.ndarray): Anatomical masks after correction.
    - mean_func (numpy.ndarray): Mean functional image.
    - max_func (numpy.ndarray): Maximum projection functional image.
    - mean_anat (numpy.ndarray): Mean anatomical image before correction.
    - mean_anat_corrected (numpy.ndarray): Mean anatomical image after correction.
    - labels (numpy.ndarray): Labels for each ROI.
    """
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


def get_labeled_masks_img(masks, labels, cate):
    """
    Generates an image with labeled masks for a specific category.

    Parameters:
    - masks (numpy.ndarray): Masks to be labeled.
    - labels (numpy.ndarray): Labels for each ROI.
    - cate (int): Category to label (1 for inhibitory, -1 for excitatory, 0 for unsure).

    Returns:
    - numpy.ndarray: Image with labeled masks.
    """
    labeled_masks_img = np.zeros(
        (masks.shape[0], masks.shape[1], 3), dtype='int32'
    )
    neuron_idx = np.where(labels == cate)[0] + 1

    for i in neuron_idx:
        neuron_mask = ((masks == i) * 255).astype('int32')
        labeled_masks_img[:, :, 0] += neuron_mask

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
        masks_anat = run_cellpose(ops, mean_anat, diameter)

        print('Computing labels for each ROI on uncorrected red channel')
        labels = get_label(masks_func, masks_anat)

        print('Computing inhibitory masks on uncorrected red channel')
        labeled_masks_img_orig = get_labeled_masks_img(masks_anat, labels, 1)

        print('Computing unsure masks on uncorrected red channel')
        unsure_masks_img_orig = get_labeled_masks_img(masks_anat, labels, 0)

        print('Computing excitatory masks on uncorrected red channel')
        # excitory_masks_img_orig = get_labeled_masks_img(masks_anat, labels, -1)
        excitory = get_excitory_rois(
            masks_anat, labeled_masks_img_orig, unsure_masks_img_orig
        )

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
            ops['Ly'],
            ops['Lx'],
            nblks=3,
            mimg=ops['meanImg'],
            mimg2=ops['meanImg_chan2'],
        )
        mean_anat_corrected = get_mask(
            ops, mean_anat_corr=mean_anat_corrected
        )

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
            masks_anat_corrected, labels_corrected, 1
        )

        print('Computing unsure masks on corrected red channel')
        unsure_masks_img_corr = get_labeled_masks_img(
            masks_anat_corrected, labels_corrected, 0
        )

        print('Computing excitatory masks on corrected red channel')
        excitory_masks_img_corr = get_labeled_masks_img(
            masks_anat_corrected, labels_corrected, -1
        )

        print(
            f"Anat before: {len(np.unique(masks_anat)) - 1}, "
            f"Anat after: {len(np.unique(masks_anat_corrected)) - 1}"
        )
        display_mean_image(mean_img=mean_anat)

        main_channel_comparison_image(
            'Anat',
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
            True,
            mean_func,
        )

        three_by_four_comparison(
            with_mask=True,
            mean_func=mean_func,
            mean_anat=mean_anat,
            mean_anat_corrected=mean_anat_corrected,
            masks_anat_both=(masks_anat, masks_anat_corrected),
            masks_func=masks_func,
            labels_both=(labels, labels_corrected),
        )

        superimposed_plots(
            with_mask=True,
            mean_func=mean_func,
            max_func=None,
            mean_anat=mean_anat,
            masks_anat_both=(masks_anat, masks_anat_corrected),
            inhibit_mask_both=(
                labeled_masks_img_orig,
                labeled_masks_img_corr,
            ),
            unsure_mask_both=(
                unsure_masks_img_orig,
                unsure_masks_img_corr,
            ),
            labels_both=(labels, labels_corrected),
        )

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
