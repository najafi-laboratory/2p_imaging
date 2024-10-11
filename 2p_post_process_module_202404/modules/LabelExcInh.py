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

from .RemoveBleedthrough import *
from .BleedthroughPlotting import *

# Z-score normalization


def normz(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-5)

# Run Cellpose for cell detection and save results


def run_cellpose(ops, mean_anat, diameter, flow_threshold=0.5):
    save_dir = os.path.join(ops['save_path0'], 'cellpose')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tifffile.imwrite(os.path.join(save_dir, 'mean_anat.tif'), mean_anat)

    model = models.Cellpose(model_type="cyto3")
    masks_anat, flows, styles, diams = model.eval(
        mean_anat, diameter=diameter, flow_threshold=flow_threshold)

    io.masks_flows_to_seg(
        images=mean_anat,
        masks=masks_anat,
        flows=flows,
        file_names=os.path.join(save_dir, 'mean_anat'),
        diams=diameter
    )

    return masks_anat

# Read and cut masks in ops


def get_mask(ops):
    masks_npy = np.load(os.path.join(
        ops['save_path0'], 'qc_results', 'masks.npy'), allow_pickle=True)
    x1, x2 = ops['xrange']
    y1, y2 = ops['yrange']

    masks_func = masks_npy[y1:y2, x1:x2]
    mean_func = ops['meanImg'][y1:y2, x1:x2]
    max_func = ops['max_proj']

    mean_anat = ops['meanImg_chan2'][y1:y2,
                                     x1:x2] if ops['nchannels'] == 2 else None

    return masks_func, mean_func, max_func, mean_anat

# Compute overlapping to get labels


def get_label(masks_func, masks_anat, thres1=0.3, thres2=0.5):
    anat_roi_ids = np.unique(masks_anat)[1:]
    masks_3d = np.zeros(
        (len(anat_roi_ids), masks_anat.shape[0], masks_anat.shape[1]))

    for i, roi_id in enumerate(anat_roi_ids):
        masks_3d[i] = (masks_anat == roi_id).astype(int)
    masks_3d[masks_3d != 0] = 1

    prob = []
    for i in tqdm(np.unique(masks_func)[1:]):
        roi_masks = (masks_func == i).astype('int32')
        roi_masks_tile = np.tile(np.expand_dims(
            roi_masks, 0), (len(anat_roi_ids), 1, 1))
        overlap = (roi_masks_tile * masks_3d).reshape(len(anat_roi_ids), -1)
        overlap = np.sum(overlap, axis=1)
        prob.append(np.max(overlap) / np.sum(roi_masks))

    prob = np.array(prob)
    labels = np.zeros_like(prob)
    labels[prob < thres1] = -1  # Excitatory
    labels[prob > thres2] = 1   # Inhibitory

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

# Label images with yellow and green


def get_labeled_masks_img(masks, labels, cate):
    labeled_masks_img = np.zeros(
        (masks.shape[0], masks.shape[1], 3), dtype='int32')
    neuron_idx = np.where(labels == cate)[0] + 1

    for i in neuron_idx:
        neuron_mask = ((masks == i) * 255).astype('int32')
        labeled_masks_img[:, :, 0] += neuron_mask

    return labeled_masks_img


def read_fluos(ops):
    path = ops['save_path0']

    fluo = np.load(os.path.join(path, 'qc_results',
                   'fluo.npy'), allow_pickle=True)
    print("fluo shape: ", fluo.shape)

    fluo_chan2 = np.load(os.path.join(path, 'qc_results',
                         'fluo_chan2.npy'), allow_pickle=True)
    print("fluo chan2 shape: ", fluo_chan2.shape)

    return fluo, fluo_chan2

# Main function to run the workflow


def run(ops, diameter):
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
        save_masks(ops, masks_func, None, None,
                   mean_func, max_func, None, labels)
    else:
        print('Running Cellpose on anatomical channel mean image')
        print(f'Found diameter as {diameter}')

        # UNCORRECTED MASKS AND LABELS

        print('Computing masks for uncorrected red channel')
        masks_anat = run_cellpose(ops, mean_anat, diameter)

        print('Computing labels for each ROI on uncorrected red channel')
        labels = get_label(masks_func, masks_anat)

        print('Computing labeled masks on uncorrected red channel')
        labeled_masks_img_orig = get_labeled_masks_img(masks_anat, labels, 1)

        print('Computing unsure masks on uncorrected red channel')
        unsure_masks_img_orig = get_labeled_masks_img(masks_anat, labels, 0)

        # Fluorescence data
        fluo, fluo_chan2 = read_fluos(ops)

        print("Fitting linear model to correct green channel bleedthrough")
        slope, offset, fluo_means, fluo_chan2_means = train_reg_model(
            fluo, fluo_chan2)

        print("Correcting red channel")
        Fchan2_corrected_means = Fchan2_corrected_anat(
            fluo_means, fluo_chan2_means, slope, 0)

        mean_anat_corrected = update_mean_anat(
            mean_anat, Fchan2_corrected_means, masks_anat)

        # CORRECTED MASKS AND LABELS

        print('Computing corrected mask')
        masks_anat_corrected = run_cellpose(ops, mean_anat_corrected, diameter)

        print('Computing corrected labels for each ROI')
        labels_corrected = get_label(masks_func, masks_anat_corrected)

        print('Computing labeled masks on corrected red channel')
        labeled_masks_img_corr = get_labeled_masks_img(
            masks_anat_corrected, labels_corrected, 1)

        print('Computing unsure masks on corrected red channel')
        unsure_masks_img_corr = get_labeled_masks_img(
            masks_anat_corrected, labels_corrected, 0)

        print("Corrected number of ROIs: ", len(
            np.argwhere(masks_anat_corrected != 0)))

        print(
            f"Anat before: {len(np.argwhere(masks_anat != 0))}, Anat after: {len(np.argwhere(masks_anat_corrected != 0))}")
        print(
            f"Unsure before: {len(np.argwhere(unsure_masks_img_orig != 0))}, Unsure after: {len(np.argwhere(unsure_masks_img_orig != 0))}")

        main_channel_comparison_image('Unsure', labels, labels_corrected, mean_anat, mean_anat_corrected, masks_anat, masks_anat_corrected,
                                      labeled_masks_img_orig, labeled_masks_img_corr, unsure_masks_img_orig, unsure_masks_img_corr, True, mean_func)

        main_channel_comparison_image('Labeled', labels, labels_corrected, mean_anat, mean_anat_corrected, masks_anat, masks_anat_corrected,
                                      labeled_masks_img_orig, labeled_masks_img_corr, unsure_masks_img_orig, unsure_masks_img_corr, True, mean_func)

        main_channel_comparison_image('Anat', labels, labels_corrected, mean_anat, mean_anat_corrected, masks_anat, masks_anat_corrected,
                                      labeled_masks_img_orig, labeled_masks_img_corr, unsure_masks_img_orig, unsure_masks_img_corr, True, mean_func)

        plot_anat_intensities(mean_anat, mean_anat_corrected)

        print(
            f'Found {np.sum(labels == 1)} labeled ROIs out of {len(labels)} in total')

        save_masks(ops, masks_func, masks_anat, masks_anat_corrected,
                   mean_func, max_func, mean_anat, mean_anat_corrected, labels)

        print('Masks results saved')
