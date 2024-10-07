#!/usr/bin/env python3

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
from .BleedthroughPlotting import *
# z score normalization.


def normz(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-5)

# run cellpose on one image for cell detection and save the results.


def run_cellpose(
        ops, mean_anat,
        diameter,
        flow_threshold=0.5,
):
    if not os.path.exists(os.path.join(ops['save_path0'], 'cellpose')):
        os.makedirs(os.path.join(ops['save_path0'], 'cellpose'))
    tifffile.imwrite(
        os.path.join(ops['save_path0'], 'cellpose', 'mean_anat.tif'),
        mean_anat)
    model = models.Cellpose(model_type="cyto3")
    masks_anat, flows, styles, diams = model.eval(
        mean_anat,
        diameter=diameter,
        flow_threshold=flow_threshold)
    io.masks_flows_to_seg(
        images=mean_anat,
        masks=masks_anat,
        flows=flows,
        file_names=os.path.join(ops['save_path0'], 'cellpose', 'mean_anat'),
        diams=diameter)
    return masks_anat


# read and cut mask in ops.

def get_mask(ops):
    masks_npy = np.load(
        os.path.join(ops['save_path0'], 'qc_results', 'masks.npy'),
        allow_pickle=True)
    x1 = ops['xrange'][0]
    x2 = ops['xrange'][1]
    y1 = ops['yrange'][0]
    y2 = ops['yrange'][1]
    masks_func = masks_npy[y1:y2, x1:x2]
    mean_func = ops['meanImg'][y1:y2, x1:x2]
    max_func = ops['max_proj']
    if ops['nchannels'] == 2:
        mean_anat = ops['meanImg_chan2'][y1:y2, x1:x2]
    else:
        mean_anat = None
    return masks_func, mean_func, max_func, mean_anat


# compute overlapping to get labels.

def get_label(
        masks_func, masks_anat,
        thres1=0.3, thres2=0.5,
):
    # reconstruct masks into 3d array.
    anat_roi_ids = np.unique(masks_anat)[1:]
    masks_3d = np.zeros(
        (len(anat_roi_ids), masks_anat.shape[0], masks_anat.shape[1]))
    for i, roi_id in enumerate(anat_roi_ids):
        masks_3d[i] = (masks_anat == roi_id).astype(int)
    masks_3d[masks_3d != 0] = 1
    # compute relative overlaps coefficient for each functional roi.
    prob = []
    for i in tqdm(np.unique(masks_func)[1:]):
        roi_masks = (masks_func == i).astype('int32')
        roi_masks_tile = np.tile(
            np.expand_dims(roi_masks, 0),
            (len(anat_roi_ids), 1, 1))
        overlap = (roi_masks_tile * masks_3d).reshape(len(anat_roi_ids), -1)
        overlap = np.sum(overlap, axis=1)
        prob.append(np.max(overlap) / np.sum(roi_masks))
    # threshold probability to get label.
    prob = np.array(prob)
    labels = np.zeros_like(prob)
    # excitory.
    labels[prob < thres1] = -1
    # inhibitory.
    labels[prob > thres2] = 1
    return labels


# save channel img and masks results.

def save_masks(ops, masks_func, masks_anat, masks_anat_corrected, mean_func, max_func, mean_anat, mean_anat_corrected, labels):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'masks.h5'),
        'w')
    f['labels'] = labels
    f['masks_func'] = masks_func
    f['mean_func'] = mean_func
    f['max_func'] = max_func
    if ops['nchannels'] == 2:
        f['mean_anat'] = mean_anat
        f['masks_anat'] = masks_anat
        f['masks_anat_corrected'] = masks_anat_corrected
        f['meanImg_chan2_corrected'] = mean_anat_corrected
    f.close()


# label images with yellow and green.
def get_labeled_masks_img(masks, labels, cate):
    labeled_masks_img = np.zeros(
        (masks.shape[0], masks.shape[1], 3), dtype='int32')
    neuron_idx = np.where(labels == cate)[0] + 1
    for i in neuron_idx:
        neuron_mask = ((masks == i) * 255).astype('int32')
        labeled_masks_img[:, :, 0] += neuron_mask
    return labeled_masks_img


def run(ops, diameter):

    for key in ops.keys():
        print(key)

    print('===============================================')
    print('===== two channel data roi identification =====')
    print('===============================================')
    print('Reading masks in functional channel')
    [masks_func, mean_func, max_func, mean_anat] = get_mask(ops)
    # deal with data here
    if np.max(masks_func) == 0:
        raise ValueError('No masks found.')
    if ops['nchannels'] == 1:
        print('Single channel recording so skip ROI labeling')
        labels = -1 * np.ones(int(np.max(masks_func))).astype('int32')
        save_masks(ops, masks_func, None, mean_func, max_func, None, labels)
    else:
        # remove green from red channel here
        print('Running cellpose on anatomical channel mean image')
        print('Found diameter as {}'.format(diameter))

        print('Computing masks for uncorrected red channel')
        masks_anat = run_cellpose(ops, mean_anat, diameter)
        print("original number of ROIs: ", len(np.argwhere(masks_anat == 1)))

        print('Computing labels for each ROI on uncorrected red channel')
        labels = get_label(masks_func, masks_anat)

        # function to remove green from red channel
        print("Fitting linear model to correct green channel bleedthrough")
        slope, offset, coords = train_reg_model(
            mean_anat=mean_anat, mean_func=mean_func, masks_anat=masks_anat)

        print("Correcting red channel")
        mean_anat_corrected = remove_green_bleedthrough(
            offset=0, slope=slope, mean_func=mean_func, mean_anat=mean_anat, coordinates=coords)

        print('computing corrected mask')
        masks_anat_corrected = run_cellpose(ops, mean_anat_corrected, diameter)
        print("corrected number of ROIs: ", len(
            np.argwhere(masks_anat_corrected == 1)))

        print('Computing corrected labels for each ROI')
        labels_corrected = get_label(masks_func, masks_anat_corrected)

        labeled_masks_img_orig = get_labeled_masks_img(masks_anat, labels, 1)
        labeled_masks_img_corr = get_labeled_masks_img(
            masks_anat_corrected, labels_corrected, 1)

        unsure_masks_img_orig = get_labeled_masks_img(masks_anat, labels, 0)

        unsure_masks_img_corr = get_labeled_masks_img(
            masks_anat_corrected, labels_corrected, 0)

        print(
            f"Anat before: {len(np.argwhere(masks_anat == 1))}, Anat after: {len(np.argwhere(masks_anat_corrected == 1))}")
        print(
            f"Unsure before: {len(np.argwhere(unsure_masks_img_orig == 1))}, Unsure after: {len(np.argwhere(unsure_masks_img_orig == 1))}")

        print(unsure_masks_img_orig[unsure_masks_img_orig != 0])

        mean_anat = mean_anat_corrected
        masks_anat = masks_anat_corrected

        main_channel_comparison_image(
            'Unsure',
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
            mean_func)

        main_channel_comparison_image(
            'Labeled',
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
            mean_func)

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
            mean_func)

        plot_anat_intensities(
            mean_anat, mean_anat_corrected)

        print('Found {} labeled ROIs out of {} in total'.format(
            np.sum(labels == 1), len(labels)))

        save_masks(
            ops,
            masks_func,
            masks_anat,
            masks_anat_corrected,
            mean_func,
            max_func,
            mean_anat,
            mean_anat_corrected,
            labels)

        print('Masks results saved')
