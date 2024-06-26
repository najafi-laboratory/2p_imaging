#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cosine


# z score normalization.

def normz(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-5)


# automatical adjustment of contrast.

def adjust_contrast(org_img, lower_percentile=50, upper_percentile=99):
    lower = np.percentile(org_img, lower_percentile)
    upper = np.percentile(org_img, upper_percentile)
    img = np.clip((org_img - lower) * 255 / (upper - lower), 0, 255)
    img = img.astype(np.uint8)
    return img


# read and cut mask in ops.

def get_mask(ops):
    masks_npy = np.load(
        os.path.join(ops['save_path0'], 'qc_results', 'masks.npy'),
        allow_pickle=True)
    x1 = ops['xrange'][0]
    x2 = ops['xrange'][1]
    y1 = ops['yrange'][0]
    y2 = ops['yrange'][1]
    masks = masks_npy[y1:y2,x1:x2]
    mean_func = ops['meanImg'][y1:y2,x1:x2]
    max_func = ops['max_proj']
    if ops['nchannels'] == 2:
        mean_anat = ops['meanImg_chan2'][y1:y2,x1:x2]
    else:
        mean_anat = None
    return masks, mean_func, max_func, mean_anat


# signal ratio between ROI and its surroundings.

def get_roi_sur_ratio(
        mask_roi,
        mean_anat
        ):
    mask_roi_ext = binary_dilation(mask_roi, iterations=2)
    # dilation for mask including surrounding.
    mask_roi_surr = binary_dilation(mask_roi_ext, iterations=3)
    # binary operation to get surrounding mask.
    mask_surr = mask_roi_surr != mask_roi_ext
    # compute mean signal.
    sig_roi = (mean_anat * mask_roi_ext).reshape(-1)
    sig_surr = (mean_anat * mask_surr).reshape(-1)
    # compute ratio.
    ratio = np.array([[
        np.sum(sig_roi) / np.sum(sig_surr),
        np.sum(sig_roi**2 - sig_surr**2)
        ]])
    return ratio


def get_spat_dis(
        mask_roi,
        mean_anat,
        max_func,
        ):
    anat_roi = (normz(mean_anat) * mask_roi).reshape(-1)
    anat_roi = anat_roi[anat_roi!=0].reshape(-1)
    func_roi = (normz(max_func) * mask_roi).reshape(-1)
    func_roi = func_roi[func_roi!=0].reshape(-1)
    dis = cosine(func_roi, anat_roi)
    return dis


# save channel img and masks results.

def save_masks(ops, masks, mean_func, max_func, mean_anat, labels):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'masks.h5'),
        'w')
    f['labels'] = labels
    f['masks'] = masks
    f['mean_func'] = mean_func
    f['max_func'] = max_func
    if ops['nchannels'] == 2:
        f['mean_anat'] = mean_anat
    f.close()


# outlier detection for standard multivariate normal.

def outlier_detect(features, thres_1=2, thres_2=3):
    mean = np.mean(features, axis=0)
    cov = np.cov(features.T)
    model = multivariate_normal(mean=mean, cov=cov)
    prob = model.pdf(features)
    prob_thres_1 = model.pdf(mean + thres_1 * np.sqrt(np.diag(cov)))
    prob_thres_2 = model.pdf(mean + thres_2 * np.sqrt(np.diag(cov)))
    labels = np.zeros(len(prob))
    # inhibitory
    labels[prob < prob_thres_2] = 1
    # excitory
    labels[prob > prob_thres_1] = -1
    return labels


# main function to use anatomical to label functional channel masks.
'''
import tifffile
tifffile.imwrite('./mean_anat.tif', mean_anat)
'''
def run(ops):
    print('===============================================')
    print('===== two channel data roi identification =====')
    print('===============================================')
    [masks, mean_func, max_func, mean_anat] = get_mask(ops)
    if np.max(masks) == 0:
        raise ValueError('No masks found.')
    if ops['nchannels'] == 1:
        print('Single channel recording so skip ROI labeling')
        labels = -1 * np.ones(int(np.max(masks))).astype('int32')
        save_masks(ops, masks, mean_func, max_func, mean_anat, labels)
    else:
        # specify functional and anatomical masks.
        surr_ratio = []
        spat_dis = []
        print('Collecting ROI statistics')
        for i in np.unique(masks)[1:]:
            # get ROI mask.
            mask_roi = (masks==i).copy()
            # metrics.
            r = get_roi_sur_ratio(mask_roi, mean_anat)
            d = get_spat_dis(
                mask_roi,
                adjust_contrast(mean_anat),
                adjust_contrast(max_func))
            surr_ratio.append(r)
            spat_dis.append(d)
        surr_ratio = np.concatenate(surr_ratio)
        for i in range(surr_ratio.shape[1]):
            surr_ratio[:,i] = normz(surr_ratio[:,i])
        spat_dis = np.log(np.array(spat_dis))
        features = np.concatenate(
            (normz(spat_mag).reshape(-1,1),
             normz(surr_ratio).reshape(-1,1)
             ), axis=1)
        # import matplotlib.pyplot as plt
        # plt.hist(spat_dis, bins=500)
        # plt.hist(surr_ratio, bins=500)
        # plt.scatter(spat_std, spat_dif)
        # plt.scatter(features[:,0], features[:,1])
        # 0 : only in functional.
        # 1 : in functional and marked in anatomical.
        print('Running outlier detection')
        labels = outlier_detect(features)
        print('Found {} inhibitory neurons with new ROI id {}'.format(
            len(np.where(labels==1)[0]), np.where(labels==1)[0]))
        print('Found {} unsure inhibitory neurons'.format(
            len(np.where(labels==0)[0])))
        print('Found the rest {} excitory neurons'.format(
            len(np.where(labels==-1)[0])))
        save_masks(ops, masks, mean_func, max_func, mean_anat, labels)
