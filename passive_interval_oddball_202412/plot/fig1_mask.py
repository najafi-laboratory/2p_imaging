#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from matplotlib.colors import ListedColormap
from scipy.ndimage import median_filter

# label images with yellow and green.
def get_labeled_masks_img(masks, labels, cate):
    labeled_masks_img = np.zeros(
        (masks.shape[0], masks.shape[1], 3), dtype='int32')
    neuron_idx = np.where(labels == cate)[0] + 1
    for i in neuron_idx:
        neuron_mask = ((masks == i) * 255).astype('int32')
        labeled_masks_img[:,:,0] += neuron_mask
    return labeled_masks_img

# get ROI sub image.
def get_roi_range(size, masks):
    row = []
    col = []
    for i in np.unique(masks)[1:]:
        rows, cols = np.where(masks == i)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        center_row = (min_row + max_row) // 2
        center_col = (min_col + max_col) // 2
        row.append(int(max(0, min(masks.shape[0] - size, center_row - size/2))))
        col.append(int(max(0, min(masks.shape[1] - size, center_col - size/2))))
    return row, col
         
# automatical adjustment of contrast.
def adjust_contrast(org_img, lower_percentile=25, upper_percentile=99):
    lower = np.percentile(org_img, lower_percentile)
    upper = np.percentile(org_img, upper_percentile)
    img = np.clip((org_img - lower) * 255 / (upper - lower), 0, 255)
    img = img.astype('int32')
    return img

class plotter_main_masks:
    
    def __init__(
            self,
            labels,
            masks,
            mean_func=None,
            max_func=None,
            mean_anat=None,
            masks_anat=None,
            ):
        self.labels = labels
        self.masks = masks
        self.mean_func = mean_func
        self.max_func = max_func
        self.mean_anat = mean_anat
        self.masks_anat = masks_anat
        self.labeled_masks_img = get_labeled_masks_img(masks, labels, 1)
        self.unsure_masks_img = get_labeled_masks_img(masks, labels, 0)
        self.size = 128
        self.roi_row, self.roi_col = get_roi_range(self.size, masks)

    # functional channel.
    def plot_func(self, ax, img, with_mask=True):
        if img == 'mean':
            f = self.mean_func
            t = 'functional channel mean projection'
        if img == 'max':
            f = self.max_func
            t = 'functional channel max projection'
        if img == 'fuse':
            f = median_filter(self.max_func, size=3)
            t = 'functional channel reference'
        func_img = np.zeros(
            (f.shape[0], f.shape[1], 3), dtype='int32')
        func_img[:,:,1] = adjust_contrast(f)
        func_img = adjust_contrast(func_img)
        if with_mask:
            x_all, y_all = np.where(find_boundaries(self.masks))
            for x,y in zip(x_all, y_all):
                func_img[x,y,:] = np.array([255,255,255])
        ax.matshow(func_img)
        ax.axis('off')
        ax.set_title(t)

    # functional channel ROI masks with color.
    def plot_func_masks_color(self, ax):
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, int(np.max(self.masks)+1)))
        np.random.shuffle(colors)
        colors[0,:] = [0,0,0,1]
        cmap = ListedColormap(colors)
        ax.matshow(self.masks, cmap=cmap)
        ax.axis('off')
        ax.set_title('functional channel ROI masks')

    # functional channel ROI masks.
    def plot_func_masks(self, ax):
        masks_img = np.zeros(
            (self.masks.shape[0], self.masks.shape[1], 3), dtype='int32')
        masks_img[:,:,1] = self.masks
        masks_img[masks_img >= 1] = 255
        ax.matshow(masks_img)
        ax.axis('off')
        ax.set_title('functional channel ROI masks')
    
    # anatomy channel and cellpose results.
    def plot_anat_cellpose(self, ax):
        anat_img = np.zeros(
            (self.mean_anat.shape[0], self.mean_anat.shape[1], 3), dtype='int32')
        anat_img[:,:,0] = adjust_contrast(self.mean_anat)
        anat_img = adjust_contrast(anat_img)
        x_all, y_all = np.where(find_boundaries(self.masks_anat))
        for x,y in zip(x_all, y_all):
            anat_img[x,y,:] = np.array([255,255,255])
        ax.matshow(anat_img)
        ax.axis('off')
        ax.set_title('cellpose results on anatomy channel mean image')
    
    # suite2p masks and cellpose masks superimpose.
    def plot_masks_superimpose(self, ax):
        masks_img = np.zeros(
            (self.masks.shape[0], self.masks.shape[1], 3), dtype='int32')
        masks_img[:,:,1] = self.masks
        masks_img[:,:,0] = self.masks_anat
        masks_img[masks_img >= 1] = 255
        x_all, y_all = np.where(find_boundaries(self.labeled_masks_img[:,:,0]))
        for x,y in zip(x_all, y_all):
            masks_img[x,y,:] = np.array([255,255,255])
        x_all, y_all = np.where(find_boundaries(self.unsure_masks_img[:,:,0]))
        for x,y in zip(x_all, y_all):
            masks_img[x,y,:] = np.array([0,196,255])
        ax.matshow(masks_img)
        ax.axis('off')
        ax.set_title('functional and anatomical masks superimpose')

    # anatomy channel mean image.
    def plot_anat(self, ax, with_mask=True):
        anat_img = np.zeros(
            (self.mean_anat.shape[0], self.mean_anat.shape[1], 3), dtype='int32')
        anat_img[:,:,0] = adjust_contrast(self.mean_anat)
        anat_img = adjust_contrast(anat_img)
        if with_mask:
            x_all, y_all = np.where(find_boundaries(self.masks))
            for x,y in zip(x_all, y_all):
                anat_img[x,y,:] = np.array([255,255,255])
            x_all, y_all = np.where(find_boundaries(self.labeled_masks_img[:,:,0]))
            for x,y in zip(x_all, y_all):
                anat_img[x,y,:] = np.array([255,255,0])
            x_all, y_all = np.where(find_boundaries(self.unsure_masks_img[:,:,0]))
            for x,y in zip(x_all, y_all):
                anat_img[x,y,:] = np.array([0,196,255])
        ax.matshow(anat_img)
        ax.axis('off')
        ax.set_title('anatomy channel mean image')

    # anatomy channel masks.
    def plot_anat_label_masks(self, ax):
        ax.matshow(self.labeled_masks_img)
        ax.axis('off')
        ax.set_title('anatomy channel label masks')

    # superimpose image.
    def plot_superimpose(self, ax, img, with_mask=True):
        if img == 'mean':
            f = self.mean_func
        if img == 'max':
            f = self.max_func
        if img == 'fuse':
            f = median_filter(self.max_func, size=3)
        super_img = np.zeros((f.shape[0], f.shape[1], 3), dtype='int32')
        super_img[:,:,0] = adjust_contrast(self.mean_anat)
        super_img[:,:,1] = adjust_contrast(f)
        super_img = adjust_contrast(super_img)
        if with_mask:
            x_all, y_all = np.where(find_boundaries(self.labeled_masks_img[:,:,0]))
            for x,y in zip(x_all, y_all):
                super_img[x,y,:] = np.array([255,255,255])
        ax.matshow(super_img)
        ax.axis('off')
        ax.set_title('channel images superimpose')

    # channel shared masks.
    def plot_shared_masks(self, ax):
        label_masks = np.zeros(
            (self.masks.shape[0], self.masks.shape[1], 3), dtype='int32')
        label_masks[:,:,0] = get_labeled_masks_img(self.masks, self.labels, 1)[:,:,0]
        label_masks[:,:,1] = self.masks
        label_masks[:,:,2] = get_labeled_masks_img(self.masks, self.labels, 0)[:,:,0]
        label_masks[label_masks >= 1] = 255
        label_masks = label_masks.astype('int32')
        ax.matshow(label_masks)
        ax.axis('off')
        ax.set_title('channel masks superimpose')
    
    # ROI global location for 1 channel data.
    def roi_loc_1chan(self, ax, roi_id, img):
        if img == 'mean':
            f = self.mean_func
        if img == 'max':
            f = self.max_func
        if img == 'fuse':
            f = median_filter(self.max_func, size=3)
        func_img = np.zeros((f.shape[0], f.shape[1], 3), dtype='int32')
        func_img[:,:,1] = adjust_contrast(f)
        func_img = adjust_contrast(func_img)
        x_all, y_all = np.where(self.masks==(roi_id+1))
        c_x = np.mean(x_all).astype('int32')
        c_y = np.mean(y_all).astype('int32')
        func_img[c_x,:,:] = np.array([128,128,255])
        func_img[:,c_y,:] = np.array([128,128,255])
        for x,y in zip(x_all, y_all):
            func_img[x,y,:] = np.array([255,255,255])
        ax.matshow(func_img)
        ax.axis('off')
        ax.set_title('ROI # {} location'.format(str(roi_id).zfill(4)))

    # ROI global location for 2 channel data.
    def roi_loc_2chan(self, ax, roi_id, img):
        if img == 'mean':
            f = self.mean_func
        if img == 'max':
            f = self.max_func
        if img == 'fuse':
            f = median_filter(self.max_func, size=3)
        super_img = np.zeros((f.shape[0], f.shape[1], 3), dtype='int32')
        super_img[:,:,0] = adjust_contrast(self.mean_anat)
        super_img[:,:,1] = adjust_contrast(f)
        super_img = adjust_contrast(super_img)
        x_all, y_all = np.where(self.masks==(roi_id+1))
        c_x = np.mean(x_all).astype('int32')
        c_y = np.mean(y_all).astype('int32')
        super_img[c_x,:,:] = np.array([128,128,255])
        super_img[:,c_y,:] = np.array([128,128,255])
        for x,y in zip(x_all, y_all):
            super_img[x,y,:] = np.array([255,255,255])
        ax.matshow(super_img)
        ax.axis('off')
        if self.labels[roi_id]==-1:
            c = 'excitory'
        if self.labels[roi_id]==0:
            c = 'unsure'
        if self.labels[roi_id]==1:
            c = 'inhibitory'
        ax.set_title('ROI # {} location ({})'.format(str(roi_id).zfill(4), c))
    
    # ROI functional channel.
    def roi_func(self, ax, roi_id, img, with_mask=True):
        if img == 'mean':
            f = self.mean_func
            t = 'functional channel mean projection'
        if img == 'max':
            f = self.max_func
            t = 'functional channel max projection'
        if img == 'fuse':
            f = median_filter(self.max_func, size=3)
            t = 'functional channel reference'
        r = self.roi_row[roi_id]
        c = self.roi_col[roi_id]
        func_img = f[r:r+self.size, c:c+self.size]
        roi_masks = (self.masks[r:r+self.size, c:c+self.size]==(roi_id+1))*1
        img = np.zeros((func_img.shape[0], func_img.shape[1], 3))
        img[:,:,1] = func_img
        img = adjust_contrast(img)
        if with_mask:
            x_all, y_all = np.where(find_boundaries(roi_masks))
            for x,y in zip(x_all, y_all):
                img[x,y,:] = np.array([255,255,255])
        ax.matshow(img)
        ax.axis('off')
        ax.set_title(t)
    
    # ROI anatomical channel mean projection.
    def roi_anat(self, ax, roi_id, with_mask=True):
        r = self.roi_row[roi_id]
        c = self.roi_col[roi_id]
        mean_anat_img = self.mean_anat[r:r+self.size, c:c+self.size]
        roi_masks = (self.masks[r:r+self.size, c:c+self.size]==(roi_id+1))*1
        img = np.zeros((mean_anat_img.shape[0], mean_anat_img.shape[1], 3))
        img[:,:,0] = mean_anat_img
        img = adjust_contrast(img)
        if with_mask:
            x_all, y_all = np.where(find_boundaries(roi_masks))
            for x,y in zip(x_all, y_all):
                img[x,y,:] = np.array([255,255,255])
        ax.matshow(img)
        ax.axis('off')
        ax.set_title('anatomy channel mean image')

    # ROI channel image superimpose with max functional.
    def roi_superimpose(self, ax, roi_id, img, with_mask=True):
        if img == 'mean':
            f = self.mean_func
        if img == 'max':
            f = self.max_func
        if img == 'fuse':
            f = median_filter(self.max_func, size=3)
        super_img = np.zeros((f.shape[0], f.shape[1], 3), dtype='int32')
        super_img[:,:,0] = adjust_contrast(self.mean_anat)
        super_img[:,:,1] = adjust_contrast(f)
        super_img = adjust_contrast(super_img)
        r = self.roi_row[roi_id]
        c = self.roi_col[roi_id]
        super_img = super_img[r:r+self.size, c:c+self.size, :]
        roi_masks = (self.masks[r:r+self.size, c:c+self.size]==(roi_id+1))*1
        if with_mask:
            x_all, y_all = np.where(find_boundaries(roi_masks))
            for x,y in zip(x_all, y_all):
                super_img[x,y,:] = np.array([255,255,255])
        ax.matshow(super_img)
        ax.axis('off')
        ax.set_title('channel images superimpose')

    # ROI masks.
    def roi_masks(self, ax, roi_id):
        r = self.roi_row[roi_id]
        c = self.roi_col[roi_id]
        roi_masks = (self.masks[r:r+self.size, c:c+self.size]==(roi_id+1))*1
        img = np.zeros((roi_masks.shape[0], roi_masks.shape[1], 3), dtype='int32')
        x_all, y_all = np.where(roi_masks)
        for x,y in zip(x_all, y_all):
            img[x,y,:] = np.array([255,255,255])
        ax.matshow(img)
        ax.axis('off')
        ax.set_title('ROI masks')
    
    def all_1chan(self, axs):
        self.plot_func(axs[0], 'mean')
        self.plot_func_masks_color(axs[1])
        self.plot_func(axs[2], 'mean', with_mask=False)
        self.plot_func(axs[3], 'max', with_mask=False)
        self.plot_func(axs[4], 'fuse', with_mask=False)

    def all_2chan(self, axs):
        self.plot_func(axs[0], 'fuse')
        self.plot_func_masks(axs[1])
        self.plot_anat_cellpose(axs[2])
        self.plot_masks_superimpose(axs[3])
        self.plot_anat(axs[4])
        self.plot_anat_label_masks(axs[5])
        self.plot_superimpose(axs[6], 'fuse')
        self.plot_shared_masks(axs[7])
        self.plot_func(axs[8], 'mean', with_mask=False)
        self.plot_anat(axs[9], with_mask=False)
        self.plot_func(axs[10], 'max', with_mask=False)
        self.plot_superimpose(axs[11], 'fuse', with_mask=False)
