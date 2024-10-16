#!/usr/bin/env python3

import os
import h5py
import numpy as np
import skimage

# read raw results from suite2p pipeline.


def read_raw(ops):
    """
    Load raw fluorescence and neuropil data from Suite2p results.

    Args:
        ops (dict): A dictionary containing Suite2p output paths and configuration.

    Returns:
        list: A list containing green channel fluorescence (F), red channel fluorescence (F_chan2), neuropil signal (Fneu), and neuron statistics (stat).
    """
    # green channel
    F = np.load(
        os.path.join(ops['save_path0'],
                     'suite2p', 'plane0', 'F.npy'), allow_pickle=True)
    # red channel
    F_chan2 = np.load(
        os.path.join(ops['save_path0'],
                     'suite2p', 'plane0', 'F_chan2.npy'), allow_pickle=True)
    Fneu = np.load(
        os.path.join(ops['save_path0'],
                     'suite2p', 'plane0', 'Fneu.npy'), allow_pickle=True)
    stat = np.load(
        os.path.join(ops['save_path0'],
                     'suite2p', 'plane0', 'stat.npy'), allow_pickle=True)
    return [F, F_chan2, Fneu, stat]

# get metrics for ROIs.


def get_metrics(ops, stat):
    """
    Compute various metrics for ROIs, including footprint, skew, aspect ratio, compactness, and connectivity.

    Args:
        ops (dict): A dictionary containing Suite2p output paths and configuration.
        stat (list): A list of dictionaries containing statistics for each ROI.

    Returns:
        tuple: A tuple containing arrays of skew, connectivity, aspect ratio, compactness, and footprint for each ROI.
    """
    # rearrange existing statistics for masks.
    # https://suite2p.readthedocs.io/en/latest/outputs.html#stat-npy-fields
    footprint = np.array([stat[i]['footprint'] for i in range(len(stat))])
    skew = np.array([stat[i]['skew'] for i in range(len(stat))])
    aspect = np.array([stat[i]['aspect_ratio'] for i in range(len(stat))])
    compact = np.array([stat[i]['compact'] for i in range(len(stat))])
    # compute connectivity of ROIs.
    masks = stat_to_masks(ops, stat)
    connect = []
    for i in np.unique(masks)[1:]:
        # find a mask with one ROI.
        m = masks.copy() * (masks == i)
        # find component number.
        connect.append(np.max(skimage.measure.label(m, connectivity=1)))
    connect = np.array(connect)
    return skew, connect, aspect, compact, footprint

# threshold the statistics to keep good ROIs.


def thres_stat(
        ops, stat,
        range_skew,
        max_connect,
        max_aspect,
        range_compact,
        range_footprint
):
    """
    Apply thresholds to ROI metrics to identify bad ROIs.

    Args:
        ops (dict): A dictionary containing Suite2p output paths and configuration.
        stat (list): A list of dictionaries containing statistics for each ROI.
        range_skew (list): The acceptable range of skew values.
        max_connect (float): The maximum allowed connectivity components.
        max_aspect (float): The maximum allowed aspect ratio.
        range_compact (list): The acceptable range of compactness values.
        range_footprint (list): The acceptable range of footprint values.

    Returns:
        np.array: An array of indices representing bad ROIs.
    """
    skew, connect, aspect, compact, footprint = get_metrics(ops, stat)
    # find bad ROI indices.
    bad_roi_id = set()
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where((footprint < range_footprint[0]) | (footprint > range_footprint[1]))[0])
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where((skew < range_skew[0]) | (skew > range_skew[1]))[0])
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where(aspect > max_aspect)[0])
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where((compact < range_compact[0]) | (compact > range_compact[1]))[0])
    bad_roi_id = bad_roi_id.union(
        bad_roi_id,
        np.where(connect > max_connect)[0])
    # convert set to numpy array for indexing.
    bad_roi_id = np.array(list(bad_roi_id))
    return bad_roi_id

# reset bad ROIs in the masks to nothing.


def reset_roi(
        bad_roi_id,
        F, F_chan2, Fneu, stat
):
    """
    Remove bad ROIs from data and keep only the good ROIs.

    Args:
        bad_roi_id (np.array): An array of indices representing bad ROIs.
        F (np.array): Green channel fluorescence data.
        F_chan2 (np.array): Red channel fluorescence data.
        Fneu (np.array): Neuropil signal data.
        stat (list): A list of dictionaries containing statistics for each ROI.

    Returns:
        tuple: A tuple containing the fluorescence data, red channel data, neuropil data, and updated statistics for good ROIs.
    """
    # reset bad ROI.
    for i in bad_roi_id:
        stat[i] = None
        F[i, :] = 0
        Fneu[i, :] = 0
    # find good ROI indices.
    good_roi_id = np.where(np.sum(F, axis=1) != 0)[0]
    # keep good ROI signals.
    fluo = F[good_roi_id, :]
    fluo_chan2 = F_chan2[good_roi_id, :]
    neuropil = Fneu[good_roi_id, :]
    stat = stat[good_roi_id]
    return fluo, fluo_chan2, neuropil, stat

# save results into npy files.


def save_qc_results(
        ops,
        fluo, fluo_chan2, neuropil, stat, masks, stat_file_name='stat'
):
    """
    Save quality control results to disk.

    Args:
        ops (dict): A dictionary containing Suite2p output paths and configuration.
        fluo (np.array): Fluorescence data after QC.
        fluo_chan2 (np.array): Red channel data after QC.
        neuropil (np.array): Neuropil data after QC.
        stat (list): A list of dictionaries containing statistics for each ROI after QC.
        masks (np.array): Masks of each ROI after QC.
        stat_file_name (str, optional): Name of the file to save the ROI statistics. Defaults to 'stat'.
    """
    if not os.path.exists(os.path.join(ops['save_path0'], 'qc_results')):
        os.makedirs(os.path.join(ops['save_path0'], 'qc_results'))
    np.save(os.path.join(ops['save_path0'], 'qc_results', 'fluo.npy'), fluo)
    np.save(os.path.join(ops['save_path0'],
            'qc_results', 'fluo_chan2.npy'), fluo_chan2)
    np.save(os.path.join(ops['save_path0'],
            'qc_results', 'neuropil.npy'), neuropil)
    np.save(os.path.join(ops['save_path0'],
            'qc_results', f'{stat_file_name}.npy'), stat)
    np.save(os.path.join(ops['save_path0'], 'qc_results', 'masks.npy'), masks)
    np.save(os.path.join(ops['save_path0'], 'ops.npy'), ops)

# convert stat.npy results to ROI masks matrix.


def stat_to_masks(ops, stat):
    """
    Convert ROI statistics to a mask matrix.

    Args:
        ops (dict): A dictionary containing Suite2p output paths and configuration.
        stat (list): A list of dictionaries containing statistics for each ROI.

    Returns:
        np.array: A mask matrix where each pixel is labeled with the corresponding ROI number.
    """
    masks = np.zeros((ops['Ly'], ops['Lx']))
    for n in range(len(stat)):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        masks[ypix, xpix] = n+1

    return masks

# save motion correction offsets.


def save_move_offset(ops):
    """
    Save motion correction offsets to an HDF5 file.

    Args:
        ops (dict): A dictionary containing Suite2p output paths and configuration, including motion correction offsets.
    """
    xoff = ops['xoff']
    yoff = ops['yoff']
    f = h5py.File(os.path.join(ops['save_path0'], 'move_offset.h5'), 'w')
    f['xoff'] = xoff
    f['yoff'] = yoff
    f.close()

# main function for quality control.


def run(
        ops,
        range_skew, max_connect, max_aspect, range_compact, range_footprint, stat_file_names,
        run_qc=True
):
    """
    Run the quality control pipeline for Suite2p imaging data.

    Args:
        ops (dict): A dictionary containing Suite2p output paths and configuration.
        range_skew (list): The acceptable range of skew values.
        max_connect (float): The maximum allowed connectivity components.
        max_aspect (float): The maximum allowed aspect ratio.
        range_compact (list): The acceptable range of compactness values.
        range_footprint (list): The acceptable range of footprint values.
        stat_file_names (list): List of names for saving the ROI statistics.
        run_qc (bool, optional): Whether to run quality control. Defaults to True.
    """
    print('===============================================')
    print('=============== quality control ===============')
    print('===============================================')
    print('Found range of footprint from {} to {}'.format(
        range_footprint[0], range_footprint[1]))
    print('Found range of skew from {} to {}'.format(
        range_skew[0], range_skew[1]))
    print('Found max number of connectivity components as {}'.format(
        max_connect))
    print('Found maximum aspect ratio as {}'.format(
        max_aspect))
    print('Found range of campact as {}'.format(
        range_compact))
    [F, F_chan2, Fneu, stat] = read_raw(ops)
    print('Found {} ROIs from suite2p'.format(F.shape[0]))
    if run_qc:
        bad_roi_id = thres_stat(
            ops, stat,
            range_skew, max_connect, max_aspect, range_compact, range_footprint)
        print('Found {} bad ROIs'.format(len(bad_roi_id)))
    else:
        bad_roi_id = []
    fluo, fluo_chan2, neuropil, stat = reset_roi(
        bad_roi_id, F, F_chan2, Fneu, stat)
    print('Saving {} ROIs after quality control'.format(fluo.shape[0]))
    masks = stat_to_masks(ops, stat)

    for name in stat_file_names:
        save_qc_results(ops, fluo, fluo_chan2, neuropil, stat, masks, name)
    save_move_offset(ops)
