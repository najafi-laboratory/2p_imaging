import numpy as np


def trim_seq(data, pivots):
    if len(data[0].shape) == 1:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i]) - pivots[i] for i in range(len(data))])
        data = [
            data[i][pivots[i] - len_l_min : pivots[i] + len_r_min]
            for i in range(len(data))
        ]
    if len(data[0].shape) == 3:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i][0, 0, :]) - pivots[i] for i in range(len(data))])
        data = [
            data[i][:, :, pivots[i] - len_l_min : pivots[i] + len_r_min]
            for i in range(len(data))
        ]
    return data


def pad_seq(align_data, align_time):
    pad_time = np.arange(
        np.min([np.min(t) for t in align_time]),
        np.max([np.max(t) for t in align_time]) + 1,
    )
    pad_data = []
    for data, time in zip(align_data, align_time):
        aligned_seq = np.full_like(pad_time, np.nan, dtype=float)
        idx = np.searchsorted(pad_time, time)
        aligned_seq[idx] = data
        pad_data.append(aligned_seq)
    return pad_data, pad_time


def align_neu_seq_utils(neu_seq, neu_time):
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    neu_time = [nt.reshape(1, -1) for nt in neu_time]
    neu_seq = np.concatenate(neu_seq, axis=0)
    neu_time = np.concatenate(neu_time, axis=0)
    neu_time = np.mean(neu_time, axis=0)
    return neu_seq, neu_time
