#!/usr/bin/env python3

import time
import functools
import tracemalloc
import numpy as np
import rastermap as rm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
from datetime import datetime
from scipy.stats import gaussian_kde

# compute mean and sem for array data.
def get_mean_sem(data, win_baseline=None):
    # compute mean.
    m = np.nanmean(data.reshape(-1, data.shape[-1]), axis=0)
    # compute sem.
    std = np.nanstd(data.reshape(-1, data.shape[-1]), axis=0)
    count = np.nansum(~np.isnan(data.reshape(-1, data.shape[-1])), axis=0)
    s = std / np.sqrt(count)
    return m, s

# compute indice with givn time window for dF/F.
def get_frame_idx_from_time(timestamps, c_time, l_time, r_time):
    l_idx = np.searchsorted(timestamps, c_time+l_time)
    r_idx = np.searchsorted(timestamps, c_time+r_time)
    return l_idx, r_idx

# return colors from dim to dark with a base color.
def get_cmap_color(n_colors, base_color=None, cmap=None, return_cmap=False):
    c_margin = 0.05
    if base_color != None:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            None, base_color)
    if cmap != None:
        pass
    colors = cmap(np.linspace(c_margin, 1 - c_margin, n_colors))
    colors = [mcolors.to_hex(color, keep_alpha=True) for color in colors]
    if return_cmap:
        return cmap, colors
    else:
        return colors

# add legend into subplots.
def add_legend(ax, colors, labels, loc):
    plot_args = [[],[]]
    handles = []
    if colors != None and labels != None:
        handles += [
            ax.plot(*plot_args, lw=0, color=colors[i], label=labels[i])[0]
            for i in range(len(labels))]
    ax.legend(
        loc=loc,
        handles=handles,
        labelcolor='linecolor',
        frameon=False,
        framealpha=0)

# hide all axis.
def hide_all_axis(ax):
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_mean_sem(ax, t, m, s, c, l=None, a=1.0):
    ax.plot(t, m, color=c, label=l, alpha=a)
    ax.fill_between(t, m - s, m + s, color=c, alpha=0.25, edgecolor='none')
    ax.set_xlim([np.min(t), np.max(t)])
    
def plot_half_violin(ax, data, x, color, side, fill=True, alpha=0.3):
    p = ax.violinplot(data[~np.isnan(data)], positions=[x], widths=1, showextrema=False)
    v = p['bodies'][0].get_paths()[0].vertices
    p['bodies'][0].set(facecolor=color if fill else 'none', alpha=alpha)
    p['bodies'][0].set_edgecolor(color)
    p['bodies'][0].set_linewidth(1)
    m = v[:,0].mean()
    v[:,0] = np.minimum(v[:,0], m) if side.lower().startswith('l') else np.maximum(v[:,0], m)

def plot_dist(ax, data, c, a, cumulative):
    bins = 25
    # raw counts
    counts, bin_edges = np.histogram(data, bins=bins)
    # fraction of samples in each bin.
    total = counts.sum()
    fractions = counts / total  
    # bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # cumulative.
    if cumulative:
        fractions = np.cumsum(fractions)
    # plot line.
    ax.plot(bin_centers, fractions, color=c, alpha=a)

def plot_mean_sem_acatter_box(ax, data, pos, color, alpha):
    m,s = get_mean_sem(data.reshape(-1,1))
    ax.scatter(
        np.ones_like(data)*pos+np.random.uniform(-0.2, 0.2, size=len(data)), data,
        color=color, alpha=alpha)
    ax.errorbar(
        pos, m, s, None,
        color='black',
        capsize=2, marker='o', linestyle='none',
        markeredgecolor='white', markeredgewidth=0.1)
    
