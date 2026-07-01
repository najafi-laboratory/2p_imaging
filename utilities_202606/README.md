# Reusable analysis utilities

`utils.py` keeps project-independent helpers for array processing, event alignment, statistics, population summaries, and Matplotlib plotting. It does not read files or define project-specific labels. Each project should load its own data, convert it to NumPy arrays, then call these helpers.

## Dependencies

Core dependencies:

```bash
pip install numpy scipy scikit-learn matplotlib rastermap
```

## Shape conventions

- `data`: `(n_units, n_time)` unless noted otherwise.
- `time_ms`, `neu_time`, `timestamps`: `(n_time,)`, in milliseconds.
- `event_times`: `(n_events,)`, in milliseconds.
- `aligned`: `(n_events, n_units, n_aligned_time)`.
- `neu_seq`: commonly `(n_trials, n_neurons, n_time)` or `(n_rows, n_time)` depending on function.
- `stim_seq`: `(n_trials, n_stim, 2)`, where the last axis stores start/end times.
- `list` inputs usually represent sessions and should contain arrays with matching trailing dimensions.

## Typical workflow

1. Load project data in a project script.
2. Put continuous signals into `data` as `(n_units, n_time)`.
3. Normalize with `zscore_rows`, `norm01`, or `rescale`.
4. Align around events with `align_events`.
5. Summarize traces with `get_mean_sem` or window responses with `get_mean_sem_win`.
6. Compare groups with `get_stat_test`, `auc_test`, or `get_win_mag_quant_stat_test`.
7. Plot traces, heatmaps, distributions, and latent dynamics with `utils_basic`.

## Function reference

### General processing

`show_resource_usage(func)`

- Main function: decorator that prints runtime and memory usage for `func`.
- Steps: starts `tracemalloc`; runs the function; reads elapsed time/current memory/peak memory; prints a short report.
- Input: any callable.
- Output: wrapped callable with the same return value as `func`.

`drop_nan(data)`

- Main function: remove NaN values from one array.
- Steps: converts input to an array; returns values where `~np.isnan(data)`.
- Input: any numeric array shape.
- Output: one-dimensional valid values, shape `(n_valid,)`.

`flatten_time(data)`

- Main function: flatten all leading axes while preserving time as the last axis.
- Steps: converts input to an array; reshapes to `(-1, data.shape[-1])`.
- Input: any array with time on the last axis, shape `(..., n_time)`.
- Output: `(n_samples, n_time)`.

`zscore_rows(data)`

- Main function: row-wise z-score normalization.
- Steps: computes row means; computes row standard deviations; replaces near-zero standard deviations with `1`; returns `(data - mean) / std`.
- Input: `(n_rows, n_time)`.
- Output: `(n_rows, n_time)`.

`rescale(data, upper, lower)`

- Main function: map data into `[lower, upper]`.
- Steps: runs `norm01`; multiplies by `upper - lower`; adds `lower`.
- Input: any numeric array shape.
- Output: same shape as `data`.

`norm01(data)`

- Main function: min-max normalize values into `[0, 1]`.
- Steps: computes nan-safe min/max; divides by the guarded range.
- Input: any numeric array shape.
- Output: same shape as `data`.

`get_norm01_params(data)`

- Main function: return affine parameters for min-max normalization.
- Steps: computes nan-safe min/max; computes scale and offset for `data * scale + offset`.
- Input: any numeric array shape.
- Output: `(x_scale, x_offset, x_min, x_max)`, all scalars.

`get_bin_idx(data, bin_win, bin_num)`

- Main function: assign values to equal-width bins.
- Steps: builds `bin_num + 1` edges with `np.linspace`; computes centers; digitizes values.
- Input: `data` any shape, `bin_win=(low, high)`, `bin_num` integer.
- Output: `bins (bin_num+1,)`, `bin_center (bin_num,)`, `bin_idx` same shape as `data`.

`get_bin_idx_list(data_list, bin_win, bin_num)`

- Main function: bin multiple arrays with shared edges.
- Steps: builds shared bins; digitizes each array in `data_list`.
- Input: list of arrays, each usually `(n_trials_session,)`.
- Output: `bins (bin_num+1,)`, `bin_center (bin_num,)`, `list_bin_idx` list matching `data_list`.

### Event alignment and summaries

`get_frame_idx_from_time(timestamps, c_time, l_time, r_time)`

- Main function: convert a relative time window into frame indices.
- Steps: searches sorted `timestamps` at `c_time + l_time` and `c_time + r_time`.
- Input: `timestamps (n_time,)`; scalar center/left/right times.
- Output: `l_idx`, `r_idx` integers.

`get_frame_window(time_ms, pre_s, post_s)`

- Main function: compute alignment frame counts and relative aligned time.
- Steps: estimates frame interval from median timestamp differences; converts seconds to frame counts; builds relative time.
- Input: `time_ms (n_time,)`, `pre_s`, `post_s` in seconds.
- Output: `l_frames`, `r_frames`, `aligned_time (n_aligned_time,)`.

`align_event(data, time_ms, event_time, pre_s, post_s)`

- Main function: align one continuous recording around one event.
- Steps: computes frame window; finds event frame; copies the complete source slice; returns NaNs if the window crosses boundaries.
- Input: `data (n_units, n_time)`, `time_ms (n_time,)`, scalar event time.
- Output: `aligned (n_units, n_aligned_time)`, `aligned_time (n_aligned_time,)`.

`align_events(data, time_ms, event_times, pre_s, post_s)`

- Main function: align many events.
- Steps: calls `align_event` for each event; stacks aligned arrays; returns the shared time axis.
- Input: `data (n_units, n_time)`, `event_times (n_events,)`.
- Output: `aligned (n_events, n_units, n_aligned_time)`, `aligned_time (n_aligned_time,)`.

`get_mean_sem_win(neu_seq, neu_time, c_time, l_time, r_time, mode, pct=25)`

- Main function: summarize each row inside a time window.
- Steps: slices the time window; computes row values by `mean`, `lower` percentile mean, or `higher` percentile mean; computes across-row mean and SEM.
- Input: `neu_seq (n_rows, n_time)`, `neu_time (n_time,)`.
- Output: `neu (n_rows,)`, `neu_mean` scalar, `neu_sem` scalar.

`get_mean_sem(data, method_m='mean', method_s='standard error', zero_start=False)`

- Main function: compute a time trace and uncertainty.
- Steps: flattens leading axes; computes mean or median; optionally subtracts first time point; computes standard deviation/count; converts to SEM, SD, confidence interval, or prediction interval.
- Input: `data (..., n_time)`.
- Output: `m (n_time,)`, `s (n_time,)`.

`get_peak_time(neu, neu_time, win_peak)`

- Main function: find peak time inside a window.
- Steps: slices the window; uses `scipy.signal.find_peaks`; falls back to `nanargmax` when no peak is found.
- Input: `neu (n_time,)`, `neu_time (n_time,)`, `win_peak=(left, right)`.
- Output: scalar peak time.

### Trial and block helpers

`sub_sampling_trial(neu_seq, samping_size=3, sampling_times=252)`

- Main function: average random trial subsets.
- Steps: converts fractional sample size to count; samples trials without replacement; averages each subset.
- Input: `neu_seq (n_trials, n_neurons, n_time)`.
- Output: `(sampling_times, n_neurons, n_time)`.

`get_block_1st_idx(trial_lbl, prepost='post')`

- Main function: mark first trials around label transitions.
- Steps: computes label differences; selects transitions into label `0` and label `1`.
- Input: `trial_lbl (n_trials,)`.
- Output: `idx_0 (n_trials,)`, `idx_1 (n_trials,)` boolean masks.

`get_block_epochs_idx(trial_lbl, epoch_len, block_combine=True)`

- Main function: split continuous label blocks into fixed-length epochs.
- Steps: finds block boundaries; builds one mask per full epoch; trims blocks to equal epoch counts per label; optionally combines blocks.
- Input: `trial_lbl (n_trials,)`, `epoch_len` integer.
- Output: if combined, `(n_epochs, n_trials)` masks per label; otherwise `(n_blocks, n_epochs, n_trials)` masks per label.

`get_block_transition_idx(trial_lbl, trials_around)`

- Main function: mark windows around 0-to-1 and 1-to-0 transitions.
- Steps: finds transition indices; removes incomplete windows; creates boolean masks around each transition.
- Input: `trial_lbl (n_trials,)`.
- Output: `trans_0to1 (n_transitions, n_trials)`, `trans_1to0 (n_transitions, n_trials)`.

`get_change_prepost_idx(trial_lbl, target)`

- Main function: mark trials before target membership and all target trials.
- Steps: computes membership changes; keeps pre-change starts; marks all target labels.
- Input: `trial_lbl (n_trials,)`, `target` list or array of labels.
- Output: `idx_pre (n_trials,)`, `idx_post (n_trials,)` boolean masks.

`get_split_idx(list_labels, cate)`

- Main function: create split points for concatenated category-filtered arrays.
- Steps: counts selected labels per session; returns cumulative counts except the last.
- Input: `list_labels` list of `(n_trials_session,)`, `cate` labels.
- Output: split indices `(n_sessions - 1,)`.

### Statistics and derived measures

`auc_test(data1, data2)`

- Main function: estimate two-group separability with permutation p-value.
- Steps: drops NaNs; builds binary labels; computes ROC AUC; permutes labels 1000 times; returns direction-free AUC.
- Input: `data1 (n_1,)`, `data2 (n_2,)`.
- Output: `auc` scalar, `p` scalar.

`get_modulation_index_neu_seq(neu, neu_time, c_time, win_eval)`

- Main function: compute normalized pre/post response change.
- Steps: flattens trials/neurons; estimates low/high baseline reference values; computes pre and post response values; divides response change by reference span.
- Input: `neu (..., n_time)`, `win_eval` list with baseline/pre/post windows.
- Output: modulation index `(n_rows,)`.

`get_isi_bin_neu(neu_seq, stim_seq, camera_pupil, isi, bin_win, bin_num, mean_sem=True)`

- Main function: bin neural, stimulus, and pupil data by inter-stimulus interval.
- Steps: digitizes ISI per session; selects trials per bin; averages trials within session; concatenates sessions; computes neural mean/SEM; averages stimulus timing; stores pupil trials.
- Input: lists of `neu_seq (n_trials, n_neurons, n_time)`, `stim_seq (n_trials, n_stim, 2)`, `camera_pupil (n_trials, n_time_pupil)`, `isi (n_trials,)`.
- Output: list containing bins, centers, trial-level neural lists, binned neural arrays, mean traces, SEM traces, stimulus summaries, and pupil arrays.

`get_temporal_scaling_data(data, t_org, t_target)`

- Main function: stretch a 2D trace to a target time axis.
- Steps: maps target time to the original span; interpolates each row with `scipy.interpolate.interp1d`.
- Input: `data (n_rows, n_time_org)`, `t_org (n_time_org,)`, `t_target (n_time_target,)`.
- Output: `(n_rows, n_time_target)`.

`get_temporal_scaling_trial_multi_sess(neu_seq, stim_seq, neu_time, target_isi)`

- Main function: temporally scale pre/stim/post trial segments across sessions.
- Steps: computes target stimulus timing; finds target windows; finds trial-specific windows; scales each segment; concatenates scaled segments.
- Input: lists of `neu_seq (n_trials, n_neurons, n_time)` and `stim_seq (n_trials, n_stim, 2)`.
- Output: list of scaled arrays, each `(n_trials, n_neurons, n_scaled_time)`.

`get_stat_test(data_1, data_2, method)`

- Main function: run a selected two-sample or paired statistical test.
- Steps: drops NaNs; runs the requested SciPy test; converts p-value into `0..3` significance level using `[0.05, 0.0005, 0.000005]`.
- Input: two arrays; `method` one of `ttest_ind`, `mannwhitneyu`, `levene`, `wilcoxon`, `cramervonmises_2samp`, `ks_2samp`, `anderson_ksamp`, `permutation`.
- Output: `p` scalar, `r` significance level scalar.

`get_win_mag_quant_stat_test(neu_seq_1, neu_seq_2, neu_time, c_time, win_eval, method)`

- Main function: compare two conditions across evaluation windows.
- Steps: computes baseline and three response-window values for both conditions; optionally baseline-corrects; runs `get_stat_test` per response window.
- Input: `neu_seq_1 (..., n_time)`, `neu_seq_2 (..., n_time)`, `win_eval` four windows.
- Output: `p (3,)`, `r (3,)`.

`get_pair_corr(neu_seq_pair)`

- Main function: compute rank-based pairwise correlations through time.
- Steps: rank-transforms trials; centers and normalizes trial vectors; computes all cross-neuron dot products per time point.
- Input: pair/list of two arrays, each `(n_trials, n_neurons, n_time)`.
- Output: `(n_neuron_pairs, n_time)`.

`sub_sampling_neuron(neu_seq)`

- Main function: average random neuron subsets.
- Steps: samples 20 percent of neurons 50 times; averages each subset.
- Input: `neu_seq (n_neurons, n_time)`.
- Output: `(50, n_time)`.

`get_population_activity(aligned_list)`

- Main function: concatenate unit-by-time population activity across sessions.
- Steps: averages events for 3D inputs; trims all sessions to the shortest time axis; concatenates units.
- Input: list of `(n_events, n_units, n_time)` or `(n_units, n_time)`.
- Output: `(n_total_units, n_time_min)`.

`get_population_mean_sem(aligned_list)`

- Main function: summarize population activity across units.
- Steps: calls `get_population_activity`; computes mean and SEM; reports unit/time counts.
- Input: same as `get_population_activity`.
- Output: dictionary with `mean (n_time,)`, `sem (n_time,)`, `n_units`, `n_time`.

### Plot helpers

`require_rastermap()`

- Main function: check optional Rastermap availability.
- Steps: returns imported module or raises a clear `ImportError`.
- Input: none.
- Output: rastermap module.

`bin_rows(data, max_rows=258)`

- Main function: downsample heatmap rows by averaging adjacent row groups.
- Steps: computes bin size; trims to complete bins; reshapes and averages.
- Input: `data (n_rows, n_time)`.
- Output: `(min(n_rows, about max_rows), n_time)`.

`get_roi_label_color(labels=None, cate=None, roi_id=None)`

- Main function: choose a color set and colormap for ROI labels/categories.
- Steps: checks ROI label or category; returns neutral, light, dark colors and colormap.
- Input: optional labels `(n_rois,)`, category list, ROI index.
- Output: `color0`, `color1`, `color2`, `cmap`.

`get_cmap_color(n_colors, base_color=None, cmap=None, return_cmap=False)`

- Main function: sample colors from a colormap.
- Steps: builds a colormap from `base_color` if given; samples evenly between margins; converts to hex.
- Input: number of colors and optional colormap.
- Output: color list, or `(cmap, colors)` when `return_cmap=True`.

`sort_heatmap_neuron(neu_seq_sort, sort_method)`

- Main function: return row order for a heatmap.
- Steps: smooths each row; sorts by Rastermap, peak timing, trough timing, mean, shuffle, or original order.
- Input: `neu_seq_sort (n_rows, n_time)`.
- Output: sorted row indices `(n_rows,)`.

`sort_heatmap_rows(data, sort_method='rastermap')`

- Main function: return sorted heatmap data and indices.
- Steps: calls `sort_heatmap_neuron`; applies index order.
- Input: `data (n_rows, n_time)`.
- Output: `sorted_data (n_rows, n_time)`, `sorted_idx (n_rows,)`.

`apply_colormap(data, norm_mode, data_share=None)`

- Main function: normalize heatmap data and convert to RGB.
- Steps: handles empty/binary data; chooses local or shared percentile clipping; optionally row-normalizes; applies Matplotlib colormap.
- Input: `data (n_rows, n_time)`, optional shared scale data.
- Output: `hm_data (n_rows, n_time, 3)`, `hm_norm`, `hm_cmap`.

`hide_all_axis(ax)`

- Main function: remove ticks and spines.
- Steps: disables ticks; hides all spines; clears tick locations.
- Input: Matplotlib axis.
- Output: modifies axis in place.

`get_random_rotate_mat_3d()`

- Main function: generate a random 3D rotation matrix.
- Steps: samples a random quaternion; converts it to a `3 x 3` matrix.
- Input: none.
- Output: `(3, 3)` rotation matrix.

`add_ax_ticks(ax, axis, nbins)`

- Main function: format x or y ticks in seconds for millisecond axes.
- Steps: sets major/minor locators; formats alternating labels as `value/1000`.
- Input: Matplotlib axis, `axis='x'` or `'y'`.
- Output: modifies axis in place.

Layout helpers:

- `adjust_layout_isi_example_epoch(ax, trial_win, bin_win)`: formats an ISI epoch axis. Inputs are axis and scalar windows; output is in-place axis styling.
- `adjust_layout_neu(ax)`: formats neural trace axes. Input axis; output in-place styling.
- `adjust_layout_cluster_neu(ax, n_clusters, xlim)`: formats cluster trace axes. Input axis, cluster count, x limits; output in-place styling.
- `adjust_layout_scatter(ax, upper, lower)`: formats paired scatter axes. Input axis and bounds; output in-place styling.
- `adjust_layout_heatmap(ax)`: formats heatmap axis. Input axis; output in-place styling.
- `adjust_layout_3d_latent(ax)`: formats 3D latent trajectory axis. Input 3D axis; output in-place styling.
- `adjust_layout_pupil(ax)`: formats pupil trace axis. Input axis; output in-place styling.

`add_legend(ax, colors=None, labels=None, n_trials=None, n_neurons=None, n_sessions=None, loc='best', dim=2)`

- Main function: add color labels and optional sample counts.
- Steps: builds invisible handles; adds legend when there is at least one handle.
- Input: Matplotlib axis plus optional labels/counts.
- Output: modifies axis in place.

`add_heatmap_colorbar(ax, cmap, norm, label, yticklabels=None)`

- Main function: draw a compact heatmap colorbar.
- Steps: hides host axis; creates inset axis; draws scalar mappable; formats labels and tick locations.
- Input: Matplotlib axis, colormap, normalization, label.
- Output: modifies axis in place.

## `utils_basic` plotting class

`utils_basic` stores plotting defaults and wraps common figure patterns. Instantiate once with:

```python
plotter = utils_basic()
```

Methods:

- `plot_mean_sem(ax, t, m, s, c, l=None, a=1.0)`: plots `m (n_time,)` with shaded `s (n_time,)` against `t (n_time,)`.
- `plot_density(ax, data, xlim, color, bw_method=0.05)`: plots a Gaussian KDE for one vector.
- `plot_half_violin(ax, data, x, color, side)`: plots one side of a violin at scalar x.
- `plot_dist(ax, data, c, cumulative, xlim=None)`: plots histogram fractions or cumulative fractions; returns fractions `(n_bins,)` or `(n_bins+2,)`.
- `plot_scatter(ax, q1, q2, c)`: plots paired/scaled scatter vectors and significance labels.
- `plot_heatmap_neuron(ax_hm, ax_cb, neu_seq, neu_time, neu_seq_sort, ...)`: plots a sorted neuron heatmap from `(n_neurons, n_time)` data. Default sorting is `peak_timing`.
- `plot_heatmap_trial(ax_hm, ax_cb, neu_seq, neu_time, ...)`: plots a trial heatmap from `(n_trials, n_time)` data.
- `plot_win_mag_quant_win_eval(ax, win_eval, color, xlim, baseline=True)`: plots evaluation-window markers.
- `plot_win_mag_quant(ax, neu_seq, neu_time, win_eval, color, c_time, offset)`: plots baseline-subtracted window magnitudes from `neu_seq (..., n_time)`.
- `plot_multi_sess_decoding_slide_win(ax, eval_time, acc_model, acc_chance, color1, color2)`: plots model and chance decoding traces from lists of accuracy arrays.
- `plot_pred_mod_index_dist(ax, mi1, mi2, color0, color1, color2)`: plots modulation-index distributions and AUC label.
- `plot_3d_latent_dynamics(ax, neu_z, stim_seq, neu_time, ...)`: plots a 3D trajectory from `neu_z (3, n_time)`.
- `plot_dis_mat(ax_hm, ax_cb, d, annotate=False)`: plots the lower triangle of a square distance matrix `d (n_items, n_items)`.

All plotting methods modify Matplotlib axes in place. Save figures from the calling script.

## Minimal example

```python
import matplotlib.pyplot as plt
from utils import zscore_rows, align_events, get_mean_sem, utils_basic

data = zscore_rows(data)
aligned, aligned_time = align_events(data, time_ms, event_times, pre_s=1, post_s=2)
mean, sem = get_mean_sem(aligned)

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
plotter = utils_basic()
plotter.plot_mean_sem(ax, aligned_time, mean, sem, c='#1f77b4')
fig.savefig('summary_trace.png', dpi=300)
```
