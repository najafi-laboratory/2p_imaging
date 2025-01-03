
# Update note

## 2024.07.27
- First release.

## 2024.10.21
- Rewritten to incorporate multiple sessions
- Added function to show used sessions for alignment.
- Moved read_ops and reset_significance to ReadResults.
- Now data reading reads multiple sessions into lists.
- Now plot_significance and plot_roi_significance take list data as input.
- Added tqdm for alignment.
- Used np.searchsorted to improve efficiency for voltage alignment.
- Added get_multi_sess_neu_trial_average to compute trial average across sessions.
- Now sem is only across neuron in session report.
- Completed modifying plot_normal_box to cross session setting.
- Completed modifying plot_odd_normal_pre and plot_odd_normal_post to cross session setting.

## 2024.10.30
- Now Trialization and StatTest remove the existing h5 file before saving.
- Now all mean trace starts from 0.
- Completed modifying plot_normal_epoch to cross session setting.
- Completed modifying plot_normal_epoch_box to cross session setting.
- Now Utils for plotting does not need label to init.
- Completed modifying utils/plot_heatmap_neuron to cross session setting.

## 2024.10.31
- Now plot_heatmap_neuron sorts based on a given window.
- Completed modifying plot_context to cross session setting.
- Completed modifying plot_context_box to cross session setting.
- Separated normal and epoch.
- Now get_multi_sess_neu_trial_average also computes sem.
- Added idx_cut for both plot_odd_normal_pre and plot_odd_normal_post.
- Combined similar subplots into single functions for fig2_align_stim.
- Completed modifying plot_change_prepost to cross session setting.

## 2024.11.01
- Deleted plot_change_prepost_box.
- Completed modifying plot_odd_normal_prepost to cross session setting.

## 2024.11.02
- Now plot_mean_sem accepts alpha for transparency.
- Added plot_odd_post_box.
- Completed modifying all stimulus types plots to cross session setting.
- Now plot_stim_type gives the total number in the title.
- Replaced all np.argmin on timestamps with np.searchsorted.
- Completed modifying plot_odd_context_post to cross session setting.
- Changed all np.min and np.max to np.nanmin and np.nanmax.
- Completed modifying plot_odd_epoch_post to cross session setting without voltage traces.
- Completed modifying plot_odd_epoch_post_box to cross session setting.

## 2024.11.03
- Deleted plot_oddball_distribution.
- Added plot_pre_odd_isi_distribution.
- Now plot_inh_exc_label_pc gives the total number in the title.
- Now get_multi_sess_neu_trial_average can return single trial response for multi session setting.
- Now get_odd_stim_prepost_idx always excludes the last element in idx_pre.
- Replaced all np function with the nan version.
- Completed modifying plot_odd_post_isi to cross session setting.
- Completed modifying plot_odd_post_isi_box to cross session setting.
- Added timescale in case the session interval settings are different.
- Now plot_motion_offset_hist plots offsets across sessions.

## 2024.11.07
- Rewritten all figures.
- Now plot_normal receives normal types and plots response averaged across stimulus.
- Now plot_normal receives fix_jitter to separate response.
- Now neu_time is the mean of all sessions.
- Now stim_value is computed with interpolation.
- Now plot_heatmap_neuron excludes nan values.
- Now get_multi_sess_neu_trial_average also processes proceeding isi.
- Now get_epoch_idx only computes early and late epoch.

## 2024.11.11
- Adjusted heatmap positions.
- Now preceeding isi is devided only by the mean into 2 bins.
- Changed color coding for epochs.
- Added super init for plotter_utils.
- Now win_sort in plot_heatmap_neuron is an input parameter.
- Added \n to titles.
- Now get_stim_response uses np.searchsorted to find alignement points.
- Now get_stim_response can align traces on expected isi.
- Now get_stim_response returns a dict of results.

## 2024.11.12
- Added voltage correction for the beginning voltage into trialization.
- Added voltage correction for the last image as oddball into trialization.
- Completed plot_odd_normal_cluster.