import numpy as np
import pandas as pd
# Ensure this is the correct import for your SpikeDeconv module
from modules import SpikeDeconv
from joblib import Parallel, delayed

def corrected_sample_var(std, n):
    """
    Apply Bessel's correction to the sample variance.
    """
    return np.square(std) * (n / (n - 1))


def pooled_variance(neuron_data, n):
    """
    Compute the pooled variance for a single tau across neurons.

    Args:
        neuron_data (pd.Series): Series containing standard deviations for neurons of a specific tau.
        n (int): Sample size per neuron.

    Returns:
        float: The pooled variance across neurons for this tau.
    """
    numerator = ((neuron_data ** 2) * (n - 1)).sum()
    denominator = len(neuron_data) * (n - 1)
    return numerator / denominator if denominator != 0 else 0


def aggregate_statistics(df, n):
    """
    Averages the statistical properties across all neurons for each tau.

    Args:
        df (pd.DataFrame): DataFrame containing neuron statistics for each tau.
        n (int): Sample size per neuron.

    Returns:
        pd.DataFrame: A DataFrame containing aggregated statistics per tau.
    """
    agg_results = []

    for tau, group in df.groupby('tau'):
        agg_stats = {
            'tau': tau,
            'agg_mean': group['mean'].mean(),
            'agg_max': group['max'].mean(),
            'agg_min': group['min'].mean(),
            'agg_med': group['median'].mean(),
            'agg_total_spike': group['total_spikes'].mean(),
            # Compute pooled standard deviation across neurons for this tau
            'pooled_std': np.sqrt(pooled_variance(group['std'], n))
        }
        agg_results.append(agg_stats)

    return pd.DataFrame(agg_results)


def compute_statistics(spikes):
    """
    Compute statistical properties of the spike traces.

    Args:
        spikes (np.array): Array of spike traces.

    Returns:
        dict: A dictionary containing statistical properties.
    """
    stats = {
        'mean': np.mean(spikes, axis=1),
        'std': np.std(spikes, axis=1),
        'max': np.max(spikes, axis=1),
        'min': np.min(spikes, axis=1),
        'median': np.median(spikes, axis=1),
        'total_spikes': np.sum(spikes, axis=1)
    }
    return stats


def compute_sta(spikes_neuron, dff_neuron, pre_spike_window=200, post_spike_window=600):
    """
    Compute the spike-triggered averages (STAs) for a single neuron on both the DF/F data and the inferred spike traces,
    including only spikes where the quadratic variation around the spike is greater than the average quadratic variation.

    Args:
        spikes_neuron (np.array): Spike train of the neuron.
        dff_neuron (np.array): DF/F data of the neuron.
        pre_spike_window (int): Number of time points before the spike.
        post_spike_window (int): Number of time points after the spike.

    Returns:
        dict or None: A dictionary containing the STAs for both DF/F and spikes, or None if STA cannot be computed.
    """
    spike_indices = np.where(spikes_neuron > 0)[0]
    dff_windows = []

    # Collect windows around each spike
    for spike_time in spike_indices:
        # Ensure indices are within bounds
        if (spike_time - pre_spike_window >= 0) and (spike_time + post_spike_window < len(dff_neuron)):
            dff_window = dff_neuron[spike_time - pre_spike_window: spike_time + post_spike_window]
            dff_windows.append(dff_window)

    # Check if any windows were collected
    if len(dff_windows) > 0:
        # Compute the STA by averaging the windows
        sta_dff = np.mean(dff_windows, axis=0)
        return sta_dff
    else:
        return None  # No windows collected, cannot compute STA


def compute_sta_parallel(spikes, dff, pre_spike_window=200, post_spike_window=600, n_jobs=-1):
    """
    Compute the spike-triggered averages (STAs) for all neurons in parallel.

    Args:
        spikes (np.array): Spike trains of shape (b, N, M) or (N, M).
        dff (np.array): DF/F data of the same shape as spikes.
        pre_spike_window (int): Time points before the spike.
        post_spike_window (int): Time points after the spike.
        n_jobs (int): Number of parallel jobs to run (-1 uses all processors).

    Returns:
        result_dict (dict): A dictionary containing:
            - 'sta_dff': np.array of shape (N, sta_length) or (b, N, sta_length)
            - 'sta_spikes': np.array of the same shape
            - 'valid': np.array of booleans indicating whether STA was computed
    """
    original_shape = spikes.shape
    if spikes.ndim == 2:
        spikes = np.expand_dims(spikes, axis=0)  # Now shape is (1, N, M)
        dff = np.expand_dims(dff, axis=0)        # Now shape is (1, N, M)
    elif spikes.ndim != 3:
        raise ValueError("Input arrays must be 2D or 3D.")

    b_size, N_neurons, M_length = spikes.shape
    sta_length = pre_spike_window + post_spike_window

    sta_dff_array = np.full((b_size, N_neurons, sta_length), np.nan)
    sta_spikes_array = np.full((b_size, N_neurons, sta_length), np.nan)
    valid_array = np.zeros((b_size, N_neurons), dtype=bool)

    indices = [(b_idx, n_idx) for b_idx in range(b_size) for n_idx in range(N_neurons)]

    def compute_sta_wrapper(idx):
        b_idx, n_idx = idx
        spikes_neuron = spikes[b_idx, n_idx, :]
        dff_neuron = dff[b_idx, n_idx, :]
        return compute_sta(spikes_neuron, dff_neuron, pre_spike_window, post_spike_window)

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_sta_wrapper)(idx) for idx in indices
    )

    for idx, res in zip(indices, results):
        b_idx, n_idx = idx
        if res is not None:
            sta_dff_array[b_idx, n_idx, :] = res
            # an array indicating if the STA for a particular neuron came back as None 
            valid_array[b_idx, n_idx] = True 

    if len(original_shape) == 2:
        sta_dff_array = sta_dff_array[0]      # Now shape is (N, sta_length)
        valid_array = valid_array[0]

    return {
        'sta_dff': sta_dff_array,
        'valid': valid_array}


def analyze_spike_traces(ops, dff, tau_spike_dict, neurons=np.arange(100), n=333):
    """
    Analyze spike traces for different tau values and save results to a CSV file.

    Args:
        ops (dict): Operation parameters.
        dff (np.array): DF/F data array.
        tau_spike_dict (dict): Dictionary mapping tau values to their corresponding spike arrays.
        neurons (list): List of neuron indices to analyze.
        n (int): Sample size for all neurons.
    """
    results = []

    # Define the window size for STA computation
    pre_spike_window = 100   # Number of time points before the spike
    post_spike_window = 300  # Number of time points after the spike

    spike_stas = []
    dff_stas = []
    for tau in tau_spike_dict.keys():
        print(f'Analyzing spikes for tau = {tau}')
        # Get the spikes for the current tau
        spikes = tau_spike_dict[tau]
        neuron_spike_stas = []
        neuron_dff_stas = []
        # Compute statistics for the spikes
        stats = compute_statistics(spikes)
        # Store results with corresponding tau
        for i, neuron in enumerate(neurons):
            result = {
                'tau': tau,
                'neuron': neuron,
                'mean': stats['mean'][neuron],
                'std': stats['std'][neuron],
                'max': stats['max'][neuron],
                'min': stats['min'][neuron],
                'median': stats['median'][neuron],
                'total_spikes': stats['total_spikes'][neuron]
            }

            # Compute the spike-triggered averages (STAs)
            spikes_neuron = spikes[neuron, :]
            dff_neuron = dff[neuron, :]

            sta_result = compute_sta(
                spikes_neuron, dff_neuron, pre_spike_window, post_spike_window)

            if sta_result is not None:
                # Convert to list for serialization
                result['sta_dff'] = sta_result['sta_dff'].tolist()
                result['sta_spikes'] = sta_result['sta_spikes'].tolist()
            else:
                # No spikes, so STAs cannot be computed
                result['sta_dff'] = None
                result['sta_spikes'] = None

            results.append(result)

            neuron_spike_stas.append(sta_result['sta_spikes'])
            neuron_dff_stas.append(sta_result['sta_dff'])
        spike_stas.append(neuron_spike_stas)
        dff_stas.append(neuron_dff_stas)

    # Convert results to DataFrame
    df_n = pd.DataFrame(results)
    df_agg = aggregate_statistics(df_n, n)

    # Save per-neuron results to CSV
    df_n.to_csv('spike_analysis_results.csv', index=False)

    # Save aggregated results across neurons for each tau to a CSV
    df_agg.to_csv('aggregated_analysis_results.csv', index=False)
    print('Spike analysis results saved to spike_analysis_results.csv')

    return df_n, np.array(spike_stas), np.array(dff_stas)
