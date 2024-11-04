import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from suite2p.extraction.dcnv import oasis
import h5py
import os
from scipy.optimize import curve_fit
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def plot_for_neuron_without_smoothed(timings, dff, spikes, threshold_val, neuron=5, tau=1.25):
    """
    Produces a figure with three subplots: (top) de-convolved spike traces, (middle) original DFF,
                                           and (bottom) traces + original overlayed

    Args:
        timings (np.array): Array of time points.
        dff (np.array): DF/F data array.
        spikes (np.array): Spike detection data array.
        neuron (int): Index of the neuron to plot. Default is 5.
        num_deconvs (int): Number of deconvolutions performed. Default is 1.
    """
    plt.figure(figsize=(30, 10))
    fig, axs = plt.subplots(3, 1, figsize=(30, 10))
    fig.tight_layout(pad=10.0)
    shift = len(timings) - len(spikes[neuron, :])

    # first plot is just inferred spikes
    # axs[0].plot(timings[shift:], 0.5 * dff[neuron, :], label='DF/F', alpha=0.5)
    axs[0].plot(timings[shift:], spikes[neuron, :],
                label='Inferred Spike', color='orange')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Inferred Spikes')
    axs[0].set_title(
        f'Inferred Spikes -- Up-Time Plot for Neuron {neuron} with Tau={tau}')
    axs[0].legend()

    # second plot is just original dff
    axs[1].plot(timings[shift:], dff[neuron, :], label='Original DF/F')
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('DF/F')
    axs[1].set_title('DF/F -- Up-time Plot')
    axs[1].legend()

    # third plot shows the two overlayed
    axs[2].plot(timings[shift:], 0.5 * dff[neuron, :], label='DF/F', alpha=0.5)
    axs[2].plot(timings[shift:], spikes[neuron, :],
                label='Inferred Spike', color='orange')

    x = np.median(timings)
    y = np.median(spikes[neuron, :])
    # axs[2].set_xlim(14000, 14600)
    # axs[2].set_ylim(-2, 2)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Inferred Spikes')
    axs[2].set_title(f'Traces + Original')
    axs[2].legend()

    # plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(f'plot_results/neuron_{neuron}__tau_{tau}_plot.pdf')
    plt.show()


def plot_for_neuron_without_smoothed_interactive(timings, dff, spikes, threshold_val, neuron=5, tau=1.25):
    """
    Produces an interactive HTML figure with three subplots: (top) de-convolved spike traces, 
                                                            (middle) original DFF,
                                                            and (bottom) traces + original overlayed

    Args:
        timings (np.array): Array of time points.
        dff (np.array): DF/F data array.
        spikes (np.array): Spike detection data array.
        neuron (int): Index of the neuron to plot. Default is 5.
        tau (float): Tau parameter value. Default is 1.25.
    """
    # Determine the shift due to potential different lengths between timings and spikes
    shift = len(timings) - len(spikes[neuron, :])

    # Create subplots using Plotly
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'Inferred Spikes -- Up-Time Plot for Neuron {neuron} with Tau={tau}',
                                        'DF/F -- Up-time Plot',
                                        'Traces + Original'))
    # First plot: Inferred spikes
    fig.add_trace(go.Scatter(x=timings[shift:], y=spikes[neuron, :],
                             mode='lines',
                             name='Inferred Spike',
                             line=dict(color='orange')),
                  row=1, col=1)

    # Second plot: Original DF/F
    fig.add_trace(go.Scatter(x=timings[shift:], y=dff[neuron, :],
                             mode='lines',
                             name='Original DF/F',
                             line=dict(color='blue')),
                  row=2, col=1)

    # Third plot: DF/F and Inferred Spikes overlayed
    fig.add_trace(go.Scatter(x=timings[shift:], y=0.5 * dff[neuron, :],
                             mode='lines',
                             name='DF/F (scaled)',
                             line=dict(color='blue')),  # Removed opacity from line
                  row=3, col=1)

    fig.add_trace(go.Scatter(x=timings[shift:], y=spikes[neuron, :],
                             mode='lines',
                             name='Inferred Spike',
                             line=dict(color='orange')),
                  row=3, col=1)

    # Update layout
    fig.update_layout(height=900, width=1000, title_text=f'Neuron {neuron} Analysis with Tau={tau} and Threshold={threshold_val[neuron]}',
                      xaxis_title='Time (ms)', yaxis_title='Signal')

    # Save plot as HTML
    plot(
        fig, filename=f'neuron_{neuron}_tau_{tau}_thresh_{threshold_val[neuron]}plot.html')


def plot_for_neuron_with_smoothed_interactive(timings, dff, spikes, threshold_val, convolved_spikes, neuron=5, tau=1.25):
    """
    Produces an interactive HTML figure with two subplots: (top) de-convolved spike traces and original DF/F,
                                                           (bottom) DF/F and smoothed inferred spikes overlayed.

    Args:
        timings (np.array): Array of time points.
        dff (np.array): DF/F data array.
        spikes (np.array): Spike detection data array.
        convolved_spikes (np.array): Convolved spikes data array.
        neuron (int): Index of the neuron to plot. Default is 5.
        tau (float): Tau parameter value. Default is 1.25.
    """
    # Determine the shift due to potential different lengths between timings and spikes
    shift = len(timings) - len(spikes[neuron, :])

    # Create subplots using Plotly
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'Inferred Spikes -- Up-Time Plot for Neuron {neuron} with Tau={tau}',
                                        'DF/F & Smoothed Inferred Spikes -- Up-Time Plot'))
    # First plot: Inferred spikes and DF/F
    fig.add_trace(go.Scatter(x=timings[shift:], y=0.5 * dff[neuron, :],
                             mode='lines',
                             name='DF/F',
                             line=dict(color='blue')),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=timings[shift:], y=spikes[neuron, :],
                             mode='lines',
                             name='Inferred Spike',
                             line=dict(color='orange')),
                  row=1, col=1)

    # Second plot: DF/F and convolved spikes overlayed
    dff_mean = np.mean(dff[neuron, :])
    smooth_mean = np.mean(convolved_spikes[neuron, :])
    # scale = 1 / (4 * smooth_mean)

    fig.add_trace(go.Scatter(x=timings[shift:], y=dff[neuron, :],
                             mode='lines',
                             name='DF/F',
                             line=dict(color='blue')),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=timings[shift:], y=15 * tau * convolved_spikes[neuron, :],
                             mode='lines',
                             name='Convolved Spike',
                             line=dict(color='red', width=3.0)),
                  row=2, col=1)

    # Update layout
    fig.update_layout(height=800, width=1000, title_text=f'Neuron {neuron} Analysis with Tau={tau}',
                      xaxis_title='Time (ms)', yaxis_title='Signal')

    # Save plot as HTML
    plot(
        fig, filename=f'neuron_{neuron}_tau_{tau}_thresh_{threshold_val[neuron]}_smoothed_plot.html')


def plot_for_neuron_with_smoothed(timings, dff, spikes, convolved_spikes, neuron=5, tau=1.25):
    """
    Plots DF/F and deconvolved spike data for a specific neuron.

    Args:
        timings (np.array): Array of time points.
        dff (np.array): DF/F data array.
        spikes (np.array): Spike detection data array.
        neuron (int): Index of the neuron to plot. Default is 5.
        num_deconvs (int): Number of deconvolutions performed. Default is 1.
        convolved_spikes (np.array): Convolved spikes data array.
    """
    plt.figure(figsize=(30, 10))
    fig, axs = plt.subplots(2, 1, figsize=(30, 10))
    fig.tight_layout(pad=10.0)
    shift = len(timings) - len(spikes[neuron, :])
    axs[0].plot(timings[shift:], 0.5 * dff[neuron, :], label='DF/F', alpha=0.5)
    axs[0].plot(timings[shift:], spikes[neuron, :],
                label='Inferred Spike', color='orange')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Inferred Spikes')
    axs[0].set_title(
        f'Inferred Spikes -- Up-Time Plot for Neuron {neuron} with Tau={tau}')
    axs[0].legend()

    dff_mean = np.mean(dff[neuron, :])
    smooth_mean = np.mean(convolved_spikes[neuron, :])

    scale = 1 / (4 * smooth_mean)

    axs[1].plot(timings[shift:], dff[neuron, :], label='DF/F', alpha=0.5)
    axs[1].plot(timings[shift:], scale * convolved_spikes[neuron, :],
                label='Convolved Spike', color='red', lw=3)
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('DF/F')
    axs[1].set_title('DF/F & Smoothed Inferred Spikes -- Up-Time Plot')
    axs[1].legend()

    plt.rcParams['savefig.dpi'] = 1000
    plt.savefig(f'neuron_{neuron}__tau_{tau}_plot.pdf')
    plt.show()


def plot_for_neuron_with_smoothed_interactive_multi_tau(
    timings, dff, threshold_list, spikes_list, convolved_spikes_list, sta_list, neuron=5, tau_list=[1.25]
):
    """
    Produces an interactive HTML figure with subplots for each tau value.
    Each row corresponds to a tau value and contains two subplots:
    - Left column: Original DF/F, inferred spikes, and convolved (smoothed) spikes.
    - Right column: The spike-triggered average (STA) for the corresponding tau.

    The tau values are displayed as titles only on the left column subplots.

    Args:
        timings (np.array): Array of time points.
        dff (np.array): DF/F data array.
        threshold_val (np.array): Threshold values for neurons.
        spikes_list (list of np.array): List of spikes arrays for each tau.
            Each array should have shape (num_neurons, num_time_points).
        convolved_spikes_list (list of np.array): List of convolved spikes arrays for each tau.
            Each array should have shape (num_neurons, num_time_points).
        sta_list (list of np.array): List of STA arrays for each tau.
            Each array is a 1D array representing the STA for the neuron.
        neuron (int): Index of the neuron to plot. Default is 5.
        tau_list (list of float): List of tau parameter values.
    """
    # Ensure that spikes_list, convolved_spikes_list, sta_list, and tau_list have the same length
    assert len(spikes_list) == len(
        tau_list), "The number of spikes arrays must match the number of tau values"
    assert len(convolved_spikes_list) == len(
        tau_list), "The number of convolved_spikes arrays must match the number of tau values"
    assert len(sta_list) == len(
        tau_list), "The number of STA arrays must match the number of tau values"

    # Prepare figure with subplots
    num_taus = len(tau_list)
    titles = [item for item in tau_list for _ in range(2)]

    fig = make_subplots(
        rows=num_taus, cols=2, shared_xaxes=False, vertical_spacing=0.05, horizontal_spacing=0.05,
        # Set subplot titles only for the left column
        subplot_titles=[f'Tau = {tau}' for tau in titles],
        specs=[[{"type": "xy"}, {"type": "xy"}] for _ in range(num_taus)]
    )

    # Extract neuron data
    dff_neuron = dff[neuron, :]

    # For scaling, compute max values
    dff_max = np.max(dff_neuron)

    # Initialize variables to control legend display
    showlegend_main = True
    showlegend_sta = True

    # For each tau and its corresponding spikes, convolved spikes, and STA, plot
    for i, (tau, spikes, convolved_spikes, sta) in enumerate(zip(tau_list, spikes_list, convolved_spikes_list, sta_list)):
        spikes_neuron = spikes[neuron, :]
        convolved_spikes_neuron = convolved_spikes[neuron, :]

        # Determine the shift due to potential different lengths between timings and spikes
        shift = len(timings) - len(spikes_neuron)

        # Adjust timings and signals based on shift
        timings_plot = timings[shift:]
        dff_neuron_shifted = dff_neuron
        spikes_neuron_shifted = spikes_neuron
        convolved_spikes_neuron_shifted = convolved_spikes_neuron

        # Scale inferred spikes to have amplitude matching half of dff_max
        spikes_scale_factor = dff_max / 2
        spikes_scaled = spikes_neuron_shifted * spikes_scale_factor

        # Scale convolved spikes to match amplitude of dff
        convolved_max = np.max(convolved_spikes_neuron_shifted)
        if convolved_max == 0:
            convolved_scale_factor = 0
        else:
            convolved_scale_factor = dff_max / convolved_max
        convolved_scaled = convolved_spikes_neuron_shifted * convolved_scale_factor

        sta_neuron = sta[neuron, :]
        # Plot in subplot (row=i+1, col=1) - Main signals
        # Plot DF/F
        fig.add_trace(
            go.Scatter(
                x=timings_plot, y=dff_neuron_shifted,
                mode='lines', name='DF/F',
                line=dict(color='blue'),
                showlegend=showlegend_main
            ),
            row=i+1, col=1
        )

        # Plot inferred spikes (scaled)
        fig.add_trace(
            go.Scatter(
                x=timings_plot, y=spikes_scaled,
                mode='lines', name='Inferred Spikes',
                line=dict(color='orange'),
                showlegend=showlegend_main
            ),
            row=i+1, col=1
        )

        # Plot convolved spikes (scaled)
        fig.add_trace(
            go.Scatter(
                x=timings_plot, y=convolved_scaled,
                mode='lines', name='Convolved Spikes',
                line=dict(color='red'),
                showlegend=showlegend_main
            ),
            row=i+1, col=1
        )

        # Only show legend in the first main subplot
        showlegend_main = False

        # Plot STA in subplot (row=i+1, col=2)
        if sta_neuron is not None:
            # Create a time vector for STA
            sta_length = len(sta_neuron)
            sta_time = np.arange(-sta_length // 2, sta_length // 2)
            fig.add_trace(
                go.Scatter(
                    x=sta_time, y=sta_neuron,
                    mode='lines', name='STA',
                    line=dict(color='green'),
                    showlegend=showlegend_sta
                ),
                row=i+1, col=2
            )
            # Only show legend in the first STA subplot
            showlegend_sta = False

            # Update x-axis for STA subplot
            fig.update_xaxes(
                title_text='Time Relative to Spike', row=i+1, col=2)
            # Update y-axis for STA subplot
            fig.update_yaxes(title_text='STA Amplitude', row=i+1, col=2)
        else:
            # If STA is None, add a text annotation
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=f'x{i*2+2} domain', yref=f'y{i*2+2} domain',
                text='No STA Available',
                showarrow=False,
                row=i+1, col=2
            )

    # Update layout
    fig.update_layout(
        height=300 * num_taus, width=1200,
        title_text=f'Neuron {neuron} Analysis with Multiple Tau Values',
    )

    # Update x-axis and y-axis titles for main plots
    for i in range(num_taus):
        fig.update_xaxes(title_text='Time (ms)', row=i+1, col=1)
        fig.update_yaxes(title_text='Signal', row=i+1, col=1)

    # Adjust the subplot titles to be on the left of each row
    for i in range(num_taus):
        fig.add_annotation(
            dict(
                text=f"Tau = {tau_list[i]}",
                x=-0.08,  # Adjust as needed
                y=(num_taus - i - 0.5) / num_taus,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                font=dict(size=14)
            )
        )

    # Remove the default subplot titles
    fig.layout.annotations = [
        ann for ann in fig.layout.annotations if 'subplot' not in ann.text]

    # Save plot as HTML
    fig.write_html(
        f'neuron_{neuron}_tau_list_smoothed_plot.html'
    )

    # Alternatively, display the plot in a notebook or browser
    # fig.show()
