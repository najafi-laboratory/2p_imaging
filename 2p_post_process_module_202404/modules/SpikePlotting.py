import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots


def plot_for_neuron_without_smoothed(timings, dff, spikes, neuron=5, tau=1.25):
    """
    Generates a figure with three subplots for a specified neuron without smoothed data:
    - Top subplot: Inferred spike trace.
    - Middle subplot: Original DF/F signal.
    - Bottom subplot: Overlay of DF/F signal (scaled) and inferred spike trace.

    Args:
        timings (np.ndarray): Array of time points.
        dff (np.ndarray): DF/F (delta F over F) data array, shape (neurons, timepoints).
        spikes (np.ndarray): Inferred spike data array, shape (neurons, timepoints).
        neuron (int, optional): Index of the neuron to plot. Default is 5.
        tau (float, optional): Tau parameter value used in analysis. Default is 1.25.

    Saves:
        A PDF file of the plots in 'plot_results' directory with filename including neuron and tau.
    """
    # Calculate shift to align timings and spikes if their lengths differ
    shift = len(timings) - len(spikes[neuron, :])

    # Set up the figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(30, 10))
    fig.tight_layout(pad=10.0)

    # First subplot: Inferred spikes
    axs[0].plot(timings[shift:], spikes[neuron, :],
                label='Inferred Spike', color='orange')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Inferred Spikes')
    axs[0].set_title(f'Inferred Spikes for Neuron {neuron} (Tau={tau})')
    axs[0].legend()

    # Second subplot: Original DF/F
    axs[1].plot(timings[shift:], dff[neuron, :], label='Original DF/F')
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('DF/F')
    axs[1].set_title('DF/F Signal')
    axs[1].legend()

    # Third subplot: Overlay of DF/F (scaled) and Inferred Spikes
    axs[2].plot(timings[shift:], 0.5 * dff[neuron, :],
                label='DF/F (scaled)', alpha=0.5)
    axs[2].plot(timings[shift:], spikes[neuron, :],
                label='Inferred Spike', color='orange')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Signal')
    axs[2].set_title('Overlay of DF/F and Inferred Spikes')
    axs[2].legend()

    # Save the figure
    plt.savefig(f'plot_results/neuron_{neuron}_tau_{tau}_plot.pdf')
    plt.show()


def plot_for_neuron_without_smoothed_interactive(timings, dff, spikes, threshold_val, neuron=5, tau=1.25):
    """
    Generates an interactive HTML figure with three subplots for a specified neuron without smoothed data:
    - First subplot: Inferred spike trace.
    - Second subplot: Original DF/F signal.
    - Third subplot: Overlay of DF/F signal (scaled) and inferred spike trace.

    Args:
        timings (np.ndarray): Array of time points.
        dff (np.ndarray): DF/F (delta F over F) data array, shape (neurons, timepoints).
        spikes (np.ndarray): Inferred spike data array, shape (neurons, timepoints).
        threshold_val (np.ndarray): Threshold values for spike detection, one per neuron.
        neuron (int, optional): Index of the neuron to plot. Default is 5.
        tau (float, optional): Tau parameter value used in analysis. Default is 1.25.

    Saves:
        An HTML file of the interactive plot in 'plot_results' directory with filename including neuron, tau, and threshold.
    """
    # Determine the shift due to potential different lengths between timings and spikes
    shift = len(timings) - len(spikes[neuron, :])

    # Create subplots using Plotly
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f'Inferred Spikes for Neuron {neuron} with Tau={tau}',
            'DF/F Signal',
            'Overlay of DF/F and Inferred Spikes'
        )
    )

    # First subplot: Inferred spikes
    fig.add_trace(
        go.Scatter(
            x=timings[shift:],
            y=spikes[neuron, :],
            mode='lines',
            name='Inferred Spike',
            line=dict(color='orange')
        ),
        row=1,
        col=1
    )

    # Second subplot: Original DF/F
    fig.add_trace(
        go.Scatter(
            x=timings[shift:],
            y=dff[neuron, :],
            mode='lines',
            name='Original DF/F',
            line=dict(color='blue')
        ),
        row=2,
        col=1
    )

    # Third subplot: Overlay of DF/F (scaled) and Inferred Spikes
    fig.add_trace(
        go.Scatter(
            x=timings[shift:],
            y=0.5 * dff[neuron, :],
            mode='lines',
            name='DF/F (scaled)',
            line=dict(color='blue')
        ),
        row=3,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=timings[shift:],
            y=spikes[neuron, :],
            mode='lines',
            name='Inferred Spike',
            line=dict(color='orange')
        ),
        row=3,
        col=1
    )

    # Update layout
    fig.update_layout(
        height=900,
        width=1000,
        title_text=f'Neuron {neuron} Analysis with Tau={tau} and Threshold={threshold_val[neuron]}',
        xaxis_title='Time (ms)',
        yaxis_title='Signal'
    )

    # Save plot as HTML
    plot(
        fig,
        filename=f'plot_results/neuron_{neuron}_tau_{tau}_thresh_{threshold_val[neuron]}_plot.html'
    )


def plot_for_neuron_with_smoothed_interactive(
        timings, dff, spikes, threshold_val, convolved_spikes, neuron=5, tau=1.25):
    """
    Generates an interactive HTML figure with two subplots for a specified neuron with smoothed data:
    - First subplot: Overlay of DF/F signal (scaled), inferred spike trace, and threshold.
    - Second subplot: Overlay of DF/F signal and convolved (smoothed) spikes.

    Args:
        timings (np.ndarray): Array of time points.
        dff (np.ndarray): DF/F (delta F over F) data array, shape (neurons, timepoints).
        spikes (np.ndarray): Inferred spike data array, shape (neurons, timepoints).
        convolved_spikes (np.ndarray): Convolved spikes data array, shape (neurons, timepoints).
        threshold_val (np.ndarray): Threshold values for spike detection, one per neuron.
        neuron (int, optional): Index of the neuron to plot. Default is 5.
        tau (float, optional): Tau parameter value used in analysis. Default is 1.25.

    Saves:
        An HTML file of the interactive plot in 'plot_results' directory with filename including neuron, tau, and threshold.
    """
    # Determine the shift due to potential different lengths between timings and spikes
    shift = len(timings) - len(spikes[neuron, :])

    # Create subplots using Plotly
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f'Inferred Spikes for Neuron {neuron} with Tau={tau}',
            'DF/F & Smoothed Inferred Spikes'
        )
    )

    # First subplot: DF/F (scaled) and inferred spikes
    fig.add_trace(
        go.Scatter(
            x=timings[shift:], y=0.5 * dff[neuron, :],
            mode='lines',
            name='DF/F (scaled)',
            line=dict(color='blue')
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=timings[shift:], y=spikes[neuron, :],
            mode='lines',
            name='Inferred Spike',
            line=dict(color='orange')
        ),
        row=1,
        col=1
    )

    # Second subplot: DF/F and convolved spikes
    fig.add_trace(
        go.Scatter(
            x=timings[shift:], y=dff[neuron, :],
            mode='lines',
            name='DF/F',
            line=dict(color='blue')
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=timings[shift:], y=15 * tau * convolved_spikes[neuron, :],
            mode='lines',
            name='Convolved Spike',
            line=dict(color='red', width=3.0)
        ),
        row=2,
        col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text=f'Neuron {neuron} Analysis with Tau={tau}',
        xaxis_title='Time (ms)',
        yaxis_title='Signal'
    )

    # Save plot as HTML
    plot(
        fig,
        filename=f'plot_results/neuron_{neuron}_tau_{tau}_thresh_{threshold_val[neuron]}_smoothed_plot.html'
    )


def plot_for_neuron_with_smoothed(timings, dff, spikes, convolved_spikes, neuron=5, tau=1.25):
    """
    Generates a figure with two subplots for a specified neuron with smoothed data:
    - Top subplot: Overlay of DF/F signal (scaled) and inferred spike trace.
    - Bottom subplot: Overlay of DF/F signal and convolved (smoothed) spikes.

    Args:
        timings (np.ndarray): Array of time points.
        dff (np.ndarray): DF/F (delta F over F) data array, shape (neurons, timepoints).
        spikes (np.ndarray): Inferred spike data array, shape (neurons, timepoints).
        convolved_spikes (np.ndarray): Convolved spikes data array, shape (neurons, timepoints).
        neuron (int, optional): Index of the neuron to plot. Default is 5.
        tau (float, optional): Tau parameter value used in analysis. Default is 1.25.

    Saves:
        A PDF file of the plots in 'plot_results' directory with filename including neuron and tau.
    """
    # Calculate shift to align timings and spikes if their lengths differ
    shift = len(timings) - len(spikes[neuron, :])

    # Set up the figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(30, 10))
    fig.tight_layout(pad=10.0)

    # First subplot: DF/F (scaled) and inferred spikes
    axs[0].plot(timings[shift:], 0.5 * dff[neuron, :], label='DF/F', alpha=0.5)
    axs[0].plot(timings[shift:], spikes[neuron, :],
                label='Inferred Spike', color='orange')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Signal')
    axs[0].set_title(f'Inferred Spikes for Neuron {neuron} with Tau={tau}')
    axs[0].legend()

    # Calculate scaling factor for convolved spikes
    smooth_mean = np.mean(convolved_spikes[neuron, :])
    if smooth_mean != 0:
        scale = 1 / (4 * smooth_mean)
    else:
        scale = 0

    # Second subplot: DF/F and scaled convolved spikes
    axs[1].plot(timings[shift:], dff[neuron, :], label='DF/F', alpha=0.5)
    axs[1].plot(timings[shift:], scale * convolved_spikes[neuron, :],
                label='Convolved Spike', color='red', lw=3)
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('DF/F')
    axs[1].set_title('DF/F & Smoothed Inferred Spikes')
    axs[1].legend()

    # Save the figure
    plt.savefig(f'plot_results/neuron_{neuron}_tau_{tau}_plot.pdf', dpi=1000)
    plt.show()


def plot_for_neuron_with_smoothed_interactive_multi_tau(
    timings,
    dff,
    spikes_list,
    convolved_spikes_list,
    sta_list,
    threshold_val=[],
    neuron=5,
    tau_list=[1.25]
):
    """
    Generates an interactive HTML figure with subplots for multiple tau values for a specified neuron.
    Each row corresponds to a tau value and contains two subplots:
    - Left column: Overlay of DF/F signal, inferred spikes (scaled), convolved (smoothed) spikes (scaled), and threshold line.
    - Right column: Spike-Triggered Average (STA) for the corresponding tau value.

    Args:
        timings (np.ndarray): Array of time points.
        dff (np.ndarray): DF/F (delta F over F) data array, shape (neurons, timepoints).
        spikes_list (list of np.ndarray): List of inferred spike data arrays for each tau, each of shape (neurons, timepoints).
        convolved_spikes_list (list of np.ndarray): List of convolved spikes data arrays for each tau, each of shape (neurons, timepoints).
        sta_list (list of np.ndarray): List of STA arrays for each tau, each of shape (neurons, timepoints).
        threshold_val (float or list): Threshold value(s) for spike detection. Can be a scalar or a list matching tau_list.
        neuron (int, optional): Index of the neuron to plot. Default is 5.
        tau_list (list of float): List of tau parameter values used in analysis.

    Saves:
        An HTML file of the interactive plot in 'plot_results' directory with filename including neuron.
    """
    # Ensure that spikes_list, convolved_spikes_list, sta_list, and tau_list have the same length
    assert len(spikes_list) == len(
        tau_list), "The number of spikes arrays must match the number of tau values"
    assert len(convolved_spikes_list) == len(
        tau_list), "The number of convolved_spikes arrays must match the number of tau values"
    assert len(sta_list) == len(
        tau_list), "The number of STA arrays must match the number of tau values"

    # Ensure threshold_val is a list with the same length as tau_list
    if isinstance(threshold_val, (int, float)):
        threshold_vals = [threshold_val] * len(tau_list)
    elif isinstance(threshold_val, (list, np.ndarray)):
        assert len(threshold_val) == len(
            tau_list), "threshold_val must have same length as tau_list"
        threshold_vals = threshold_val
    else:
        raise ValueError("threshold_val must be a scalar or a list/array")

    # Prepare figure with subplots
    num_taus = len(tau_list)
    fig = make_subplots(
        rows=num_taus, cols=2, shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.05,
        specs=[[{"type": "xy"}, {"type": "xy"}] for _ in range(num_taus)]
    )

    # Extract neuron data
    dff_neuron = dff[neuron, :]
    timings_neuron = timings

    # For scaling, compute max values
    dff_max = np.max(dff_neuron)

    # Initialize variables to control legend display
    showlegend_main = True
    showlegend_sta = True

    # For each tau and its corresponding spikes, convolved spikes, and STA, plot
    for i, (tau, spikes, convolved_spikes, sta, threshold) in enumerate(zip(
            tau_list, spikes_list, convolved_spikes_list, sta_list, threshold_vals)):
        spikes_neuron = spikes[neuron, :]
        convolved_spikes_neuron = convolved_spikes[neuron, :]

        # Determine the shift due to potential different lengths between timings and spikes
        shift = len(timings_neuron) - len(spikes_neuron)

        # Adjust timings and signals based on shift
        timings_plot = timings_neuron[shift:]
        dff_neuron_shifted = dff_neuron[shift:]
        spikes_neuron_shifted = spikes_neuron
        convolved_spikes_neuron_shifted = convolved_spikes_neuron

        # Scale inferred spikes to have amplitude matching half of dff_max
        spikes_scale_factor = dff_max / 2
        spikes_scaled = spikes_neuron_shifted * spikes_scale_factor

        # STA for the neuron
        sta_neuron = sta[neuron, :] if sta is not None else None

        # Plot in subplot (row=i+1, col=1) - Main signals
        # Plot DF/F
        fig.add_trace(
            go.Scatter(
                x=timings_plot, y=dff_neuron_shifted,
                mode='lines', name='DF/F',
                line=dict(color='blue'),
                showlegend=showlegend_main,
                opacity=0.8
            ),
            row=i+1, col=1
        )

        # Plot inferred spikes (scaled)
        fig.add_trace(
            go.Scatter(
                x=timings_plot, y=spikes_scaled,
                mode='lines', name='Inferred Spikes',
                line=dict(color='orange'),
                showlegend=showlegend_main,
                opacity=0.8
            ),
            row=i+1, col=1
        )

        try:
            fig.add_hline(y=threshold[neuron][0], line_dash='dash',
                          line_color='red', row=i+1, col=1)

        except:
            fig.add_hline(y=threshold, line_dash='dash',
                          line_color='red', row=i+1, col=1)

        # fig.add_hline(y=threshold, line=dict(color='purple',
        #               dash='dash'), annotation='Spike Threshold')

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
        f'plot_results/neuron_{neuron}_tau_list_smoothed_plot.html'
    )
