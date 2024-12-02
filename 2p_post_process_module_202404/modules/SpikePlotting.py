import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import os


def plot_for_neuron_interactive(
    timings,
    dff,
    spikes,
    threshold_val,
    thresholded_spikes,
    below_thresh_sta,
    above_thresh_sta,
    neuron=5,
    tau=0.4,
    smoothed=False,
    smoothed_spikes=None
):
    """
    Generates an interactive HTML figure with multiple subplots for a specified neuron:
    - First row:
        - Columns 1 and 2 (spanning): Inferred spikes with threshold value plotted as a horizontal line.
        - Column 3: Below-threshold STA.
    - Second row:
        - Columns 1 and 2 (spanning): Thresholded inferred spikes.
        - Column 3: Above-threshold STA.
    - Third row:
        - Columns 1, 2, and 3 (spanning): DF/F + inferred spikes + threshold line.

    Args:
        timings (np.ndarray): Array of time points.
        dff (np.ndarray): DF/F (delta F over F) data array, shape (neurons, timepoints).
        spikes (np.ndarray): Inferred spike data array, shape (neurons, timepoints).
        threshold_val (float): Threshold value for spike detection for the specified neuron.
        thresholded_spikes (np.ndarray): Thresholded spikes data array, shape (neurons, timepoints).
        below_thresh_sta (np.ndarray): Spike-triggered average for below-threshold spikes.
        above_thresh_sta (np.ndarray): Spike-triggered average for above-threshold spikes.
        neuron (int, optional): Index of the neuron to plot. Default is 5.
        tau (float, optional): Tau parameter value used in analysis. Default is 0.4.
        smoothed (bool, optional): If True, plots the smoothed spikes. Default is False.
        smoothed_spikes (np.ndarray, optional): Smoothed spikes data array, shape (neurons, timepoints). Required if `smoothed` is True.

    Saves:
        An HTML file of the interactive plot in 'plot_results' directory with filename including neuron, tau, and threshold.
    """

    # Determine the shift due to potential different lengths between timings and spikes
    shift = len(timings) - len(spikes[neuron, :])
    dff_scale_factor = 0.3

    # Adjust the layout
    subplot_titles = (
        f'Inferred Spikes for Neuron {neuron} (Tau={tau}, Threshold={threshold_val})',
        'Below-threshold STA',
        'Thresholded Inferred Spikes',
        'Above-threshold STA',
        'Superimposed DF/F + Inferred Spikes + Threshold'
    )
    specs = [
        [{'colspan': 2}, None, {}],  # First row
        [{'colspan': 2}, None, {}],  # Second row
        [{'colspan': 2}, None, {}]   # Third row spans all columns
    ]

    # Create subplots using Plotly with the new layout
    fig = make_subplots(
        rows=3,
        cols=3,
        shared_xaxes=False,
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
        specs=specs,
        subplot_titles=subplot_titles
    )

    # First row, columns 1 and 2: Inferred spikes with threshold line
    fig.add_trace(
        go.Scatter(
            x=timings[shift:],
            y=spikes[neuron, :],
            mode='lines',
            name='Inferred Spikes',
            line=dict(color='orange')
        ),
        row=1,
        col=1
    )
    fig.add_hline(
        y=threshold_val,
        line_dash='dash',
        line_color='red',
        annotation_text='Threshold',
        annotation_position='top left',
        row=1,
        col=1
    )

    # First row, column 3: Below-threshold STA
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(below_thresh_sta)),
            y=below_thresh_sta,
            mode='lines',
            name='Below-threshold STA',
            line=dict(color='purple')
        ),
        row=1,
        col=3
    )

    # Second row, columns 1 and 2: Thresholded inferred spikes
    fig.add_trace(
        go.Scatter(
            x=timings[shift:],
            y=thresholded_spikes[neuron, :],
            mode='lines',
            name='Thresholded Inferred Spikes',
            line=dict(color='green')
        ),
        row=2,
        col=1
    )

    # Second row, column 3: Above-threshold STA
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(above_thresh_sta)),
            y=above_thresh_sta,
            mode='lines',
            name='Above-threshold STA',
            line=dict(color='brown')
        ),
        row=2,
        col=3
    )

    # Third row, spanning all columns: Superimposed DF/F + inferred spikes + threshold line
    fig.add_trace(
        go.Scatter(
            x=timings[shift:],
            y=(dff_scale_factor * dff[neuron, :]),
            mode='lines',
            name='DF/F Signal',
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
            name='Inferred Spikes',
            line=dict(color='orange')
        ),
        row=3,
        col=1
    )
    fig.add_hline(
        y=threshold_val,
        line_dash='dash',
        line_color='red',
        annotation_text='Threshold',
        annotation_position='top left',
        row=3,
        col=1
    )

    if smoothed:
        # Add smoothed spikes to the third row
        fig.add_trace(
            go.Scatter(
                x=timings[shift:],
                y=smoothed_spikes[neuron, :],
                mode='lines',
                name='Smoothed Spikes',
                line=dict(color='magenta')
            ),
            row=3,
            col=1
        )

    # Update layout and assign axis titles
    fig.update_layout(
        height=1000,
        width=1400,
        title_text=f'Neuron {neuron} Analysis with Tau={tau} and Threshold={threshold_val}',
        showlegend=True
    )

    fig.update_xaxes(title_text='Time (ms)', row=1, col=1)
    fig.update_yaxes(title_text='Signal', row=1, col=1)

    fig.update_xaxes(title_text='Samples', row=1, col=3)
    fig.update_yaxes(title_text='STA Value', row=1, col=3)

    fig.update_xaxes(title_text='Time (ms)', row=2, col=1)
    fig.update_yaxes(title_text='Signal', row=2, col=1)

    fig.update_xaxes(title_text='Samples', row=2, col=3)
    fig.update_yaxes(title_text='STA Value', row=2, col=3)

    fig.update_xaxes(title_text='Time (ms)', row=3, col=1)
    fig.update_yaxes(title_text='Superimposed Signal', row=3, col=1)

    # Save plot as HTML
    plot(
        fig,
        filename=f'plot_results/neuron_{neuron}_tau_{tau}_thresh_{threshold_val}_plot.html'
    )


def plot_for_neuron_with_smoothed_interactive(
        timings, dff, spikes, convolved_spikes, neuron=5, tau=1.25):
    """
    Generates an interactive HTML figure with two subplots for a specified neuron with smoothed data:
    - First subplot: Overlay of DF/F signal (scaled), inferred spike trace, and threshold.
    - Second subplot: Overlay of DF/F signal and convolved (smoothed) spikes.

    Args:
        timings (np.ndarray): Array of time points.
        dff (np.ndarray): DF/F (delta F over F) data array, shape (neurons, timepoints).
        spikes (np.ndarray): Inferred spike data array, shape (neurons, timepoints).
        convolved_spikes (np.ndarray): Convolved spikes data array, shape (neurons, timepoints).
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
            x=timings[shift:], y=dff[neuron, :],
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
            x=timings[shift:], y=convolved_spikes[neuron, :],
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
        filename=f'plot_results/neuron_{neuron}_tau_{tau}_smoothed_plot.html'
    )


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


def plot_stas_single_thresh_neuron(above_sta, below_sta, thresh_val, neuron, tau):
    """
    Plots the above- and below-threshold STAs on a single plot for a given threshold value.

    Args:
        above_sta (np.ndarray): Spike-triggered average for spikes above the threshold, shape (neurons, timepoints).
        below_sta (np.ndarray): Spike-triggered average for spikes below the threshold, shape (neurons, timepoints).
        thresh_val (float): Threshold value used for separating spikes.
        neuron (int): Index of the neuron to plot.

    Saves:
        A plot as an HTML file in the 'plot_results' directory with the filename including neuron and threshold value.
    """
    output_dir = 'plot_results'
    os.makedirs(output_dir, exist_ok=True)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(below_sta)),
            y=below_sta,
            mode='lines',
            name='Below-Threshold STA',
            line=dict(color='blue')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(above_sta)),
            y=above_sta,
            mode='lines',
            name='Above-Threshold STA',
            line=dict(color='orange')
        )
    )

    fig.add_hline(
        y=thresh_val,
        line_dash='dash',
        line_color='red',
        annotation_text=f'Threshold ({thresh_val})',
        annotation_position='top left')

    fig.update_layout(
        title=f'STAs for Neuron {neuron} at Threshold {thresh_val}',
        title_x=0.5,  # Center-align the title
        xaxis_title='Time Relative to Spike (samples)',
        yaxis_title='STA Amplitude',
        legend=dict(
            x=0.5,
            y=-0.2,
            xanchor='center',
            orientation='h',
            font=dict(size=10)
        ),
        height=600,
        width=800,
        margin=dict(t=80)
    )

    filename = f'{output_dir}/neuron_{neuron}_thresh_{thresh_val}_tau_{tau}_sta_plot.html'
    fig.write_html(filename)


def plot_stas_multi_thresh_neuron(thresh_vals, above_thresh_stas, below_thresh_stas, tau, neuron):
    """Plots above- and below-threshold STAs across multiple threshold values"""

    output_dir = 'plot_results'
    assert thresh_vals.shape[0] == len(above_thresh_stas) == len(below_thresh_stas), \
        "Length of thresh_vals, above_thresh_stas, and below_thresh_stas must be the same."

    # Create subplots for each threshold value
    num_thresholds = len(thresh_vals)
    fig = make_subplots(
        rows=num_thresholds,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.05,
        subplot_titles=[
            f'Threshold = {round(thresh, 2)}' for thresh in thresh_vals]
    )

    # Add traces for each threshold
    for i, (thresh, above_sta, below_sta) in enumerate(zip(thresh_vals, above_thresh_stas, below_thresh_stas)):
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(below_sta)),
                y=below_sta,
                mode='lines',
                name='Below-Threshold STA',
                line=dict(color='blue'),
                showlegend=(i == 0)  # Only show legend for the first subplot
            ),
            row=i + 1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(above_sta)),
                y=above_sta,
                mode='lines',
                name='Above-Threshold STA',
                line=dict(color='orange'),
                showlegend=(i == 0)  # Only show legend for the first subplot
            ),
            row=i + 1,
            col=1
        )

        fig.add_hline(
            y=thresh,
            line_dash='dash',
            line_color='red',
            annotation_text=f'Threshold ({round(thresh, 2)})',
            annotation_position='top left',
            row=i + 1,
            col=1
        )

    # Update layout
    fig.update_layout(
        title=f'STAs for Neuron {neuron} Across Multiple Thresholds (Tau={tau})',
        title_x=0.5,  # Center-align the title
        height=300 * num_thresholds,  # Adjust height based on the number of thresholds
        width=800,
        showlegend=True,
        legend=dict(
            x=0.5,
            y=-0.1,  # Place legend below the plot
            xanchor='center',
            orientation='h',
            font=dict(size=10)
        )
    )

    # Add axis labels for each subplot
    for i in range(num_thresholds):
        fig.update_xaxes(
            title_text='Time Relative to Spike (samples)', row=i + 1, col=1)
        fig.update_yaxes(title_text='STA Amplitude', row=i + 1, col=1)

    # Save plot as an HTML file
    filename = f'{output_dir}/neuron_{neuron}_tau_{tau}_multi_thresh_sta_plot.html'
    fig.write_html(filename)


def plot_baselined_filtered_data(
    timings,
    orig_dff,
    inferred_spikes,
    sta,
    tau,
    neuron=0,
    baselined_dff=None,
    filtered_dff=None,
    # if true, then the filtered dff has been baselined first
    filtered_after_baselined=False,
    lam='N/A',
    p='N/A'
):
    """
    Produces a figure with:
    - Left side (spanning columns 1 and 2):
        - Row 1: Raw DF/F, Baselined DF/F (if available), Filtered DF/F (if available), and Inferred Spikes (if available) superimposed.
        - Row 2: Baselined DF/F vs Raw DF/F (if Baselined DF/F is provided).
        - Row 3: Filtered DF/F vs Raw DF/F (if Filtered DF/F is provided).
    - Right side (column 3):
        - Row 1: Spike-Triggered Average (STA) for the specified neuron.

    Args:
        timings (np.ndarray): Array of time points.
        orig_dff (np.ndarray): Original DF/F data array, shape (neurons, timepoints).
        inferred_spikes (np.ndarray): Inferred spikes data array, shape (neurons, timepoints).
        sta (np.ndarray): Spike-triggered average data array, shape (neurons, timepoints).
        tau (float): Tau parameter used in the analysis.
        neuron (int, optional): Index of the neuron to plot. Default is 0.
        baselined_dff (np.ndarray, optional): Baselined DF/F data array, shape (neurons, timepoints). Default is None.
        filtered_dff (np.ndarray, optional): Filtered DF/F data array, shape (neurons, timepoints). Default is None.
        filtered_after_baselined (bool, optional): If True, indicates that filtering was done after baselining. Default is False.

    Saves:
        An HTML file of the interactive plot in 'plot_results' directory with filename including neuron.
    """

    output_dir = 'plot_results'
    os.makedirs(output_dir, exist_ok=True)

    num_rows = 1  # Start with 1 for the first row
    if baselined_dff is not None:
        num_rows += 1  # Add row for baselined DF/F vs raw DF/F
    if filtered_dff is not None:
        num_rows += 1  # Add row for filtered DF/F vs raw DF/F

    specs = []
    for i in range(num_rows):
        if i == 0:
            specs.append(
                [{"colspan": 2}, None, {"rowspan": num_rows}]
            )
        else:
            specs.append(
                [{"colspan": 2}, None, None]
            )

    subplot_titles = []

    title_row1_col1 = 'Raw DF/F'
    if baselined_dff is not None:
        title_row1_col1 += ', Baselined DF/F'
    if filtered_dff is not None:
        title_row1_col1 += ', Filtered DF/F'
    if inferred_spikes is not None:
        title_row1_col1 += ', Inferred Spikes'
    subplot_titles.append(title_row1_col1)

    subplot_titles.append('Spike-Triggered Average (STA)')

    if baselined_dff is not None:
        subplot_titles.append('Baselined DF/F vs Raw DF/F')

    if filtered_dff is not None:
        if filtered_after_baselined:
            title_row = 'Baselined, Filtered DF/F vs Raw DF/F'
        else:
            title_row = '(Without Baselining) Filtered DF/F vs Raw DF/F'
        subplot_titles.append(title_row)

    fig = make_subplots(
        rows=num_rows,
        cols=3,
        specs=specs,
        subplot_titles=subplot_titles,
        shared_xaxes=False
    )

    orig_dff_neuron = orig_dff[neuron, :]
    timings_neuron = timings

    legend_traces = set()

    showlegend_raw_dff = 'Raw DF/F' not in legend_traces
    fig.add_trace(
        go.Scatter(
            x=timings_neuron,
            y=orig_dff_neuron,
            mode='lines',
            name='Raw DF/F',
            line=dict(color='blue'),
            showlegend=showlegend_raw_dff
        ),
        row=1,
        col=1
    )
    if showlegend_raw_dff:
        legend_traces.add('Raw DF/F')

    if baselined_dff is not None:
        baselined_dff_neuron = baselined_dff[neuron, :]
        showlegend_baselined_dff = 'Baselined DF/F' not in legend_traces
        fig.add_trace(
            go.Scatter(
                x=timings_neuron,
                y=baselined_dff_neuron,
                mode='lines',
                name='Baselined DF/F',
                line=dict(color='orange'),
                showlegend=showlegend_baselined_dff
            ),
            row=1,
            col=1
        )
        if showlegend_baselined_dff:
            legend_traces.add('Baselined DF/F')

    if filtered_dff is not None:
        filtered_dff_neuron = filtered_dff[neuron, :]
        showlegend_filtered_dff = 'Filtered DF/F' not in legend_traces
        fig.add_trace(
            go.Scatter(
                x=timings_neuron,
                y=filtered_dff_neuron,
                mode='lines',
                name='Filtered DF/F',
                line=dict(color='yellow'),
                showlegend=showlegend_filtered_dff,
                opacity=0.8
            ),
            row=1,
            col=1
        )
        if showlegend_filtered_dff:
            legend_traces.add('Filtered DF/F')

    if inferred_spikes is not None:
        inferred_spikes_neuron = inferred_spikes[neuron, :]
        showlegend_inferred_spikes = 'Inferred Spikes' not in legend_traces
        fig.add_trace(
            go.Scatter(
                x=timings_neuron,
                y=inferred_spikes_neuron,
                mode='lines',
                name='Inferred Spikes',
                line=dict(color='red'),
                showlegend=showlegend_inferred_spikes
            ),
            row=1,
            col=1
        )
        if showlegend_inferred_spikes:
            legend_traces.add('Inferred Spikes')

    sta_neuron = sta[neuron, :]
    sta_length = len(sta_neuron)
    sta_time = np.arange(-sta_length // 2, sta_length // 2)

    showlegend_sta = 'STA' not in legend_traces
    fig.add_trace(
        go.Scatter(
            x=sta_time,
            y=sta_neuron,
            mode='lines',
            name='STA',
            line=dict(color='purple'),
            showlegend=showlegend_sta
        ),
        row=1,
        col=3
    )
    if showlegend_sta:
        legend_traces.add('STA')

    current_row = 2

    if baselined_dff is not None:
        # Plot Raw DF/F (without legend)
        fig.add_trace(
            go.Scatter(
                x=timings_neuron,
                y=orig_dff_neuron,
                mode='lines',
                name='Raw DF/F',
                line=dict(color='blue'),
                showlegend=False  # Already added
            ),
            row=current_row,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=timings_neuron,
                y=baselined_dff_neuron,
                mode='lines',
                name='Baselined DF/F',
                line=dict(color='orange'),
                showlegend=False  # Already added
            ),
            row=current_row,
            col=1
        )
        current_row += 1

    if filtered_dff is not None:
        # Plot Raw DF/F (without legend)
        fig.add_trace(
            go.Scatter(
                x=timings_neuron,
                y=orig_dff_neuron,
                mode='lines',
                name='Raw DF/F',
                line=dict(color='blue'),
                showlegend=False  # Already added
            ),
            row=current_row,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=timings_neuron,
                y=filtered_dff_neuron,
                mode='lines',
                name='Filtered DF/F',
                line=dict(color='yellow'),
                showlegend=False,  # Already added
                opacity=0.8
            ),
            row=current_row,
            col=1
        )
        current_row += 1

    # Update layout
    fig.update_layout(
        height=300 * num_rows,
        width=1200,
        title=f'Neuron {neuron} Analysis with Tau={tau}',
        showlegend=True
    )

    fig.update_xaxes(title_text='Time (ms)', row=1, col=1)
    fig.update_yaxes(title_text='Signal Amplitude', row=1, col=1)

    fig.update_xaxes(
        title_text='Time Relative to Spike (samples)', row=1, col=3)
    fig.update_yaxes(title_text='STA Amplitude', row=1, col=3)

    row_idx = 2
    if baselined_dff is not None:
        fig.update_xaxes(title_text='Time (ms)', row=row_idx, col=1)
        fig.update_yaxes(title_text='Signal Amplitude', row=row_idx, col=1)
        row_idx += 1

    if filtered_dff is not None:
        fig.update_xaxes(title_text='Time (ms)', row=row_idx, col=1)
        fig.update_yaxes(title_text='Signal Amplitude', row=row_idx, col=1)

    filename = ''
    if filtered_after_baselined:
        filename = f'{output_dir}/neuron_{neuron}_tau_{tau}_baselined_p_{p}_lam_{lam}_filtered_data.html'
    else:
        filename = f'{output_dir}/neuron_{neuron}_tau_{tau}_filtered_data.html'
    fig.write_html(filename)
