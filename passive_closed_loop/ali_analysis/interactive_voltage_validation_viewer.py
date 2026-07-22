"""
Create an interactive raw-voltage validation viewer.

The viewer is designed for closed-loop two-photon voltage validation. It reads
the raw Prairie ``*VoltageRecording*.csv`` file, plots all physical voltage
inputs, and overlays the timing events used by the alignment pipeline:

* Bonsai logger analysis-trial starts and ends from the trial-level event table.
* Raw CSV Input 3 scope-exposure rising edges used as the imaging-frame clock.
* The nearest dF/F frame time assigned to each analysis trial.
* Raw CSV Input 1 photodiode threshold crossings, for troubleshooting.

The raw voltage CSV is large, so the full-session view uses a min/max envelope
per time bin. To inspect raw-like detail, call the same function with
``time_start_sec`` and ``time_end_sec`` around a short window.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


VOLTAGE_INPUTS = {
    " Input 0": "Trial sync / EBC",
    " Input 1": "Photodiode",
    " Input 2": "HiFi TTL",
    " Input 3": "Scope exposure / 2P frame TTL",
    " Input 4": "HiFi audio waveform",
    " Input 5": "FLIR output / camera strobe",
    " Input 6": "Encoder phase A",
    " Input 7": "Encoder phase B",
}

EVENT_TABLE_NAME = "block2_performed_events_with_dff_frames.csv"
ANALYSIS_TRIAL_TABLE_NAME = "block2_analysis_trials_with_dff_frames.csv"
DEFAULT_ANALYSIS_TABLE = f"alignment_validation/{ANALYSIS_TRIAL_TABLE_NAME}"
DEFAULT_EVENT_TABLE = f"alignment_validation/{EVENT_TABLE_NAME}"
DEFAULT_PHYSICAL_HEALTH = "alignment_validation/physical_voltage_input_health.csv"
DEFAULT_OUTPUT = "interactive_voltage_validation_viewer.html"
PRIMARY_SCOPE_INPUT = " Input 3"
PHOTODIODE_INPUT = " Input 1"


@dataclass(frozen=True)
class VoltageViewerResult:
    """Output paths and compact validation metadata."""

    html_path: Path
    voltage_csv_path: Path
    event_table_path: Path
    n_bins: int
    time_start_sec: float
    time_end_sec: float
    summary: pd.DataFrame


def find_voltage_csv(data_dir: Path) -> Path:
    """Find the Prairie raw voltage CSV in a session directory."""
    matches = sorted(data_dir.glob("*VoltageRecording*.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find *VoltageRecording*.csv in {data_dir}")
    return matches[0]


def resolve_event_table(data_dir: Path, output_path: Path, event_table_path: str | Path | None) -> Path:
    """Resolve the trial-level analysis table, falling back to performed events."""
    if event_table_path is not None:
        path = Path(event_table_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing event table: {path}")
        return path

    candidates = [
        output_path.parent / DEFAULT_ANALYSIS_TABLE,
        output_path.parent / ANALYSIS_TRIAL_TABLE_NAME,
        output_path.parent / DEFAULT_EVENT_TABLE,
        output_path.parent / EVENT_TABLE_NAME,
        data_dir / DEFAULT_ANALYSIS_TABLE,
        data_dir / ANALYSIS_TRIAL_TABLE_NAME,
        data_dir / DEFAULT_EVENT_TABLE,
        data_dir / EVENT_TABLE_NAME,
        Path(
            "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/"
            "VW01_20260520_Closed_Loop_test-1556/alignment_validation/"
            "block2_analysis_trials_with_dff_frames.csv"
        ),
        Path(
            "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/"
            "VW01_20260520_Closed_Loop_test-1556/alignment_validation/"
            "block2_performed_events_with_dff_frames.csv"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find block2_analysis_trials_with_dff_frames.csv or block2_performed_events_with_dff_frames.csv. "
        "Run ali_analysis/event_alignment_validation.py first or pass event_table_path."
    )


def load_thresholds(data_dir: Path, output_path: Path, physical_health_path: str | Path | None) -> dict[str, float]:
    """Load adaptive raw voltage thresholds from validation output when present."""
    if physical_health_path is not None:
        candidates = [Path(physical_health_path).expanduser().resolve()]
    else:
        candidates = [
            output_path.parent / DEFAULT_PHYSICAL_HEALTH,
            output_path.parent / "physical_voltage_input_health.csv",
            data_dir / DEFAULT_PHYSICAL_HEALTH,
            data_dir / "physical_voltage_input_health.csv",
            Path(
                "/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/"
                "VW01_20260520_Closed_Loop_test-1556/alignment_validation/"
                "physical_voltage_input_health.csv"
            ),
        ]

    for candidate in candidates:
        if candidate.exists():
            health = pd.read_csv(candidate)
            if {"input", "threshold"}.issubset(health.columns):
                return {
                    f" {row.input}": float(row.threshold)
                    for row in health.itertuples(index=False)
                    if pd.notna(row.threshold)
                }

    # Conservative fallbacks for sessions without a validation table.
    return {
        " Input 0": 2.5,
        " Input 1": 2.5,
        " Input 2": 0.5,
        " Input 3": 2.5,
        " Input 5": 1.5,
        " Input 6": 2.5,
        " Input 7": 2.5,
    }


def load_alignment_events(event_table_path: Path) -> pd.DataFrame:
    """Load event times used for alignment from the validated event table."""
    cols = [
        "Analysis_Trial",
        "Analysis_Trial_In_Condition",
        "Analysis_Condition",
        "Id",
        "Trial_Number",
        "Trial_Type",
        "Stim_Start_sec",
        "Stim_End_sec",
        "Nearest_Dff_Frame",
        "Nearest_Dff_Time_sec",
        "Stim_To_Dff_Frame_Delta_sec",
        "Imaging_Frame_Source",
    ]
    events = pd.read_csv(event_table_path, usecols=lambda col: col in cols)
    for col in ["Stim_Start_sec", "Stim_End_sec", "Nearest_Dff_Time_sec"]:
        if col in events:
            events[col] = pd.to_numeric(events[col], errors="coerce")
    return events


def read_voltage_time_bounds(voltage_csv: Path) -> tuple[float, float]:
    """Read first and last raw voltage timestamps in seconds."""
    first = pd.read_csv(voltage_csv, usecols=["Time(ms)"], nrows=1)["Time(ms)"].iloc[0] / 1000.0
    last = None
    for chunk in pd.read_csv(voltage_csv, usecols=["Time(ms)"], chunksize=1_000_000):
        last = float(chunk["Time(ms)"].iloc[-1]) / 1000.0
    if last is None:
        raise ValueError(f"No rows found in {voltage_csv}")
    return float(first), float(last)


def build_envelope_arrays(
    voltage_csv: Path,
    *,
    thresholds: dict[str, float],
    time_start_sec: float,
    time_end_sec: float,
    max_bins: int,
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], np.ndarray]:
    """Downsample raw voltage traces and collect selected edge times."""
    if time_end_sec <= time_start_sec:
        raise ValueError("time_end_sec must be greater than time_start_sec")

    duration = time_end_sec - time_start_sec
    n_bins = int(max(10, min(max_bins, np.ceil(duration * 5000))))
    bin_width = duration / n_bins
    centers = time_start_sec + (np.arange(n_bins) + 0.5) * bin_width
    usecols = ["Time(ms)"] + list(VOLTAGE_INPUTS.keys())

    mins = {col: np.full(n_bins, np.inf, dtype=np.float32) for col in VOLTAGE_INPUTS}
    maxs = {col: np.full(n_bins, -np.inf, dtype=np.float32) for col in VOLTAGE_INPUTS}
    sums = {col: np.zeros(n_bins, dtype=np.float64) for col in VOLTAGE_INPUTS}
    counts = np.zeros(n_bins, dtype=np.int64)
    edge_times = {
        PRIMARY_SCOPE_INPUT: {"rising": [], "falling": []},
        PHOTODIODE_INPUT: {"rising": [], "falling": []},
    }
    prev_digital: dict[str, Optional[bool]] = {col: None for col in edge_times}
    rows_read = 0

    for chunk in pd.read_csv(voltage_csv, usecols=usecols, chunksize=chunksize):
        t_sec_all = chunk["Time(ms)"].to_numpy(dtype=np.float64) / 1000.0
        keep = (t_sec_all >= time_start_sec) & (t_sec_all <= time_end_sec)
        if not keep.any():
            if t_sec_all[0] > time_end_sec:
                break
            continue

        t_sec = t_sec_all[keep]
        bin_idx = np.floor((t_sec - time_start_sec) / bin_width).astype(np.int64)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        np.add.at(counts, bin_idx, 1)
        rows_read += len(t_sec)

        for col in VOLTAGE_INPUTS:
            x = chunk[col].to_numpy(dtype=np.float32)[keep]
            np.minimum.at(mins[col], bin_idx, x)
            np.maximum.at(maxs[col], bin_idx, x)
            np.add.at(sums[col], bin_idx, x.astype(np.float64))

        for col in edge_times:
            threshold = thresholds.get(col)
            if threshold is None or not np.isfinite(threshold):
                continue
            x = chunk[col].to_numpy(dtype=np.float32)[keep]
            digital = x > threshold
            if digital.size == 0:
                continue
            if prev_digital[col] is not None:
                if (not prev_digital[col]) and bool(digital[0]):
                    edge_times[col]["rising"].append(np.array([float(t_sec[0])], dtype=np.float64))
                if prev_digital[col] and (not bool(digital[0])):
                    edge_times[col]["falling"].append(np.array([float(t_sec[0])], dtype=np.float64))
            rising = np.flatnonzero((~digital[:-1]) & digital[1:]) + 1
            falling = np.flatnonzero(digital[:-1] & (~digital[1:])) + 1
            if len(rising):
                edge_times[col]["rising"].append(t_sec[rising].astype(np.float64, copy=True))
            if len(falling):
                edge_times[col]["falling"].append(t_sec[falling].astype(np.float64, copy=True))
            prev_digital[col] = bool(digital[-1])

    if rows_read == 0:
        raise ValueError("No voltage rows were found inside the requested time window.")

    for col in VOLTAGE_INPUTS:
        empty = counts == 0
        mins[col][empty] = np.nan
        maxs[col][empty] = np.nan

    for col in edge_times:
        for edge_name in ["rising", "falling"]:
            parts = edge_times[col][edge_name]
            edge_times[col][edge_name] = np.concatenate(parts) if parts else np.array([], dtype=np.float64)

    means = {
        col: np.divide(sums[col], counts, out=np.full(n_bins, np.nan, dtype=np.float64), where=counts > 0)
        for col in VOLTAGE_INPUTS
    }
    envelope = pd.DataFrame({"time_sec": centers, "count": counts})
    for col in VOLTAGE_INPUTS:
        label = col.strip().replace(" ", "_")
        envelope[f"{label}_min"] = mins[col]
        envelope[f"{label}_max"] = maxs[col]
        envelope[f"{label}_mean"] = means[col]

    return envelope, edge_times, {"mins": mins, "maxs": maxs, "means": means}, centers


def envelope_trace_arrays(time: np.ndarray, ymin: np.ndarray, ymax: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build vertical min/max envelope segments for Plotly."""
    valid = np.isfinite(ymin) & np.isfinite(ymax)
    x = np.empty(valid.sum() * 3, dtype=np.float64)
    y = np.empty(valid.sum() * 3, dtype=np.float32)
    x[0::3] = time[valid]
    x[1::3] = time[valid]
    x[2::3] = np.nan
    y[0::3] = ymin[valid]
    y[1::3] = ymax[valid]
    y[2::3] = np.nan
    return x, y


def event_marker_trace(
    *,
    name: str,
    times: np.ndarray,
    y_value: float,
    color: str,
    marker_size: int = 9,
    opacity: float = 0.8,
) -> dict:
    """Create an event-raster marker trace."""
    return {
        "type": "scattergl",
        "x": times,
        "y": np.full(len(times), y_value, dtype=float),
        "mode": "markers",
        "marker": {"symbol": "line-ns-open", "size": marker_size, "color": color, "opacity": opacity},
        "name": name,
        "hovertemplate": f"{name}<br>time=%{{x:.6f}} s<extra></extra>",
    }


def filter_times(times: pd.Series | np.ndarray, start: float, end: float) -> np.ndarray:
    """Return finite event times inside a window."""
    arr = np.asarray(times, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr[(arr >= start) & (arr <= end)]


def build_summary(
    *,
    events: pd.DataFrame,
    edge_times: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
    time_start_sec: float,
    time_end_sec: float,
    n_bins: int,
) -> pd.DataFrame:
    """Create a compact summary table for the HTML report."""
    is_analysis_table = "Analysis_Condition" in events.columns
    event_prefix = "analysis_trial" if is_analysis_table else "performed_stim"
    event_label = "trial-level analysis table" if is_analysis_table else "performed stimulus table"
    rows = [
        {
            "item": "viewer_time_window_sec",
            "value": f"{time_start_sec:.3f} to {time_end_sec:.3f}",
        },
        {"item": "event_table_type", "value": event_label},
        {"item": "downsample_bins", "value": f"{n_bins:,}"},
        {"item": f"{event_prefix}_starts_in_window", "value": f"{len(filter_times(events['Stim_Start_sec'], time_start_sec, time_end_sec)):,}"},
        {"item": f"{event_prefix}_ends_in_window", "value": f"{len(filter_times(events['Stim_End_sec'], time_start_sec, time_end_sec)):,}"},
        {
            "item": f"nearest_dff_frame_times_per_{event_prefix}_in_window",
            "value": f"{len(filter_times(events['Nearest_Dff_Time_sec'], time_start_sec, time_end_sec)):,}",
        },
    ]
    for col, label in [(PRIMARY_SCOPE_INPUT, "Input 3 scope exposure"), (PHOTODIODE_INPUT, "Input 1 photodiode")]:
        rows.extend(
            [
                {"item": f"{label} threshold", "value": f"{thresholds.get(col, np.nan):.6g} V"},
                {"item": f"{label} rising_edges_in_window", "value": f"{len(edge_times[col]['rising']):,}"},
                {"item": f"{label} falling_edges_in_window", "value": f"{len(edge_times[col]['falling']):,}"},
            ]
        )
    return pd.DataFrame(rows)


def axis_ref(prefix: str, row: int) -> str:
    """Return Plotly trace axis reference for a subplot row."""
    return prefix if row == 1 else f"{prefix}{row}"


def axis_key(prefix: str, row: int) -> str:
    """Return Plotly layout axis key for a subplot row."""
    return f"{prefix}axis" if row == 1 else f"{prefix}axis{row}"


def row_domains(row_heights: list[float], vertical_gap: float = 0.012) -> list[tuple[float, float]]:
    """Compute stacked y-axis domains from top to bottom."""
    total = float(sum(row_heights))
    usable = 1.0 - vertical_gap * (len(row_heights) - 1)
    heights = [h / total * usable for h in row_heights]
    domains = []
    top = 1.0
    for height in heights:
        bottom = top - height
        domains.append((bottom, top))
        top = bottom - vertical_gap
    return domains


def create_voltage_figure(
    *,
    title: str,
    envelope_data: dict[str, np.ndarray],
    centers: np.ndarray,
    events: pd.DataFrame,
    edge_times: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
    time_start_sec: float,
    time_end_sec: float,
) -> dict:
    """Build the interactive Plotly figure."""
    subplot_titles = ["Alignment event raster"] + [
        f"{col.strip()} - {role}" for col, role in VOLTAGE_INPUTS.items()
    ]
    row_heights = [0.18] + [0.1025] * len(VOLTAGE_INPUTS)
    domains = row_domains(row_heights)
    data = []
    layout: dict = {
        "title": {"text": title, "x": 0.01, "xanchor": "left"},
        "height": 1550,
        "width": 1500,
        "template": "plotly_white",
        "hovermode": "x unified",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        "margin": {"l": 95, "r": 35, "t": 125, "b": 55},
        "annotations": [],
    }

    for row_idx, (domain, row_title) in enumerate(zip(domains, subplot_titles), start=1):
        x_key = axis_key("x", row_idx)
        y_key = axis_key("y", row_idx)
        layout[x_key] = {
            "domain": [0.0, 1.0],
            "anchor": axis_ref("y", row_idx),
            "showgrid": False,
            "range": [time_start_sec, time_end_sec],
            "matches": "x" if row_idx > 1 else None,
            "showticklabels": row_idx == len(domains),
            "title": {"text": "Time (s)"} if row_idx == len(domains) else None,
        }
        layout[y_key] = {
            "domain": list(domain),
            "anchor": axis_ref("x", row_idx),
            "showgrid": False,
            "title": {"text": "V"} if row_idx > 1 else None,
        }
        layout["annotations"].append(
            {
                "text": row_title,
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": min(domain[1] + 0.004, 1.0),
                "xanchor": "left",
                "yanchor": "bottom",
                "showarrow": False,
                "font": {"size": 12},
            }
        )

    stim_start = filter_times(events["Stim_Start_sec"], time_start_sec, time_end_sec)
    stim_end = filter_times(events["Stim_End_sec"], time_start_sec, time_end_sec)
    nearest_dff = filter_times(events["Nearest_Dff_Time_sec"], time_start_sec, time_end_sec)
    scope_rising = edge_times[PRIMARY_SCOPE_INPUT]["rising"]
    photo_rising = edge_times[PHOTODIODE_INPUT]["rising"]
    photo_falling = edge_times[PHOTODIODE_INPUT]["falling"]
    is_analysis_table = "Analysis_Condition" in events.columns
    start_label = "Analysis trial start from logger" if is_analysis_table else "StimStart from logger"
    end_label = "Analysis trial end from logger" if is_analysis_table else "StimEnd from logger"
    nearest_label = (
        "Nearest dF/F frame per analysis trial"
        if is_analysis_table
        else "Nearest dF/F frame per stimulus"
    )

    event_traces = [
        event_marker_trace(name=start_label, times=stim_start, y_value=4, color="#d62728"),
        event_marker_trace(name=end_label, times=stim_end, y_value=3, color="#ff7f0e"),
        event_marker_trace(name=nearest_label, times=nearest_dff, y_value=2, color="#111111", opacity=0.65),
        event_marker_trace(name="Input 3 rising edges: frame TTL", times=scope_rising, y_value=1, color="#1f77b4", marker_size=7, opacity=0.42),
        event_marker_trace(name="Input 1 photodiode threshold edges", times=np.sort(np.concatenate([photo_rising, photo_falling])), y_value=0, color="#9467bd"),
    ]
    for trace in event_traces:
        trace["xaxis"] = axis_ref("x", 1)
        trace["yaxis"] = axis_ref("y", 1)
        data.append(trace)

    layout["yaxis"].update(
        {
            "range": [-0.6, 4.6],
            "tickmode": "array",
            "tickvals": [0, 1, 2, 3, 4],
            "ticktext": ["Photodiode edges", "Input 3 frame edges", "Nearest dF/F", "Event end", "Event start"],
        }
    )

    colors = ["#4c78a8", "#f58518", "#54a24b", "#1f77b4", "#b279a2", "#72b7b2", "#e45756", "#9467bd"]
    for row_idx, (color, col) in enumerate(zip(colors, VOLTAGE_INPUTS), start=2):
        ymin = envelope_data["mins"][col]
        ymax = envelope_data["maxs"][col]
        x_env, y_env = envelope_trace_arrays(centers, ymin, ymax)
        data.append(
            {
                "type": "scattergl",
                "x": x_env,
                "y": y_env,
                "xaxis": axis_ref("x", row_idx),
                "yaxis": axis_ref("y", row_idx),
                "mode": "lines",
                "line": {"color": color, "width": 1},
                "name": f"{col.strip()} raw min/max envelope",
                "hovertemplate": f"{col.strip()}<br>time=%{{x:.6f}} s<br>voltage=%{{y:.5f}} V<extra></extra>",
            }
        )
        means = envelope_data["means"][col]
        data.append(
            {
                "type": "scattergl",
                "x": centers,
                "y": means,
                "xaxis": axis_ref("x", row_idx),
                "yaxis": axis_ref("y", row_idx),
                "mode": "lines",
                "line": {"color": "#222222", "width": 0.6},
                "opacity": 0.55,
                "name": f"{col.strip()} bin mean",
                "hovertemplate": f"{col.strip()} mean<br>time=%{{x:.6f}} s<br>voltage=%{{y:.5f}} V<extra></extra>",
                "showlegend": False,
            }
        )
        threshold = thresholds.get(col)
        if threshold is not None and np.isfinite(threshold):
            data.append(
                {
                    "type": "scattergl",
                    "x": np.array([time_start_sec, time_end_sec]),
                    "y": np.array([threshold, threshold], dtype=float),
                    "xaxis": axis_ref("x", row_idx),
                    "yaxis": axis_ref("y", row_idx),
                    "mode": "lines",
                    "line": {"color": "#111111", "width": 0.8, "dash": "dash"},
                    "name": f"{col.strip()} threshold",
                    "hovertemplate": f"{col.strip()} threshold={threshold:.5f} V<extra></extra>",
                    "showlegend": row_idx == 2,
                }
            )
    return {"data": data, "layout": layout, "config": {"responsive": True, "scrollZoom": True}}


def summary_table_html(summary: pd.DataFrame) -> str:
    """Render summary table HTML."""
    rows = "\n".join(
        f"<tr><td>{row.item}</td><td>{row.value}</td></tr>"
        for row in summary.itertuples(index=False)
    )
    return f"""
<table class="summary-table">
  <thead><tr><th>Item</th><th>Value</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


def write_html_report(
    *,
    output_path: Path,
    fig: dict,
    summary: pd.DataFrame,
    data_dir: Path,
    voltage_csv: Path,
    event_table: Path,
) -> None:
    """Write standalone HTML with explanation plus Plotly figure."""
    fig_json = json.dumps(fig, default=json_default)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Interactive Voltage Validation Viewer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #1f2933;
      background: white;
    }}
    h1 {{
      margin-bottom: 6px;
    }}
    .meta {{
      color: #52606d;
      font-size: 14px;
      margin-bottom: 16px;
    }}
    .summary-table {{
      border-collapse: collapse;
      margin: 14px 0 20px 0;
      font-size: 14px;
    }}
    .summary-table th,
    .summary-table td {{
      border: 1px solid #d9e2ec;
      padding: 6px 10px;
      text-align: left;
    }}
    .summary-table th {{
      background: #f0f4f8;
    }}
    .note {{
      max-width: 1200px;
      font-size: 14px;
      line-height: 1.45;
      color: #334e68;
    }}
  </style>
</head>
<body>
  <h1>Interactive Voltage Validation Viewer</h1>
  <div class="meta">Session: {data_dir}<br>Voltage CSV: {voltage_csv}<br>Event table: {event_table}</div>
  <div class="note">
    The raw traces are shown as min/max envelopes per time bin so the full
    recording can be viewed in a browser. Use horizontal zoom/pan to inspect
    timing. For raw-like detail around a short interval, regenerate this viewer
    with <code>time_start_sec</code> and <code>time_end_sec</code>.
  </div>
  {summary_table_html(summary)}
  <div id="voltage-plot"></div>
  <script>
    const figure = {fig_json};
    Plotly.newPlot("voltage-plot", figure.data, figure.layout, figure.config);
  </script>
</body>
</html>
"""
    output_path.write_text(html)


def json_default(value):
    """JSON serializer for NumPy and pandas values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def create_interactive_voltage_validation_viewer(
    data_dir: str | Path,
    output_path: str | Path,
    event_table_path: str | Path | None = None,
    physical_health_path: str | Path | None = None,
    time_start_sec: float | None = None,
    time_end_sec: float | None = None,
    max_bins: int = 60_000,
    chunksize: int = 500_000,
) -> VoltageViewerResult:
    """Create a standalone interactive voltage validation HTML file.

    Parameters
    ----------
    data_dir
        Session directory containing the raw Prairie VoltageRecording CSV.
    output_path
        HTML path to write.
    event_table_path
        Optional path to ``block2_analysis_trials_with_dff_frames.csv``. If an
        older performed-event table is passed and the sibling trial-level table
        exists, the trial-level table is used.
    physical_health_path
        Optional path to ``physical_voltage_input_health.csv`` containing the
        adaptive raw-voltage thresholds.
    time_start_sec, time_end_sec
        Optional time window. If omitted, the whole recording is shown.
    max_bins
        Maximum number of min/max bins per trace. Higher values preserve more
        detail but make a larger HTML file.
    chunksize
        Number of voltage CSV rows per read chunk.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    voltage_csv = find_voltage_csv(data_dir)
    event_table = resolve_event_table(data_dir, output_path, event_table_path)
    thresholds = load_thresholds(data_dir, output_path, physical_health_path)
    events = load_alignment_events(event_table)

    full_start, full_end = read_voltage_time_bounds(voltage_csv)
    start = full_start if time_start_sec is None else float(time_start_sec)
    end = full_end if time_end_sec is None else float(time_end_sec)
    start = max(full_start, start)
    end = min(full_end, end)

    envelope_df, edge_times, envelope_data, centers = build_envelope_arrays(
        voltage_csv,
        thresholds=thresholds,
        time_start_sec=start,
        time_end_sec=end,
        max_bins=max_bins,
        chunksize=chunksize,
    )
    summary = build_summary(
        events=events,
        edge_times=edge_times,
        thresholds=thresholds,
        time_start_sec=start,
        time_end_sec=end,
        n_bins=len(envelope_df),
    )
    title = f"Raw Voltage Validation: {data_dir.name}"
    fig = create_voltage_figure(
        title=title,
        envelope_data=envelope_data,
        centers=centers,
        events=events,
        edge_times=edge_times,
        thresholds=thresholds,
        time_start_sec=start,
        time_end_sec=end,
    )
    write_html_report(
        output_path=output_path,
        fig=fig,
        summary=summary,
        data_dir=data_dir,
        voltage_csv=voltage_csv,
        event_table=event_table,
    )
    return VoltageViewerResult(
        html_path=output_path,
        voltage_csv_path=voltage_csv,
        event_table_path=event_table,
        n_bins=len(envelope_df),
        time_start_sec=start,
        time_end_sec=end,
        summary=summary,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Create an interactive raw-voltage validation viewer.")
    parser.add_argument("--data-dir", required=True, help="Session data directory.")
    parser.add_argument("--output-path", required=True, help="Output HTML path.")
    parser.add_argument("--event-table", default=None, help="Validated event table path.")
    parser.add_argument("--physical-health", default=None, help="Physical voltage health CSV with thresholds.")
    parser.add_argument("--time-start-sec", type=float, default=None, help="Optional window start in seconds.")
    parser.add_argument("--time-end-sec", type=float, default=None, help="Optional window end in seconds.")
    parser.add_argument("--max-bins", type=int, default=60_000, help="Maximum envelope bins per trace.")
    parser.add_argument("--chunksize", type=int, default=500_000, help="CSV rows per chunk.")
    return parser


def main() -> None:
    """Command-line entry point."""
    args = build_arg_parser().parse_args()
    result = create_interactive_voltage_validation_viewer(
        data_dir=args.data_dir,
        output_path=args.output_path,
        event_table_path=args.event_table,
        physical_health_path=args.physical_health,
        time_start_sec=args.time_start_sec,
        time_end_sec=args.time_end_sec,
        max_bins=args.max_bins,
        chunksize=args.chunksize,
    )
    print(f"Saved voltage viewer: {result.html_path}")
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
