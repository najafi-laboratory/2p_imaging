"""
Create event-aligned dF/F and wheel-speed figures.

This module uses the validated alignment output from
``event_alignment_validation.py``. The preferred input is:

    block2_analysis_trials_with_dff_frames.csv

That table contains trial-level alignment rows: standard 0 degree trial starts
and motor oddball trials. The lower-level
``block2_performed_events_with_dff_frames.csv`` still exists for timing
validation, but it has one row per short stimulus presentation and is not used
for neural trial averaging when the trial-level table is available.

For the current session, only block 2 has validated performed event timing in
the Bonsai output/logger files. The code therefore treats "all blocks" as
"all validated performed blocks available in the event table". With the current
files this is block 2. When both validation tables are available, 0 degree
standard plots use standard trial starts from the full performed-event table:
the first valid standard 0 deg ``StimStart`` between consecutive oddballs.
Oddball plots use the trial-level table.

The output is one multi-page PDF. Page 1 contains orientation-aligned
conditions in ascending orientation order. Page 2 contains the block-2
motor-oddball conditions in the requested task order. Each condition is drawn
as a GridSpec row with neural and wheel-speed traces side by side, and legends
in dedicated columns next to their subplot.

Each condition contains:

1. Population event-aligned z-scored dF/F.
2. Event-aligned wheel speed.

The legends include the number of neurons and the number of trials used after
boundary and duration-outlier filtering. PNG files are not generated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


EVENT_TABLE_NAME = "block2_performed_events_with_dff_frames.csv"
ANALYSIS_TRIAL_TABLE_NAME = "block2_analysis_trials_with_dff_frames.csv"
LOGGER_CSV = "orientations_logger.csv"
DFF_H5 = "qc_results/dff.h5"
OPS_NPY = "ops.npy"

ORIENTATION_SET_DEG = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
ORIENTATION_TRIAL_TYPES = {
    "single",
    "standard",
    "prerecorded",
    "motor_orientation_45",
    "motor_orientation_90",
    "rf_mapping",
}
ODDBALL_TRIAL_TYPES = {
    "motor_omission",
    "motor_halt",
    "motor_orientation_45",
    "motor_orientation_90",
}
BLOCK2_CONDITIONS = [
    ("block2_standard_control", "Block 2 standard 0 deg", "standard_control", "standard", 0.0),
    ("block2_omission", "Block 2 omission", "omission", "motor_omission", None),
    ("block2_halt", "Block 2 halt", "halt", "motor_halt", None),
    ("block2_orientation_90", "Block 2 orientation 90", "orientation_90", "motor_orientation_90", None),
    ("block2_orientation_45", "Block 2 orientation 45", "orientation_45", "motor_orientation_45", None),
]


@dataclass(frozen=True)
class AlignmentFigureResult:
    """Summary of one aligned condition in the combined PDF."""

    condition: str
    title: str
    n_neurons: int
    n_trials_total: int
    n_trials_used: int
    pdf_path: Path
    note: str


@dataclass(frozen=True)
class ConditionAlignmentData:
    """Computed traces for one condition before PDF rendering."""

    result: AlignmentFigureResult
    time: np.ndarray
    dff_mean: Optional[np.ndarray]
    dff_sem: Optional[np.ndarray]
    wheel_mean: Optional[np.ndarray]
    wheel_sem: Optional[np.ndarray]


@dataclass(frozen=True)
class AlignedAnalysisResult:
    """Outputs from ``run_aligned_dff_behavior_analysis``."""

    output_dir: Path
    pdf_path: Path
    condition_summary_path: Path
    skipped_conditions_path: Path
    condition_summary: pd.DataFrame
    skipped_conditions: pd.DataFrame
    figure_results: list[AlignmentFigureResult]


def load_ops(data_dir: Path) -> dict:
    """Load Suite2p ops metadata."""
    path = data_dir / OPS_NPY
    if not path.exists():
        return {}
    return np.load(path, allow_pickle=True).item()


def load_dff(data_dir: Path) -> np.ndarray:
    """Load dF/F traces as ROI x frame float32 array."""
    path = data_dir / DFF_H5
    if not path.exists():
        raise FileNotFoundError(f"Missing dF/F file: {path}")
    with h5py.File(path, "r") as h5:
        return np.asarray(h5["dff"], dtype=np.float32)


def zscore_dff(dff: np.ndarray) -> np.ndarray:
    """Z-score each ROI across time."""
    means = dff.mean(axis=1, keepdims=True, dtype=np.float64)
    stds = dff.std(axis=1, keepdims=True, dtype=np.float64)
    stds[stds == 0] = 1.0
    return ((dff - means) / stds).astype(np.float32)


def load_events(event_table_path: Path, duration_tolerance_sec: float = 0.020) -> pd.DataFrame:
    """Load validated trial/event table and add orientation/deviation columns."""
    if not event_table_path.exists():
        raise FileNotFoundError(f"Missing validated event table: {event_table_path}")
    events = pd.read_csv(event_table_path)

    # Bonsai output stores orientation in radians for this session. Convert if
    # all non-null values are in the radian range.
    orientation = pd.to_numeric(events["Orientation"], errors="coerce")
    finite = orientation.dropna().abs()
    if not finite.empty and finite.max() <= 2 * np.pi + 1e-6:
        events["Orientation_Deg"] = np.rad2deg(orientation)
    else:
        events["Orientation_Deg"] = orientation
    events["Orientation_Deg"] = events["Orientation_Deg"].round(6)
    events["Orientation_Mod180"] = np.mod(events["Orientation_Deg"], 180)

    events["Duration_Error_sec"] = (
        pd.to_numeric(events["Logger_Duration_sec"], errors="coerce")
        - pd.to_numeric(events["Duration"], errors="coerce")
    )
    events["Duration_Outlier"] = events["Duration_Error_sec"].abs() > duration_tolerance_sec
    events["Nearest_Dff_Frame"] = pd.to_numeric(events["Nearest_Dff_Frame"], errors="coerce").astype("Int64")
    return events


def find_event_table(data_dir: Path, output_dir: Path, event_table_path: Optional[str | Path]) -> Path:
    """Resolve the validated trial-level table path."""
    if event_table_path is not None:
        path = Path(event_table_path).expanduser().resolve()
        if path.name == EVENT_TABLE_NAME:
            sibling = path.with_name(ANALYSIS_TRIAL_TABLE_NAME)
            if sibling.exists():
                return sibling
        return path

    candidates = [
        output_dir / "alignment_validation" / ANALYSIS_TRIAL_TABLE_NAME,
        output_dir / ANALYSIS_TRIAL_TABLE_NAME,
        data_dir / "alignment_validation" / ANALYSIS_TRIAL_TABLE_NAME,
        data_dir / ANALYSIS_TRIAL_TABLE_NAME,
        Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/VW01_20260520_Closed_Loop_test-1556/alignment_validation") / ANALYSIS_TRIAL_TABLE_NAME,
        output_dir / "alignment_validation" / EVENT_TABLE_NAME,
        output_dir / EVENT_TABLE_NAME,
        data_dir / "alignment_validation" / EVENT_TABLE_NAME,
        data_dir / EVENT_TABLE_NAME,
        Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/VW01_20260520_Closed_Loop_test-1556/alignment_validation") / EVENT_TABLE_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find block2_analysis_trials_with_dff_frames.csv or block2_performed_events_with_dff_frames.csv. "
        "Run event_alignment_validation.py first or pass event_table_path explicitly."
    )


def find_performed_event_table(
    data_dir: Path,
    output_dir: Path,
    primary_event_table_path: Path,
) -> Optional[Path]:
    """Find the full performed-event table used to derive standard starts."""
    candidates = []
    if primary_event_table_path.name == EVENT_TABLE_NAME:
        candidates.append(primary_event_table_path)
    else:
        candidates.append(primary_event_table_path.with_name(EVENT_TABLE_NAME))
    candidates.extend(
        [
            output_dir / "alignment_validation" / EVENT_TABLE_NAME,
            output_dir / EVENT_TABLE_NAME,
            data_dir / "alignment_validation" / EVENT_TABLE_NAME,
            data_dir / EVENT_TABLE_NAME,
            Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Figures/VW01_20260520_Closed_Loop_test-1556/alignment_validation") / EVENT_TABLE_NAME,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def standard_trial_starts_between_oddballs(events: pd.DataFrame) -> pd.DataFrame:
    """Return first standard 0 deg start in each interval between oddballs."""
    events = events.sort_values("Stim_Start_sec").reset_index(drop=True).copy()
    if "Duration_Outlier" in events:
        events = events[~events["Duration_Outlier"]].reset_index(drop=True)

    oddballs = events[events["Trial_Type"].isin(ODDBALL_TRIAL_TYPES)].reset_index(drop=True)
    rows = []
    for oddball_index in range(len(oddballs) - 1):
        previous_oddball = oddballs.iloc[oddball_index]
        next_oddball = oddballs.iloc[oddball_index + 1]
        in_interval = (
            (events["Stim_Start_sec"] > previous_oddball["Stim_Start_sec"])
            & (events["Stim_Start_sec"] < next_oddball["Stim_Start_sec"])
            & events["Trial_Type"].eq("standard")
            & np.isclose(events["Orientation_Mod180"], 0.0, atol=1e-4)
        )
        candidates = events.loc[in_interval].sort_values("Stim_Start_sec")
        if not candidates.empty:
            rows.append(candidates.iloc[0].copy())

    if not rows:
        return events.iloc[0:0].copy()
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["Standard_Trial_Definition"] = "first standard 0 deg StimStart between consecutive oddballs"
    return out


def load_wheel_speed(data_dir: Path, dff_frame_times: np.ndarray) -> np.ndarray:
    """Load wheel encoder degrees from logger and interpolate speed to dF/F frames.

    Returns speed in deg/sec sampled at dF/F frame times. Values outside logger
    support are filled with NaN.
    """
    logger_path = data_dir / LOGGER_CSV
    if not logger_path.exists():
        return np.full(len(dff_frame_times), np.nan, dtype=np.float32)

    logger = pd.read_csv(logger_path)
    wheel = logger[logger["Value"].astype(str).str.startswith("Wheel-Deg-")].copy()
    if wheel.empty:
        return np.full(len(dff_frame_times), np.nan, dtype=np.float32)

    wheel["Deg"] = wheel["Value"].str.replace("Wheel-Deg-", "", regex=False).astype(float)
    wheel = wheel.sort_values("Timestamp")
    # Drop duplicate timestamps, which can break interpolation and gradients.
    wheel = wheel.drop_duplicates("Timestamp", keep="last")
    t = wheel["Timestamp"].to_numpy(dtype=float)
    deg = wheel["Deg"].to_numpy(dtype=float)

    if len(t) < 3:
        return np.full(len(dff_frame_times), np.nan, dtype=np.float32)

    speed = np.gradient(deg, t)
    frame_speed = np.interp(dff_frame_times, t, speed, left=np.nan, right=np.nan)
    return frame_speed.astype(np.float32)


def aligned_segments(trace: np.ndarray, event_frames: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Return event-aligned trace segments as trials x time."""
    idx = event_frames[:, None] + offsets[None, :]
    return trace[idx]


def mean_sem(segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SEM across trials, ignoring NaNs."""
    mean = np.nanmean(segments, axis=0)
    valid = np.sum(np.isfinite(segments), axis=0)
    std = np.nanstd(segments, axis=0, ddof=1)
    sem = np.divide(std, np.sqrt(valid), out=np.full_like(std, np.nan), where=valid > 1)
    return mean, sem


def filter_events_for_condition(
    events: pd.DataFrame,
    *,
    analysis_condition: Optional[str] = None,
    trial_type: Optional[str] = None,
    orientation_deg: Optional[float] = None,
    exclude_duration_outliers: bool = True,
) -> pd.DataFrame:
    """Select events for one condition."""
    selected = events.copy()
    if analysis_condition is not None and "Analysis_Condition" in selected.columns:
        selected = selected[selected["Analysis_Condition"].eq(analysis_condition)]
    if trial_type is not None:
        selected = selected[selected["Trial_Type"].eq(trial_type)]
    if orientation_deg is not None:
        selected = selected[np.isclose(selected["Orientation_Mod180"], orientation_deg, atol=1e-4)]
    if exclude_duration_outliers:
        selected = selected[~selected["Duration_Outlier"]]
    return selected


def select_orientation_analysis_events(
    events: pd.DataFrame,
    orientation_deg: float,
    *,
    using_analysis_trials: bool,
    standard_trial_events: Optional[pd.DataFrame],
    exclude_duration_outliers: bool,
) -> pd.DataFrame:
    """Select trial-level orientation events without counting non-visual oddballs."""
    if not using_analysis_trials:
        orientation_events = events[events["Trial_Type"].isin(ORIENTATION_TRIAL_TYPES)].copy()
        return filter_events_for_condition(
            orientation_events,
            orientation_deg=orientation_deg,
            exclude_duration_outliers=exclude_duration_outliers,
        )

    if float(orientation_deg) == 0.0 and standard_trial_events is not None:
        return filter_events_for_condition(
            standard_trial_events,
            trial_type="standard",
            orientation_deg=0.0,
            exclude_duration_outliers=exclude_duration_outliers,
        )

    # Omission and halt rows may carry a nominal orientation value but are not
    # visual orientation trials and should not enter this page.
    orientation_to_condition = {
        0.0: "standard_control",
        45.0: "orientation_45",
        90.0: "orientation_90",
    }
    analysis_condition = orientation_to_condition.get(float(orientation_deg))
    if analysis_condition is None:
        return events.iloc[0:0].copy()
    return filter_events_for_condition(
        events,
        analysis_condition=analysis_condition,
        exclude_duration_outliers=exclude_duration_outliers,
    )


def compute_condition_alignment(
    *,
    title: str,
    condition_name: str,
    events: pd.DataFrame,
    population_z_trace: np.ndarray,
    wheel_speed_trace: np.ndarray,
    frame_rate: float,
    pre_sec: float,
    post_sec: float,
    n_neurons: int,
    pdf_path: Path,
    empty_note: str = "No trials survived filtering and window-boundary checks.",
) -> ConditionAlignmentData:
    """Compute event-aligned population dF/F and wheel-speed means."""
    pre_frames = int(round(pre_sec * frame_rate))
    post_frames = int(round(post_sec * frame_rate))
    offsets = np.arange(-pre_frames, post_frames + 1)
    time = offsets / frame_rate

    total_trials = len(events)
    event_frames = events["Nearest_Dff_Frame"].dropna().astype(int).to_numpy()
    in_bounds = (event_frames + offsets[0] >= 0) & (event_frames + offsets[-1] < len(population_z_trace))
    event_frames = event_frames[in_bounds]
    used_trials = len(event_frames)

    if used_trials == 0:
        return ConditionAlignmentData(
            result=AlignmentFigureResult(
                condition=condition_name,
                title=title,
                n_neurons=n_neurons,
                n_trials_total=total_trials,
                n_trials_used=0,
                pdf_path=pdf_path,
                note=empty_note,
            ),
            time=time,
            dff_mean=None,
            dff_sem=None,
            wheel_mean=None,
            wheel_sem=None,
        )

    dff_segments = aligned_segments(population_z_trace, event_frames, offsets)
    wheel_segments = aligned_segments(wheel_speed_trace, event_frames, offsets)
    dff_mean, dff_sem = mean_sem(dff_segments)
    wheel_mean, wheel_sem = mean_sem(wheel_segments)

    return ConditionAlignmentData(
        result=AlignmentFigureResult(
            condition=condition_name,
            title=title,
            n_neurons=n_neurons,
            n_trials_total=total_trials,
            n_trials_used=used_trials,
            pdf_path=pdf_path,
            note="OK",
        ),
        time=time,
        dff_mean=dff_mean,
        dff_sem=dff_sem,
        wheel_mean=wheel_mean,
        wheel_sem=wheel_sem,
    )


def style_trace_axis(ax: plt.Axes, *, show_xlabel: bool) -> None:
    """Apply consistent publication-style axis formatting."""
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=8)
    if show_xlabel:
        ax.set_xlabel("Time from event onset (s)", fontsize=9)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def add_side_legend(legend_ax: plt.Axes, handles: list, labels: list[str]) -> None:
    """Place a legend in a dedicated GridSpec column beside a subplot."""
    legend_ax.axis("off")
    if handles:
        legend_ax.legend(
            handles,
            labels,
            loc="center left",
            frameon=False,
            fontsize=8,
            handlelength=1.8,
            borderaxespad=0.0,
        )


def plot_condition_grid_row(
    *,
    fig: plt.Figure,
    grid,
    row_index: int,
    data: ConditionAlignmentData,
    show_xlabel: bool,
    x_limits: tuple[float, float],
) -> None:
    """Render one condition as neural and wheel-speed subplots plus side legends."""
    ax_dff = fig.add_subplot(grid[row_index, 0])
    legend_dff = fig.add_subplot(grid[row_index, 1])
    ax_wheel = fig.add_subplot(grid[row_index, 2], sharex=ax_dff)
    legend_wheel = fig.add_subplot(grid[row_index, 3])

    ax_dff.set_title(data.result.title, loc="left", fontsize=10, weight="bold", pad=5)
    ax_wheel.set_title("Wheel speed", loc="left", fontsize=10, weight="bold", pad=5)
    ax_dff.set_ylabel("z-dF/F", fontsize=9)
    ax_wheel.set_ylabel("deg/s", fontsize=9)

    if data.result.n_trials_used == 0:
        for ax in (ax_dff, ax_wheel):
            ax.text(
                0.5,
                0.5,
                data.result.note,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#555555",
            )
            ax.set_xlim(*x_limits)
            style_trace_axis(ax, show_xlabel=show_xlabel)
        add_side_legend(legend_dff, [], [])
        add_side_legend(legend_wheel, [], [])
        return

    dff_color = "#1d4ed8"
    wheel_color = "#b42318"

    dff_label = (
        "Population z-dF/F\nmean +/- SEM\n"
        f"neurons={data.result.n_neurons}\n"
        f"trials={data.result.n_trials_used}"
    )
    (dff_line,) = ax_dff.plot(data.time, data.dff_mean, color=dff_color, linewidth=1.8, label=dff_label)
    ax_dff.fill_between(
        data.time,
        data.dff_mean - data.dff_sem,
        data.dff_mean + data.dff_sem,
        color=dff_color,
        alpha=0.28,
        linewidth=0,
    )

    wheel_label = f"Wheel speed\nmean +/- SEM\ntrials={data.result.n_trials_used}"
    (wheel_line,) = ax_wheel.plot(data.time, data.wheel_mean, color=wheel_color, linewidth=1.8, label=wheel_label)
    ax_wheel.fill_between(
        data.time,
        data.wheel_mean - data.wheel_sem,
        data.wheel_mean + data.wheel_sem,
        color=wheel_color,
        alpha=0.28,
        linewidth=0,
    )

    for ax in (ax_dff, ax_wheel):
        ax.set_xlim(*x_limits)
        style_trace_axis(ax, show_xlabel=show_xlabel)

    add_side_legend(legend_dff, [dff_line], [dff_label])
    add_side_legend(legend_wheel, [wheel_line], [wheel_label])


def write_condition_group_page(
    *,
    pdf: PdfPages,
    page_title: str,
    conditions: list[ConditionAlignmentData],
    x_limits: tuple[float, float],
) -> None:
    """Write one ordered condition group to the combined PDF."""
    n_rows = max(len(conditions), 1)
    fig_height = max(7.5, 2.05 * n_rows + 1.4)
    fig = plt.figure(figsize=(16.5, fig_height))
    fig.suptitle(page_title, fontsize=15, weight="bold", y=0.992)
    grid = fig.add_gridspec(
        nrows=n_rows,
        ncols=4,
        width_ratios=[5.0, 1.55, 5.0, 1.55],
        left=0.055,
        right=0.985,
        top=0.935,
        bottom=0.055,
        wspace=0.08,
        hspace=0.62,
    )

    if not conditions:
        ax = fig.add_subplot(grid[0, 0:4])
        ax.axis("off")
        ax.text(0.5, 0.5, "No conditions available.", ha="center", va="center", fontsize=11)
    else:
        for row_index, data in enumerate(conditions):
            plot_condition_grid_row(
                fig=fig,
                grid=grid,
                row_index=row_index,
                data=data,
                show_xlabel=row_index == len(conditions) - 1,
                x_limits=x_limits,
            )

    pdf.savefig(fig)
    plt.close(fig)


def write_combined_pdf(
    *,
    pdf_path: Path,
    orientation_conditions: list[ConditionAlignmentData],
    block2_conditions: list[ConditionAlignmentData],
    pre_sec: float,
    post_sec: float,
    using_analysis_trials: bool,
) -> None:
    """Write all condition sets into one ordered PDF."""
    x_limits = (-pre_sec, post_sec)
    orientation_page_title = (
        "Orientation-Aligned Trial-Level Analysis Events, Block 2"
        if using_analysis_trials
        else "Orientation-Aligned Trials, All Validated Performed Blocks"
    )
    with PdfPages(pdf_path) as pdf:
        write_condition_group_page(
            pdf=pdf,
            page_title=orientation_page_title,
            conditions=orientation_conditions,
            x_limits=x_limits,
        )
        write_condition_group_page(
            pdf=pdf,
            page_title="Block 2 Sensorimotor Oddball Conditions",
            conditions=block2_conditions,
            x_limits=x_limits,
        )


def run_aligned_dff_behavior_analysis(
    data_dir: str | Path,
    output_dir: str | Path,
    event_table_path: str | Path | None = None,
    pre_sec: float = 2.0,
    post_sec: float = 4.0,
    exclude_duration_outliers: bool = True,
    use_standard_trial_starts: bool = True,
    pdf_name: str = "aligned_dff_behavior_all_conditions.pdf",
    save_pdf: Optional[bool] = None,
) -> AlignedAnalysisResult:
    """Run event-aligned neural and wheel-speed analysis.

    Parameters
    ----------
    data_dir
        Session data directory containing dF/F, logger, and Suite2p metadata.
    output_dir
        Directory where aligned figures and summary CSVs will be saved.
    event_table_path
        Optional explicit path to the trial-level
        ``block2_analysis_trials_with_dff_frames.csv``. If an older
        ``block2_performed_events_with_dff_frames.csv`` path is provided and
        the sibling trial-level table exists, the trial-level table is used.
    pre_sec, post_sec
        Alignment window in seconds around each event onset.
    exclude_duration_outliers
        If True, remove events whose logger duration differs from the CSV
        duration by more than 20 ms. This removes the known standard duration
        outliers from this session.
    use_standard_trial_starts
        If True and the full performed-event table is available, derive 0
        degree standard trials as the first valid standard ``StimStart``
        between consecutive oddballs. This avoids aligning to every 33 ms
        standard presentation.
    pdf_name
        File name for the combined multi-page PDF.
    save_pdf
        Deprecated compatibility argument for older notebooks. This function
        always writes the combined PDF and never writes PNG files.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_name_path = Path(pdf_name).expanduser()
    pdf_path = pdf_name_path if pdf_name_path.is_absolute() else output_dir / pdf_name_path
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    event_table_path = find_event_table(data_dir, output_dir, event_table_path)
    events = load_events(event_table_path)
    performed_event_table_path = find_performed_event_table(data_dir, output_dir, event_table_path)
    performed_events = (
        load_events(performed_event_table_path)
        if use_standard_trial_starts and performed_event_table_path is not None
        else None
    )

    dff = load_dff(data_dir)
    n_neurons, n_frames = dff.shape
    ops = load_ops(data_dir)
    frame_rate = float(ops.get("fs", 30.0))
    dff_frame_times = np.arange(n_frames, dtype=float) / frame_rate

    z_dff = zscore_dff(dff)
    population_z_trace = z_dff.mean(axis=0)
    wheel_speed_trace = load_wheel_speed(data_dir, dff_frame_times)
    using_analysis_trials = "Analysis_Condition" in events.columns
    standard_trial_events = None
    if performed_events is not None:
        block2_performed_events = performed_events[performed_events["Block_Number"].eq(2)].copy()
        standard_trial_events = standard_trial_starts_between_oddballs(block2_performed_events)
        if standard_trial_events.empty:
            standard_trial_events = None

    results: list[AlignmentFigureResult] = []
    skipped_rows = []
    orientation_condition_data: list[ConditionAlignmentData] = []
    block2_condition_data: list[ConditionAlignmentData] = []

    # Orientation figure set. When possible, 0 deg uses standard trial starts
    # between oddballs and the 45/90 deg rows use motor orientation oddballs.
    for orientation in ORIENTATION_SET_DEG:
        condition = f"orientation_{str(orientation).replace('.', 'p')}_deg_all_validated_blocks"
        title = (
            f"{orientation:g} deg standard trial starts between oddballs, block 2"
            if using_analysis_trials and orientation == 0.0 and standard_trial_events is not None
            else f"{orientation:g} deg aligned analysis trials, block 2"
            if using_analysis_trials
            else f"{orientation:g} deg aligned trials, all validated performed blocks"
        )
        selected = select_orientation_analysis_events(
            events,
            orientation,
            using_analysis_trials=using_analysis_trials,
            standard_trial_events=standard_trial_events,
            exclude_duration_outliers=exclude_duration_outliers,
        )
        empty_note = (
            "No trial-level analysis events for this orientation."
            if using_analysis_trials
            else "No validated performed trials for this orientation in the event table."
        )
        if selected.empty:
            skipped_rows.append(
                {
                    "condition": condition,
                    "title": title,
                    "reason": empty_note,
                }
            )
        condition_data = compute_condition_alignment(
            title=title,
            condition_name=condition,
            events=selected,
            population_z_trace=population_z_trace,
            wheel_speed_trace=wheel_speed_trace,
            frame_rate=frame_rate,
            pre_sec=pre_sec,
            post_sec=post_sec,
            n_neurons=n_neurons,
            pdf_path=pdf_path,
            empty_note=empty_note,
        )
        orientation_condition_data.append(condition_data)
        results.append(condition_data.result)

    # Block-2 motor oddball condition set.
    block2_events = events[events["Block_Number"].eq(2)].copy()
    for condition, title, analysis_condition, trial_type, orientation in BLOCK2_CONDITIONS:
        if (
            use_standard_trial_starts
            and analysis_condition == "standard_control"
            and standard_trial_events is not None
        ):
            selected = filter_events_for_condition(
                standard_trial_events,
                trial_type="standard",
                orientation_deg=0.0,
                exclude_duration_outliers=exclude_duration_outliers,
            )
            title = "Block 2 standard 0 deg trial starts between oddballs"
        elif using_analysis_trials:
            selected = filter_events_for_condition(
                block2_events,
                analysis_condition=analysis_condition,
                exclude_duration_outliers=exclude_duration_outliers,
            )
        else:
            selected = filter_events_for_condition(
                block2_events,
                trial_type=trial_type,
                orientation_deg=orientation,
                exclude_duration_outliers=exclude_duration_outliers,
            )
        empty_note = "No trials after filtering."
        if selected.empty:
            skipped_rows.append({"condition": condition, "title": title, "reason": empty_note})
        condition_data = compute_condition_alignment(
            title=title,
            condition_name=condition,
            events=selected,
            population_z_trace=population_z_trace,
            wheel_speed_trace=wheel_speed_trace,
            frame_rate=frame_rate,
            pre_sec=pre_sec,
            post_sec=post_sec,
            n_neurons=n_neurons,
            pdf_path=pdf_path,
            empty_note=empty_note,
        )
        block2_condition_data.append(condition_data)
        results.append(condition_data.result)

    condition_summary = pd.DataFrame([result.__dict__ for result in results])
    skipped_conditions = pd.DataFrame(skipped_rows)
    condition_summary_path = output_dir / "aligned_dff_behavior_condition_summary.csv"
    skipped_conditions_path = output_dir / "aligned_dff_behavior_skipped_conditions.csv"
    condition_summary.to_csv(condition_summary_path, index=False)
    skipped_conditions.to_csv(skipped_conditions_path, index=False)
    write_combined_pdf(
        pdf_path=pdf_path,
        orientation_conditions=orientation_condition_data,
        block2_conditions=block2_condition_data,
        pre_sec=pre_sec,
        post_sec=post_sec,
        using_analysis_trials=using_analysis_trials,
    )

    return AlignedAnalysisResult(
        output_dir=output_dir,
        pdf_path=pdf_path,
        condition_summary_path=condition_summary_path,
        skipped_conditions_path=skipped_conditions_path,
        condition_summary=condition_summary,
        skipped_conditions=skipped_conditions,
        figure_results=results,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Create aligned dF/F and wheel-speed figures.")
    parser.add_argument("--data-dir", required=True, help="Session data directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for aligned figures.")
    parser.add_argument("--event-table", default=None, help="Path to validated event table.")
    parser.add_argument("--pre-sec", type=float, default=2.0, help="Pre-event window in seconds.")
    parser.add_argument("--post-sec", type=float, default=4.0, help="Post-event window in seconds.")
    parser.add_argument("--include-duration-outliers", action="store_true", help="Keep logger-duration outlier events.")
    parser.add_argument(
        "--use-table-standard-control",
        action="store_true",
        help="Use standard rows already stored in the trial-level table instead of rederiving standard trial starts from the performed-event table.",
    )
    parser.add_argument(
        "--pdf-name",
        default="aligned_dff_behavior_all_conditions.pdf",
        help="File name for the combined multi-page PDF.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""
    args = build_arg_parser().parse_args()
    result = run_aligned_dff_behavior_analysis(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        event_table_path=args.event_table,
        pre_sec=args.pre_sec,
        post_sec=args.post_sec,
        exclude_duration_outliers=not args.include_duration_outliers,
        use_standard_trial_starts=not args.use_table_standard_control,
        pdf_name=args.pdf_name,
    )
    print(f"Saved combined aligned PDF: {result.pdf_path}")
    print(f"Saved condition summary: {result.condition_summary_path}")
    print(result.condition_summary[["condition", "n_neurons", "n_trials_total", "n_trials_used", "note"]].to_string(index=False))
    if not result.skipped_conditions.empty:
        print("\nSkipped conditions:")
        print(result.skipped_conditions.to_string(index=False))


if __name__ == "__main__":
    main()
