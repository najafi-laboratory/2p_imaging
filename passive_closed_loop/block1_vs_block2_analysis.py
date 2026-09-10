"""Compare VW01 passive closed-loop Block 1 and Block 2 responses.

This analysis uses the performed Bonsai stimulus table and logger timing from
the 2026-05-22 session. It treats the raw Block 2 ``standard`` rows as frame
updates and collapses contiguous standard runs into presentation-level
``block2_standard_sequence`` events. dF/F quality is not assumed; the figures
are intended to verify whether the analysis can run and expose the requested
Block 1 vs Block 2 comparisons.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


DEFAULT_RAW_SESSION = Path(
    "/storage/project/r-fnajafi3-0/shared/2P_Imaging/VW01/VW01_20260522_ClosedLoop-1567"
)
DEFAULT_BONSAI_DIR = DEFAULT_RAW_SESSION / "VW01_20260522T181733"
DEFAULT_PROCESSED_DIR = Path(
    "/storage/scratch1/3/grubin6/2p_processing_results/VW01_20260522_ClosedLoop-1567_xml_referenced"
)
DEFAULT_OUTPUT_DIR = Path("Outputs/VW01_20260522/block1_vs_block2_analysis")


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    title: str
    selector: str


def load_dff(processed_dir: Path) -> np.ndarray:
    """Load dF/F from the pipeline output."""
    candidates = [
        processed_dir / "dff.h5",
        processed_dir / "qc_results" / "dff.h5",
    ]
    for path in candidates:
        if path.exists():
            with h5py.File(path, "r") as h5:
                return np.asarray(h5["dff"], dtype=np.float32)
    raise FileNotFoundError(f"Could not find dff.h5 under {processed_dir}")


def load_ops(processed_dir: Path) -> dict:
    """Load Suite2p ops metadata."""
    candidates = [
        processed_dir / "suite2p" / "plane0" / "ops.npy",
        processed_dir / "ops.npy",
    ]
    for path in candidates:
        if path.exists():
            return np.load(path, allow_pickle=True).item()
    return {}


def zscore_rows(data: np.ndarray) -> np.ndarray:
    """Z-score each ROI across time."""
    mean = data.mean(axis=1, keepdims=True, dtype=np.float64)
    std = data.std(axis=1, keepdims=True, dtype=np.float64)
    std[std == 0] = 1.0
    return ((data - mean) / std).astype(np.float32)


def parse_logger(logger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract StimStart, StimEnd, and Wheel-Deg rows from Bonsai logger."""
    values = logger["Value"].astype(str)
    starts = logger.loc[values.str.startswith("StimStart-", na=False), ["Frame", "Timestamp", "Value"]].copy()
    starts["Id"] = starts["Value"].str.replace("StimStart-", "", regex=False)
    starts = starts.rename(columns={"Frame": "StimStart_Frame", "Timestamp": "StimStart_sec"})

    ends = logger.loc[values.str.startswith("StimEnd-", na=False), ["Frame", "Timestamp", "Value"]].copy()
    ends["Id"] = ends["Value"].str.replace("StimEnd-", "", regex=False)
    ends = ends.rename(columns={"Frame": "StimEnd_Frame", "Timestamp": "StimEnd_sec"})

    wheel = logger.loc[values.str.startswith("Wheel-Deg-", na=False), ["Frame", "Timestamp", "Value"]].copy()
    wheel["Wheel_Deg"] = wheel["Value"].str.extract(r"Wheel-Deg-\s*([-+0-9.eE]+)")[0].astype(float)
    wheel = wheel.sort_values("Timestamp").reset_index(drop=True)
    return starts[["Id", "StimStart_Frame", "StimStart_sec"]], ends[["Id", "StimEnd_Frame", "StimEnd_sec"]], wheel


def build_event_table(stim_csv: Path, logger_csv: Path) -> pd.DataFrame:
    """Merge Bonsai stimulus rows with logger start/end timestamps."""
    stim = pd.read_csv(stim_csv)
    logger = pd.read_csv(logger_csv)
    starts, ends, _ = parse_logger(logger)
    events = stim.merge(starts, on="Id", how="left").merge(ends, on="Id", how="left")
    events["Orientation_Deg"] = np.rad2deg(pd.to_numeric(events["Orientation"], errors="coerce")).round(6)
    events["Logger_Duration_sec"] = events["StimEnd_sec"] - events["StimStart_sec"]
    events["Duration_Error_sec"] = events["Logger_Duration_sec"] - events["Duration"]
    events["Duration_Outlier"] = events["Duration_Error_sec"].abs() > 0.080
    return events.sort_values("TrialNumber").reset_index(drop=True)


def collapse_block2_standard_runs(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse contiguous Block 2 standard frame updates into sequence events."""
    block2 = events[events["BlockNumber"].eq(2)].sort_values("TrialNumber").reset_index(drop=True)
    rows: list[dict] = []
    i = 0
    while i < len(block2):
        row = block2.iloc[i]
        if row["TrialType"] == "standard":
            j = i
            while j + 1 < len(block2) and block2.iloc[j + 1]["TrialType"] == "standard":
                j += 1
            chunk = block2.iloc[i : j + 1]
            rows.append(
                {
                    "Analysis_Condition": "block2_standard_sequence",
                    "Source_Update_Rows": len(chunk),
                    "BlockNumber": 2,
                    "BlockLabel": row["BlockLabel"],
                    "BlockType": row["BlockType"],
                    "TrialType": "standard",
                    "Orientation_Deg": 0.0,
                    "TrialNumber_Start": int(chunk["TrialNumber"].iloc[0]),
                    "TrialNumber_End": int(chunk["TrialNumber"].iloc[-1]),
                    "StimStart_sec": chunk["StimStart_sec"].iloc[0],
                    "StimEnd_sec": chunk["StimEnd_sec"].iloc[-1],
                    "Logger_Duration_sec": chunk["StimEnd_sec"].iloc[-1] - chunk["StimStart_sec"].iloc[0],
                    "Duration": chunk["Duration"].sum(),
                }
            )
            i = j + 1
        else:
            rows.append(
                {
                    "Analysis_Condition": "block2_" + str(row["TrialType"]).replace("motor_", ""),
                    "Source_Update_Rows": 1,
                    "BlockNumber": 2,
                    "BlockLabel": row["BlockLabel"],
                    "BlockType": row["BlockType"],
                    "TrialType": row["TrialType"],
                    "Orientation_Deg": row["Orientation_Deg"],
                    "TrialNumber_Start": int(row["TrialNumber"]),
                    "TrialNumber_End": int(row["TrialNumber"]),
                    "StimStart_sec": row["StimStart_sec"],
                    "StimEnd_sec": row["StimEnd_sec"],
                    "Logger_Duration_sec": row["Logger_Duration_sec"],
                    "Duration": row["Duration"],
                }
            )
            i += 1
    return pd.DataFrame(rows)


def build_analysis_trials(events: pd.DataFrame) -> pd.DataFrame:
    """Build presentation-level analysis trials for the requested comparisons."""
    rows: list[pd.DataFrame] = []

    def add_condition(mask: pd.Series, condition: str) -> None:
        valid_timing = events["StimStart_sec"].notna() & events["StimEnd_sec"].notna() & ~events["Duration_Outlier"]
        sub = events.loc[mask & valid_timing].copy()
        if sub.empty:
            return
        sub["Analysis_Condition"] = condition
        sub["Source_Update_Rows"] = 1
        sub["TrialNumber_Start"] = sub["TrialNumber"]
        sub["TrialNumber_End"] = sub["TrialNumber"]
        rows.append(
            sub[
                [
                    "Analysis_Condition",
                    "Source_Update_Rows",
                    "BlockNumber",
                    "BlockLabel",
                    "BlockType",
                    "TrialType",
                    "Orientation_Deg",
                    "TrialNumber_Start",
                    "TrialNumber_End",
                    "StimStart_sec",
                    "StimEnd_sec",
                    "Logger_Duration_sec",
                    "Duration",
                ]
            ]
        )

    add_condition(
        events["BlockNumber"].eq(1) & events["TrialType"].eq("single") & np.isclose(events["Orientation_Deg"], 0.0),
        "block1_single_0",
    )
    add_condition(
        events["BlockNumber"].eq(1) & events["TrialType"].eq("single") & np.isclose(events["Orientation_Deg"], 45.0),
        "block1_single_45",
    )
    add_condition(
        events["BlockNumber"].eq(1) & events["TrialType"].eq("single") & np.isclose(events["Orientation_Deg"], 90.0),
        "block1_single_90",
    )
    add_condition(events["BlockNumber"].eq(1) & events["TrialType"].eq("omission"), "block1_omission")
    add_condition(events["BlockNumber"].eq(1) & events["TrialType"].eq("halt"), "block1_halt")

    rows.append(collapse_block2_standard_runs(events))
    trials = pd.concat(rows, ignore_index=True)
    trials = trials.sort_values(["StimStart_sec", "TrialNumber_Start"]).reset_index(drop=True)
    trials["Analysis_Trial"] = np.arange(1, len(trials) + 1)
    return trials


def frame_times_from_voltage(processed_dir: Path, nframes: int, fs: float) -> np.ndarray:
    """Use scope exposure voltage edges when available; otherwise fall back to fs."""
    path = processed_dir / "raw_voltages.h5"
    if not path.exists():
        return np.arange(nframes, dtype=float) / fs
    with h5py.File(path, "r") as h5:
        t = np.asarray(h5["raw/vol_time"])
        dt = float(np.nanmedian(np.diff(t[: min(10000, len(t))])))
        t_sec = t * (0.001 if dt > 0.01 else 1.0)
        img = np.asarray(h5["raw/vol_img"])
    high = img > 0.5
    rising = np.flatnonzero((~high[:-1]) & high[1:]) + 1
    edge_times = t_sec[rising]
    if len(edge_times) >= nframes:
        return edge_times[:nframes]
    fallback = np.arange(nframes, dtype=float) / fs
    fallback[: len(edge_times)] = edge_times
    return fallback


def nearest_frame_indices(frame_times: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    """Map event times to nearest imaging frame indices."""
    idx = np.searchsorted(frame_times, event_times)
    idx0 = np.clip(idx - 1, 0, len(frame_times) - 1)
    idx1 = np.clip(idx, 0, len(frame_times) - 1)
    choose_right = np.abs(frame_times[idx1] - event_times) < np.abs(frame_times[idx0] - event_times)
    return np.where(choose_right, idx1, idx0)


def wheel_speed_trace(logger_csv: Path, timebase: np.ndarray) -> np.ndarray:
    """Interpolate Bonsai wheel speed onto the imaging frame timebase."""
    logger = pd.read_csv(logger_csv)
    _, _, wheel = parse_logger(logger)
    if len(wheel) < 3:
        return np.full_like(timebase, np.nan, dtype=float)
    wheel_time = wheel["Timestamp"].to_numpy(dtype=float)
    wheel_deg = wheel["Wheel_Deg"].to_numpy(dtype=float)
    speed = np.gradient(wheel_deg, wheel_time)
    return np.interp(timebase, wheel_time, speed, left=np.nan, right=np.nan)


def extract_aligned(
    data: np.ndarray,
    event_frames: Iterable[int],
    fs: float,
    pre_sec: float,
    post_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract event-aligned traces from ROI x frame or 1D frame data."""
    event_frames = np.asarray(list(event_frames), dtype=int)
    pre = int(round(pre_sec * fs))
    post = int(round(post_sec * fs))
    offsets = np.arange(-pre, post + 1)
    valid = event_frames[(event_frames - pre >= 0) & (event_frames + post < data.shape[-1])]
    if data.ndim == 1:
        aligned = np.stack([data[f + offsets] for f in valid], axis=0) if len(valid) else np.empty((0, len(offsets)))
    else:
        aligned = (
            np.stack([data[:, f + offsets] for f in valid], axis=0)
            if len(valid)
            else np.empty((0, data.shape[0], len(offsets)))
        )
    return offsets / fs, aligned


def mean_sem(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return nanmean and SEM."""
    mean = np.nanmean(values, axis=axis)
    n = np.sum(np.isfinite(values), axis=axis)
    sem = np.nanstd(values, axis=axis) / np.sqrt(np.maximum(n, 1))
    return mean, sem


def condition_response(
    trials: pd.DataFrame,
    condition: str,
    z_dff: np.ndarray,
    wheel_speed: np.ndarray,
    fs: float,
    pre_sec: float,
    post_sec: float,
) -> dict:
    """Compute aligned population and wheel traces for one condition."""
    sub = trials[trials["Analysis_Condition"].eq(condition)].copy()
    t, aligned_dff = extract_aligned(z_dff, sub["Nearest_Frame"], fs, pre_sec, post_sec)
    _, aligned_wheel = extract_aligned(wheel_speed, sub["Nearest_Frame"], fs, pre_sec, post_sec)
    if aligned_dff.size:
        # trials x rois x time -> average per trial over rois, then over trials
        trial_pop = np.nanmean(aligned_dff, axis=1)
        dff_mean, dff_sem = mean_sem(trial_pop, axis=0)
    else:
        trial_pop = np.empty((0, len(t)))
        dff_mean = dff_sem = np.full(len(t), np.nan)
    wheel_mean, wheel_sem = mean_sem(aligned_wheel, axis=0) if aligned_wheel.size else (np.full(len(t), np.nan), np.full(len(t), np.nan))

    baseline_mask = (t >= -0.5) & (t < 0)
    response_mask = (t >= 0) & (t <= 0.8)
    trial_baseline = np.nanmean(trial_pop[:, baseline_mask], axis=1) if trial_pop.size else np.array([])
    trial_peak = np.nanmax(trial_pop[:, response_mask], axis=1) if trial_pop.size else np.array([])
    trial_response = trial_peak - trial_baseline if len(trial_peak) else np.array([])
    return {
        "condition": condition,
        "trials_total": len(sub),
        "trials_used": len(trial_pop),
        "time": t,
        "dff_mean": dff_mean,
        "dff_sem": dff_sem,
        "wheel_mean": wheel_mean,
        "wheel_sem": wheel_sem,
        "trial_response": trial_response,
        "median_duration_sec": float(np.nanmedian(sub["Logger_Duration_sec"])) if len(sub) else np.nan,
    }


def plot_comparison_page(pdf: PdfPages, title: str, responses: list[dict], stim_duration_lookup: dict[str, float]) -> None:
    """Render one comparison page."""
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(responses), 3)))
    for color, response in zip(colors, responses):
        label = f"{response['condition']} (n={response['trials_used']}/{response['trials_total']})"
        t = response["time"]
        dur = stim_duration_lookup.get(response["condition"], response["median_duration_sec"])
        if np.isfinite(dur):
            axes[0].axvspan(0, dur, color=color, alpha=0.08, lw=0)
            axes[1].axvspan(0, dur, color=color, alpha=0.08, lw=0)
        axes[0].plot(t, response["dff_mean"], color=color, lw=1.5, label=label)
        axes[0].fill_between(t, response["dff_mean"] - response["dff_sem"], response["dff_mean"] + response["dff_sem"], color=color, alpha=0.18)
        axes[1].plot(t, response["wheel_mean"], color=color, lw=1.2, label=label)
        axes[1].fill_between(t, response["wheel_mean"] - response["wheel_sem"], response["wheel_mean"] + response["wheel_sem"], color=color, alpha=0.18)
    for ax in axes:
        ax.axvline(0, color="black", lw=0.8)
        ax.grid(False)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    axes[0].set_ylabel("Population z-dF/F")
    axes[1].set_ylabel("Wheel speed (deg/s)")
    axes[1].set_xlabel("Time from onset (s)")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.78, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def run_analysis(raw_session: Path, bonsai_dir: Path, processed_dir: Path, output_dir: Path) -> dict:
    """Run Block 1 vs Block 2 comparison analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stim_csv = bonsai_dir / "orientations_orientations0.csv"
    logger_csv = bonsai_dir / "orientations_logger.csv"

    events = build_event_table(stim_csv, logger_csv)
    trials = build_analysis_trials(events)
    dff = load_dff(processed_dir)
    ops = load_ops(processed_dir)
    fs = float(ops.get("fs", 30.0))
    frame_times = frame_times_from_voltage(processed_dir, dff.shape[1], fs)
    trials["Nearest_Frame"] = nearest_frame_indices(frame_times, trials["StimStart_sec"].to_numpy(dtype=float))
    trials["Nearest_Frame_Time_sec"] = frame_times[trials["Nearest_Frame"]]
    trials["Frame_Delta_sec"] = trials["StimStart_sec"] - trials["Nearest_Frame_Time_sec"]

    z_dff = zscore_rows(dff)
    wheel_speed = wheel_speed_trace(logger_csv, frame_times)

    conditions = [
        "block1_single_0",
        "block2_standard_sequence",
        "block1_single_45",
        "block2_orientation_45",
        "block1_single_90",
        "block2_orientation_90",
        "block1_omission",
        "block2_omission",
        "block1_halt",
        "block2_halt",
    ]
    response_by_condition = {
        condition: condition_response(trials, condition, z_dff, wheel_speed, fs, pre_sec=1.0, post_sec=2.0)
        for condition in conditions
    }

    summary_rows = []
    for condition, response in response_by_condition.items():
        values = response["trial_response"]
        summary_rows.append(
            {
                "condition": condition,
                "trials_total": response["trials_total"],
                "trials_used": response["trials_used"],
                "n_rois": dff.shape[0],
                "median_duration_sec": response["median_duration_sec"],
                "peak_minus_baseline_mean": float(np.nanmean(values)) if len(values) else np.nan,
                "peak_minus_baseline_sem": float(np.nanstd(values) / np.sqrt(max(len(values), 1))) if len(values) else np.nan,
            }
        )
    condition_summary = pd.DataFrame(summary_rows)

    comparisons = [
        ("orientation_0_block1_vs_block2", "block1_single_0", "block2_standard_sequence"),
        ("orientation_45_block1_vs_block2", "block1_single_45", "block2_orientation_45"),
        ("orientation_90_block1_vs_block2", "block1_single_90", "block2_orientation_90"),
        ("omission_block1_vs_block2", "block1_omission", "block2_omission"),
        ("halt_block1_vs_block2", "block1_halt", "block2_halt"),
        ("block2_orientation_45_vs_90", "block2_orientation_45", "block2_orientation_90"),
    ]
    comparison_rows = []
    for name, a, b in comparisons:
        va = response_by_condition[a]["trial_response"]
        vb = response_by_condition[b]["trial_response"]
        comparison_rows.append(
            {
                "comparison": name,
                "condition_a": a,
                "condition_b": b,
                "n_a": len(va),
                "n_b": len(vb),
                "mean_a": float(np.nanmean(va)) if len(va) else np.nan,
                "mean_b": float(np.nanmean(vb)) if len(vb) else np.nan,
                "mean_b_minus_a": float(np.nanmean(vb) - np.nanmean(va)) if len(va) and len(vb) else np.nan,
            }
        )
    comparison_summary = pd.DataFrame(comparison_rows)

    block_summary = (
        events.groupby(["BlockNumber", "BlockLabel", "BlockType"], dropna=False)
        .agg(
            rows=("Id", "size"),
            trial_start=("TrialNumber", "min"),
            trial_end=("TrialNumber", "max"),
            csv_duration_sec=("Duration", "sum"),
            logger_start_sec=("StimStart_sec", "min"),
            logger_end_sec=("StimEnd_sec", "max"),
            missing_starts=("StimStart_sec", lambda x: int(x.isna().sum())),
            missing_ends=("StimEnd_sec", lambda x: int(x.isna().sum())),
        )
        .reset_index()
    )
    block_summary["logger_span_sec"] = block_summary["logger_end_sec"] - block_summary["logger_start_sec"]

    trials.to_csv(output_dir / "analysis_trials_all_blocks.csv", index=False)
    block_summary.to_csv(output_dir / "block_summary.csv", index=False)
    condition_summary.to_csv(output_dir / "condition_response_summary.csv", index=False)
    comparison_summary.to_csv(output_dir / "block1_vs_block2_comparison_summary.csv", index=False)

    stim_duration_lookup = {
        "block1_single_0": 0.343,
        "block2_standard_sequence": float(np.nanmedian(trials.loc[trials["Analysis_Condition"].eq("block2_standard_sequence"), "Logger_Duration_sec"])),
        "block1_single_45": 0.343,
        "block2_orientation_45": 0.343,
        "block1_single_90": 0.343,
        "block2_orientation_90": 0.343,
        "block1_omission": 0.343,
        "block2_omission": 0.343,
        "block1_halt": 0.343,
        "block2_halt": 0.343,
    }

    pdf_path = output_dir / "block1_vs_block2_aligned_responses.pdf"
    with PdfPages(pdf_path) as pdf:
        plot_comparison_page(
            pdf,
            "Orientation 0: Block 1 single vs Block 2 standard sequence onset",
            [response_by_condition["block1_single_0"], response_by_condition["block2_standard_sequence"]],
            stim_duration_lookup,
        )
        plot_comparison_page(
            pdf,
            "Orientation 45: Block 1 single vs Block 2 motor orientation oddball",
            [response_by_condition["block1_single_45"], response_by_condition["block2_orientation_45"]],
            stim_duration_lookup,
        )
        plot_comparison_page(
            pdf,
            "Orientation 90: Block 1 single vs Block 2 motor orientation oddball",
            [response_by_condition["block1_single_90"], response_by_condition["block2_orientation_90"]],
            stim_duration_lookup,
        )
        plot_comparison_page(
            pdf,
            "Omission: Block 1 control vs Block 2 motor omission",
            [response_by_condition["block1_omission"], response_by_condition["block2_omission"]],
            stim_duration_lookup,
        )
        plot_comparison_page(
            pdf,
            "Halt: Block 1 control vs Block 2 motor halt",
            [response_by_condition["block1_halt"], response_by_condition["block2_halt"]],
            stim_duration_lookup,
        )
        plot_comparison_page(
            pdf,
            "Block 2 orientation oddballs",
            [response_by_condition["block2_orientation_45"], response_by_condition["block2_orientation_90"]],
            stim_duration_lookup,
        )

    return {
        "output_dir": output_dir,
        "pdf_path": pdf_path,
        "trials": trials,
        "condition_summary": condition_summary,
        "comparison_summary": comparison_summary,
        "block_summary": block_summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VW01 Block 1 vs Block 2 closed-loop analysis.")
    parser.add_argument("--raw-session", type=Path, default=DEFAULT_RAW_SESSION)
    parser.add_argument("--bonsai-dir", type=Path, default=DEFAULT_BONSAI_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_analysis(
        raw_session=args.raw_session,
        bonsai_dir=args.bonsai_dir,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
    )
    print(f"Output directory: {result['output_dir']}")
    print(f"PDF: {result['pdf_path']}")
    print("Condition summary:")
    print(result["condition_summary"].to_string(index=False))
    print("Comparisons:")
    print(result["comparison_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
