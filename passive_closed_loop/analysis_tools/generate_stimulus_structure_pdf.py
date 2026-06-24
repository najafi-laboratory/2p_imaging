"""Generate an Allen-style stimulus structure and verification PDF."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


GITHUB_EXAMPLE_URL = (
    "https://raw.githubusercontent.com/AllenNeuralDynamics/"
    "openscope-community-predictive-processing/main/code/stimulus-control/"
    "src/Mindscope/examples/sensorimotor_mismatch_example.csv"
)
GITHUB_PHASE_URL = (
    "https://raw.githubusercontent.com/AllenNeuralDynamics/"
    "openscope-community-predictive-processing/main/code/stimulus-control/"
    "src/Mindscope/running_phases/"
    "250815165708_366122_3dd8df03-615d-4633-a5e3-a6bb3635c3b7_running_phase_wheel.csv"
)

BLOCK_COLORS = {
    "standard_control": "#b8d8eb",
    "motor_oddball": "#f3b562",
    "sequential_control_block": "#d9d9d9",
    "jitter_control": "#e8c99b",
    "open_loop_prerecorded": "#b9b6dc",
    "movie": "#8fc17a",
}
ODDBALL_COLORS = {
    "motor_orientation_45": "#d62728",
    "motor_orientation_90": "#ff7f0e",
    "motor_halt": "#6a3d9a",
    "motor_omission": "#2ca25f",
    "halt": "#8c6bb1",
    "omission": "#31a354",
}


def normalize_expected(path: Path) -> pd.DataFrame:
    expected = pd.read_csv(path)
    expected = expected.rename(columns={column: column.replace("_", "") for column in expected.columns})
    return expected


def load_events(stim_path: Path, logger_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    stim = pd.read_csv(stim_path)
    logger = pd.read_csv(logger_path)
    values = logger["Value"].astype(str)

    starts = logger.loc[values.str.startswith("StimStart-"), ["Frame", "Timestamp", "Value"]].copy()
    starts["Id"] = starts["Value"].str.replace("StimStart-", "", regex=False)
    starts = starts.rename(columns={"Frame": "StartFrame", "Timestamp": "StartSec"}).drop(columns="Value")

    ends = logger.loc[values.str.startswith("StimEnd-"), ["Frame", "Timestamp", "Value"]].copy()
    ends["Id"] = ends["Value"].str.replace("StimEnd-", "", regex=False)
    ends = ends.rename(columns={"Frame": "EndFrame", "Timestamp": "EndSec"}).drop(columns="Value")

    events = stim.merge(starts, on="Id", how="left").merge(ends, on="Id", how="left")
    events["OrientationDeg"] = np.rad2deg(pd.to_numeric(events["Orientation"], errors="coerce"))
    events["TimeMin"] = events["StartSec"] / 60
    return events, logger


def semantic_match(expected: pd.DataFrame, observed: pd.DataFrame, block: int) -> tuple[bool, str]:
    left = expected.loc[expected["BlockNumber"].eq(block)].reset_index(drop=True)
    right = observed.loc[observed["BlockNumber"].eq(block)].reset_index(drop=True).copy()
    right["Orientation"] = np.rad2deg(pd.to_numeric(right["Orientation"], errors="coerce"))

    if len(left) != len(right):
        return False, f"row count differs ({len(left):,} expected, {len(right):,} recorded)"

    columns = [
        "BlockNumber", "BlockLabel", "TrialNumber", "SequenceNumber", "TrialInSequence",
        "Contrast", "Delay", "Duration", "Orientation", "SpatialFrequency",
        "TemporalFrequency", "TrialType", "BlockType",
    ]
    for column in columns:
        if pd.api.types.is_numeric_dtype(left[column]):
            same = np.isclose(
                pd.to_numeric(left[column], errors="coerce"),
                pd.to_numeric(right[column], errors="coerce"),
                equal_nan=True,
            )
        else:
            same = left[column].astype(str).eq(right[column].astype(str)).to_numpy()
        if not np.all(same):
            return False, f"{column} differs in {int(np.sum(~same)):,} rows"

    if block == 2:
        phase_ok = pd.to_numeric(right["Phase"], errors="coerce").notna().all()
        return bool(phase_ok), "planned wheel phase replaced by recorded numeric phase"

    expected_phase = pd.to_numeric(left["Phase"], errors="coerce")
    observed_phase = pd.to_numeric(right["Phase"], errors="coerce")
    phase_ok = np.allclose(expected_phase, observed_phase, equal_nan=True)
    return bool(phase_ok), "all semantic fields and row order match"


def block_bounds(events: pd.DataFrame) -> pd.DataFrame:
    bounds = (
        events.groupby(["BlockNumber", "BlockLabel", "BlockType"], as_index=False)
        .agg(
            StartSec=("StartSec", "min"),
            EndSec=("EndSec", "max"),
            DeclaredDurationSec=("Duration", "sum"),
            Rows=("Id", "size"),
            MissingStarts=("StartSec", lambda values: int(values.isna().sum())),
            MissingEnds=("EndSec", lambda values: int(values.isna().sum())),
        )
        .sort_values("BlockNumber")
    )
    bounds["PlotEndSec"] = bounds["EndSec"].fillna(
        bounds["StartSec"] + bounds["DeclaredDurationSec"]
    )
    return bounds


def add_block_boundaries(axis: plt.Axes, blocks: pd.DataFrame) -> None:
    for start in blocks["StartSec"].dropna():
        axis.axvline(start / 60, color="black", linewidth=0.6, alpha=0.45)
    final_end = blocks["PlotEndSec"].dropna().max()
    if np.isfinite(final_end):
        axis.axvline(final_end / 60, color="black", linewidth=0.6, alpha=0.45)


def structure_page(pdf: PdfPages, events: pd.DataFrame, blocks: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(15, 11))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 2.2, 1.4], hspace=0.34)
    ax_blocks = fig.add_subplot(grid[0])
    ax_orient = fig.add_subplot(grid[1])
    ax_events = fig.add_subplot(grid[2])

    session_end_min = blocks["PlotEndSec"].max() / 60
    for block in blocks.itertuples():
        start = block.StartSec / 60
        end = block.PlotEndSec / 60
        width = end - start
        ax_blocks.add_patch(
            Rectangle(
                (start, 0),
                width,
                1,
                facecolor=BLOCK_COLORS.get(block.BlockType, "#eeeeee"),
                edgecolor="black",
                linewidth=0.8,
            )
        )
        label = f"{block.BlockNumber}: {block.BlockLabel}\n{block.BlockType}\n{width:.1f} min"
        ax_blocks.text(start + width / 2, 0.5, label, ha="center", va="center", fontsize=7)

    ax_blocks.set(xlim=(0, session_end_min), ylim=(0, 1), ylabel="Blocks")
    ax_blocks.set_yticks([])
    ax_blocks.set_title("Recorded Block Structure", loc="left", weight="bold")

    visual = events.loc[
        ~events["TrialType"].isin(["omission", "motor_omission"])
        & events["StartSec"].notna()
    ]
    for block_number, group in visual.groupby("BlockNumber"):
        ax_orient.scatter(
            group["TimeMin"],
            group["OrientationDeg"],
            s=2 if len(group) > 2000 else 8,
            alpha=0.45,
            label=f"Block {block_number}",
        )
    add_block_boundaries(ax_orient, blocks)
    ax_orient.set(
        xlim=(0, session_end_min),
        ylabel="Orientation (degrees)",
        title="Performed Grating Orientations Over Time",
    )
    ax_orient.set_yticks(np.arange(0, 316, 22.5))
    ax_orient.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    event_types = [name for name in ODDBALL_COLORS if name in set(events["TrialType"])]
    for row_index, event_type in enumerate(event_types):
        subset = events.loc[events["TrialType"].eq(event_type) & events["StartSec"].notna()]
        ax_events.scatter(
            subset["TimeMin"],
            np.full(len(subset), row_index),
            color=ODDBALL_COLORS[event_type],
            s=14,
            label=event_type,
        )
    add_block_boundaries(ax_events, blocks)
    ax_events.set(
        xlim=(0, session_end_min),
        xlabel="Logger time (minutes)",
        ylabel="Event type",
        title="Recorded Halt, Omission, And Orientation-Oddball Events",
    )
    ax_events.set_yticks(range(len(event_types)), event_types)
    ax_events.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    for axis in (ax_blocks, ax_orient, ax_events):
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("VW01 2026-05-22 Sensory-Motor Paradigm Structure", fontsize=16, weight="bold")
    fig.text(
        0.01,
        0.01,
        "Allen-style structure view generated from orientations_orientations0.csv joined to logger StimStart/StimEnd timestamps.",
        fontsize=8,
    )
    fig.subplots_adjust(right=0.82, top=0.93, bottom=0.06)
    pdf.savefig(fig)
    plt.close(fig)


def verification_page(
    pdf: PdfPages,
    events: pd.DataFrame,
    blocks: pd.DataFrame,
    expected: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for block in blocks.itertuples():
        expected_block = expected.loc[expected["BlockNumber"].eq(block.BlockNumber)]
        observed_block = events.loc[events["BlockNumber"].eq(block.BlockNumber)]
        match, note = semantic_match(expected, events, block.BlockNumber)
        expected_counts = expected_block["TrialType"].value_counts().sort_index().to_dict()
        observed_counts = observed_block["TrialType"].value_counts().sort_index().to_dict()
        timing_ok = block.MissingStarts == 0 and (
            block.MissingEnds == 0 or block.BlockType == "movie"
        )
        rows.append(
            {
                "Block": block.BlockNumber,
                "Label": block.BlockLabel,
                "Expected rows": len(expected_block),
                "Recorded rows": len(observed_block),
                "Type counts match": expected_counts == observed_counts,
                "Stim timing": "PASS" if timing_ok else "FAIL",
                "Semantic/order check": "PASS" if match else "FAIL",
                "Note": note,
            }
        )
    verification = pd.DataFrame(rows)

    fig, axis = plt.subplots(figsize=(15, 8.5))
    axis.axis("off")
    axis.set_title(
        "GitHub Example Versus Recorded Performed-Stimulus Verification",
        loc="left",
        fontsize=15,
        weight="bold",
        pad=18,
    )
    display_table = verification.copy()
    display_table["Type counts match"] = display_table["Type counts match"].map({True: "PASS", False: "FAIL"})
    display_table = display_table.rename(
        columns={
            "Expected rows": "Expected",
            "Recorded rows": "Recorded",
            "Type counts match": "Types",
            "Stim timing": "Timing",
            "Semantic/order check": "Semantics",
        }
    )
    table = axis.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        bbox=[0, 0.26, 1, 0.68],
        colWidths=[0.05, 0.16, 0.08, 0.08, 0.08, 0.08, 0.10, 0.37],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#dddddd")
            cell.set_text_props(weight="bold")
        elif column in (4, 5, 6):
            text = cell.get_text().get_text()
            cell.set_facecolor("#d9ead3" if text == "PASS" else "#f4cccc")

    expected_extra = expected.loc[expected["BlockNumber"] > blocks["BlockNumber"].max()]
    notes = [
        "Blocks 1-6 match the official Mindscope sensorimotor_mismatch example in row count, condition counts, semantic values, and order.",
        "Block 2 is expected to differ only in Phase: the planned value 'wheel' is replaced online by the numeric wheel-derived phase.",
        "The GitHub example has two Trippy rows; the recording has one and its logger StimEnd is missing.",
    ]
    if not expected_extra.empty:
        labels = ", ".join(
            f"Block {number} {label}"
            for number, label in expected_extra[["BlockNumber", "BlockLabel"]].drop_duplicates().itertuples(index=False)
        )
        notes.append(f"The GitHub example then contains {labels}; these blocks are absent from the recorded output.")
    axis.text(0, 0.19, "\n".join(f"- {note}" for note in notes), va="top", fontsize=9)
    axis.text(
        0,
        0.03,
        "PASS verifies Bonsai commands/logger timing, not physical monitor output. A usable hardware photodiode is required for independent display validation.",
        fontsize=9,
        weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)
    return verification


def block6_page(pdf: PdfPages, events: pd.DataFrame, phase_source: pd.DataFrame) -> dict:
    block6 = events.loc[events["BlockNumber"].eq(6)].sort_values("TrialNumber").reset_index(drop=True)
    recorded_phase = pd.to_numeric(block6["Phase"], errors="coerce").to_numpy()
    source_phase = pd.to_numeric(phase_source["Phase_Radians"], errors="coerce").to_numpy()

    signature = recorded_phase[:20]
    candidates = np.flatnonzero(np.isclose(source_phase, signature[0], rtol=0, atol=1e-12))
    matches = [
        int(start)
        for start in candidates
        if start + len(recorded_phase) <= len(source_phase)
        and np.allclose(source_phase[start : start + len(signature)], signature, rtol=0, atol=1e-12)
        and np.allclose(source_phase[start : start + len(recorded_phase)], recorded_phase, rtol=0, atol=1e-12)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Block 6 source match, found {matches}")

    start = matches[0]
    stop = start + len(recorded_phase)
    source_segment = source_phase[start:stop]
    max_error = float(np.max(np.abs(source_segment - recorded_phase)))
    source_start_time = float(phase_source.iloc[start]["Timestamp"])
    source_end_time = float(phase_source.iloc[stop - 1]["Timestamp"])

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.33)
    axis = fig.add_subplot(grid[0])
    info_axis = fig.add_subplot(grid[1])
    info_axis.axis("off")

    sample_index = np.arange(len(recorded_phase))
    axis.plot(sample_index, source_segment, color="#666666", linewidth=1.2, label="Repository source segment")
    axis.plot(sample_index, recorded_phase, color="#1f77b4", linewidth=0.55, alpha=0.8, label="Recorded Block 6 Phase")
    oddballs = block6["TrialType"].ne("prerecorded")
    axis.scatter(
        sample_index[oddballs],
        recorded_phase[oddballs],
        s=20,
        color=[ODDBALL_COLORS.get(value, "black") for value in block6.loc[oddballs, "TrialType"]],
        label="Injected mismatch events",
        zorder=3,
    )
    axis.set(
        xlabel="Block 6 output row / nominal 30 Hz phase sample",
        ylabel="Phase (radians)",
        title="Block 6 Recorded Phase Exactly Matches Repository Prerecorded Source",
    )
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    mismatch_counts = block6.loc[oddballs, "TrialType"].value_counts().sort_index()
    details = [
        "Source file:",
        "running_phases/250815165708_366122_3dd8df03-615d-4633-a5e3-a6bb3635c3b7_running_phase_wheel.csv",
        "",
        f"Unique matching source interval: zero-based rows {start}:{stop}",
        f"Source Timestamp values: {source_start_time:.6f} to {source_end_time:.6f} s",
        f"Rows compared: {len(recorded_phase):,}; maximum absolute phase error: {max_error:.3g}",
        "",
        "Generation logic: select a contiguous 6.4-minute nominal 30 Hz segment using deterministic seed variant=0,",
        "then replace 8 rows of each mismatch type while preserving the source phase values.",
        "Mismatch counts: " + ", ".join(f"{name}={count}" for name, count in mismatch_counts.items()),
        "",
        "Conclusion: Block 6 replays a phase trajectory extracted from an earlier repository wheel recording;",
        "it is not a replay of VW01's May 22 Block 2 wheel trajectory.",
    ]
    info_axis.text(0, 1, "\n".join(details), va="top", fontsize=10)
    fig.subplots_adjust(right=0.82, top=0.93, bottom=0.08)
    pdf.savefig(fig)
    plt.close(fig)

    return {
        "source_start_row": start,
        "source_stop_row_exclusive": stop,
        "source_timestamp_start": source_start_time,
        "source_timestamp_end": source_end_time,
        "rows_compared": len(recorded_phase),
        "max_abs_phase_error": max_error,
    }


def wheel_phase_page(pdf: PdfPages, events: pd.DataFrame, logger: pd.DataFrame) -> dict:
    wheel = logger.loc[logger["Value"].astype(str).str.startswith("Wheel-Deg-")].copy()
    wheel["WheelDeg"] = (
        wheel["Value"].str.replace("Wheel-Deg-", "", regex=False).str.strip().astype(float)
    )
    wheel = wheel.sort_values("Timestamp")

    standard = events.loc[
        events["BlockNumber"].eq(2)
        & events["TrialType"].eq("standard")
        & events["StartSec"].notna()
    ].sort_values("StartSec")
    wheel_at_update = np.interp(
        standard["StartSec"].to_numpy(),
        wheel["Timestamp"].to_numpy(),
        wheel["WheelDeg"].to_numpy(),
    )
    phase = pd.to_numeric(standard["Phase"], errors="coerce").to_numpy()
    wheel_change = np.diff(wheel_at_update)
    phase_change = np.angle(np.exp(1j * np.diff(phase)))

    valid = (
        (np.abs(wheel_change) > 0.001)
        & (np.abs(wheel_change) < 20)
        & (np.abs(phase_change) < 2)
    )
    gain = float(
        np.linalg.lstsq(wheel_change[valid, None], phase_change[valid], rcond=None)[0][0]
    )
    predicted_change = gain * wheel_change[valid]
    residual = phase_change[valid] - predicted_change
    correlation = float(np.corrcoef(wheel_change[valid], phase_change[valid])[0, 1])
    r_squared = float(
        1
        - np.sum(residual**2)
        / np.sum((phase_change[valid] - np.mean(phase_change[valid])) ** 2)
    )

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.38, wspace=0.28)
    trace_axis = fig.add_subplot(grid[0, :])
    scatter_axis = fig.add_subplot(grid[1, 0])
    info_axis = fig.add_subplot(grid[1, 1])
    info_axis.axis("off")

    display_rows = min(5000, len(standard))
    trace_axis.plot(
        standard["StartSec"].to_numpy()[:display_rows] / 60,
        phase[:display_rows],
        color="#1f77b4",
        linewidth=0.7,
        label="Recorded visual phase",
    )
    scaled_wheel = np.mod(
        gain * wheel_at_update[:display_rows] + phase[0] - gain * wheel_at_update[0],
        2 * np.pi,
    )
    trace_axis.plot(
        standard["StartSec"].to_numpy()[:display_rows] / 60,
        scaled_wheel,
        color="#d62728",
        linewidth=0.7,
        alpha=0.65,
        label="Scaled wheel trajectory",
    )
    trace_axis.set(
        xlabel="Logger time (minutes)",
        ylabel="Wrapped phase (radians)",
        title="Block 2 Wheel And Standard-Grating Phase",
    )
    trace_axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    sample = np.linspace(0, np.sum(valid) - 1, min(8000, np.sum(valid))).astype(int)
    scatter_axis.scatter(
        wheel_change[valid][sample],
        phase_change[valid][sample],
        s=3,
        alpha=0.2,
        color="#444444",
    )
    x_line = np.linspace(
        np.percentile(wheel_change[valid], 1),
        np.percentile(wheel_change[valid], 99),
        100,
    )
    scatter_axis.plot(x_line, gain * x_line, color="#d62728", linewidth=1.5)
    scatter_axis.set(
        xlabel="Wheel change between phase updates (degrees)",
        ylabel="Wrapped phase change (radians)",
        title="Local Wheel-to-Phase Mapping",
    )

    metrics = [
        f"Block 2 standard updates: {len(standard):,}",
        f"Wheel logger samples: {len(wheel):,}",
        f"Valid local update pairs: {int(np.sum(valid)):,}",
        "",
        f"Correlation: {correlation:.4f}",
        f"R-squared: {r_squared:.4f}",
        f"Fitted gain: {gain:.6f} rad phase / wheel degree",
        f"             {np.rad2deg(gain):.4f} deg phase / wheel degree",
        f"Median absolute residual: {np.median(np.abs(residual)):.6f} rad",
        f"95th percentile residual: {np.percentile(np.abs(residual), 95):.6f} rad",
        f"Residual RMSE: {np.sqrt(np.mean(residual**2)):.6f} rad",
        "",
        "PASS criterion used here: R-squared >= 0.95 and correlation >= 0.95.",
        "Result: PASS" if r_squared >= 0.95 and correlation >= 0.95 else "Result: FAIL",
        "",
        "The gain is estimated from this recording. The workflow externalizes",
        "wheel/screen coupling parameters, so the configured gain cannot be",
        "recovered from the CSV files alone.",
    ]
    info_axis.text(0, 1, "\n".join(metrics), va="top", fontsize=10)

    for axis in (trace_axis, scatter_axis):
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Block 2 Wheel-to-Visual-Phase Verification", fontsize=16, weight="bold")
    fig.subplots_adjust(right=0.82, top=0.92, bottom=0.08)
    pdf.savefig(fig)
    plt.close(fig)

    return {
        "standard_updates": len(standard),
        "wheel_samples": len(wheel),
        "valid_update_pairs": int(np.sum(valid)),
        "correlation": correlation,
        "r_squared": r_squared,
        "gain_rad_per_wheel_degree": gain,
        "gain_phase_degrees_per_wheel_degree": float(np.rad2deg(gain)),
        "median_abs_residual_rad": float(np.median(np.abs(residual))),
        "p95_abs_residual_rad": float(np.percentile(np.abs(residual), 95)),
        "residual_rmse_rad": float(np.sqrt(np.mean(residual**2))),
        "status": "PASS" if r_squared >= 0.95 and correlation >= 0.95 else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stim-csv", required=True, type=Path)
    parser.add_argument("--logger-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_path = args.output_dir / "github_sensorimotor_mismatch_example.csv"
    phase_path = args.output_dir / "github_prerecorded_running_phase_source.csv"
    if not expected_path.exists():
        urlretrieve(GITHUB_EXAMPLE_URL, expected_path)
    if not phase_path.exists():
        urlretrieve(GITHUB_PHASE_URL, phase_path)

    events, logger = load_events(args.stim_csv, args.logger_csv)
    expected = normalize_expected(expected_path)
    phase_source = pd.read_csv(phase_path)
    blocks = block_bounds(events)

    pdf_path = args.output_dir / "VW01_20260522_stimulus_structure_verification.pdf"
    with PdfPages(pdf_path) as pdf:
        structure_page(pdf, events, blocks)
        verification = verification_page(pdf, events, blocks, expected)
        wheel_phase = wheel_phase_page(pdf, events, logger)
        provenance = block6_page(pdf, events, phase_source)

    verification.to_csv(args.output_dir / "block_structure_verification.csv", index=False)
    pd.DataFrame([wheel_phase]).to_csv(args.output_dir / "block2_wheel_phase_verification.csv", index=False)
    pd.DataFrame([provenance]).to_csv(args.output_dir / "block6_phase_provenance.csv", index=False)
    print(pdf_path)


if __name__ == "__main__":
    main()
