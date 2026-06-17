#!/usr/bin/env python3
"""Generate static and interactive preprocessing QC summaries for one session."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils_2p.preprocessing_qc_pipeline import QC_PRESETS
from utils_2p.roi_labels import (
    dff_qc_metrics,
    load_iscell,
    map_qc_to_suite2p_rois,
    morphology_exclusion_reasons,
    roi_morphology_metrics,
    suite2p_stat_fingerprint,
)


PDF_NAME_TEMPLATE = "{session_name}_preprocessing_summary.pdf"
HTML_NAME_TEMPLATE = "{session_name}_interactive_fov_roi_dff.html"


def _load_npy(paths: list[Path], allow_pickle: bool = False) -> Any:
    for path in paths:
        if path.exists():
            return np.load(path, allow_pickle=allow_pickle)
    searched = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not find any of:\n{searched}")


def _load_ops(session_dir: Path) -> dict[str, Any]:
    ops = _load_npy(
        [session_dir / "ops.npy", session_dir / "suite2p" / "plane0" / "ops.npy"],
        allow_pickle=True,
    ).item()
    ops["save_path0"] = str(session_dir)
    return ops


def _load_stat(session_dir: Path) -> np.ndarray:
    return _load_npy(
        [session_dir / "qc_results" / "stat.npy", session_dir / "suite2p" / "plane0" / "stat.npy"],
        allow_pickle=True,
    )


def _load_roi_mask(session_dir: Path) -> np.ndarray:
    masks_h5 = session_dir / "masks.h5"
    if masks_h5.exists():
        with h5py.File(masks_h5, "r") as h5:
            if "masks_func" in h5:
                return np.asarray(h5["masks_func"])
    return _load_npy([session_dir / "qc_results" / "masks.npy", session_dir / "masks.npy"])


def _load_masks_h5_image(session_dir: Path, key: str) -> np.ndarray | None:
    path = session_dir / "masks.h5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as h5:
        if key not in h5:
            return None
        return np.asarray(h5[key])


def _load_dff(session_dir: Path) -> np.ndarray:
    for path in [session_dir / "dff.h5", session_dir / "qc_results" / "dff.h5"]:
        if path.exists():
            with h5py.File(path, "r") as h5:
                if "dff" not in h5:
                    raise KeyError(f"{path} does not contain dataset 'dff'")
                return np.asarray(h5["dff"], dtype=np.float32)
    raise FileNotFoundError(f"Could not find dff.h5 in {session_dir} or {session_dir / 'qc_results'}")


def _load_raw_fractional_dff(session_dir: Path, ops: dict[str, Any]) -> np.ndarray:
    fluo_path = session_dir / "qc_results" / "fluo.npy"
    neuropil_path = session_dir / "qc_results" / "neuropil.npy"
    if not fluo_path.exists() or not neuropil_path.exists():
        raise FileNotFoundError("Missing qc_results/fluo.npy or qc_results/neuropil.npy")

    fluo = np.load(fluo_path, allow_pickle=True).astype(np.float32, copy=False)
    neuropil = np.load(neuropil_path, allow_pickle=True).astype(np.float32, copy=False)
    signal = fluo - float(ops.get("neucoeff", 0.7)) * neuropil
    baseline = gaussian_filter(signal, [0.0, 600.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = (signal - baseline) / baseline
    dff[~np.isfinite(dff)] = np.nan
    return dff.astype(np.float32, copy=False)


def _load_display_dff(session_dir: Path, ops: dict[str, Any]) -> tuple[np.ndarray, str]:
    try:
        return _load_raw_fractional_dff(session_dir, ops), "raw dF/F"
    except FileNotFoundError:
        return _load_dff(session_dir), "saved dF/F dataset"


def _load_offsets(session_dir: Path, ops: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = session_dir / "move_offset.h5"
    if path.exists():
        with h5py.File(path, "r") as h5:
            if "xoff" in h5 and "yoff" in h5:
                return np.asarray(h5["xoff"]), np.asarray(h5["yoff"])
    xoff = np.asarray(ops.get("xoff", []), dtype=float)
    yoff = np.asarray(ops.get("yoff", []), dtype=float)
    return xoff, yoff


def _load_qc_parameters(session_dir: Path) -> dict[str, Any] | None:
    path = session_dir / "qc_results" / "qc_parameters.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pipeline_parameters(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "preprocessing_pipeline_parameters.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _target_structure(pipeline_parameters: dict[str, Any], qc_parameters: dict[str, Any] | None) -> str:
    target = pipeline_parameters.get("target_structure")
    if target:
        return str(target)
    source = str((qc_parameters or {}).get("source", "")).lower()
    if "cerebellum" in source:
        return "cerebellum_lax"
    if "dendrite" in source:
        return "dendrite"
    if "neuron" in source:
        return "neuron"
    return "unknown"


def _load_suite2p_dff(session_dir: Path, ops: dict[str, Any]) -> np.ndarray:
    plane_dir = session_dir / "suite2p" / "plane0"
    fluo = np.load(plane_dir / "F.npy", allow_pickle=False).astype(np.float32, copy=False)
    neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=False).astype(np.float32, copy=False)
    signal = fluo - float(ops.get("neucoeff", 0.7)) * neuropil
    baseline = gaussian_filter(signal, [0.0, 600.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = (signal - baseline) / baseline
    dff[~np.isfinite(dff)] = np.nan
    return dff.astype(np.float32, copy=False)


def _stat_to_mask(stat: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.int32)
    for index, entry in enumerate(stat):
        ypix = np.asarray(entry.get("ypix", []), dtype=int)
        xpix = np.asarray(entry.get("xpix", []), dtype=int)
        mask[ypix, xpix] = index + 1
    return mask


def _robust_limits(image: np.ndarray, lower: float = 1, upper: float = 99.7) -> tuple[float, float]:
    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _normalize_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = _robust_limits(image)
    return (np.clip((image - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _array_png_data_uri(array: np.ndarray) -> str:
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _channel_png_data_uri(image: np.ndarray, color: str) -> str:
    return _array_png_data_uri(_channel_rgb(image, color))


def _channel_rgb(image: np.ndarray, color: str) -> np.ndarray:
    uint8 = _normalize_uint8(image)
    rgb = np.zeros((*uint8.shape, 3), dtype=np.uint8)
    if color == "red":
        rgb[..., 0] = uint8
    elif color == "green":
        rgb[..., 1] = uint8
    else:
        rgb[...] = uint8[..., None]
    return rgb


def _mask_data_uri(mask: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    masked = np.ma.masked_where(mask <= 0, mask)
    ax.imshow(masked, cmap="turbo", interpolation="nearest")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, transparent=True)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _boundary(mask: np.ndarray) -> np.ndarray:
    labels = np.asarray(mask)
    positive = labels > 0
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return boundary & positive


def _pixel_runs_path(xpix: np.ndarray, ypix: np.ndarray) -> str:
    segments: list[str] = []
    if not xpix.size:
        return ""
    for y in np.unique(ypix):
        row_x = np.sort(np.unique(xpix[ypix == y]))
        start = int(row_x[0])
        previous = start
        for x in row_x[1:]:
            x = int(x)
            if x != previous + 1:
                width = previous - start + 1
                segments.append(f"M{start} {int(y)}h{width}v1h-{width}z")
                start = x
            previous = x
        width = previous - start + 1
        segments.append(f"M{start} {int(y)}h{width}v1h-{width}z")
    return "".join(segments)


def _roi_table(stat: np.ndarray, mask: np.ndarray, n_rois: int) -> list[dict[str, float | int | str]]:
    rois: list[dict[str, float | int | str]] = []
    for idx in range(n_rois):
        entry = stat[idx]
        ypix = np.asarray(entry.get("ypix", []), dtype=int)
        xpix = np.asarray(entry.get("xpix", []), dtype=int)
        rois.append({"roi": idx, "path": _pixel_runs_path(xpix, ypix), "npix": int(xpix.size)})
    return rois


def _float32_b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array.astype("<f4", copy=False)).tobytes()).decode("ascii")


def _float64_b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array.astype("<f8", copy=False)).tobytes()).decode("ascii")


def _plot_channel_image(ax: plt.Axes, image: np.ndarray | None, title: str, color: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    if image is None:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
        return
    ax.imshow(_channel_rgb(image, color), interpolation="nearest")


def _plot_channel_overlay(ax: plt.Axes, image: np.ndarray | None, mask: np.ndarray, title: str, color: str) -> None:
    _plot_channel_image(ax, image, title, color)
    if image is None:
        return
    outline = np.ma.masked_where(~_boundary(mask), _boundary(mask))
    ax.imshow(outline, cmap="gray", vmin=0, vmax=1, alpha=1.0, interpolation="nearest")


def _build_fov_figure(
    *,
    session_name: str,
    n_rois: int,
    mean_green: np.ndarray,
    max_green: np.ndarray | None,
    mean_red: np.ndarray | None,
    max_red: np.ndarray | None,
    mask: np.ndarray,
) -> plt.Figure:
    if mean_red is None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.7), constrained_layout=True)
        fig.suptitle(
            f"{session_name} FOV summary ({n_rois} ROIs)\n"
            "* Anatomical channel not present for this recording",
            fontsize=14,
        )
        _plot_channel_image(axes[0], max_green, "Functional max projection", "green")
        _plot_channel_image(axes[1], mean_green, "Functional mean projection", "green")
        _plot_channel_overlay(axes[2], mean_green, mask, "Functional mean + white outlines", "green")
        axes[3].set_title("Raw white outlines", fontsize=10)
        axes[3].set_axis_off()
        axes[3].imshow(np.zeros_like(mask), cmap="gray", vmin=0, vmax=1)
        axes[3].imshow(
            np.ma.masked_where(~_boundary(mask), _boundary(mask)),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        return fig

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    fig.suptitle(f"{session_name} FOV summary ({n_rois} ROIs)", fontsize=14)

    _plot_channel_image(fig.add_subplot(gs[0, 0]), max_green, "Functional max projection", "green")
    _plot_channel_image(fig.add_subplot(gs[0, 1]), mean_green, "Functional mean projection", "green")
    _plot_channel_image(fig.add_subplot(gs[0, 2]), max_red, "Anatomical max projection", "red")
    _plot_channel_image(fig.add_subplot(gs[0, 3]), mean_red, "Anatomical mean projection", "red")

    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Raw white outlines", fontsize=10)
    ax.set_axis_off()
    ax.imshow(np.zeros_like(mask), cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(~_boundary(mask), _boundary(mask)), cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    _plot_channel_overlay(fig.add_subplot(gs[1, 1]), mean_green, mask, "Functional mean + white outlines", "green")

    ax = fig.add_subplot(gs[1, 2])
    ax.set_axis_off()

    _plot_channel_overlay(fig.add_subplot(gs[1, 3]), mean_red, mask, "Anatomical mean + white outlines", "red")

    return fig


def _build_motion_figure(
    *,
    session_name: str,
    n_rois: int,
    xoff: np.ndarray,
    yoff: np.ndarray,
    frame_rate: float,
) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.9])
    fig.suptitle(f"{session_name} motion correction summary ({n_rois} ROIs)", fontsize=14)

    ax = fig.add_subplot(gs[0, :])
    ax.set_title("X shift per frame", fontsize=10)
    if xoff.size:
        frames = np.arange(xoff.size)
        ax.plot(frames, xoff, lw=0.25, color="black")
        ax.axhline(0, color="#9ca3af", lw=0.7)
        ax.set_xlabel("Frame")
        ax.set_ylabel("X shift (pixels)")
    else:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)

    ax = fig.add_subplot(gs[1, :])
    ax.set_title("Y shift per frame", fontsize=10)
    if yoff.size:
        frames = np.arange(yoff.size)
        ax.plot(frames, yoff, lw=0.25, color="black")
        ax.axhline(0, color="#9ca3af", lw=0.7)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Y shift (pixels)")
    else:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)

    ax = fig.add_subplot(gs[2, 0])
    ax.set_title("Offset distributions", fontsize=10)
    if xoff.size and yoff.size:
        values = np.concatenate([np.asarray(xoff, dtype=float), np.asarray(yoff, dtype=float)])
        finite = values[np.isfinite(values)]
        if finite.size:
            lo, hi = np.percentile(finite, [0.5, 99.5])
            if hi <= lo:
                lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
            bins = np.linspace(lo, hi, 80)
            ax.hist(xoff, bins=bins, density=True, histtype="stepfilled", alpha=0.28, color="#1f77b4", label="x")
            ax.hist(yoff, bins=bins, density=True, histtype="stepfilled", alpha=0.28, color="#d62728", label="y")
        ax.set_xlabel("Shift (pixels)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)

    ax = fig.add_subplot(gs[2, 1])
    ax.set_title("Shift cumulative distributions", fontsize=10)
    if xoff.size and yoff.size:
        for values, label, color in [(xoff, "x", "#1f77b4"), (yoff, "y", "#d62728")]:
            values = np.asarray(values, dtype=float)
            if values.size:
                sorted_values = np.sort(values)
                ax.plot(sorted_values, np.linspace(0, 1, sorted_values.size), label=label, color=color)
        ax.set_xlabel("Shift (pixels)")
        ax.set_ylabel("Cumulative fraction")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)

    ax = fig.add_subplot(gs[2, 2])
    ax.set_title("Absolute shift cumulative distributions", fontsize=10)
    if xoff.size and yoff.size:
        for values, label, color in [(np.abs(xoff), "|x|", "#1f77b4"), (np.abs(yoff), "|y|", "#d62728")]:
            values = np.asarray(values, dtype=float)
            if values.size:
                sorted_values = np.sort(values)
                ax.plot(sorted_values, np.linspace(0, 1, sorted_values.size), label=label, color=color)
        ax.axvline(16, color="#6b7280", lw=0.9, ls="--", label="16 px")
        ax.axvline(20, color="#111827", lw=0.9, ls=":", label="20 px")
        ax.set_xlabel("Absolute shift (pixels)")
        ax.set_ylabel("Cumulative fraction")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)

    return fig


def _write_summary_pdf(
    output_path: Path,
    *,
    session_name: str,
    n_rois: int,
    mean_green: np.ndarray,
    max_green: np.ndarray | None,
    mean_red: np.ndarray | None,
    max_red: np.ndarray | None,
    mask: np.ndarray,
    xoff: np.ndarray,
    yoff: np.ndarray,
    frame_rate: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fov_fig = _build_fov_figure(
        session_name=session_name,
        n_rois=n_rois,
        mean_green=mean_green,
        max_green=max_green,
        mean_red=mean_red,
        max_red=max_red,
        mask=mask,
    )
    motion_fig = _build_motion_figure(
        session_name=session_name,
        n_rois=n_rois,
        xoff=xoff,
        yoff=yoff,
        frame_rate=frame_rate,
    )
    with PdfPages(output_path) as pdf:
        pdf.savefig(fov_fig, dpi=200)
        pdf.savefig(motion_fig, dpi=200)
    fov_fig.savefig(output_path.with_name("preprocessing_fov_summary.png"), dpi=200)
    motion_fig.savefig(output_path.with_name("motion_correction_summary.png"), dpi=200)
    plt.close(fov_fig)
    plt.close(motion_fig)


def _write_html(
    output_path: Path,
    *,
    session_name: str,
    mean_green: np.ndarray,
    mean_red: np.ndarray | None,
    mask: np.ndarray,
    stat: np.ndarray,
    suite2p_indices: np.ndarray,
    iscell: np.ndarray,
    suite2p_fingerprint: str,
    morphology_metrics: list[dict[str, float | int]],
    preset_exclusion_reasons: list[list[str]],
    qc_parameters: dict[str, Any] | None,
    target_structure: str,
    dff: np.ndarray,
    dff_label: str,
    frame_rate: float,
) -> None:
    rois = _roi_table(stat, mask, dff.shape[0])
    dff_metrics: list[dict[str, float | None]] = []
    for metrics in dff_qc_metrics(dff):
        dff_metrics.append(
            {
                key: (float(value) if np.isfinite(value) else None)
                for key, value in metrics.items()
            }
        )
    red_available = mean_red is not None
    image_height, image_width = np.asarray(mean_green).shape[:2]
    payload = {
        "session": session_name,
        "frameRate": float(frame_rate),
        "nRois": int(dff.shape[0]),
        "nFrames": int(dff.shape[1]),
        "imageWidth": int(image_width),
        "imageHeight": int(image_height),
        "green": _channel_png_data_uri(mean_green, "green"),
        "red": _channel_png_data_uri(mean_red, "red") if red_available else None,
        "redAvailable": red_available,
        "mask": _mask_data_uri(mask),
        "rois": rois,
        "suite2pIndices": suite2p_indices.tolist(),
        "suite2pRoiCount": int(iscell.shape[0]),
        "suite2pStatFingerprint": suite2p_fingerprint,
        "iscell": _float64_b64(iscell),
        "morphology": morphology_metrics,
        "presetExclusionReasons": preset_exclusion_reasons,
        "qcParameters": qc_parameters,
        "targetStructure": target_structure,
        "dffMetrics": dff_metrics,
        "morphologyPresets": {
            name: {
                "skewMin": values["range_skew"][0],
                "skewMax": values["range_skew"][1],
                "maxConnect": values["max_connect"],
                "aspectMin": values["range_aspect"][0],
                "aspectMax": values["range_aspect"][1],
                "footprintMin": values["range_footprint"][0],
                "footprintMax": values["range_footprint"][1],
                "compactMin": values["range_compact"][0],
                "compactMax": values["range_compact"][1],
            }
            for name, values in QC_PRESETS.items()
        },
        "dff": _float32_b64(dff),
        "dffLabel": dff_label,
    }
    fov_grid_class = "with-red" if red_available else "single-channel"
    red_panel = (
        '<div class="panel"><div class="title">Red anatomical mean</div>'
        '<div class="imagewrap"><img id="red"><svg class="overlay" preserveAspectRatio="xMidYMid meet">'
        '</svg></div></div>'
        if red_available
        else ""
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{session_name} preprocessing QC ({dff.shape[0]} ROIs)</title>
<style>
body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: #202124; background: #f6f7f8; }}
.page {{ width: min(1680px, calc(100vw - 28px)); margin: 16px auto 26px; }}
.head {{ display: flex; justify-content: space-between; gap: 14px; align-items: end; margin-bottom: 12px; }}
h1 {{ margin: 0; font-size: 21px; letter-spacing: 0; }}
.meta {{ color: #667085; font-size: 13px; text-align: right; }}
.grid {{ display: grid; gap: 10px; }}
.fov-review {{ display: grid; grid-template-columns: minmax(0, 1fr) 310px; gap: 10px; align-items: start; }}
.grid.with-red {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.grid.single-channel {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.panel {{ background: #fff; border: 1px solid #d0d5dd; border-radius: 7px; padding: 10px; box-sizing: border-box; }}
.title {{ font-size: 14px; font-weight: 700; margin-bottom: 8px; }}
.imagewrap {{ position: relative; width: 100%; aspect-ratio: 1/1; background: #111; overflow: hidden; }}
.imagewrap img, .imagewrap svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
.imagewrap img {{ object-fit: contain; image-rendering: pixelated; }}
.roi {{ fill: transparent; stroke: rgba(255,255,255,.86); stroke-width: .7; cursor: pointer; vector-effect: non-scaling-stroke; pointer-events: all; }}
.roi:hover {{ fill: rgba(6,182,212,.2); stroke: #06b6d4; stroke-width: 1.6; }}
.roi.selected {{ fill: rgba(255,255,255,.18); stroke: #ffffff; stroke-width: 2.8; }}
.controls {{ display: grid; grid-template-columns: 1fr repeat(5, auto); gap: 9px; align-items: center; margin-top: 10px; }}
.label-controls {{ display: flex; flex-direction: column; gap: 8px; align-items: stretch; }}
.label-controls .button-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
.label-controls .nav-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
.nav-button {{ font-size: 18px; font-weight: 700; }}
.filter-controls {{ display: grid; grid-template-columns: repeat(6, minmax(130px, 1fr)); gap: 8px; margin-top: 10px; align-items: end; }}
.filter-controls label {{ font-size: 12px; color: #475467; }}
.filter-controls input {{ display: block; margin-top: 3px; width: 100%; box-sizing: border-box; }}
.filter-summary {{ margin-top: 8px; color: #475467; font-size: 13px; }}
.filter-details summary {{ cursor: pointer; font-weight: 700; }}
.session-message {{ margin: 10px 0; padding: 9px 11px; background: #eef4ff; border: 1px solid #b2ccff; border-radius: 6px; font-size: 13px; }}
button, input, select {{ font: inherit; }}
button {{ border: 1px solid #d0d5dd; background: #fff; border-radius: 6px; padding: 7px 10px; cursor: pointer; }}
button.good {{ border-color: #16a34a; color: #166534; }}
button.bad {{ border-color: #dc2626; color: #991b1b; }}
button.unlabeled {{ border-color: #667085; color: #475467; }}
button.active {{ color: #fff; }}
button.good.active {{ background: #16a34a; }}
button.bad.active {{ background: #dc2626; }}
button.unlabeled.active {{ background: #667085; }}
input, select {{ border: 1px solid #d0d5dd; border-radius: 6px; padding: 7px 8px; width: 86px; }}
.filter-controls input {{ width: 100%; }}
.filter-controls select {{ width: 100%; }}
canvas {{ width: 100%; display: block; background: #fff; border: 1px solid #d0d5dd; box-sizing: border-box; }}
#stackCanvas {{ height: 560px; cursor: crosshair; }}
#traceCanvas {{ height: 250px; cursor: grab; }}
#traceCanvas.dragging {{ cursor: grabbing; }}
.plots {{ display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 10px; }}
.trace-sort {{ display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: end; margin: 6px 0 10px; padding: 8px 10px; background: #f8fafc; border: 1px solid #d0d5dd; border-radius: 6px; }}
.trace-sort label {{ font-size: 12px; color: #475467; }}
.trace-sort select {{ display: block; margin-top: 3px; min-width: 180px; width: auto; }}
.metric-formula {{ flex: 1 1 320px; }}
.metric-formula summary {{ cursor: pointer; font-weight: 700; color: #344054; }}
.note {{ margin-top: 6px; color: #667085; font-size: 12px; }}
.docs-link {{ display: inline-block; color: #175cd3; font-size: 13px; font-weight: 600; text-decoration: none; }}
.docs-link:hover {{ text-decoration: underline; }}
@media (max-width: 1100px) {{ .fov-review, .grid, .controls {{ grid-template-columns: 1fr; }} .head {{ display: block; }} .meta {{ text-align: left; }} }}
</style>
</head>
<body>
<div class="page">
  <div class="head"><h1>{session_name} preprocessing QC ({dff.shape[0]} ROIs)</h1><div class="meta" id="meta"></div></div>
  <div class="session-message" id="sessionMessage"></div>
  <div class="fov-review">
    <div class="grid {fov_grid_class}">
      <div class="panel"><div class="title">Green functional mean</div><div class="imagewrap"><img id="green"><svg class="overlay" preserveAspectRatio="xMidYMid meet"></svg></div></div>
      {red_panel}
      <div class="panel"><div class="title">ROI masks</div><div class="imagewrap"><img id="mask"><svg class="overlay" preserveAspectRatio="xMidYMid meet"></svg></div></div>
    </div>
    <div class="panel label-controls">
      <strong>Manual ROI review</strong>
      <label>Selected ROI <input id="roiInput" type="number" min="0" value="0"></label>
      <div id="readout"></div>
      <div class="button-row">
        <button id="markGood" class="good">Good (G)</button>
        <button id="markBad" class="bad">Bad (B)</button>
      </div>
      <button id="markUnlabeled" class="unlabeled">Unlabeled (U)</button>
      <button id="markAllGood">Mark all visible as good</button>
      <button id="clearLabels">Clear all labels / show all ROIs</button>
      <div class="nav-row">
        <button id="previousRoi" class="nav-button" title="Previous visible ROI (Left arrow)">&#8592; Previous</button>
        <button id="nextRoi" class="nav-button" title="Next visible ROI (Right arrow)">Next &#8594;</button>
      </div>
      <span class="note">Keyboard: G/B/U label; left/right arrows select the previous/next visible ROI.</span>
      <span id="labelCounts"></span>
      <label><input id="showAllRois" type="checkbox"> Show excluded ROIs</label>
      <button id="openExclusions">Open exclusion reasons</button>
      <button id="saveIscell">Save iscell_qc.npy</button>
      <button id="exportLabels">Export label JSON</button>
      <a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/feature/interactive-roi-labels/docs/roi-reviewer-exports.md" target="_blank" rel="noopener noreferrer">How to use reviewer output</a>
      <span class="note">Saving requires a browser file prompt. Existing unreviewed Suite2p rows are preserved.</span>
    </div>
  </div>
  <div class="panel">
    <div id="targetStructureSummary" class="title"></div>
    <div class="filter-summary" id="filterSummary"></div>
    <details class="filter-details">
      <summary>Show morphology settings and threshold sandbox</summary>
      <div class="filter-controls">
        <label>Preset <select id="filterPreset"></select></label>
        <label>Custom preset name <input id="presetName" type="text" placeholder="my preset"></label>
        <button id="savePreset">Save preset</button>
        <label>Skew min <input id="skewMin" type="number" step="0.01"></label>
        <label>Skew max <input id="skewMax" type="number" step="0.01"></label>
        <label>Max connect <input id="maxConnect" type="number" min="0" step="1"></label>
        <label>Aspect min <input id="aspectMin" type="number" step="0.01"></label>
        <label>Aspect max <input id="aspectMax" type="number" step="0.01"></label>
        <label>Footprint min <input id="footprintMin" type="number" step="0.01"></label>
        <label>Footprint max <input id="footprintMax" type="number" step="0.01"></label>
        <label>Compact min <input id="compactMin" type="number" step="0.01"></label>
        <label>Compact max <input id="compactMax" type="number" step="0.01"></label>
        <button id="resetFilter">Reset QC thresholds</button>
        <button id="applyFilterToLabels">Apply filter to labels</button>
      </div>
      <div class="note">This tests the same stat.npy morphology fields used by the QC stage. Changing thresholds previews pass/fail counts; labels change only when Apply filter to labels is clicked.</div>
    </details>
  </div>
  <div class="plots">
    <div class="panel">
      <div class="title" id="traceTitle">Selected ROI dF/F</div>
      <canvas id="traceCanvas"></canvas>
      <div class="note">Wheel or drag to zoom/pan time. Double-click to reset.</div>
      <div class="controls">
        <strong>Trace window</strong>
        <label>Start s <input id="timeStart" type="number" min="0" step="0.001" value="0"></label>
        <label>End s <input id="timeEnd" type="number" min="0" step="0.001" value="0"></label>
        <label>First ROI <input id="yStart" type="number" min="0" value="0"></label>
        <label>Last ROI <input id="yEnd" type="number" min="0" value="0"></label>
        <button id="reset">Reset zoom</button>
      </div>
    </div>
    <div class="panel">
      <div class="title" id="stackTitle">dF/F, stacked ROIs</div>
      <div class="trace-sort">
        <label>Sort visible ROIs by
          <select id="sortMetric">
            <option value="event_snr" selected>Event SNR</option>
            <option value="temporal_snr">Temporal smoothness</option>
            <option value="original">Original Suite2p index</option>
          </select>
        </label>
        <label>Sort order
          <select id="sortDirection">
            <option value="desc" selected>Highest first</option>
            <option value="asc">Lowest first</option>
          </select>
        </label>
        <button id="applySort">Apply sort</button>
        <details class="metric-formula">
          <summary>Metric formula used for sorting</summary>
          <div class="note">
            Event SNR = (P95(dF/F) - P50(dF/F)) / noise SD, where noise SD is the MAD-based estimate from the smoothed-trace residual.
            Temporal smoothness = 1 - var(diff(dF/F)) / (2 * var(dF/F)).
          </div>
        </details>
      </div>
      <canvas id="stackCanvas"></canvas>
      <div class="note">Change the dropdowns and click Apply sort to reorder the stacked traces. The First/Last ROI range follows the applied row order, not the Suite2p index.</div>
      <div class="note">Wheel to zoom time. Use First/Last ROI to choose the displayed rows in the applied sorted order. Stack labels and ROI readouts use the original Suite2p ROI index.</div>
    </div>
  </div>
</div>
<script id="payload" type="application/json">{json.dumps(payload, separators=(",", ":"))}</script>
<script>
"use strict";
const data = JSON.parse(document.getElementById("payload").textContent);
document.getElementById("green").src = data.green;
const redImage = document.getElementById("red");
if (redImage && data.redAvailable) redImage.src = data.red;
document.getElementById("mask").style.display = "none";
document.getElementById("meta").textContent = `${{data.nRois}} ROIs | ${{data.nFrames.toLocaleString()}} frames | ${{data.frameRate.toFixed(3)}} Hz${{data.redAvailable ? "" : " | no red channel detected"}}`;
document.querySelectorAll(".overlay").forEach(svg => svg.setAttribute("viewBox", `0 0 ${{data.imageWidth}} ${{data.imageHeight}}`));
document.getElementById("stackTitle").textContent = `${{data.dffLabel}}, stacked ROIs`;
document.getElementById("targetStructureSummary").textContent = `Target structure: ${{data.targetStructure}}`;
document.getElementById("roiInput").max = data.nRois - 1;
const sessionDurationSec = (data.nFrames - 1) / data.frameRate;
document.getElementById("timeStart").max = sessionDurationSec.toFixed(3);
document.getElementById("timeEnd").max = sessionDurationSec.toFixed(3);
document.getElementById("timeEnd").value = sessionDurationSec.toFixed(3);

function b64f32(base64) {{
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}}
function b64f64(base64) {{
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float64Array(bytes.buffer);
}}
const dff = b64f32(data.dff);
const sourceIscell = b64f64(data.iscell);
const iscell = new Float64Array(sourceIscell.length);
for (let i = 0; i < sourceIscell.length; i++) iscell[i] = sourceIscell[i];
const presetPass = Uint8Array.from(data.presetExclusionReasons, reasons => reasons.length === 0 ? 1 : 0);
const labels = new Int8Array(data.nRois);
for (let roi = 0; roi < data.nRois; roi++) {{
  labels[roi] = presetPass[roi] ? -1 : 0;
  if (!presetPass[roi]) {{
    const suite2pRoi = data.suite2pIndices[roi];
    iscell[suite2pRoi * 2] = 0;
    iscell[suite2pRoi * 2 + 1] = 0;
  }}
}}
const filterPass = new Uint8Array(data.nRois);
const customPresetKey = "utils_2p_morphology_presets_v1";
let customPresets = {{}};
try {{ customPresets = JSON.parse(localStorage.getItem(customPresetKey) || "{{}}"); }} catch (_error) {{}}
const defaultFilter = data.morphologyPresets[data.targetStructure] || data.morphologyPresets.neuron;
let selected = 0, x0 = 0, x1 = data.nFrames - 1, y0 = 0, y1 = 0, visibleRois = [];
let appliedSortMetric = "event_snr";
let appliedSortDirection = "desc";

function fit(canvas) {{
  const r = window.devicePixelRatio || 1, box = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(box.width * r));
  canvas.height = Math.max(1, Math.round(box.height * r));
}}
function trace(roi) {{ return dff.subarray(roi * data.nFrames, (roi + 1) * data.nFrames); }}
function val(roi, frame) {{ return dff[roi * data.nFrames + frame]; }}
function metricValue(roi, metric) {{
  if (metric === "event_snr") return data.dffMetrics[roi].event_snr;
  if (metric === "temporal_snr") return data.dffMetrics[roi].temporal_snr;
  if (metric === "original" || metric === "suite2p_index") return data.suite2pIndices[roi];
  return roi;
}}
function sortVisibleRois(rois) {{
  const metric = appliedSortMetric;
  const direction = appliedSortDirection;
  return rois.slice().sort((a, b) => {{
    const av = metricValue(a, metric);
    const bv = metricValue(b, metric);
    const aFinite = Number.isFinite(av);
    const bFinite = Number.isFinite(bv);
    if (aFinite !== bFinite) return aFinite ? -1 : 1;
    if (!aFinite && !bFinite) return a - b;
    if (av === bv) return a - b;
    return direction === "asc" ? av - bv : bv - av;
  }});
}}
function applySort() {{
  appliedSortMetric = document.getElementById("sortMetric").value;
  appliedSortDirection = document.getElementById("sortDirection").value;
  updateVisibleRois();
}}
function syncTimeInputs() {{
  document.getElementById("timeStart").value = (x0 / data.frameRate).toFixed(3);
  document.getElementById("timeEnd").value = (x1 / data.frameRate).toFixed(3);
}}
function setTimeWindow(startSec, endSec) {{
  startSec = Number(startSec);
  endSec = Number(endSec);
  if (!Number.isFinite(startSec)) startSec = 0;
  if (!Number.isFinite(endSec)) endSec = sessionDurationSec;
  startSec = Math.max(0, Math.min(sessionDurationSec, startSec));
  endSec = Math.max(0, Math.min(sessionDurationSec, endSec));
  if (endSec <= startSec) endSec = Math.min(sessionDurationSec, startSec + 1 / data.frameRate);
  x0 = Math.max(0, Math.min(data.nFrames - 1, Math.round(startSec * data.frameRate)));
  x1 = Math.max(0, Math.min(data.nFrames - 1, Math.round(endSec * data.frameRate)));
  if (x1 <= x0) x1 = Math.min(data.nFrames - 1, x0 + 1);
  syncTimeInputs();
  draw();
}}
function setFrameWindow(startFrame, endFrame) {{
  x0 = Math.max(0, Math.min(data.nFrames - 1, startFrame));
  x1 = Math.max(0, Math.min(data.nFrames - 1, endFrame));
  if (x1 <= x0) x1 = Math.min(data.nFrames - 1, x0 + 1);
  syncTimeInputs();
  draw();
}}
function setSelected(roi) {{
  selected = Math.max(0, Math.min(data.nRois - 1, Math.round(roi)));
  document.getElementById("roiInput").value = selected;
  const metrics = data.morphology[selected];
  const dffMetrics = data.dffMetrics[selected];
  const suite2pRoi = data.suite2pIndices[selected];
  document.getElementById("readout").textContent = `Selected Suite2p ROI ${{suite2pRoi}} (summary row ${{selected}}) | ${{data.nRois}} total ROIs | skew ${{fmt(metrics.skew)}} connect ${{metrics.connect}} aspect ${{fmt(metrics.aspect)}} compact ${{fmt(metrics.compact)}} footprint ${{fmt(metrics.footprint)}} | event SNR ${{fmt(dffMetrics.event_snr)}} | temporal ${{fmt(dffMetrics.temporal_snr)}}`;
  document.getElementById("traceTitle").textContent = `Selected Suite2p ROI ${{suite2pRoi}} ${{data.dffLabel}}`;
  document.querySelectorAll(".roi").forEach(c => c.classList.toggle("selected", Number(c.dataset.roi) === selected));
  updateLabelControls();
  draw();
}}
function updateVisibleRois() {{
  const showAll = document.getElementById("showAllRois").checked;
  const rois = [];
  for (let roi = 0; roi < data.nRois; roi++) if (showAll || labels[roi] !== 0) rois.push(roi);
  visibleRois = sortVisibleRois(rois);
  if (!visibleRois.length) visibleRois = Array.from({{length: data.nRois}}, (_v, roi) => roi);
  document.getElementById("yStart").max = visibleRois.length - 1;
  document.getElementById("yEnd").max = visibleRois.length - 1;
  y0 = Math.max(0, Math.min(y0, visibleRois.length - 1));
  y1 = Math.max(y0, Math.min(y1 || Math.min(19, visibleRois.length - 1), visibleRois.length - 1));
  document.getElementById("yStart").value = y0;
  document.getElementById("yEnd").value = y1;
  let good = 0, bad = 0, unlabeled = 0;
  for (const label of labels) {{ if (label === 1) good++; else if (label === 0) bad++; else unlabeled++; }}
  const sortMetric = appliedSortMetric;
  const sortDirection = appliedSortDirection;
  const sortLabel = sortMetric === "original"
    ? `original Suite2p index (${{sortDirection === "asc" ? "lowest first" : "highest first"}})`
    : `${{sortMetric.replace("_", " ")}} (${{sortDirection === "asc" ? "lowest first" : "highest first"}})`;
  document.getElementById("sessionMessage").textContent = `Target structure: ${{data.targetStructure}}. Embedded ${{data.nRois}} original Suite2p ROIs. ${{good}} good, ${{bad}} bad, ${{unlabeled}} unlabeled; ${{visibleRois.length}} ROIs are visible. Sorting: ${{sortLabel}}.`;
  document.querySelectorAll(".roi").forEach(path => {{
    const roi = Number(path.dataset.roi);
    path.style.display = (showAll || labels[roi] !== 0) ? "" : "none";
  }});
  if (!visibleRois.includes(selected)) setSelected(visibleRois[0]);
  else draw();
}}
function updateLabelControls() {{
  const label = labels[selected];
  document.getElementById("markGood").classList.toggle("active", label === 1);
  document.getElementById("markBad").classList.toggle("active", label === 0);
  document.getElementById("markUnlabeled").classList.toggle("active", label === -1);
  let good = 0, bad = 0, unlabeled = 0;
  for (const value of labels) {{ if (value === 1) good++; else if (value === 0) bad++; else unlabeled++; }}
  document.getElementById("labelCounts").textContent = `${{good}} good | ${{bad}} bad | ${{unlabeled}} unlabeled`;
}}
function fmt(value) {{
  if (value === null || value === undefined) return "nan";
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(3) : "nan";
}}
function filterValue(id) {{
  const value = Number(document.getElementById(id).value);
  return Number.isFinite(value) ? value : NaN;
}}
function readFilter() {{
  return {{
    skewMin: filterValue("skewMin"), skewMax: filterValue("skewMax"),
    maxConnect: filterValue("maxConnect"),
    aspectMin: filterValue("aspectMin"), aspectMax: filterValue("aspectMax"),
    footprintMin: filterValue("footprintMin"), footprintMax: filterValue("footprintMax"),
    compactMin: filterValue("compactMin"), compactMax: filterValue("compactMax"),
  }};
}}
function writeFilter(filter) {{
  for (const [id, value] of Object.entries(filter)) {{
    const input = document.getElementById(id);
    if (input) input.value = value;
  }}
  evaluateFilter();
}}
function populatePresetSelect(selectedName = data.targetStructure) {{
  const select = document.getElementById("filterPreset");
  select.textContent = "";
  for (const name of Object.keys(data.morphologyPresets)) {{
    const option = document.createElement("option"); option.value = `built-in:${{name}}`; option.textContent = name; select.appendChild(option);
  }}
  for (const name of Object.keys(customPresets).sort()) {{
    const option = document.createElement("option"); option.value = `custom:${{name}}`; option.textContent = `${{name}} (saved)`; select.appendChild(option);
  }}
  const desired = data.morphologyPresets[selectedName] ? `built-in:${{selectedName}}` : `custom:${{selectedName}}`;
  if ([...select.options].some(option => option.value === desired)) select.value = desired;
}}
function loadSelectedPreset() {{
  const [kind, name] = document.getElementById("filterPreset").value.split(":", 2);
  const preset = kind === "built-in" ? data.morphologyPresets[name] : customPresets[name];
  if (preset) writeFilter(preset);
}}
function passesFilter(metrics, filter) {{
  return (
    metrics.footprint >= filter.footprintMin && metrics.footprint <= filter.footprintMax &&
    metrics.skew >= filter.skewMin && metrics.skew <= filter.skewMax &&
    metrics.aspect >= filter.aspectMin && metrics.aspect <= filter.aspectMax &&
    metrics.compact >= filter.compactMin && metrics.compact <= filter.compactMax &&
    metrics.connect <= filter.maxConnect
  );
}}
function morphologyReasons(metrics, filter) {{
  const reasons = [];
  if (!(metrics.footprint >= filter.footprintMin && metrics.footprint <= filter.footprintMax)) reasons.push(`footprint ${{fmt(metrics.footprint)}} outside [${{filter.footprintMin}}, ${{filter.footprintMax}}]`);
  if (!(metrics.skew >= filter.skewMin && metrics.skew <= filter.skewMax)) reasons.push(`skew ${{fmt(metrics.skew)}} outside [${{filter.skewMin}}, ${{filter.skewMax}}]`);
  if (!(metrics.aspect >= filter.aspectMin && metrics.aspect <= filter.aspectMax)) reasons.push(`aspect_ratio ${{fmt(metrics.aspect)}} outside [${{filter.aspectMin}}, ${{filter.aspectMax}}]`);
  if (!(metrics.compact >= filter.compactMin && metrics.compact <= filter.compactMax)) reasons.push(`compact ${{fmt(metrics.compact)}} outside [${{filter.compactMin}}, ${{filter.compactMax}}]`);
  if (metrics.connect > filter.maxConnect) reasons.push(`connectivity ${{metrics.connect}} exceeds ${{filter.maxConnect}}`);
  return reasons;
}}
function evaluateFilter() {{
  const filter = readFilter();
  let pass = 0;
  for (let roi = 0; roi < data.nRois; roi++) {{
    filterPass[roi] = passesFilter(data.morphology[roi], filter) ? 1 : 0;
    pass += filterPass[roi];
  }}
  document.getElementById("filterSummary").textContent = `${{pass}} / ${{data.nRois}} original Suite2p ROIs pass the current morphology thresholds.`;
  draw();
}}
function resetFilter() {{
  loadSelectedPreset();
}}
function applyFilterToLabels() {{
  const current = selected;
  for (let roi = 0; roi < data.nRois; roi++) {{
    labels[roi] = filterPass[roi] ? -1 : 0;
    const suite2pRoi = data.suite2pIndices[roi];
    if (labels[roi] === -1) {{
      iscell[suite2pRoi * 2] = sourceIscell[suite2pRoi * 2];
      iscell[suite2pRoi * 2 + 1] = sourceIscell[suite2pRoi * 2 + 1];
    }} else {{
      iscell[suite2pRoi * 2] = 0;
      iscell[suite2pRoi * 2 + 1] = 0;
    }}
  }}
  updateVisibleRois();
  if (labels[current] !== 0 || document.getElementById("showAllRois").checked) setSelected(current);
}}
function setLabel(label) {{
  labels[selected] = label;
  if (label !== -1) {{
    const suite2pRoi = data.suite2pIndices[selected];
    iscell[suite2pRoi * 2] = label;
    iscell[suite2pRoi * 2 + 1] = label;
  }} else {{
    const suite2pRoi = data.suite2pIndices[selected];
    iscell[suite2pRoi * 2] = sourceIscell[suite2pRoi * 2];
    iscell[suite2pRoi * 2 + 1] = sourceIscell[suite2pRoi * 2 + 1];
  }}
  updateLabelControls();
  updateVisibleRois();
}}
function clearAllLabels() {{
  for (let roi = 0; roi < data.nRois; roi++) {{
    labels[roi] = -1;
    const suite2pRoi = data.suite2pIndices[roi];
    iscell[suite2pRoi * 2] = sourceIscell[suite2pRoi * 2];
    iscell[suite2pRoi * 2 + 1] = sourceIscell[suite2pRoi * 2 + 1];
  }}
  document.getElementById("showAllRois").checked = false;
  updateLabelControls();
  updateVisibleRois();
}}
function moveVisible(direction) {{
  const currentIndex = visibleRois.indexOf(selected);
  const nextIndex = Math.max(0, Math.min(visibleRois.length - 1, currentIndex + direction));
  setSelected(visibleRois[nextIndex]);
}}
function makeOverlays() {{
  document.querySelectorAll(".overlay").forEach(svg => {{
    data.rois.forEach(r => {{
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", r.path);
      path.dataset.roi = r.roi; path.classList.add("roi"); path.addEventListener("click", () => setSelected(r.roi)); svg.appendChild(path);
    }});
  }});
}}
function drawAxes(ctx, w, h, l, t, pw, ph, xLabel, yLabel) {{
  ctx.strokeStyle = "#d0d5dd"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(l,t); ctx.lineTo(l,t+ph); ctx.lineTo(l+pw,t+ph); ctx.stroke();
  ctx.fillStyle = "#475467"; ctx.font = `${{12 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.fillText(xLabel, l + pw / 2, h - 8);
  ctx.save(); ctx.translate(14, t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore();
}}
function tickStep(seconds) {{
  const options = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600];
  for (const step of options) if (seconds / step <= 8) return step;
  return 1200;
}}
function timeLabel(seconds, majorStep) {{
  if (majorStep < 1) return `${{Math.round(seconds * 1000)}} ms`;
  if (seconds < 60) return `${{seconds.toFixed(majorStep < 2 ? 1 : 0)}} s`;
  return `${{(seconds / 60).toFixed(1)}} min`;
}}
function drawTimeGrid(ctx, l, t, pw, ph) {{
  const startSec = x0 / data.frameRate, endSec = x1 / data.frameRate, spanSec = Math.max(1e-9, endSec - startSec);
  const major = tickStep(spanSec), minor = major >= 1 ? major / 10 : major / 5;
  function xOfSec(sec) {{ return l + (sec - startSec) / spanSec * pw; }}
  ctx.save();
  ctx.beginPath(); ctx.rect(l, t, pw, ph); ctx.clip();
  if (spanSec <= 20) {{
    ctx.strokeStyle = "#f1f5f9"; ctx.lineWidth = 1;
    for (let sec = Math.ceil(startSec / minor) * minor; sec <= endSec; sec += minor) {{
      const x = xOfSec(sec); ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke();
    }}
  }}
  ctx.strokeStyle = "#e2e8f0"; ctx.lineWidth = 1;
  const labelTicks = [];
  for (let sec = Math.ceil(startSec / major) * major; sec <= endSec; sec += major) {{
    const x = xOfSec(sec); ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke();
    labelTicks.push([x, timeLabel(sec, major)]);
  }}
  ctx.restore();
  ctx.fillStyle = "#475467"; ctx.font = `${{11 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.textBaseline = "top";
  labelTicks.forEach(([x, label]) => ctx.fillText(label, x, t + ph + 8));
}}
function colorForRoi(roi) {{
  const palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#2563eb", "#dc2626", "#059669", "#ca8a04", "#7c3aed", "#0891b2", "#be123c", "#4d7c0f", "#c2410c", "#4338ca"];
  return palette[roi % palette.length];
}}
function drawStack() {{
  const canvas = document.getElementById("stackCanvas"); fit(canvas); const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, l = 62, r = 16, t = 14, b = 56, pw = w-l-r, ph = h-t-b;
  ctx.clearRect(0,0,w,h); ctx.fillStyle = "#fff"; ctx.fillRect(0,0,w,h); drawAxes(ctx,w,h,l,t,pw,ph,"time (s)","ROI index");
  const ys = Math.max(0, Math.floor(y0)), ye = Math.min(visibleRois.length - 1, Math.ceil(y1));
  const xs = Math.max(0, Math.floor(x0)), xe = Math.min(data.nFrames - 1, Math.ceil(x1));
  drawTimeGrid(ctx, l, t, pw, ph);
  const count = Math.max(1, ye - ys + 1), rowH = ph / count, pixelCount = Math.max(1, Math.floor(pw));
  let amplitudes = [];
  for (let rowIndex = ys; rowIndex <= ye; rowIndex++) {{
    const roi = visibleRois[rowIndex];
    const tr = trace(roi);
    let sum = 0, n = 0;
    for (let f = xs; f <= xe; f++) {{ const v = tr[f]; if (Number.isFinite(v)) {{ sum += v; n++; }} }}
    const center = n ? sum / n : 0;
    let dev = 0, dn = 0;
    for (let f = xs; f <= xe; f++) {{ const v = tr[f]; if (Number.isFinite(v)) {{ dev += Math.abs(v - center); dn++; }} }}
    amplitudes.push(Math.max(dn ? dev / dn : 0, 0.2));
  }}
  const sortedAmp = amplitudes.slice().sort((a,b) => a-b);
  const typicalAmp = Math.max(sortedAmp[Math.floor(sortedAmp.length / 2)] || 1, 0.2);
  const scale = (rowH * 0.24) / typicalAmp;
  ctx.save(); ctx.beginPath(); ctx.rect(l, t, pw, ph); ctx.clip();
  for (let rowIndex = ys; rowIndex <= ye; rowIndex++) {{
    const roi = visibleRois[rowIndex], tr = trace(roi), row = rowIndex - ys, baseline = t + rowH * (row + 0.5), color = colorForRoi(roi);
    ctx.strokeStyle = roi === selected ? "#111827" : color;
    ctx.lineWidth = roi === selected ? Math.max(1.8, 1.8 * (window.devicePixelRatio || 1)) : Math.max(0.8, window.devicePixelRatio || 1);
    ctx.beginPath();
    let first = true;
    let sum = 0, n = 0;
    for (let f = xs; f <= xe; f++) {{ const v = tr[f]; if (Number.isFinite(v)) {{ sum += v; n++; }} }}
    const center = n ? sum / n : 0;
    const framesPerPixel = (xe - xs + 1) / pixelCount;
    if (framesPerPixel <= 1.2) {{
      for (let f = xs; f <= xe; f++) {{
        const x = l + (f - x0) / (x1 - x0) * pw;
        const y = baseline - (tr[f] - center) * scale;
        if (first) {{ ctx.moveTo(x, y); first = false; }} else ctx.lineTo(x, y);
      }}
    }} else {{
      for (let px = 0; px < pixelCount; px++) {{
        const f0 = Math.max(xs, Math.floor(xs + px * framesPerPixel));
        const f1 = Math.min(xe, Math.floor(xs + (px + 1) * framesPerPixel));
        let minV = Infinity, maxV = -Infinity;
        for (let f = f0; f <= f1; f++) {{
          const v = tr[f];
          if (Number.isFinite(v)) {{ minV = Math.min(minV, v); maxV = Math.max(maxV, v); }}
        }}
        if (!Number.isFinite(minV)) continue;
        const x = l + px;
        ctx.moveTo(x, baseline - (minV - center) * scale);
        ctx.lineTo(x, baseline - (maxV - center) * scale);
      }}
    }}
    ctx.stroke();
    if (count <= 80) {{
      ctx.fillStyle = color; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(String(data.suite2pIndices[roi]), l - 8, baseline);
    }}
  }}
  ctx.restore();
}}
function drawTrace() {{
  const canvas = document.getElementById("traceCanvas"); fit(canvas); const ctx = canvas.getContext("2d");
  const w=canvas.width,h=canvas.height,l=62,r=16,t=14,b=56,pw=w-l-r,ph=h-t-b, tr=trace(selected);
  ctx.clearRect(0,0,w,h); ctx.fillStyle="#fff"; ctx.fillRect(0,0,w,h); drawAxes(ctx,w,h,l,t,pw,ph,"time (s)","dF/F"); drawTimeGrid(ctx,l,t,pw,ph);
  const xs=Math.max(0,Math.floor(x0)), xe=Math.min(data.nFrames-1,Math.ceil(x1)); let lo=Infinity,hi=-Infinity;
  for (let f=xs; f<=xe; f++) {{ const v=tr[f]; if (Number.isFinite(v)) {{ lo=Math.min(lo,v); hi=Math.max(hi,v); }} }}
  if (!Number.isFinite(lo) || hi<=lo) {{ lo=-1; hi=1; }} const pad=(hi-lo)*.08||1; lo-=pad; hi+=pad;
  const xOf=f=>l+(f-x0)/(x1-x0)*pw, yOf=v=>t+(1-(v-lo)/(hi-lo))*ph;
  ctx.strokeStyle="#1d4ed8"; ctx.lineWidth=Math.max(1,window.devicePixelRatio||1); ctx.beginPath();
  const pixelCount = Math.max(1, Math.floor(pw)), framesPerPixel = (xe - xs + 1) / pixelCount;
  if (framesPerPixel <= 1.2) {{
    let first=true; for (let f=xs; f<=xe; f++) {{ const x=xOf(f), y=yOf(tr[f]); if (first) {{ ctx.moveTo(x,y); first=false; }} else ctx.lineTo(x,y); }}
  }} else {{
    for (let px=0; px<pixelCount; px++) {{
      const f0=Math.max(xs, Math.floor(xs + px * framesPerPixel)), f1=Math.min(xe, Math.floor(xs + (px + 1) * framesPerPixel));
      let minV=Infinity, maxV=-Infinity;
      for (let f=f0; f<=f1; f++) {{ const v=tr[f]; if (Number.isFinite(v)) {{ minV=Math.min(minV,v); maxV=Math.max(maxV,v); }} }}
      if (!Number.isFinite(minV)) continue;
      const x=l+px; ctx.moveTo(x, yOf(minV)); ctx.lineTo(x, yOf(maxV));
    }}
  }}
  ctx.stroke();
  ctx.fillStyle="#475467"; ctx.textAlign="right"; ctx.textBaseline="middle"; for (let i=0; i<=4; i++) {{ const v=lo+i/4*(hi-lo); ctx.fillText(v.toFixed(2), l-8, yOf(v)); }}
}}
function draw() {{ drawTrace(); drawStack(); }}
function reset() {{ x0=0; x1=data.nFrames-1; y0=0; y1=Math.min(19, visibleRois.length-1); document.getElementById("yStart").value=0; document.getElementById("yEnd").value=y1; syncTimeInputs(); draw(); }}
document.getElementById("roiInput").addEventListener("change", e => setSelected(Number(e.target.value)));
document.getElementById("timeStart").addEventListener("change", () => setTimeWindow(document.getElementById("timeStart").value, document.getElementById("timeEnd").value));
document.getElementById("timeEnd").addEventListener("change", () => setTimeWindow(document.getElementById("timeStart").value, document.getElementById("timeEnd").value));
document.getElementById("yStart").addEventListener("change", e => {{ y0=Number(e.target.value); draw(); }});
document.getElementById("yEnd").addEventListener("change", e => {{ y1=Number(e.target.value); draw(); }});
document.getElementById("reset").addEventListener("click", reset);
document.getElementById("markGood").addEventListener("click", () => setLabel(1));
document.getElementById("markBad").addEventListener("click", () => setLabel(0));
document.getElementById("markUnlabeled").addEventListener("click", () => setLabel(-1));
document.getElementById("markAllGood").addEventListener("click", () => {{
  for (const roi of visibleRois) {{
    labels[roi] = 1;
    const suite2pRoi = data.suite2pIndices[roi];
    iscell[suite2pRoi * 2] = 1;
    iscell[suite2pRoi * 2 + 1] = 1;
  }}
  updateLabelControls();
  updateVisibleRois();
}});
document.getElementById("clearLabels").addEventListener("click", clearAllLabels);
document.getElementById("previousRoi").addEventListener("click", () => moveVisible(-1));
document.getElementById("nextRoi").addEventListener("click", () => moveVisible(1));
document.getElementById("showAllRois").addEventListener("change", updateVisibleRois);
document.getElementById("applySort").addEventListener("click", applySort);
document.getElementById("openExclusions").addEventListener("click", () => {{
  const filter = readFilter();
  const rows = data.morphology.map((metrics, roi) => {{
    const reasons = morphologyReasons(metrics, filter);
    if (labels[roi] === 0) reasons.push("manual/current label: bad");
    else if (labels[roi] === -1 && reasons.length === 0) reasons.push("unlabeled");
    const text = reasons.join("; ") || "included";
    const dffMetrics = data.dffMetrics[roi];
    return `<tr><td>${{roi}}</td><td>${{data.suite2pIndices[roi]}}</td><td>${{fmt(dffMetrics.event_snr)}}</td><td>${{fmt(dffMetrics.temporal_snr)}}</td><td>${{text}}</td></tr>`;
  }}).join("");
  const win = window.open("", "_blank");
  win.document.write(`<!doctype html><title>${{data.session}} ROI exclusions</title><style>body{{font-family:Arial,sans-serif;margin:20px}}td,th{{border:1px solid #ddd;padding:4px 7px}}table{{border-collapse:collapse}}</style><h1>${{data.session}} ROI exclusion reasons</h1><p>Target structure: ${{data.targetStructure}}</p><table><thead><tr><th>ROI</th><th>Suite2p row</th><th>Event SNR</th><th>Temporal SNR</th><th>Reason</th></tr></thead><tbody>${{rows}}</tbody></table>`);
  win.document.close();
}});
function npyBlob(values, rows) {{
  const encoder = new TextEncoder();
  let header = `{{'descr': '<f8', 'fortran_order': False, 'shape': (${{rows}}, 2), }}`;
  const preambleLength = 10;
  const padding = (16 - ((preambleLength + header.length + 1) % 16)) % 16;
  header += " ".repeat(padding) + "\\n";
  const headerBytes = encoder.encode(header);
  const buffer = new ArrayBuffer(preambleLength + headerBytes.length + values.byteLength);
  const bytes = new Uint8Array(buffer);
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
  new DataView(buffer).setUint16(8, headerBytes.length, true);
  bytes.set(headerBytes, preambleLength);
  bytes.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), preambleLength + headerBytes.length);
  return new Blob([buffer], {{type: "application/octet-stream"}});
}}
function downloadBlob(blob, filename) {{
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}}
async function saveIscell() {{
  const blob = npyBlob(iscell, iscell.length / 2);
  if ("showSaveFilePicker" in window) {{
    try {{
      const handle = await window.showSaveFilePicker({{
        suggestedName: "iscell_qc.npy",
        types: [{{description: "NumPy array", accept: {{"application/octet-stream": [".npy"]}}}}],
      }});
      const writable = await handle.createWritable();
      await writable.write(blob); await writable.close();
      return;
    }} catch (error) {{
      if (error.name === "AbortError") return;
      console.warn("Direct save failed; downloading instead", error);
    }}
  }}
  downloadBlob(blob, "iscell_qc.npy");
}}
function exportLabelJson() {{
  const payload = {{
    format: "utils_2p_manual_roi_labels_v1",
    session: data.session,
    suite2p_roi_count: data.suite2pRoiCount,
    suite2p_stat_fingerprint: data.suite2pStatFingerprint,
    morphology_filter: readFilter(),
    labels: Array.from(labels, (label, roi) => ({{
      summary_roi: roi, suite2p_roi: data.suite2pIndices[roi], label: label === -1 ? null : Number(label),
    }})),
  }};
  downloadBlob(new Blob([JSON.stringify(payload, null, 2) + "\\n"], {{type: "application/json"}}), `${{data.session}}_manual_roi_labels.json`);
}}
document.getElementById("saveIscell").addEventListener("click", saveIscell);
document.getElementById("exportLabels").addEventListener("click", exportLabelJson);
document.getElementById("resetFilter").addEventListener("click", resetFilter);
document.getElementById("applyFilterToLabels").addEventListener("click", applyFilterToLabels);
document.getElementById("filterPreset").addEventListener("change", loadSelectedPreset);
document.getElementById("savePreset").addEventListener("click", () => {{
  const name = document.getElementById("presetName").value.trim();
  if (!name) {{ alert("Enter a custom preset name first."); return; }}
  customPresets[name] = readFilter();
  try {{
    localStorage.setItem(customPresetKey, JSON.stringify(customPresets));
  }} catch (error) {{
    alert(`Could not save preset in this browser: ${{error}}`);
    return;
  }}
  populatePresetSelect(name);
  document.getElementById("filterPreset").value = `custom:${{name}}`;
}});
["skewMin","skewMax","maxConnect","aspectMin","aspectMax","footprintMin","footprintMax","compactMin","compactMax"].forEach(id => {{
  document.getElementById(id).addEventListener("change", evaluateFilter);
}});
window.addEventListener("keydown", event => {{
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if (event.key.toLowerCase() === "g") setLabel(1);
  else if (event.key.toLowerCase() === "b") setLabel(0);
  else if (event.key.toLowerCase() === "u") setLabel(-1);
  else if (event.key === "ArrowLeft") moveVisible(-1);
  else if (event.key === "ArrowRight") moveVisible(1);
}});
document.getElementById("stackCanvas").addEventListener("click", e => {{
  const rect=e.target.getBoundingClientRect(), frac=(e.clientY-rect.top)/rect.height;
  const row = Math.max(0, Math.min(visibleRois.length - 1, Math.floor(y0 + frac * (y1-y0+1))));
  setSelected(visibleRois[row]);
}});
document.getElementById("stackCanvas").addEventListener("wheel", e => {{
  e.preventDefault(); const rect=e.target.getBoundingClientRect(), xf=(e.clientX-rect.left)/rect.width, c=x0+xf*(x1-x0), s=(e.deltaY<0?.78:1.28)*(x1-x0);
  setFrameWindow(Math.max(0,c-xf*s), Math.min(data.nFrames-1, Math.max(0,c-xf*s)+s));
}}, {{passive:false}});
let dragging=false, sx=0, start0=0, start1=0;
document.getElementById("traceCanvas").addEventListener("wheel", e => {{
  e.preventDefault(); const rect=e.target.getBoundingClientRect(), xf=(e.clientX-rect.left)/rect.width, c=x0+xf*(x1-x0), s=(e.deltaY<0?.78:1.28)*(x1-x0);
  setFrameWindow(Math.max(0,c-xf*s), Math.min(data.nFrames-1, Math.max(0,c-xf*s)+s));
}}, {{passive:false}});
document.getElementById("traceCanvas").addEventListener("mousedown", e => {{ dragging=true; sx=e.clientX; start0=x0; start1=x1; e.target.classList.add("dragging"); }});
window.addEventListener("mousemove", e => {{ if (!dragging) return; const rect=document.getElementById("traceCanvas").getBoundingClientRect(), shift=-(e.clientX-sx)/rect.width*(start1-start0); setFrameWindow(start0+shift, start1+shift); }});
window.addEventListener("mouseup", () => {{ dragging=false; document.getElementById("traceCanvas").classList.remove("dragging"); }});
document.getElementById("traceCanvas").addEventListener("dblclick", reset);
window.addEventListener("resize", draw);
makeOverlays(); syncTimeInputs(); populatePresetSelect(data.targetStructure); resetFilter(); applySort(); setSelected(visibleRois[0]);
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def create_preprocessing_summary(
    session_data_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    pdf_name: str | None = None,
    html_name: str | None = None,
) -> tuple[Path, Path]:
    session_dir = Path(session_data_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else session_dir
    ops = _load_ops(session_dir)
    suite2p_dir = session_dir / "suite2p" / "plane0"
    suite2p_stat = np.load(suite2p_dir / "stat.npy", allow_pickle=True)
    qc_parameters = _load_qc_parameters(session_dir)
    pipeline_parameters = _load_pipeline_parameters(session_dir)
    xoff, yoff = _load_offsets(session_dir, ops)

    mean_green = _load_masks_h5_image(session_dir, "mean_func")
    if mean_green is None:
        mean_green = np.asarray(ops.get("meanImg"))
    max_green = _load_masks_h5_image(session_dir, "max_func")
    if max_green is None:
        max_green = np.asarray(ops.get("max_proj", ops.get("maxImg", mean_green)))
    mean_red = _load_masks_h5_image(session_dir, "mean_anat")
    red_available = mean_red is not None
    if mean_red is None and ops.get("nchannels", 1) != 1 and "meanImg_chan2" in ops:
        mean_red = np.asarray(ops.get("meanImg_chan2"))
        red_available = True
    max_red = _load_masks_h5_image(session_dir, "max_anat")
    if max_red is None and red_available:
        max_red = np.asarray(ops.get("meanImg_chan2_corrected", mean_red))
    if not red_available:
        mean_red = None
        max_red = None
    if mean_green is None:
        raise KeyError("Could not find green functional mean image in masks.h5 or ops.npy")

    frame_rate = float(ops.get("fs", 30.0))
    dff = _load_suite2p_dff(session_dir, ops)
    dff_label = "raw dF/F from Suite2p F.npy/Fneu.npy"
    n_rois = min(dff.shape[0], len(suite2p_stat))
    dff = dff[:n_rois]
    stat = suite2p_stat[:n_rois]
    suite2p_indices = np.arange(n_rois, dtype=np.int64)
    mask = _stat_to_mask(stat, np.asarray(mean_green).shape[:2])
    iscell_path = suite2p_dir / "iscell.npy"
    iscell = load_iscell(iscell_path, n_rois)
    if not iscell_path.exists():
        iscell[:, :] = 1.0
    suite2p_fingerprint = suite2p_stat_fingerprint(stat)
    morphology_metrics = roi_morphology_metrics(stat)
    target_structure = _target_structure(pipeline_parameters, qc_parameters)
    preset_exclusion_reasons = (
        morphology_exclusion_reasons(morphology_metrics, qc_parameters)
        if qc_parameters is not None
        else [[] for _ in range(n_rois)]
    )
    pdf_name = pdf_name or PDF_NAME_TEMPLATE.format(session_name=session_dir.name)
    html_name = html_name or HTML_NAME_TEMPLATE.format(session_name=session_dir.name)
    pdf_path = out_dir / pdf_name
    html_path = out_dir / html_name
    _write_summary_pdf(
        pdf_path,
        session_name=session_dir.name,
        n_rois=n_rois,
        mean_green=mean_green,
        max_green=max_green,
        mean_red=mean_red,
        max_red=max_red,
        mask=mask,
        xoff=xoff,
        yoff=yoff,
        frame_rate=frame_rate,
    )
    _write_html(
        html_path,
        session_name=session_dir.name,
        mean_green=mean_green,
        mean_red=mean_red,
        mask=mask,
        stat=stat,
        suite2p_indices=suite2p_indices,
        iscell=iscell,
        suite2p_fingerprint=suite2p_fingerprint,
        morphology_metrics=morphology_metrics,
        preset_exclusion_reasons=preset_exclusion_reasons,
        qc_parameters=qc_parameters,
        target_structure=target_structure,
        dff=dff,
        dff_label=dff_label,
        frame_rate=frame_rate,
    )
    return pdf_path, html_path


def run(ops: dict[str, Any], output_dir: str | os.PathLike[str] | None = None) -> tuple[Path, Path]:
    return create_preprocessing_summary(ops["save_path0"], output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create preprocessing QC PDF and interactive HTML outputs.")
    parser.add_argument("session_data_path", help="Processed session directory.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: session_data_path.")
    parser.add_argument("--pdf-name", default=None)
    parser.add_argument("--html-name", default=None)
    args = parser.parse_args()
    pdf_path, html_path = create_preprocessing_summary(
        args.session_data_path,
        output_dir=args.output_dir,
        pdf_name=args.pdf_name,
        html_name=args.html_name,
    )
    print(f"Saved preprocessing PDF: {pdf_path}")
    print(f"Saved interactive HTML: {html_path}")


if __name__ == "__main__":
    main()
