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
EMBEDDED_DFF_BYTE_LIMIT = 80 * 1024 * 1024
EMBEDDED_DFF_BASE64_EXPANSION = 4 / 3
EMBEDDED_DFF_HTML_OVERHEAD = 2 * 1024 * 1024


def _estimate_embedded_dff_bytes(n_rois: int, n_frames: int) -> int:
    raw_bytes = max(0, int(n_rois)) * max(0, int(n_frames)) * 4
    return int(raw_bytes * EMBEDDED_DFF_BASE64_EXPANSION + EMBEDDED_DFF_HTML_OVERHEAD)


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
    return "all_rois"


def _all_rois_filter() -> dict[str, float | int | None]:
    return {
        "skewMin": None,
        "skewMax": None,
        "maxConnect": None,
        "aspectMin": None,
        "aspectMax": None,
        "footprintMin": None,
        "footprintMax": None,
        "compactMin": None,
        "compactMax": None,
        "eventSnrMin": None,
        "eventSnrMax": None,
        "decayTauMin": None,
        "decayTauMax": None,
    }


def _morphology_preset_payload() -> dict[str, dict[str, float | int | None]]:
    presets: dict[str, dict[str, float | int | None]] = {"all_rois": _all_rois_filter()}
    for name, values in QC_PRESETS.items():
        presets[name] = {
            "skewMin": values["range_skew"][0],
            "skewMax": values["range_skew"][1],
            "maxConnect": values["max_connect"],
            "aspectMin": values["range_aspect"][0],
            "aspectMax": values["range_aspect"][1],
            "footprintMin": values["range_footprint"][0],
            "footprintMax": values["range_footprint"][1],
            "compactMin": values["range_compact"][0],
            "compactMax": values["range_compact"][1],
            "eventSnrMin": None,
            "eventSnrMax": None,
            "decayTauMin": None,
            "decayTauMax": None,
        }
    return presets


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


def _load_suite2p_fluorescence(session_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    plane_dir = session_dir / "suite2p" / "plane0"
    fluo = np.load(plane_dir / "F.npy", allow_pickle=False, mmap_mode="r")
    neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=False, mmap_mode="r")
    return fluo, neuropil


def _dff_metrics_to_jsonable(dff_metrics: list[dict[str, float | int | np.floating | np.integer]]) -> list[dict[str, float | None]]:
    payload: list[dict[str, float | None]] = []
    for metrics in dff_metrics:
        payload.append(
            {
                key: (float(value) if np.isfinite(value) else None)
                for key, value in metrics.items()
            }
        )
    return payload


def _build_dff_and_metrics(
    session_dir: Path,
    ops: dict[str, Any],
    n_rois: int,
    n_frames: int,
    frame_rate: float,
    storage_mode: str,
    sidecar_path: Path | None,
    *,
    chunk_size: int = 64,
) -> tuple[np.ndarray | None, list[dict[str, float | None]]]:
    if storage_mode == "embedded":
        dff = _load_suite2p_dff(session_dir, ops)
        dff = dff[:n_rois, :n_frames]
        return dff, _dff_metrics_to_jsonable(list(dff_qc_metrics(dff, frame_rate=frame_rate)))

    if sidecar_path is None:
        raise ValueError("sidecar_path is required when storage_mode is file")

    fluo, neuropil = _load_suite2p_fluorescence(session_dir)
    coeff = float(ops.get("neucoeff", 0.7))
    writer = np.lib.format.open_memmap(sidecar_path, mode="w+", dtype=np.float32, shape=(n_rois, n_frames))
    metrics: list[dict[str, float | None]] = []
    for start in range(0, n_rois, chunk_size):
        stop = min(n_rois, start + chunk_size)
        signal = np.asarray(fluo[start:stop], dtype=np.float32) - coeff * np.asarray(neuropil[start:stop], dtype=np.float32)
        baseline = gaussian_filter(signal, [0.0, 600.0])
        with np.errstate(divide="ignore", invalid="ignore"):
            dff_chunk = (signal - baseline) / baseline
        dff_chunk[~np.isfinite(dff_chunk)] = np.nan
        writer[start:stop] = dff_chunk.astype(np.float32, copy=False)
        metrics.extend(_dff_metrics_to_jsonable(list(dff_qc_metrics(dff_chunk, frame_rate=frame_rate))))
    writer.flush()
    return None, metrics


def _add_roi_area_metrics(
    dff_metrics: list[dict[str, float | None]],
    stat: np.ndarray,
) -> list[dict[str, float | None]]:
    for metrics, entry in zip(dff_metrics, stat):
        xpix = np.asarray(entry.get("xpix", []), dtype=int)
        metrics["roi_area"] = float(xpix.size)
    return dff_metrics


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


def _roi_outline_path(xpix: np.ndarray, ypix: np.ndarray) -> str:
    segments: list[str] = []
    if not xpix.size:
        return ""
    pixels = {(int(x), int(y)) for x, y in zip(xpix, ypix)}
    for x, y in sorted(pixels, key=lambda item: (item[1], item[0])):
        if (x, y - 1) not in pixels:
            segments.append(f"M{x} {y}h1")
        if (x, y + 1) not in pixels:
            segments.append(f"M{x} {y + 1}h1")
        if (x - 1, y) not in pixels:
            segments.append(f"M{x} {y}v1")
        if (x + 1, y) not in pixels:
            segments.append(f"M{x + 1} {y}v1")
    return "".join(segments)


def _roi_table(stat: np.ndarray, mask: np.ndarray, n_rois: int) -> list[dict[str, float | int | str]]:
    rois: list[dict[str, float | int | str]] = []
    for idx in range(n_rois):
        entry = stat[idx]
        ypix = np.asarray(entry.get("ypix", []), dtype=int)
        xpix = np.asarray(entry.get("xpix", []), dtype=int)
        rois.append({"roi": idx, "path": _roi_outline_path(xpix, ypix), "npix": int(xpix.size)})
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
    n_rois: int,
    n_frames: int,
    dff_label: str,
    frame_rate: float,
    dff_metrics: list[dict[str, float | None]],
    dff: np.ndarray | None,
    dff_storage_mode: str,
    dff_sidecar_name: str | None,
    estimated_embedded_dff_bytes: int,
) -> None:
    rois = _roi_table(stat, mask, n_rois)
    red_available = mean_red is not None
    image_height, image_width = np.asarray(mean_green).shape[:2]
    payload = {
        "session": session_name,
        "frameRate": float(frame_rate),
        "nRois": int(n_rois),
        "nFrames": int(n_frames),
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
        "initialLabels": None,
        "initialMorphologyFilter": None,
        "customMorphologyPresets": None,
        "dffMetrics": dff_metrics,
        "dffStorageMode": dff_storage_mode,
        "dffSidecarName": dff_sidecar_name,
        "estimatedEmbeddedDffBytes": int(estimated_embedded_dff_bytes),
        "morphologyPresets": _morphology_preset_payload(),
        "dff": _float32_b64(dff) if dff_storage_mode == "embedded" and dff is not None else None,
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
<title>{session_name} preprocessing QC ({n_rois} ROIs)</title>
<style>
body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: #202124; background: #f6f7f8; }}
.page {{ width: min(1680px, calc(100vw - 28px)); margin: 16px auto 26px; }}
.head {{ display: flex; justify-content: space-between; gap: 14px; align-items: end; margin-bottom: 12px; }}
h1 {{ margin: 0; font-size: 21px; letter-spacing: 0; }}
.meta {{ color: #667085; font-size: 13px; text-align: right; }}
.grid {{ display: grid; gap: 8px; }}
.review-main {{ margin-top: 8px; }}
.viewer-column {{ display: flex; flex-direction: column; gap: 8px; }}
.fov-row {{ display: grid; grid-template-columns: minmax(0, 1fr) clamp(320px, 23vw, 380px); gap: 6px; align-items: start; }}
.fov-review {{ display: grid; gap: 8px; align-items: start; }}
.grid.with-red {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.grid.single-channel {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.panel {{ background: #fff; border: 1px solid #d0d5dd; border-radius: 7px; padding: 8px; box-sizing: border-box; }}
.title {{ font-size: 14px; font-weight: 700; margin-bottom: 6px; }}
.imagewrap {{ position: relative; width: 100%; aspect-ratio: 1/1; background: #111; overflow: hidden; }}
.imagewrap img, .imagewrap svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
.imagewrap img {{ object-fit: contain; image-rendering: pixelated; }}
.roi {{ fill: none; stroke: rgba(255,255,255,.86); stroke-width: .7; cursor: pointer; vector-effect: non-scaling-stroke; pointer-events: all; }}
.roi:hover {{ fill: none; stroke: #06b6d4; stroke-width: 1.6; }}
.roi.selected {{ fill: none; stroke: #ffffff; stroke-width: 2.8; }}
.controls {{ display: grid; grid-template-columns: 1fr repeat(5, auto); gap: 7px; align-items: center; margin-top: 8px; }}
.label-controls {{ display: flex; flex-direction: column; gap: 6px; align-items: stretch; }}
.label-controls .button-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
.label-controls .nav-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
.roi-details summary {{ cursor: pointer; font-weight: 700; color: #344054; }}
.roi-details #readout {{ margin-top: 5px; color: #475467; font-size: 12px; line-height: 1.35; }}
.save-option {{ display: grid; gap: 3px; justify-items: start; }}
.save-option button, .save-options .docs-link {{ width: fit-content; }}
.info-button {{ padding: 3px 7px; width: fit-content; font-size: 12px; }}
.nav-button {{ font-size: 18px; font-weight: 700; }}
.control-column {{ display: flex; flex-direction: column; gap: 6px; min-height: 0; overflow-y: auto; }}
.morphology-card {{ display: flex; flex-direction: column; gap: 6px; }}
.qc-header {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; }}
.qc-current {{ color: #667085; font-size: 12px; }}
.sort-card {{ display: flex; flex-direction: column; gap: 6px; padding-top: 6px; border-top: 1px solid #eaecf0; }}
.sort-header {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; }}
.sort-current {{ color: #667085; font-size: 12px; }}
.filter-controls {{ display: grid; grid-template-columns: repeat(3, minmax(130px, 1fr)); gap: 8px; margin-top: 10px; align-items: end; }}
.filter-controls label {{ font-size: 12px; color: #475467; }}
.filter-controls input {{ display: block; margin-top: 3px; width: 100%; box-sizing: border-box; }}
.filter-subsection-title {{ margin-top: 10px; font-size: 13px; font-weight: 700; color: #344054; }}
.source-heading {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; }}
.filter-summary {{ color: #475467; font-size: 13px; }}
.dialog-actions {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; margin-top: 12px; }}
.dialog-header {{ display: flex; justify-content: space-between; gap: 10px; align-items: start; }}
.dialog-title {{ font-size: 16px; font-weight: 700; }}
.dialog-section {{ margin-top: 12px; padding-top: 12px; border-top: 1px solid #eaecf0; }}
.dialog-section:first-child {{ margin-top: 0; padding-top: 0; border-top: 0; }}
.dialog-section-title {{ font-size: 14px; font-weight: 700; margin-bottom: 6px; }}
.info-box {{ margin-top: 8px; padding: 8px 10px; background: #f8fafc; border: 1px solid #d0d5dd; border-radius: 6px; color: #475467; font-size: 12px; line-height: 1.35; }}
.save-options {{ display: grid; gap: 10px; margin-top: 10px; }}
.bulk-label-controls {{ display: grid; gap: 8px; margin-top: 10px; justify-items: start; }}
.bulk-label-controls select {{ width: auto; min-width: 160px; }}
dialog {{ width: min(980px, calc(100vw - 40px)); border: 1px solid #d0d5dd; border-radius: 8px; padding: 14px; box-shadow: 0 24px 60px rgba(16,24,40,.24); }}
dialog::backdrop {{ background: rgba(15,23,42,.38); }}
button, input, select {{ font: inherit; }}
button {{ border: 1px solid #d0d5dd; background: #fff; border-radius: 6px; padding: 6px 9px; cursor: pointer; }}
button.good {{ border-color: #16a34a; color: #166534; }}
button.bad {{ border-color: #dc2626; color: #991b1b; }}
button.unsure {{ border-color: #d97706; color: #92400e; }}
button.unlabeled {{ border-color: #667085; color: #475467; }}
button.active {{ color: #fff; }}
button.good.active {{ background: #16a34a; }}
button.bad.active {{ background: #dc2626; }}
button.unsure.active {{ background: #d97706; }}
button.unlabeled.active {{ background: #667085; }}
input, select {{ border: 1px solid #d0d5dd; border-radius: 6px; padding: 7px 8px; width: 86px; }}
.filter-controls input {{ width: 100%; }}
.filter-controls select {{ width: 100%; }}
canvas {{ width: 100%; display: block; background: #fff; border: 1px solid #d0d5dd; box-sizing: border-box; }}
#stackCanvas {{ height: 560px; cursor: crosshair; }}
#traceCanvas {{ height: 220px; cursor: grab; }}
#traceCanvas.dragging {{ cursor: grabbing; }}
.plots {{ display: grid; grid-template-columns: 1fr; gap: 8px; margin-top: 8px; }}
.trace-sort {{ display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: end; margin: 6px 0 10px; padding: 8px 10px; background: #f8fafc; border: 1px solid #d0d5dd; border-radius: 6px; }}
.trace-sort label {{ font-size: 12px; color: #475467; }}
.trace-sort select {{ display: block; margin-top: 3px; min-width: 180px; width: auto; }}
.metric-formula {{ flex: 1 1 320px; }}
.metric-formula summary {{ cursor: pointer; font-weight: 700; color: #344054; }}
.metric-table-wrap {{ max-height: 75vh; overflow: auto; border: 1px solid #d0d5dd; margin-top: 12px; }}
.metric-table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
.metric-table th, .metric-table td {{ border: 1px solid #e5e7eb; padding: 4px 7px; text-align: right; white-space: nowrap; }}
.metric-table th {{ position: sticky; top: 0; background: #f8fafc; z-index: 1; }}
.metric-table td:first-child, .metric-table td:nth-child(2), .metric-table td:last-child {{ text-align: left; }}
.metric-fail {{ background: rgba(248, 113, 113, .28); }}
.trace-loader {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 8px; }}
.trace-loader input {{ width: auto; }}
.note {{ margin-top: 6px; color: #667085; font-size: 12px; }}
.docs-link {{ display: inline-block; color: #175cd3; font-size: 13px; font-weight: 600; text-decoration: none; }}
.docs-link:hover {{ text-decoration: underline; }}
@media (max-width: 1100px) {{ .fov-row, .fov-review, .grid, .controls {{ grid-template-columns: 1fr; }} .head {{ display: block; }} .meta {{ text-align: left; }} }}
</style>
</head>
<body>
<div class="page">
  <div class="head"><h1>{session_name} preprocessing QC ({n_rois} ROIs)</h1><div class="meta" id="meta"></div></div>
  <dialog id="morphologyDialog">
    <div class="dialog-title">ROI QC Filters</div>
    <div class="dialog-section">
      <div class="dialog-section-title">Suite2p morphology QC filters</div>
      <div id="targetStructureSummary" class="title"></div>
      <div class="filter-summary" id="filterSummary"></div>
      <div class="note">These preset thresholds preview pass/fail counts; labels change only when Apply Filters is clicked.</div>
      <div class="filter-controls">
        <label>Preset <select id="filterPreset"></select></label>
        <label>Custom preset name <input id="presetName" type="text" placeholder="my preset"></label>
        <button id="savePreset">Save preset</button>
        <button id="savePresetHtml">Save preset into HTML</button>
        <button id="exportPreset">Export preset JSON</button>
        <input id="presetFile" type="file" accept=".json" style="display:none;">
        <button id="importPreset">Import preset JSON</button>
      </div>
      <div class="filter-subsection-title source-heading">
        <span>Suite2p Morphology Metrics</span>
        <button class="info-button" type="button" data-info-target="suite2pMetricSources" aria-expanded="false">(i)</button>
      </div>
      <div id="suite2pMetricSources" class="info-box" hidden>
        Suite2p morphology metrics here come from ROI <code>stat.npy</code> fields such as <code>aspect_ratio</code>, <code>compact</code>, <code>footprint</code>, and <code>skew</code>.
        <a class="docs-link" href="https://suite2p.readthedocs.io/en/latest/outputs/#statnpy-fields" target="_blank" rel="noopener noreferrer">Suite2p stat.npy field definitions</a>
      </div>
      <div class="filter-controls">
        <label>Skew min <input id="skewMin" type="number" step="0.01"></label>
        <label>Skew max <input id="skewMax" type="number" step="0.01"></label>
        <label>Aspect min <input id="aspectMin" type="number" step="0.01"></label>
        <label>Aspect max <input id="aspectMax" type="number" step="0.01"></label>
        <label>Footprint min <input id="footprintMin" type="number" step="0.01"></label>
        <label>Footprint max <input id="footprintMax" type="number" step="0.01"></label>
        <label>Compact min <input id="compactMin" type="number" step="0.01"></label>
        <label>Compact max <input id="compactMax" type="number" step="0.01"></label>
      </div>
      <div class="filter-subsection-title source-heading">
        <span>Custom Metrics</span>
        <button class="info-button" type="button" data-info-target="customMetricSources" aria-expanded="false">(i)</button>
      </div>
      <div id="customMetricSources" class="info-box" hidden>
        Connectivity is calculated by preprocessing QC as the number of 4-connected components in each ROI pixel mask.
        Event SNR and decay tau are calculated from the raw Suite2p-derived dF/F trace for each ROI.
        <a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/main/2p_post_process_module_202404/modules/QualControlDataIO.py#L29-L36" target="_blank" rel="noopener noreferrer">Connectivity calculation code</a>
        <a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/docs/summary-generation-examples/utils_2p/roi_labels.py#L121-L157" target="_blank" rel="noopener noreferrer">Event SNR calculation code</a>
        <a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/docs/summary-generation-examples/utils_2p/roi_labels.py#L176-L216" target="_blank" rel="noopener noreferrer">Decay tau calculation code</a>
      </div>
      <div class="filter-controls">
        <label>Max connectivity <input id="maxConnect" type="number" min="0" step="1"></label>
        <label>Event SNR min <input id="eventSnrMin" type="number" step="0.01" placeholder="optional"></label>
        <label>Event SNR max <input id="eventSnrMax" type="number" step="0.01" placeholder="optional"></label>
        <label>Decay tau min (s) <input id="decayTauMin" type="number" step="0.01" placeholder="optional"></label>
        <label>Decay tau max (s) <input id="decayTauMax" type="number" step="0.01" placeholder="optional"></label>
        <button id="resetFilter">Reset QC thresholds</button>
        <button id="applyFilterToLabels">Apply Filters</button>
      </div>
    </div>
    <div class="dialog-actions"><button id="closeMorphologyDialog" type="button">Close</button></div>
  </dialog>
  <dialog id="sortDialog">
    <div class="dialog-title">Sort ROIs and dF/Fs</div>
    <div class="source-heading">
      <button class="info-button" type="button" data-info-target="sortSuite2pSources" aria-expanded="false">(i) Suite2p metrics</button>
      <button class="info-button" type="button" data-info-target="sortCustomSources" aria-expanded="false">(i) custom metrics</button>
    </div>
    <div id="sortSuite2pSources" class="info-box" hidden>
      Suite2p morphology sort options come from ROI <code>stat.npy</code> fields.
      <a class="docs-link" href="https://suite2p.readthedocs.io/en/latest/outputs/#statnpy-fields" target="_blank" rel="noopener noreferrer">Suite2p stat.npy field definitions</a>
    </div>
    <div id="sortCustomSources" class="info-box" hidden>
      Connectivity is calculated by preprocessing QC as the number of 4-connected components in each ROI pixel mask.
      Event SNR and decay tau are calculated from the raw Suite2p-derived dF/F trace for each ROI.
      <a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/main/2p_post_process_module_202404/modules/QualControlDataIO.py#L29-L36" target="_blank" rel="noopener noreferrer">Connectivity calculation code</a>
      <a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/docs/summary-generation-examples/utils_2p/roi_labels.py#L121-L157" target="_blank" rel="noopener noreferrer">Event SNR calculation code</a>
      <a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/docs/summary-generation-examples/utils_2p/roi_labels.py#L176-L216" target="_blank" rel="noopener noreferrer">Decay tau calculation code</a>
    </div>
    <div class="trace-sort">
      <label>Sort visible ROIs by
        <select id="sortMetric">
          <optgroup label="Suite2p Morphology Metrics">
            <option value="roi_area">ROI area (px)</option>
            <option value="skew">Skew</option>
            <option value="aspect">Aspect ratio</option>
            <option value="compact">Compactness</option>
            <option value="footprint">Footprint</option>
            <option value="original" selected>Original Suite2p index</option>
          </optgroup>
          <optgroup label="Custom Metrics">
            <option value="event_snr">Event SNR</option>
            <option value="decay_tau_seconds">Calcium decay constant (tau)</option>
            <option value="connectivity">Connectivity</option>
          </optgroup>
        </select>
      </label>
      <label>Sort order
        <select id="sortDirection">
          <option value="desc">Highest first</option>
          <option value="asc" selected>Lowest first</option>
        </select>
      </label>
      <button id="applySort">Apply sort</button>
    </div>
    <div class="dialog-actions"><button id="closeSortDialog" type="button">Close</button></div>
  </dialog>
  <dialog id="saveLabelsDialog">
    <div class="dialog-header">
      <div class="dialog-title">Save Labels</div>
      <button id="saveLabelsInfo" class="info-button" type="button" aria-expanded="false" aria-controls="saveLabelsHelp">(i)</button>
    </div>
    <div id="saveLabelsHelp" class="info-box" hidden>
      Save current state into HTML downloads a reviewed HTML copy that preserves labels and custom morphology presets inside the file.
      Save roi_manual_labels.npy downloads a three-column NumPy mask for downstream scripts: full Suite2p good mask, morphology-filtered good mask, and morphology-filtered good-or-unsure mask.
    </div>
    <div class="save-options">
      <div class="save-option">
        <button id="saveHtmlWithLabels">Save current state into HTML</button>
        <span class="note">Use this to save the current state (labels, presets) of the .html to return to after closing the browser.</span>
      </div>
      <div class="save-option">
        <button id="saveManualLabels">Save roi_manual_labels.npy</button>
        <span class="note">Exports the downstream NumPy mask with full Suite2p, morphology-filtered good, and morphology-filtered good-or-unsure columns.</span>
      </div>
      <a class="docs-link" href="https://najafi-laboratory.github.io/2p_imaging/roi-reviewer-exports/#2-export-format-and-downstream-use" target="_blank" rel="noopener noreferrer">Output format details</a>
    </div>
    <div class="dialog-actions"><button id="closeSaveLabelsDialog" type="button">Close</button></div>
  </dialog>
  <dialog id="labelAllDialog">
    <div class="dialog-title">Label all visible ROIs as ...</div>
    <div class="note">This applies only to the ROIs currently visible in the reviewer.</div>
    <div class="bulk-label-controls">
      <label>Label
        <select id="labelAllValue">
          <option value="1">Good</option>
          <option value="0">Bad</option>
          <option value="2">Unsure</option>
          <option value="-1">Unlabeled</option>
        </select>
      </label>
    </div>
    <div class="dialog-actions">
      <button id="applyLabelAll" type="button">Apply</button>
      <button id="closeLabelAllDialog" type="button">Close</button>
    </div>
  </dialog>
  <div class="review-main">
    <div class="viewer-column">
      <div class="fov-row">
        <div class="fov-review">
          <div class="grid {fov_grid_class}">
            <div class="panel"><div class="title">Green functional mean</div><div class="imagewrap"><img id="green"><svg class="overlay" preserveAspectRatio="xMidYMid meet"></svg></div></div>
            {red_panel}
            <div class="panel"><div class="title">ROI masks</div><div class="imagewrap"><img id="mask"><svg class="overlay" preserveAspectRatio="xMidYMid meet"></svg></div></div>
          </div>
        </div>
        <div class="control-column">
          <div class="panel morphology-card">
            <div class="qc-header"><strong>ROI QC Filters</strong><span id="targetStructureInline" class="qc-current"></span></div>
            <div id="filterSummaryInline" class="filter-summary"></div>
            <button id="openMorphologyDialog" type="button">Edit ROI Metric Filters</button>
            <label><input id="showAllRois" type="checkbox"> Show all Filtered ROIs</label>
            <div class="sort-card">
              <div class="sort-header"><strong>Sorting</strong><span id="sortCurrent" class="sort-current"></span></div>
              <button id="openSortDialog" type="button">Sort ROIs by Metrics</button>
            </div>
          </div>
          <div class="panel label-controls">
            <strong>Manual ROI Labeler</strong>
            <label>Selected ROI (Suite2p Index) <input id="roiInput" type="number" min="0" value="0"> <span id="selectedSortPosition" class="note"></span></label>
            <details class="roi-details">
              <summary id="roiDetailsSummary">Selected ROI Details</summary>
              <div id="readout"></div>
            </details>
            <div class="button-row">
              <button id="markGood" class="good">Good (G)</button>
              <button id="markBad" class="bad">Bad (B)</button>
            </div>
            <div class="button-row">
              <button id="markUnsure" class="unsure">Unsure (S)</button>
              <button id="markUnlabeled" class="unlabeled">Unlabeled (U)</button>
            </div>
            <button id="openLabelAllDialog" type="button">Label all as ...</button>
            <div class="nav-row">
              <button id="previousRoi" class="nav-button" title="Previous visible ROI (Left arrow)">&#8592; Previous</button>
              <button id="nextRoi" class="nav-button" title="Next visible ROI (Right arrow)">Next &#8594;</button>
            </div>
            <span class="note">Keyboard: G/B/S/U label; left/right arrows select the previous/next visible ROI.</span>
            <span id="labelCounts"></span>
            <button id="openExclusions">Open ROI metric spreadsheet</button>
            <button id="openSaveLabelsDialog" type="button">Save Labels</button>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="title" id="traceTitle">Selected ROI dF/F</div>
        <div class="trace-loader" id="traceLoader" style="display:none;">
          <input id="dffFile" type="file" accept=".npy">
          <button id="loadDffFile">Load dF/F file</button>
        </div>
        <div class="note" id="traceLoadNote"></div>
        <canvas id="traceCanvas"></canvas>
        <div class="note">Wheel or drag to zoom/pan time. Double-click to reset.</div>
        <div class="controls">
          <strong>Trace window</strong>
          <label>Start s <input id="timeStart" type="number" min="0" step="0.001" value="0"></label>
          <label>End s <input id="timeEnd" type="number" min="0" step="0.001" value="0"></label>
          <button id="reset">Reset zoom</button>
        </div>
      </div>
    </div>
  </div>
  <div class="plots">
    <div class="panel">
      <div class="title" id="stackTitle">dF/F, stacked ROIs</div>
      <div class="controls">
        <strong>Stacked trace range</strong>
        <label>First ROI <input id="yStart" type="number" min="0" value="0"></label>
        <label>Last ROI <input id="yEnd" type="number" min="0" value="0"></label>
      </div>
      <canvas id="stackCanvas"></canvas>
      <div class="note">Wheel to zoom time.</div>
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
document.querySelectorAll(".imagewrap img").forEach(img => img.addEventListener("load", syncControlColumnHeight));
document.getElementById("mask").style.display = "none";
document.getElementById("meta").textContent = `${{data.nRois}} ROIs | ${{data.nFrames.toLocaleString()}} frames | ${{data.frameRate.toFixed(3)}} Hz${{data.redAvailable ? "" : " | no red channel detected"}}`;
document.querySelectorAll(".overlay").forEach(svg => svg.setAttribute("viewBox", `0 0 ${{data.imageWidth}} ${{data.imageHeight}}`));
document.getElementById("targetStructureSummary").textContent = `Target structure: ${{data.targetStructure}}`;
document.getElementById("targetStructureInline").textContent = `Current preset target structure: ${{data.targetStructure}}`;
document.getElementById("roiInput").max = Math.max(...data.suite2pIndices);
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
function parseNpy(arrayBuffer) {{
  const headerBytes = new Uint8Array(arrayBuffer);
  if (headerBytes.length < 10 || headerBytes[0] !== 0x93 || headerBytes[1] !== 0x4e || headerBytes[2] !== 0x55 || headerBytes[3] !== 0x4d || headerBytes[4] !== 0x50 || headerBytes[5] !== 0x59) {{
    throw new Error("Not a NumPy .npy file");
  }}
  const major = headerBytes[6];
  let offset = 0;
  if (major === 1) {{
    offset = 10 + new DataView(arrayBuffer).getUint16(8, true);
  }} else if (major === 2 || major === 3) {{
    offset = 12 + new DataView(arrayBuffer).getUint32(8, true);
  }} else {{
    throw new Error(`Unsupported .npy version: ${{major}}`);
  }}
  const headerText = new TextDecoder("latin1").decode(headerBytes.subarray(major === 1 ? 10 : 12, offset)).trim();
  const descrMatch = /'descr':\\s*'([^']+)'/.exec(headerText);
  const shapeMatch = /'shape':\\s*\\(([^\\)]*)\\)/.exec(headerText);
  const fortranMatch = /'fortran_order':\\s*(True|False)/.exec(headerText);
  if (!descrMatch || !shapeMatch || !fortranMatch) throw new Error("Could not parse .npy header");
  if (fortranMatch[1] !== "False") throw new Error("Fortran-ordered .npy files are not supported");
  const shape = shapeMatch[1].split(",").map(part => Number(part.trim())).filter(Number.isFinite);
  const descr = descrMatch[1];
  const dataOffset = offset;
  if (descr === "<f4" || descr === "|f4") {{
    return {{array: new Float32Array(arrayBuffer, dataOffset), shape}};
  }}
  if (descr === "<f8" || descr === "|f8") {{
    const source = new Float64Array(arrayBuffer, dataOffset);
    const target = new Float32Array(source.length);
    for (let i = 0; i < source.length; i++) target[i] = source[i];
    return {{array: target, shape}};
  }}
  throw new Error(`Unsupported .npy dtype: ${{descr}}`);
}}
let dff = null;
function setDffFromArrayBuffer(arrayBuffer) {{
  const parsed = parseNpy(arrayBuffer);
  dff = parsed.array;
  if (parsed.shape.length >= 2) {{
    data.nRois = parsed.shape[0];
    data.nFrames = parsed.shape[1];
  }}
  document.getElementById("traceLoadNote").textContent = `Loaded dF/F file with ${{data.nRois}} ROIs x ${{data.nFrames}} frames.`;
  draw();
}}
if (data.dffStorageMode === "embedded" && data.dff) {{
  dff = b64f32(data.dff);
}} else {{
  const loader = document.getElementById("traceLoader");
  const note = document.getElementById("traceLoadNote");
  loader.style.display = "";
  note.textContent = `This session was too large to fully embed (estimated ${{(data.estimatedEmbeddedDffBytes / (1024 * 1024)).toFixed(1)}} MB). Load ${{data.dffSidecarName || "the sidecar .npy file"}} to enable the trace viewer.`;
}}
const presetPass = Uint8Array.from(data.presetExclusionReasons, reasons => reasons.length === 0 ? 1 : 0);
const labels = new Int8Array(data.nRois);
for (let roi = 0; roi < data.nRois; roi++) {{
  labels[roi] = presetPass[roi] ? -1 : 0;
}}
if (Array.isArray(data.initialLabels) && data.initialLabels.length === data.nRois) {{
  for (let roi = 0; roi < data.nRois; roi++) {{
    const label = Number(data.initialLabels[roi]);
    if (label === -1 || label === 0 || label === 1 || label === 2) labels[roi] = label;
  }}
}}
const filterPass = new Uint8Array(data.nRois);
let customPresets = data.customMorphologyPresets && typeof data.customMorphologyPresets === "object" ? {{...data.customMorphologyPresets}} : {{}};
const defaultFilter = data.morphologyPresets[data.targetStructure] || data.morphologyPresets.all_rois;
let selected = 0, x0 = 0, x1 = data.nFrames - 1, y0 = 0, y1 = 0, visibleRois = [];
let appliedSortMetric = "original";
let appliedSortDirection = "asc";

function fit(canvas) {{
  const r = window.devicePixelRatio || 1, box = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(box.width * r));
  canvas.height = Math.max(1, Math.round(box.height * r));
}}
function syncControlColumnHeight() {{
  const fovReview = document.querySelector(".fov-review");
  const controls = document.querySelector(".control-column");
  if (!fovReview || !controls) return;
  if (window.matchMedia("(max-width: 1100px)").matches) {{
    controls.style.height = "";
    controls.style.maxHeight = "";
    controls.style.overflowY = "";
    return;
  }}
  const fovHeight = Math.round(fovReview.getBoundingClientRect().height);
  if (fovHeight > 0) {{
    controls.style.height = `${{fovHeight}}px`;
    controls.style.maxHeight = `${{fovHeight}}px`;
    controls.style.overflowY = "auto";
  }}
}}
function trace(roi) {{
  if (!dff) return null;
  return dff.subarray(roi * data.nFrames, (roi + 1) * data.nFrames);
}}
function val(roi, frame) {{
  if (!dff) return NaN;
  return dff[roi * data.nFrames + frame];
}}
function metricValue(roi, metric) {{
  if (metric === "event_snr") return data.dffMetrics[roi].event_snr;
  if (metric === "decay_tau_seconds") return data.dffMetrics[roi].decay_tau_seconds;
  if (metric === "roi_area") return data.dffMetrics[roi].roi_area;
  if (metric === "connectivity") return data.morphology[roi].connect;
  if (metric === "skew") return data.morphology[roi].skew;
  if (metric === "aspect") return data.morphology[roi].aspect;
  if (metric === "compact") return data.morphology[roi].compact;
  if (metric === "footprint") return data.morphology[roi].footprint;
  if (metric === "original" || metric === "suite2p_index") return data.suite2pIndices[roi];
  return roi;
}}
function metricLabel(metric) {{
  if (metric === "event_snr") return "Event SNR";
  if (metric === "decay_tau_seconds") return "Calcium decay constant (tau)";
  if (metric === "roi_area") return "ROI area (px)";
  if (metric === "connectivity") return "Connectivity";
  if (metric === "skew") return "Skew";
  if (metric === "aspect") return "Aspect ratio";
  if (metric === "compact") return "Compactness";
  if (metric === "footprint") return "Footprint";
  if (metric === "original" || metric === "suite2p_index") return "original Suite2p index";
  return metric.replace("_", " ");
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
function roiFromSuite2pIndex(value) {{
  const suite2pIndex = Math.round(Number(value));
  const exact = data.suite2pIndices.indexOf(suite2pIndex);
  if (exact >= 0) return exact;
  return Math.max(0, Math.min(data.nRois - 1, suite2pIndex));
}}
function currentSortPositionText() {{
  const position = visibleRois.includes(selected) ? visibleRois.indexOf(selected) + 1 : 0;
  const total = visibleRois.length;
  return `${{position}}/${{total}} by ${{metricLabel(appliedSortMetric)}} ${{appliedSortDirection}}`;
}}
function updateSortCurrent() {{
  document.getElementById("sortCurrent").textContent = `Current order: ${{metricLabel(appliedSortMetric)}}, ${{appliedSortDirection}}`;
  document.getElementById("selectedSortPosition").textContent = currentSortPositionText();
}}
function applySort() {{
  appliedSortMetric = document.getElementById("sortMetric").value;
  appliedSortDirection = document.getElementById("sortDirection").value;
  updateVisibleRois();
  updateSortCurrent();
  if (visibleRois.includes(selected)) setSelected(selected);
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
  const metrics = data.morphology[selected];
  const dffMetrics = data.dffMetrics[selected];
  const suite2pRoi = data.suite2pIndices[selected];
  document.getElementById("roiInput").value = suite2pRoi;
  document.getElementById("roiDetailsSummary").textContent = "Selected ROI Details";
  document.getElementById("readout").textContent = `area ${{fmt(dffMetrics.roi_area)}} px | skew ${{fmt(metrics.skew)}} connect ${{metrics.connect}} aspect ${{fmt(metrics.aspect)}} compact ${{fmt(metrics.compact)}} footprint ${{fmt(metrics.footprint)}} | event SNR ${{fmt(dffMetrics.event_snr)}} | decay tau ${{fmt(dffMetrics.decay_tau_seconds)}} s`;
  document.getElementById("traceTitle").textContent = `Selected ROI - Suite2p Original Index ${{suite2pRoi}}/${{data.nRois}}, Current Sort ${{currentSortPositionText()}}`;
  document.querySelectorAll(".roi").forEach(c => c.classList.toggle("selected", Number(c.dataset.roi) === selected));
  updateLabelControls();
  updateSortCurrent();
  draw();
}}
function updateVisibleRois() {{
  const showAll = document.getElementById("showAllRois").checked;
  const rois = [];
  for (let roi = 0; roi < data.nRois; roi++) if (showAll || filterPass[roi]) rois.push(roi);
  visibleRois = sortVisibleRois(rois);
  if (!visibleRois.length) visibleRois = Array.from({{length: data.nRois}}, (_v, roi) => roi);
  document.getElementById("yStart").max = visibleRois.length - 1;
  document.getElementById("yEnd").max = visibleRois.length - 1;
  y0 = Math.max(0, Math.min(y0, visibleRois.length - 1));
  y1 = Math.max(y0, Math.min(y1 || Math.min(19, visibleRois.length - 1), visibleRois.length - 1));
  document.getElementById("yStart").value = y0;
  document.getElementById("yEnd").value = y1;
  document.querySelectorAll(".roi").forEach(path => {{
    const roi = Number(path.dataset.roi);
    path.style.display = (showAll || filterPass[roi]) ? "" : "none";
  }});
  if (!visibleRois.includes(selected)) setSelected(visibleRois[0]);
  else draw();
}}
function updateLabelControls() {{
  const label = labels[selected];
  document.getElementById("markGood").classList.toggle("active", label === 1);
  document.getElementById("markBad").classList.toggle("active", label === 0);
  document.getElementById("markUnsure").classList.toggle("active", label === 2);
  document.getElementById("markUnlabeled").classList.toggle("active", label === -1);
  let good = 0, bad = 0, unsure = 0, unlabeled = 0;
  for (const value of labels) {{ if (value === 1) good++; else if (value === 0) bad++; else if (value === 2) unsure++; else unlabeled++; }}
  document.getElementById("labelCounts").textContent = `${{good}} good | ${{bad}} bad | ${{unsure}} unsure | ${{unlabeled}} unlabeled`;
}}
function labelName(label) {{
  if (label === 1) return "good";
  if (label === 0) return "bad";
  if (label === 2) return "unsure";
  return "unlabeled";
}}
function fmt(value) {{
  if (value === null || value === undefined) return "nan";
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(3) : "nan";
}}
function filterValue(id) {{
  const element = document.getElementById(id);
  if (!element) return NaN;
  const raw = element.value;
  if (String(raw).trim() === "") return NaN;
  const value = Number(raw);
  return Number.isFinite(value) ? value : NaN;
}}
function passesLower(value, lower) {{
  return !Number.isFinite(lower) || (Number.isFinite(value) && value >= lower);
}}
function passesUpper(value, upper) {{
  return !Number.isFinite(upper) || (Number.isFinite(value) && value <= upper);
}}
function readFilter() {{
  return {{
    skewMin: filterValue("skewMin"), skewMax: filterValue("skewMax"),
    maxConnect: filterValue("maxConnect"),
    aspectMin: filterValue("aspectMin"), aspectMax: filterValue("aspectMax"),
    footprintMin: filterValue("footprintMin"), footprintMax: filterValue("footprintMax"),
    compactMin: filterValue("compactMin"), compactMax: filterValue("compactMax"),
    eventSnrMin: filterValue("eventSnrMin"), eventSnrMax: filterValue("eventSnrMax"),
    decayTauMin: filterValue("decayTauMin"), decayTauMax: filterValue("decayTauMax"),
  }};
}}
function normalizeFilter(filter) {{
  const normalized = {{}};
  for (const key of ["skewMin","skewMax","maxConnect","aspectMin","aspectMax","footprintMin","footprintMax","compactMin","compactMax"]) {{
    if (filter[key] === null || filter[key] === undefined || String(filter[key]).trim() === "") {{
      normalized[key] = null;
      continue;
    }}
    const value = Number(filter[key]);
    normalized[key] = Number.isFinite(value) ? value : null;
  }}
  for (const key of ["eventSnrMin","eventSnrMax","decayTauMin","decayTauMax"]) {{
    if (filter[key] === null || filter[key] === undefined || String(filter[key]).trim() === "") {{
      normalized[key] = null;
      continue;
    }}
    const value = Number(filter[key]);
    normalized[key] = Number.isFinite(value) ? value : null;
    if ((key.endsWith("Max")) && normalized[key] === 0) normalized[key] = null;
  }}
  return normalized;
}}
function writeFilter(filter) {{
  const normalized = normalizeFilter(filter);
  for (const [id, value] of Object.entries(normalized)) {{
    const input = document.getElementById(id);
    if (input) input.value = value === null || value === undefined ? "" : value;
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
function currentPresetName() {{
  return document.getElementById("presetName").value.trim();
}}
function saveCurrentPresetToPage() {{
  const name = currentPresetName();
  if (!name) {{ alert("Enter a custom preset name first."); return null; }}
  try {{
    customPresets[name] = normalizeFilter(readFilter());
  }} catch (error) {{
    alert(`Could not save preset: ${{error.message}}`);
    return null;
  }}
  populatePresetSelect(name);
  document.getElementById("filterPreset").value = `custom:${{name}}`;
  return name;
}}
function passesFilter(roi, metrics, filter) {{
  const dffMetrics = data.dffMetrics[roi];
  return (
    passesLower(metrics.footprint, filter.footprintMin) && passesUpper(metrics.footprint, filter.footprintMax) &&
    passesLower(metrics.skew, filter.skewMin) && passesUpper(metrics.skew, filter.skewMax) &&
    passesLower(metrics.aspect, filter.aspectMin) && passesUpper(metrics.aspect, filter.aspectMax) &&
    passesLower(metrics.compact, filter.compactMin) && passesUpper(metrics.compact, filter.compactMax) &&
    passesUpper(metrics.connect, filter.maxConnect) &&
    passesLower(dffMetrics.event_snr, filter.eventSnrMin) &&
    passesUpper(dffMetrics.event_snr, filter.eventSnrMax) &&
    passesLower(dffMetrics.decay_tau_seconds, filter.decayTauMin) &&
    passesUpper(dffMetrics.decay_tau_seconds, filter.decayTauMax)
  );
}}
function morphologyReasons(metrics, dffMetrics, filter) {{
  const reasons = [];
  if (!passesLower(metrics.footprint, filter.footprintMin)) reasons.push(`footprint ${{fmt(metrics.footprint)}} below ${{filter.footprintMin}}`);
  if (!passesUpper(metrics.footprint, filter.footprintMax)) reasons.push(`footprint ${{fmt(metrics.footprint)}} above ${{filter.footprintMax}}`);
  if (!passesLower(metrics.skew, filter.skewMin)) reasons.push(`skew ${{fmt(metrics.skew)}} below ${{filter.skewMin}}`);
  if (!passesUpper(metrics.skew, filter.skewMax)) reasons.push(`skew ${{fmt(metrics.skew)}} above ${{filter.skewMax}}`);
  if (!passesLower(metrics.aspect, filter.aspectMin)) reasons.push(`aspect_ratio ${{fmt(metrics.aspect)}} below ${{filter.aspectMin}}`);
  if (!passesUpper(metrics.aspect, filter.aspectMax)) reasons.push(`aspect_ratio ${{fmt(metrics.aspect)}} above ${{filter.aspectMax}}`);
  if (!passesLower(metrics.compact, filter.compactMin)) reasons.push(`compact ${{fmt(metrics.compact)}} below ${{filter.compactMin}}`);
  if (!passesUpper(metrics.compact, filter.compactMax)) reasons.push(`compact ${{fmt(metrics.compact)}} above ${{filter.compactMax}}`);
  if (!passesUpper(metrics.connect, filter.maxConnect)) reasons.push(`connectivity ${{metrics.connect}} exceeds ${{filter.maxConnect}}`);
  if (!passesLower(dffMetrics.event_snr, filter.eventSnrMin)) reasons.push(`event SNR ${{fmt(dffMetrics.event_snr)}} below ${{filter.eventSnrMin}}`);
  if (!passesUpper(dffMetrics.event_snr, filter.eventSnrMax)) reasons.push(`event SNR ${{fmt(dffMetrics.event_snr)}} above ${{filter.eventSnrMax}}`);
  if (!passesLower(dffMetrics.decay_tau_seconds, filter.decayTauMin)) reasons.push(`decay tau ${{fmt(dffMetrics.decay_tau_seconds)}} below ${{filter.decayTauMin}}`);
  if (!passesUpper(dffMetrics.decay_tau_seconds, filter.decayTauMax)) reasons.push(`decay tau ${{fmt(dffMetrics.decay_tau_seconds)}} above ${{filter.decayTauMax}}`);
  return reasons;
}}
function evaluateFilter() {{
  const filter = readFilter();
  let pass = 0;
  for (let roi = 0; roi < data.nRois; roi++) {{
    filterPass[roi] = passesFilter(roi, data.morphology[roi], filter) ? 1 : 0;
    pass += filterPass[roi];
  }}
  const summary = `${{pass}} / ${{data.nRois}} original Suite2p ROIs pass the current morphology and custom metric filters.`;
  document.getElementById("filterSummary").textContent = summary;
  document.getElementById("filterSummaryInline").textContent = summary;
  draw();
}}
function resetFilter() {{
  loadSelectedPreset();
}}
function applyFilterToLabels() {{
  const current = selected;
  for (let roi = 0; roi < data.nRois; roi++) {{
    labels[roi] = filterPass[roi] ? -1 : 0;
  }}
  updateVisibleRois();
  if (labels[current] !== 0 || document.getElementById("showAllRois").checked) setSelected(current);
}}
function setLabel(label) {{
  labels[selected] = label;
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
  const firstRow = Math.max(0, Math.floor(y0));
  const lastRow = Math.min(visibleRois.length - 1, Math.ceil(y1));
  const selectedPosition = visibleRois.includes(selected) ? visibleRois.indexOf(selected) + 1 : 0;
  document.getElementById("stackTitle").textContent = `stacked raw dF/F, selected is ${{selectedPosition}}/${{visibleRois.length}} sorted by ${{metricLabel(appliedSortMetric)}} ${{appliedSortDirection}}`;
  const canvas = document.getElementById("stackCanvas"); fit(canvas); const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, l = 62, r = 16, t = 14, b = 56, pw = w-l-r, ph = h-t-b;
  ctx.clearRect(0,0,w,h); ctx.fillStyle = "#fff"; ctx.fillRect(0,0,w,h); drawAxes(ctx,w,h,l,t,pw,ph,"time (s)","ROI index");
  if (!dff) {{
    ctx.fillStyle = "#475467"; ctx.font = `${{14 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.fillText("Load the dF/F file to enable stacked traces.", l + pw / 2, t + ph / 2);
    return;
  }}
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
  if (!tr) {{
    ctx.fillStyle = "#475467"; ctx.font = `${{14 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.fillText("Load the dF/F file to enable this trace.", l + pw / 2, t + ph / 2);
    return;
  }}
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
document.getElementById("roiInput").addEventListener("change", e => setSelected(roiFromSuite2pIndex(e.target.value)));
document.getElementById("timeStart").addEventListener("change", () => setTimeWindow(document.getElementById("timeStart").value, document.getElementById("timeEnd").value));
document.getElementById("timeEnd").addEventListener("change", () => setTimeWindow(document.getElementById("timeStart").value, document.getElementById("timeEnd").value));
document.getElementById("yStart").addEventListener("change", e => {{ y0=Number(e.target.value); draw(); }});
document.getElementById("yEnd").addEventListener("change", e => {{ y1=Number(e.target.value); draw(); }});
document.getElementById("reset").addEventListener("click", reset);
document.getElementById("markGood").addEventListener("click", () => setLabel(1));
document.getElementById("markBad").addEventListener("click", () => setLabel(0));
document.getElementById("markUnsure").addEventListener("click", () => setLabel(2));
document.getElementById("markUnlabeled").addEventListener("click", () => setLabel(-1));
const labelAllDialog = document.getElementById("labelAllDialog");
document.getElementById("openLabelAllDialog").addEventListener("click", () => {{
  if (typeof labelAllDialog.showModal === "function") labelAllDialog.showModal();
  else labelAllDialog.setAttribute("open", "");
}});
document.getElementById("closeLabelAllDialog").addEventListener("click", () => {{
  if (typeof labelAllDialog.close === "function") labelAllDialog.close();
  else labelAllDialog.removeAttribute("open");
}});
document.getElementById("applyLabelAll").addEventListener("click", () => {{
  const label = Number(document.getElementById("labelAllValue").value);
  for (const roi of visibleRois) {{
    labels[roi] = label;
  }}
  updateLabelControls();
  updateVisibleRois();
  if (typeof labelAllDialog.close === "function") labelAllDialog.close();
  else labelAllDialog.removeAttribute("open");
}});
document.getElementById("previousRoi").addEventListener("click", () => moveVisible(-1));
document.getElementById("nextRoi").addEventListener("click", () => moveVisible(1));
document.getElementById("showAllRois").addEventListener("change", updateVisibleRois);
document.getElementById("applySort").addEventListener("click", applySort);
const sortDialog = document.getElementById("sortDialog");
document.getElementById("openSortDialog").addEventListener("click", () => {{
  if (typeof sortDialog.showModal === "function") sortDialog.showModal();
  else sortDialog.setAttribute("open", "");
}});
document.getElementById("closeSortDialog").addEventListener("click", () => {{
  if (typeof sortDialog.close === "function") sortDialog.close();
  else sortDialog.removeAttribute("open");
}});
const dffFileInput = document.getElementById("dffFile");
document.getElementById("loadDffFile").addEventListener("click", () => dffFileInput.click());
dffFileInput.addEventListener("change", () => {{
  const file = dffFileInput.files && dffFileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {{
    try {{
      setDffFromArrayBuffer(reader.result);
    }} catch (error) {{
      document.getElementById("traceLoadNote").textContent = `Could not load dF/F file: ${{error.message}}`;
    }}
  }};
  reader.readAsArrayBuffer(file);
}});
document.getElementById("openExclusions").addEventListener("click", () => {{
  const filter = readFilter();
  function csvEscape(value) {{
    const text = String(value ?? "");
    return /[",\\n]/.test(text) ? `"${{text.replaceAll('"', '""')}}"` : text;
  }}
  function td(value, failed = false) {{
    return `<td${{failed ? ' class="metric-fail"' : ""}}>${{value}}</td>`;
  }}
  function labelTd(label) {{
    const cls = label === "good" ? "label-good" : label === "bad" ? "label-bad" : label === "unsure" ? "label-unsure" : "";
    return `<td${{cls ? ` class="${{cls}}"` : ""}}>${{label}}</td>`;
  }}
  const rowsData = data.morphology.map((metrics, roi) => {{
    const dffMetrics = data.dffMetrics[roi];
    const reasons = morphologyReasons(metrics, dffMetrics, filter);
    if (labels[roi] === 0) reasons.push("manual/current label: bad");
    else if (labels[roi] === 2) reasons.push("manual/current label: unsure");
    else if (labels[roi] === -1 && reasons.length === 0) reasons.push("unlabeled");
    return {{
      suite2p: data.suite2pIndices[roi],
      label: labelName(labels[roi]),
      footprint: metrics.footprint,
      skew: metrics.skew,
      aspect: metrics.aspect,
      compact: metrics.compact,
      connectivity: metrics.connect,
      eventSnr: dffMetrics.event_snr,
      decayTau: dffMetrics.decay_tau_seconds,
      fail: {{
        footprint: !(metrics.footprint >= filter.footprintMin && metrics.footprint <= filter.footprintMax),
        skew: !(metrics.skew >= filter.skewMin && metrics.skew <= filter.skewMax),
        aspect: !(metrics.aspect >= filter.aspectMin && metrics.aspect <= filter.aspectMax),
        compact: !(metrics.compact >= filter.compactMin && metrics.compact <= filter.compactMax),
        connectivity: !passesUpper(metrics.connect, filter.maxConnect),
        eventSnr: !(passesLower(dffMetrics.event_snr, filter.eventSnrMin) && passesUpper(dffMetrics.event_snr, filter.eventSnrMax)),
        decayTau: !(passesLower(dffMetrics.decay_tau_seconds, filter.decayTauMin) && passesUpper(dffMetrics.decay_tau_seconds, filter.decayTauMax)),
      }},
      reason: reasons.join("; ") || "included",
    }};
  }});
  const rows = rowsData.map(row => `<tr>${{
    td(row.suite2p) +
    labelTd(row.label) +
    td(fmt(row.footprint), row.fail.footprint) +
    td(fmt(row.skew), row.fail.skew) +
    td(fmt(row.aspect), row.fail.aspect) +
    td(fmt(row.compact), row.fail.compact) +
    td(row.connectivity, row.fail.connectivity) +
    td(fmt(row.eventSnr), row.fail.eventSnr) +
    td(fmt(row.decayTau), row.fail.decayTau) +
    td(row.reason)
  }}</tr>`).join("");
  const csvHeader = ["suite2p_index","label","footprint","skew","aspect_ratio","compact","connectivity","event_snr","decay_tau_seconds","reason"];
  const csvRows = rowsData.map(row => [
    row.suite2p, row.label, fmt(row.footprint), fmt(row.skew), fmt(row.aspect), fmt(row.compact),
    row.connectivity, fmt(row.eventSnr), fmt(row.decayTau), row.reason,
  ].map(csvEscape).join(","));
  const csv = [csvHeader.join(","), ...csvRows].join("\\n") + "\\n";
  const win = window.open("", "_blank");
  win.document.write(`<!doctype html><title>${{data.session}} ROI metrics</title><style>body{{font-family:Arial,sans-serif;margin:20px}}button{{margin:8px 0 12px;padding:6px 10px}}.metric-table-wrap{{max-height:80vh;overflow:auto;border:1px solid #d0d5dd}}.metric-table{{border-collapse:collapse;width:100%;font-size:12px}}.metric-table th,.metric-table td{{border:1px solid #e5e7eb;padding:4px 7px;text-align:right;white-space:nowrap}}.metric-table th{{position:sticky;top:0;background:#f8fafc;z-index:1}}.metric-table td:nth-child(1),.metric-table td:nth-child(2),.metric-table td:last-child{{text-align:left}}.metric-fail,.label-bad{{background:rgba(248,113,113,.28)}}.label-good{{background:rgba(34,197,94,.28)}}.label-unsure{{background:rgba(250,204,21,.28)}}</style><h1>${{data.session}} ROI metric spreadsheet</h1><p>Target structure: ${{data.targetStructure}}</p><button id="downloadCsv">Download CSV</button><div class="metric-table-wrap"><table class="metric-table"><thead><tr><th>Suite2p index</th><th>Label</th><th>Footprint</th><th>Skew</th><th>Aspect ratio</th><th>Compact</th><th>Connectivity</th><th>Event SNR</th><th>Decay tau (s)</th><th>Reason</th></tr></thead><tbody>${{rows}}</tbody></table></div><script>const csv = ${{JSON.stringify(csv)}}; document.getElementById("downloadCsv").addEventListener("click", () => {{ const blob = new Blob([csv], {{type: "text/csv"}}); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "${{data.session}}_roi_metric_spreadsheet.csv"; a.click(); URL.revokeObjectURL(a.href); }});<\\/script>`);
  win.document.close();
}});
function npyBlob(values, rows, cols) {{
  const encoder = new TextEncoder();
  let header = `{{'descr': '<f8', 'fortran_order': False, 'shape': (${{rows}}, ${{cols}}), }}`;
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
function reviewedMaskArray() {{
  const rows = data.suite2pRoiCount;
  const values = new Float64Array(rows * 3);
  for (let roi = 0; roi < data.nRois; roi++) {{
    const suite2pRoi = data.suite2pIndices[roi];
    const label = labels[roi];
    const morphPass = filterPass[roi] === 1;
    values[suite2pRoi * 3] = label === 1 ? 1 : 0;
    values[suite2pRoi * 3 + 1] = morphPass ? (label === 1 ? 1 : 0) : NaN;
    values[suite2pRoi * 3 + 2] = morphPass ? (label === 1 || label === 2 ? 1 : 0) : NaN;
  }}
  return values;
}}
async function saveManualLabels() {{
  const reviewedMasks = reviewedMaskArray();
  const blob = npyBlob(reviewedMasks, data.suite2pRoiCount, 3);
  if ("showSaveFilePicker" in window) {{
    try {{
      const handle = await window.showSaveFilePicker({{
        suggestedName: "roi_manual_labels.npy",
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
  downloadBlob(blob, "roi_manual_labels.npy");
}}
async function saveHtmlWithLabels() {{
  const payloadScript = document.getElementById("payload");
  const savedPayload = {{
    ...data,
    initialLabels: Array.from(labels),
    initialMorphologyFilter: readFilter(),
    customMorphologyPresets: customPresets,
    savedAt: new Date().toISOString(),
  }};
  const filename = `${{data.session}}_interactive_fov_roi_dff.html`;
  const htmlClone = document.documentElement.cloneNode(true);
  htmlClone.querySelector("#payload").textContent = JSON.stringify(savedPayload).replace(/</g, "\\u003c");
  htmlClone.querySelectorAll("dialog").forEach(dialog => {{
    dialog.removeAttribute("open");
  }});
  htmlClone.querySelectorAll(".info-box").forEach(box => {{
    box.setAttribute("hidden", "");
  }});
  htmlClone.querySelectorAll("[aria-expanded]").forEach(element => {{
    element.setAttribute("aria-expanded", "false");
  }});
  const blob = new Blob(["<!doctype html>\\n" + htmlClone.outerHTML], {{type: "text/html"}});
  if ("showSaveFilePicker" in window) {{
    try {{
      const handle = await window.showSaveFilePicker({{
        suggestedName: filename,
        types: [{{description: "HTML", accept: {{"text/html": [".html"]}}}}],
      }});
      const writable = await handle.createWritable();
      await writable.write(blob); await writable.close();
      return;
    }} catch (error) {{
      if (error.name === "AbortError") return;
      console.warn("Direct HTML save failed; downloading instead", error);
    }}
  }}
  downloadBlob(blob, filename);
}}
function exportPresetJson() {{
  const select = document.getElementById("filterPreset");
  const [kind, selectedName] = select.value.split(":", 2);
  const name = currentPresetName() || selectedName || data.targetStructure || "morphology_preset";
  let filter;
  try {{
    filter = normalizeFilter(readFilter());
  }} catch (error) {{
    alert(`Could not export preset: ${{error.message}}`);
    return;
  }}
  const payload = {{
    format: "utils_2p_morphology_preset_v1",
    name,
    source_session: data.session,
    target_structure: data.targetStructure,
    filter,
  }};
  downloadBlob(new Blob([JSON.stringify(payload, null, 2) + "\\n"], {{type: "application/json"}}), `${{name}}_morphology_preset.json`);
}}
function importPresetObject(payload) {{
  const name = String(payload.name || payload.preset_name || "").trim();
  const filter = payload.filter || payload.morphology_filter || payload;
  if (!name) throw new Error("Preset JSON must include a name.");
  customPresets[name] = normalizeFilter(filter);
  populatePresetSelect(name);
  document.getElementById("filterPreset").value = `custom:${{name}}`;
  document.getElementById("presetName").value = name;
  writeFilter(customPresets[name]);
}}
document.getElementById("saveManualLabels").addEventListener("click", saveManualLabels);
document.getElementById("saveHtmlWithLabels").addEventListener("click", saveHtmlWithLabels);
const saveLabelsDialog = document.getElementById("saveLabelsDialog");
const saveLabelsHelp = document.getElementById("saveLabelsHelp");
const saveLabelsInfo = document.getElementById("saveLabelsInfo");
document.getElementById("openSaveLabelsDialog").addEventListener("click", () => {{
  if (typeof saveLabelsDialog.showModal === "function") saveLabelsDialog.showModal();
  else saveLabelsDialog.setAttribute("open", "");
}});
document.getElementById("closeSaveLabelsDialog").addEventListener("click", () => {{
  if (typeof saveLabelsDialog.close === "function") saveLabelsDialog.close();
  else saveLabelsDialog.removeAttribute("open");
}});
saveLabelsInfo.addEventListener("click", () => {{
  const shouldShow = saveLabelsHelp.hasAttribute("hidden");
  saveLabelsHelp.toggleAttribute("hidden", !shouldShow);
  saveLabelsInfo.setAttribute("aria-expanded", String(shouldShow));
}});
document.querySelectorAll("[data-info-target]").forEach(button => {{
  button.addEventListener("click", () => {{
    const target = document.getElementById(button.dataset.infoTarget);
    if (!target) return;
    const shouldShow = target.hasAttribute("hidden");
    target.toggleAttribute("hidden", !shouldShow);
    button.setAttribute("aria-expanded", String(shouldShow));
  }});
}});
document.getElementById("exportPreset").addEventListener("click", exportPresetJson);
const presetFileInput = document.getElementById("presetFile");
document.getElementById("importPreset").addEventListener("click", () => presetFileInput.click());
presetFileInput.addEventListener("change", () => {{
  const file = presetFileInput.files && presetFileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {{
    try {{
      importPresetObject(JSON.parse(reader.result));
    }} catch (error) {{
      alert(`Could not import preset: ${{error.message}}`);
    }}
  }};
  reader.readAsText(file);
}});
document.getElementById("resetFilter").addEventListener("click", resetFilter);
document.getElementById("applyFilterToLabels").addEventListener("click", applyFilterToLabels);
const morphologyDialog = document.getElementById("morphologyDialog");
document.getElementById("openMorphologyDialog").addEventListener("click", () => {{
  if (typeof morphologyDialog.showModal === "function") morphologyDialog.showModal();
  else morphologyDialog.setAttribute("open", "");
}});
document.getElementById("closeMorphologyDialog").addEventListener("click", () => {{
  if (typeof morphologyDialog.close === "function") morphologyDialog.close();
  else morphologyDialog.removeAttribute("open");
}});
document.getElementById("filterPreset").addEventListener("change", loadSelectedPreset);
document.getElementById("savePreset").addEventListener("click", () => {{
  saveCurrentPresetToPage();
}});
document.getElementById("savePresetHtml").addEventListener("click", () => {{
  if (saveCurrentPresetToPage()) saveHtmlWithLabels();
}});
["skewMin","skewMax","maxConnect","aspectMin","aspectMax","footprintMin","footprintMax","compactMin","compactMax"].forEach(id => {{
  document.getElementById(id).addEventListener("change", evaluateFilter);
}});
window.addEventListener("keydown", event => {{
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if (event.key.toLowerCase() === "g") setLabel(1);
  else if (event.key.toLowerCase() === "b") setLabel(0);
  else if (event.key.toLowerCase() === "s") setLabel(2);
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
window.addEventListener("resize", () => {{ syncControlColumnHeight(); draw(); }});
makeOverlays(); syncTimeInputs(); populatePresetSelect(data.targetStructure); resetFilter();
if (data.initialMorphologyFilter) writeFilter(data.initialMorphologyFilter);
applySort(); syncControlColumnHeight(); setSelected(visibleRois[0]);
requestAnimationFrame(() => {{ syncControlColumnHeight(); draw(); }});
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
    target_structure: str | None = None,
) -> tuple[Path, Path]:
    session_dir = Path(session_data_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else session_dir
    out_dir.mkdir(parents=True, exist_ok=True)
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
    dff_label = "raw dF/F from Suite2p F.npy/Fneu.npy"
    fluo_probe = np.load(suite2p_dir / "F.npy", allow_pickle=False, mmap_mode="r")
    n_rois = min(int(fluo_probe.shape[0]), len(suite2p_stat))
    n_frames = int(fluo_probe.shape[1])
    stat = suite2p_stat[:n_rois]
    suite2p_indices = np.arange(n_rois, dtype=np.int64)
    estimated_embedded_dff_bytes = _estimate_embedded_dff_bytes(n_rois, n_frames)
    dff_storage_mode = "embedded" if estimated_embedded_dff_bytes <= EMBEDDED_DFF_BYTE_LIMIT else "file"
    dff_sidecar_name = None
    dff_sidecar_path = None
    if dff_storage_mode == "file":
        dff_sidecar_name = f"{session_dir.name}_dff.npy"
        dff_sidecar_path = out_dir / dff_sidecar_name
    dff, dff_metrics = _build_dff_and_metrics(
        session_dir,
        ops,
        n_rois,
        n_frames,
        frame_rate,
        dff_storage_mode,
        dff_sidecar_path,
    )
    dff_metrics = _add_roi_area_metrics(dff_metrics, stat)
    mask = _stat_to_mask(stat, np.asarray(mean_green).shape[:2])
    iscell_path = suite2p_dir / "iscell.npy"
    iscell = load_iscell(iscell_path, n_rois)
    if not iscell_path.exists():
        iscell[:, :] = 1.0
    suite2p_fingerprint = suite2p_stat_fingerprint(stat)
    morphology_metrics = roi_morphology_metrics(stat)
    target_structure = target_structure or _target_structure(pipeline_parameters, qc_parameters)
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
        n_rois=n_rois,
        n_frames=n_frames,
        dff_label=dff_label,
        frame_rate=frame_rate,
        dff_metrics=dff_metrics,
        dff=dff,
        dff_storage_mode=dff_storage_mode,
        dff_sidecar_name=dff_sidecar_name,
        estimated_embedded_dff_bytes=estimated_embedded_dff_bytes,
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
    parser.add_argument(
        "--target-structure",
        choices=["all_rois", *sorted(QC_PRESETS)],
        default=None,
        help="Override the morphology QC preset used to initialize the interactive reviewer.",
    )
    args = parser.parse_args()
    pdf_path, html_path = create_preprocessing_summary(
        args.session_data_path,
        output_dir=args.output_dir,
        pdf_name=args.pdf_name,
        html_name=args.html_name,
        target_structure=args.target_structure,
    )
    print(f"Saved preprocessing PDF: {pdf_path}")
    print(f"Saved interactive HTML: {html_path}")


if __name__ == "__main__":
    main()
