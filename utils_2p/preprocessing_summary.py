#!/usr/bin/env python3
"""Generate static and interactive preprocessing QC summaries for one session."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr

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


@dataclass(frozen=True)
class SummaryInputLayout:
    name: str
    stat_path: Path
    fluo_path: Path
    neuropil_path: Path
    iscell_path: Path
    spikes_paths: tuple[Path, ...]

    @property
    def dff_label(self) -> str:
        if self.name == "suite2p":
            return "raw dF/F from Suite2p F.npy/Fneu.npy"
        return f"raw dF/F from {self.name} fluo.npy/neuropil.npy"


def _path_status(path: Path) -> str:
    return str(path) if path.exists() else f"{path} (missing)"


def _layout_complete(layout: SummaryInputLayout) -> bool:
    return layout.stat_path.exists() and layout.fluo_path.exists() and layout.neuropil_path.exists()


def _direct_suite2p_layout(plane_dir: Path, session_dir: Path) -> SummaryInputLayout:
    return SummaryInputLayout(
        name="suite2p",
        stat_path=plane_dir / "stat.npy",
        fluo_path=plane_dir / "F.npy",
        neuropil_path=plane_dir / "Fneu.npy",
        iscell_path=plane_dir / "iscell.npy",
        spikes_paths=(session_dir / "spikes.h5", session_dir / "qc_results" / "spikes.h5"),
    )


def _direct_filtered_layout(qc_dir: Path, session_dir: Path) -> SummaryInputLayout:
    return SummaryInputLayout(
        name=qc_dir.name,
        stat_path=qc_dir / "stat.npy",
        fluo_path=qc_dir / "fluo.npy",
        neuropil_path=qc_dir / "neuropil.npy",
        iscell_path=qc_dir / "iscell.npy",
        spikes_paths=(qc_dir / "spikes.h5", session_dir / "spikes.h5"),
    )


def _suite2p_layout(session_dir: Path) -> SummaryInputLayout:
    plane_dir = session_dir / "suite2p" / "plane0"
    return _direct_suite2p_layout(plane_dir, session_dir)


def _qc_results_layout(session_dir: Path, dirname: str) -> SummaryInputLayout:
    qc_dir = session_dir / dirname
    return _direct_filtered_layout(qc_dir, session_dir)


def _resolve_summary_session_and_layout(
    input_path: Path, requested: str = "auto"
) -> tuple[Path, SummaryInputLayout]:
    """Resolve a session root and input layout from either a session or data directory."""

    if input_path.name == "plane0" and input_path.parent.name == "suite2p":
        session_dir = input_path.parents[1]
        return session_dir, _direct_suite2p_layout(input_path, session_dir)
    if input_path.name == "suite2p" and (input_path / "plane0").is_dir():
        session_dir = input_path.parent
        return session_dir, _direct_suite2p_layout(input_path / "plane0", session_dir)
    if input_path.name in {"qc_results", "manual_qc_results"}:
        session_dir = input_path.parent
        return session_dir, _direct_filtered_layout(input_path, session_dir)
    if (input_path / "stat.npy").exists() and (input_path / "F.npy").exists() and (input_path / "Fneu.npy").exists():
        session_dir = input_path.parents[1] if input_path.name == "plane0" else input_path.parent
        return session_dir, _direct_suite2p_layout(input_path, session_dir)
    if (
        (input_path / "stat.npy").exists()
        and (input_path / "fluo.npy").exists()
        and (input_path / "neuropil.npy").exists()
    ):
        return input_path.parent, _direct_filtered_layout(input_path, input_path.parent)
    session_dir = input_path
    return session_dir, _resolve_summary_input_layout(session_dir, requested)


def _resolve_summary_input_layout(session_dir: Path, requested: str = "auto") -> SummaryInputLayout:
    layouts = {
        "suite2p": _suite2p_layout(session_dir),
        "qc_results": _qc_results_layout(session_dir, "qc_results"),
        "manual_qc_results": _qc_results_layout(session_dir, "manual_qc_results"),
    }
    if requested != "auto":
        layout = layouts[requested]
        if not _layout_complete(layout):
            raise FileNotFoundError(
                f"Requested input layout '{requested}' is incomplete. Required files:\n"
                f"  stat: {_path_status(layout.stat_path)}\n"
                f"  fluorescence: {_path_status(layout.fluo_path)}\n"
                f"  neuropil: {_path_status(layout.neuropil_path)}"
            )
        return layout

    for name in ("qc_results", "manual_qc_results", "suite2p"):
        layout = layouts[name]
        if _layout_complete(layout):
            return layout
    searched = "\n".join(
        f"{name}: stat={layout.stat_path}, fluorescence={layout.fluo_path}, neuropil={layout.neuropil_path}"
        for name, layout in layouts.items()
    )
    raise FileNotFoundError(f"Could not detect a complete summary input layout. Checked:\n{searched}")


def _log_summary_input_layout(layout: SummaryInputLayout, requested: str) -> None:
    mode = "auto-detected" if requested == "auto" else "explicit"
    print(f"Using {mode} preprocessing summary input layout: {layout.name}")
    print(f"  stat: {layout.stat_path}")
    print(f"  fluorescence: {layout.fluo_path}")
    print(f"  neuropil: {layout.neuropil_path}")
    print(f"  iscell: {_path_status(layout.iscell_path)}")
    existing_spikes = [path for path in layout.spikes_paths if path.exists()]
    if existing_spikes:
        print(f"  inferred spikes: {existing_spikes[0]}")
    else:
        print("  inferred spikes: not found")


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


def _load_masks_h5_label_mask(
    session_dir: Path,
    key: str,
    target_shape: tuple[int, int],
    ops: dict[str, Any],
) -> np.ndarray | None:
    path = session_dir / "masks.h5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as h5:
        if key not in h5:
            return None
        mask = np.asarray(h5[key])
    if mask.shape[:2] == target_shape:
        return mask
    xrange = ops.get("xrange")
    yrange = ops.get("yrange")
    if xrange is None or yrange is None:
        return None
    x1, x2 = map(int, xrange)
    y1, y2 = map(int, yrange)
    if mask.shape[:2] != (y2 - y1, x2 - x1):
        return None
    expanded = np.zeros(target_shape, dtype=mask.dtype)
    expanded[y1:y2, x1:x2] = mask
    return expanded


def _coerce_cell_type_label_values(labels: np.ndarray, source: str) -> list[int | None]:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    finite = labels[np.isfinite(labels)]
    if not np.isin(finite, [-1, 0, 1]).all():
        raise ValueError(f"Skipped {source}: values must be -1, 0, 1, or NaN.")
    return [None if not np.isfinite(value) else int(value) for value in labels]


def _load_cell_type_labels(
    session_dir: Path,
    n_rois: int,
    suite2p_stat: np.ndarray,
) -> tuple[list[int | None], str]:
    """Load optional excitatory/inhibitory labels indexed by original Suite2p ROI."""
    candidates = [
        session_dir / "suite2p" / "plane0" / "roi_cell_type_labels.npy",
        session_dir / "roi_cell_type_labels.npy",
    ]
    for path in candidates:
        if not path.exists():
            continue
        labels = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if labels.shape != (n_rois,):
            return [None] * n_rois, f"Skipped {path.name}: expected {n_rois} labels, found {labels.size}."
        try:
            return _coerce_cell_type_label_values(labels, path.name), str(path)
        except ValueError as exc:
            return [None] * n_rois, str(exc)

    path = session_dir / "masks.h5"
    if not path.exists():
        return [None] * n_rois, "No cell-type labels loaded; masks.h5 or roi_cell_type_labels.npy was not found."
    with h5py.File(path, "r") as h5:
        if "labels" not in h5:
            return [None] * n_rois, "No cell-type labels loaded; masks.h5 does not contain a labels dataset."
        labels = np.asarray(h5["labels"], dtype=np.float64).reshape(-1)
    if labels.shape != (n_rois,):
        qc_stat_path = session_dir / "qc_results" / "stat.npy"
        if not qc_stat_path.exists():
            return [None] * n_rois, f"Skipped masks.h5 labels: expected {n_rois} labels, found {labels.size}."
        qc_stat = np.load(qc_stat_path, allow_pickle=True)
        if labels.shape != (len(qc_stat),):
            return [None] * n_rois, (
                f"Skipped masks.h5 labels: expected {n_rois} original Suite2p labels "
                f"or {len(qc_stat)} QC labels, found {labels.size}."
            )
        try:
            mapped_indices = map_qc_to_suite2p_rois(qc_stat, suite2p_stat)
            mapped = np.full(n_rois, np.nan, dtype=np.float64)
            mapped[mapped_indices] = labels
            return _coerce_cell_type_label_values(mapped, "masks.h5 labels"), "masks.h5:labels mapped from qc_results/stat.npy"
        except ValueError as exc:
            return [None] * n_rois, f"Skipped masks.h5 labels: {exc}"
    try:
        return _coerce_cell_type_label_values(labels, "masks.h5 labels"), "masks.h5:labels"
    except ValueError as exc:
        return [None] * n_rois, str(exc)


def _load_dff(session_dir: Path) -> np.ndarray:
    for path in [session_dir / "dff.h5", session_dir / "qc_results" / "dff.h5"]:
        if path.exists():
            with h5py.File(path, "r") as h5:
                if "dff" not in h5:
                    raise KeyError(f"{path} does not contain dataset 'dff'")
                return np.asarray(h5["dff"], dtype=np.float32)
    raise FileNotFoundError(f"Could not find dff.h5 in {session_dir} or {session_dir / 'qc_results'}")


def _load_oasis_spikes(
    session_dir: Path,
    n_rois: int,
    n_frames: int,
    input_layout: SummaryInputLayout | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    paths = input_layout.spikes_paths if input_layout is not None else (session_dir / "spikes.h5", session_dir / "qc_results" / "spikes.h5")
    for path in paths:
        if not path.exists():
            continue
        with h5py.File(path, "r") as h5:
            if "spikes" not in h5:
                continue
            spikes = np.asarray(h5["spikes"], dtype=np.float32)
            attrs = {key: h5.attrs[key].item() if hasattr(h5.attrs[key], "item") else h5.attrs[key] for key in h5.attrs}
        spikes = spikes[:n_rois, :n_frames]
        if spikes.shape != (n_rois, n_frames):
            padded = np.full((n_rois, n_frames), np.nan, dtype=np.float32)
            padded[: spikes.shape[0], : spikes.shape[1]] = spikes
            spikes = padded
        return spikes, attrs
    return None, {}


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
        "andreaPostdocSnrMin": None,
        "roiAreaMin": None,
        "roiAreaMax": None,
        "autocorrEfoldMin": None,
        "autocorrEfoldMax": None,
        "oasisEventSnrMin": None,
        "oasisRiseTauMin": None,
        "oasisRiseTauMax": None,
        "oasisDecayTauMin": None,
        "oasisDecayTauMax": None,
        "oasisResidualKsMax": None,
        "cellTypeFilter": None,
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
            "andreaPostdocSnrMin": None,
            "roiAreaMin": None,
            "roiAreaMax": None,
            "autocorrEfoldMin": None,
            "autocorrEfoldMax": None,
            "oasisEventSnrMin": None,
            "oasisRiseTauMin": None,
            "oasisRiseTauMax": None,
            "oasisDecayTauMin": None,
            "oasisDecayTauMax": None,
            "oasisResidualKsMax": None,
            "cellTypeFilter": None,
        }
    return presets


def _load_suite2p_dff(session_dir: Path, ops: dict[str, Any], input_layout: SummaryInputLayout | None = None) -> np.ndarray:
    layout = input_layout or _suite2p_layout(session_dir)
    fluo = np.load(layout.fluo_path, allow_pickle=False).astype(np.float32, copy=False)
    neuropil = np.load(layout.neuropil_path, allow_pickle=False).astype(np.float32, copy=False)
    signal = fluo - float(ops.get("neucoeff", 0.7)) * neuropil
    baseline = gaussian_filter(signal, [0.0, 600.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = (signal - baseline) / baseline
    dff[~np.isfinite(dff)] = np.nan
    return dff.astype(np.float32, copy=False)


def _load_suite2p_fluorescence(
    session_dir: Path,
    input_layout: SummaryInputLayout | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    layout = input_layout or _suite2p_layout(session_dir)
    fluo = np.load(layout.fluo_path, allow_pickle=False, mmap_mode="r")
    neuropil = np.load(layout.neuropil_path, allow_pickle=False, mmap_mode="r")
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
    input_layout: SummaryInputLayout,
    n_rois: int,
    n_frames: int,
    frame_rate: float,
    storage_mode: str,
    sidecar_path: Path | None,
    *,
    chunk_size: int = 64,
) -> tuple[np.ndarray | None, list[dict[str, float | None]]]:
    if storage_mode == "embedded":
        dff = _load_suite2p_dff(session_dir, ops, input_layout)
        dff = dff[:n_rois, :n_frames]
        return dff, _dff_metrics_to_jsonable(list(dff_qc_metrics(dff, frame_rate=frame_rate)))

    if sidecar_path is None:
        raise ValueError("sidecar_path is required when storage_mode is file")

    fluo, neuropil = _load_suite2p_fluorescence(session_dir, input_layout)
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


def _event_window_residuals(trace: np.ndarray, spikes: np.ndarray, threshold: float) -> np.ndarray:
    event_frames = np.flatnonzero(np.isfinite(spikes) & (spikes > threshold))
    if event_frames.size == 0:
        return np.array([], dtype=np.float32)
    trace = np.asarray(trace, dtype=np.float64)
    n_frames = trace.shape[0]
    marked = np.zeros(n_frames, dtype=bool)
    for frame in event_frames:
        start = max(0, int(frame) - 3)
        stop = min(n_frames, int(frame) + 4)
        marked[start:stop] = True
    frames = np.flatnonzero(marked)
    if frames.size == 0:
        return np.array([], dtype=np.float32)
    radius = 8
    valid = np.isfinite(trace)
    filled = np.where(valid, trace, 0.0)
    padded_sum = np.concatenate(([0.0], np.cumsum(filled, dtype=np.float64)))
    padded_count = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    starts = np.maximum(0, frames - radius)
    stops = np.minimum(n_frames, frames + radius + 1)
    sums = padded_sum[stops] - padded_sum[starts]
    counts = padded_count[stops] - padded_count[starts]
    keep = (counts > 0) & valid[frames]
    if not np.any(keep):
        return np.array([], dtype=np.float32)
    residuals = trace[frames[keep]] - sums[keep] / counts[keep]
    return residuals.astype(np.float32, copy=False)


def _gaussian_ks_distance(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 10:
        return np.nan
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    sorted_values = np.sort(values)
    gaussian_cdf = ndtr((sorted_values - mean) / sd)
    empirical_hi = np.arange(1, sorted_values.size + 1, dtype=np.float64) / sorted_values.size
    empirical_lo = np.arange(0, sorted_values.size, dtype=np.float64) / sorted_values.size
    return float(np.max(np.maximum(np.abs(empirical_hi - gaussian_cdf), np.abs(gaussian_cdf - empirical_lo))))


def _best_oasis_residual_threshold(
    trace: np.ndarray, spikes: np.ndarray, default_threshold: float
) -> tuple[float, float, int]:
    finite_spikes = spikes[np.isfinite(spikes) & (spikes > 0)]
    if finite_spikes.size == 0:
        return np.nan, np.nan, 0
    candidates = {float(default_threshold)}
    for quantile in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        candidates.add(float(np.nanquantile(finite_spikes, quantile)))
    best_threshold = np.nan
    best_ks = np.nan
    best_count = 0
    for threshold in sorted(value for value in candidates if np.isfinite(value) and value >= 0):
        residuals = _event_window_residuals(trace, spikes, threshold)
        ks = _gaussian_ks_distance(residuals)
        if not np.isfinite(ks):
            continue
        if not np.isfinite(best_ks) or ks < best_ks:
            best_ks = ks
            best_threshold = float(threshold)
            best_count = int(residuals.size)
    return best_threshold, best_ks, best_count


def _oasis_event_waveform_metrics(
    trace: np.ndarray,
    spikes: np.ndarray,
    threshold: float,
    frame_rate: float,
) -> tuple[float, float, float]:
    event_frames = np.flatnonzero(np.isfinite(spikes) & (spikes > threshold))
    if event_frames.size == 0 or not np.isfinite(frame_rate) or frame_rate <= 0:
        return np.nan, np.nan, np.nan
    pre = max(2, int(round(frame_rate * 0.5)))
    post = max(4, int(round(frame_rate * 2.0)))
    windows = []
    for frame in event_frames:
        frame = int(frame)
        if frame - pre < 0 or frame + post >= trace.size:
            continue
        window = np.asarray(trace[frame - pre : frame + post + 1], dtype=np.float32)
        if np.all(np.isfinite(window)):
            windows.append(window)
    if len(windows) < 3:
        return np.nan, np.nan, np.nan
    event_mean = np.mean(np.stack(windows, axis=0), axis=0)
    baseline = float(np.mean(event_mean[:pre]))
    baseline_sd = float(np.std(event_mean[:pre], ddof=1))
    peak_index = int(np.nanargmax(event_mean[pre:])) + pre
    peak = float(event_mean[peak_index])
    amplitude = peak - baseline
    snr = amplitude / baseline_sd if baseline_sd > 0 and amplitude > 0 else np.nan

    rise_tau = np.nan
    if amplitude > 0 and peak_index > 0:
        low = baseline + 0.1 * amplitude
        high = baseline + 0.9 * amplitude
        rise_segment = event_mean[: peak_index + 1]
        low_hits = np.flatnonzero(rise_segment >= low)
        high_hits = np.flatnonzero(rise_segment >= high)
        if low_hits.size and high_hits.size:
            rise_tau = float(max(0, high_hits[0] - low_hits[0]) / frame_rate)

    decay_tau = np.nan
    if amplitude > 0:
        decay_target = baseline + amplitude / np.e
        decay_segment = event_mean[peak_index:]
        decay_hits = np.flatnonzero(decay_segment <= decay_target)
        if decay_hits.size:
            decay_tau = float(decay_hits[0] / frame_rate)
    return float(snr), rise_tau, decay_tau


def _add_oasis_residual_metrics(
    dff_metrics: list[dict[str, float | None]],
    dff: np.ndarray | None,
    dff_sidecar_path: Path | None,
    oasis_spikes: np.ndarray | None,
    default_threshold: float,
    frame_rate: float,
) -> list[dict[str, float | None]]:
    if oasis_spikes is None:
        return dff_metrics
    source = dff
    if source is None and dff_sidecar_path is not None and dff_sidecar_path.exists():
        source = np.load(dff_sidecar_path, mmap_mode="r")
    if source is None:
        return dff_metrics
    n_rois = min(len(dff_metrics), int(source.shape[0]), int(oasis_spikes.shape[0]))
    for roi in range(n_rois):
        threshold, ks, count = _best_oasis_residual_threshold(
            np.asarray(source[roi], dtype=np.float32),
            np.asarray(oasis_spikes[roi], dtype=np.float32),
            default_threshold,
        )
        dff_metrics[roi]["oasis_optimal_threshold"] = float(threshold) if np.isfinite(threshold) else None
        dff_metrics[roi]["oasis_event_residual_ks"] = float(ks) if np.isfinite(ks) else None
        dff_metrics[roi]["oasis_event_residual_count"] = float(count)
        if np.isfinite(threshold):
            snr, rise_tau, decay_tau = _oasis_event_waveform_metrics(
                np.asarray(source[roi], dtype=np.float32),
                np.asarray(oasis_spikes[roi], dtype=np.float32),
                threshold,
                frame_rate,
            )
        else:
            snr, rise_tau, decay_tau = np.nan, np.nan, np.nan
        dff_metrics[roi]["oasis_event_snr"] = float(snr) if np.isfinite(snr) else None
        dff_metrics[roi]["oasis_rise_tau_seconds"] = float(rise_tau) if np.isfinite(rise_tau) else None
        dff_metrics[roi]["oasis_decay_tau_seconds"] = float(decay_tau) if np.isfinite(decay_tau) else None
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


def _roi_hit_path(xpix: np.ndarray, ypix: np.ndarray) -> str:
    if not xpix.size:
        return ""
    return "".join(f"M{int(x)} {int(y)}h1v1h-1Z" for x, y in zip(xpix, ypix))


def _roi_table(stat: np.ndarray, mask: np.ndarray, n_rois: int) -> list[dict[str, float | int | str]]:
    rois: list[dict[str, float | int | str]] = []
    for idx in range(n_rois):
        entry = stat[idx]
        ypix = np.asarray(entry.get("ypix", []), dtype=int)
        xpix = np.asarray(entry.get("xpix", []), dtype=int)
        rois.append(
            {
                "roi": idx,
                "path": _roi_outline_path(xpix, ypix),
                "hitPath": _roi_hit_path(xpix, ypix),
                "npix": int(xpix.size),
            }
        )
    return rois


def _label_mask_roi_table(mask: np.ndarray | None) -> list[dict[str, float | int | str]]:
    if mask is None:
        return []
    rois: list[dict[str, float | int | str]] = []
    for idx, label_id in enumerate(np.unique(mask)):
        if label_id <= 0:
            continue
        ypix, xpix = np.where(mask == label_id)
        rois.append(
            {
                "roi": idx,
                "label": int(label_id),
                "path": _roi_outline_path(xpix, ypix),
                "npix": int(xpix.size),
            }
        )
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
    cellpose_mask: np.ndarray | None,
    mask: np.ndarray,
    stat: np.ndarray,
    suite2p_indices: np.ndarray,
    iscell: np.ndarray,
    suite2p_fingerprint: str,
    morphology_metrics: list[dict[str, float | int]],
    cell_type_labels: list[int | None],
    cell_type_label_source: str,
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
    xoff: np.ndarray,
    yoff: np.ndarray,
    oasis_spikes: np.ndarray | None,
    oasis_attrs: dict[str, Any],
    oasis_storage_mode: str,
    oasis_sidecar_name: str | None,
) -> None:
    rois = _roi_table(stat, mask, n_rois)
    red_available = mean_red is not None
    cellpose_rois = _label_mask_roi_table(cellpose_mask)
    cellpose_available = red_available and bool(cellpose_rois)
    image_height, image_width = np.asarray(mean_green).shape[:2]
    xoff = np.asarray(xoff, dtype=np.float32)
    yoff = np.asarray(yoff, dtype=np.float32)
    motion_frame_count = min(xoff.size, yoff.size)
    motion_available = motion_frame_count > 0
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
        "cellposeMaskAvailable": cellpose_available,
        "cellposeRois": cellpose_rois,
        "mask": _mask_data_uri(mask),
        "rois": rois,
        "suite2pIndices": suite2p_indices.tolist(),
        "suite2pRoiCount": int(iscell.shape[0]),
        "suite2pStatFingerprint": suite2p_fingerprint,
        "iscell": _float64_b64(iscell),
        "morphology": morphology_metrics,
        "cellTypeLabels": cell_type_labels,
        "cellTypeLabelSource": cell_type_label_source,
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
        "motionAvailable": motion_available,
        "motionFrameCount": int(motion_frame_count),
        "xoff": _float32_b64(xoff[:motion_frame_count]) if motion_available else None,
        "yoff": _float32_b64(yoff[:motion_frame_count]) if motion_available else None,
        "morphologyPresets": _morphology_preset_payload(),
        "dff": _float32_b64(dff) if dff_storage_mode == "embedded" and dff is not None else None,
        "dffLabel": dff_label,
        "oasisAvailable": oasis_spikes is not None,
        "oasisSpikes": _float32_b64(oasis_spikes)
        if oasis_storage_mode == "embedded" and oasis_spikes is not None
        else None,
        "oasisStorageMode": oasis_storage_mode,
        "oasisSidecarName": oasis_sidecar_name,
        "oasisEventThreshold": float(oasis_attrs.get("event_threshold", 0.05)),
        "oasisAttrs": oasis_attrs,
    }
    fov_grid_class = "with-red" if red_available else "single-channel"
    oasis_diagnostics_panel = (
        '<div class="panel oasis-diagnostics minimized">'
        '<div class="oasis-diagnostics-header">'
        '<div class="title">Selected ROI inferred spike fit metrics</div>'
        '<div class="oasis-diagnostics-actions">'
        '<button id="oasisDiagnosticsInfo" class="info-button" type="button" data-info-target="oasisDiagnosticsHelp" aria-expanded="false">Read more</button>'
        '</div>'
        '</div>'
        '<div id="oasisDiagnosticsHelp" class="info-box" hidden>'
        '<p><strong>Average transient:</strong> dF/F windows around inferred spikes are baseline-subtracted and averaged for the selected ROI.</p>'
        '<p><strong>Exponential model:</strong> a simple decay curve is drawn from the average transient peak using the empirical e-fold time, where the average transient falls to 1/e of its peak amplitude.</p>'
        '<p><strong>Gaussian residual KS:</strong> event-window residuals are compared with a fitted Gaussian CDF; lower KS distance means the residuals are closer to Gaussian.</p>'
        '<p><strong>Amplitude threshold:</strong> only inferred spike amplitudes above the selected threshold contribute to the average transient.</p>'
        '</div>'
        '<div class="oasis-diagnostics-body">'
        '<div class="note" id="oasisDiagnosticsSummary">Inferred spikes not loaded for this session.</div>'
        '<canvas id="oasisTransientCanvas"></canvas>'
        '<canvas id="oasisGaussianCanvas"></canvas>'
        '</div>'
        '</div>'
        if oasis_spikes is not None
        else ""
    )
    red_panel = (
        '<div class="panel"><div class="panel-title-row"><div class="title">Anatomical mean</div>'
        '<label class="overlay-mode-control">Overlay'
        '<select id="redOverlayMode">'
        '<option value="functional" selected>Suite2p ROIs</option>'
        '<option value="cellpose">Cellpose ROIs</option>'
        '</select></label></div>'
        '<div class="imagewrap"><img id="red"><svg class="overlay red-overlay" preserveAspectRatio="xMidYMid meet">'
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
html, body {{ overflow-y: auto; }}
body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: #202124; background: #f6f7f8; }}
.page {{ width: min(1600px, calc(100vw - 72px)); margin: 16px auto 26px; }}
.head {{ display: flex; justify-content: space-between; gap: 14px; align-items: end; margin-bottom: 12px; }}
h1 {{ margin: 0; font-size: 21px; letter-spacing: 0; }}
.meta {{ color: #667085; font-size: 13px; text-align: right; }}
.grid {{ display: grid; gap: 6px; }}
.review-main {{ margin-top: 8px; display: grid; grid-template-columns: minmax(0, 1fr) clamp(270px, 22vw, 340px); gap: 8px; align-items: start; }}
.review-main.menu-collapsed {{ grid-template-columns: minmax(0, 1fr) 38px; }}
.viewer-column {{ display: flex; flex-direction: column; gap: 6px; }}
.fov-row {{ display: grid; grid-template-columns: 1fr; gap: 6px; align-items: start; }}
.fov-review {{ display: grid; grid-template-columns: minmax(420px, 720px) minmax(300px, 1fr); gap: 8px; align-items: start; justify-items: start; }}
.fov-review.oasis-collapsed {{ grid-template-columns: minmax(520px, 900px); }}
.fov-review > .grid {{ width: min(100%, 720px); }}
.fov-review.oasis-collapsed > .grid {{ width: min(100%, 860px); }}
.grid.with-red {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.grid.single-channel {{ grid-template-columns: minmax(0, 1fr); }}
.panel {{ background: #fff; border: 1px solid #d0d5dd; border-radius: 7px; padding: 7px; box-sizing: border-box; }}
.title {{ font-size: 13px; font-weight: 700; margin-bottom: 4px; }}
.panel-title-row {{ height: 28px; display: flex; gap: 6px; align-items: center; justify-content: space-between; margin-bottom: 4px; overflow: hidden; }}
.panel-title-row .title {{ margin-bottom: 0; }}
.imagewrap {{ position: relative; width: 100%; aspect-ratio: 1/1; background: #111; overflow: hidden; }}
.imagewrap img, .imagewrap svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
.imagewrap img {{ object-fit: contain; image-rendering: pixelated; }}
.imagewrap img, .imagewrap svg {{ transform-origin: 0 0; will-change: transform; }}
.imagewrap {{ cursor: grab; }}
.imagewrap.panning {{ cursor: grabbing; }}
.roi-hit {{ fill: rgba(255,255,255,0); stroke: none; cursor: pointer; pointer-events: all; }}
.roi {{ fill: none; stroke: rgba(255,255,255,.86); stroke-width: .7; vector-effect: non-scaling-stroke; pointer-events: none; }}
.roi-hit:hover + .roi {{ fill: none; stroke: #06b6d4; stroke-width: 1.6; }}
.roi.selected {{ fill: none; stroke: #00e5ff; stroke-width: 3.2; }}
.cellpose-roi {{ fill: none; stroke: #fbbf24; stroke-width: 1.4; vector-effect: non-scaling-stroke; pointer-events: none; }}
.overlay-mode-control {{ display: flex; flex: 0 0 auto; gap: 4px; align-items: center; margin: 0; font-size: 11px; color: #475467; white-space: nowrap; }}
.overlay-mode-control select {{ width: auto; max-width: 128px; min-width: 104px; padding: 2px 4px; font-size: 11px; }}
.controls {{ display: grid; grid-template-columns: 1fr repeat(5, auto); gap: 5px; align-items: center; margin-top: 6px; }}
.label-controls {{ display: flex; flex-direction: column; gap: 4px; align-items: stretch; }}
.label-controls .button-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }}
.label-controls .nav-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }}
.show-roi-menu {{ position: relative; display: inline-block; }}
.show-roi-menu > summary {{ list-style: none; cursor: pointer; border: 1px solid #d0d5dd; border-radius: 6px; padding: 4px 7px; background: #fff; }}
.show-roi-menu > summary::-webkit-details-marker {{ display: none; }}
.show-roi-options {{ position: absolute; z-index: 20; top: calc(100% + 4px); left: 0; display: grid; gap: 4px; min-width: 165px; padding: 8px; border: 1px solid #d0d5dd; border-radius: 7px; background: #fff; box-shadow: 0 12px 30px rgba(16,24,40,.18); }}
.show-roi-options label {{ display: flex; gap: 6px; align-items: center; white-space: nowrap; }}
.show-roi-options input {{ width: auto; margin: 0; }}
.roi-details summary {{ cursor: pointer; font-weight: 700; color: #344054; }}
.roi-details #readout {{ margin-top: 5px; color: #475467; font-size: 14px; line-height: 1.35; }}
.save-option {{ display: grid; gap: 3px; justify-items: start; }}
.save-option button, .save-options .docs-link {{ width: fit-content; }}
.info-button {{ padding: 3px 7px; width: fit-content; font-size: 14px; color: #175cd3; font-weight: 600; }}
.nav-button {{ font-size: 15px; font-weight: 700; }}
.side-menu {{ position: fixed; top: 10px; right: max(16px, calc((100vw - 1600px) / 2)); width: clamp(270px, 22vw, 340px); z-index: 15; display: flex; flex-direction: column; gap: 6px; max-height: calc(100vh - 20px); overflow-y: auto; align-self: start; }}
.side-menu-toggle {{ align-self: stretch; font-weight: 700; }}
.control-column {{ display: flex; flex-direction: column; gap: 5px; align-items: stretch; min-height: 0; overflow: visible; font-size: 14px; }}
.review-main.menu-collapsed .control-column {{ display: none; }}
.review-main.menu-collapsed .side-menu {{ width: 38px; overflow: visible; }}
.review-main.menu-collapsed .side-menu-toggle {{ writing-mode: vertical-rl; min-height: 150px; padding: 8px 4px; }}
.menu-card > summary {{ cursor: pointer; font-weight: 700; color: #202124; }}
.menu-card > summary::marker {{ color: #667085; }}
.menu-card-content {{ display: flex; flex-direction: column; gap: 4px; margin-top: 6px; }}
.morphology-card {{ display: flex; flex-direction: column; gap: 4px; }}
.qc-header {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; }}
.qc-current {{ color: #667085; font-size: 14px; }}
.sort-card {{ display: flex; flex-direction: column; gap: 4px; padding-top: 4px; border-top: 1px solid #eaecf0; }}
.sort-header {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; }}
.sort-current {{ color: #667085; font-size: 14px; }}
.trace-header-row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; }}
.trace-header-controls {{ display: none; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: flex-end; font-size: 14px; }}
.trace-header-controls.visible {{ display: flex; }}
.trace-header-controls button {{ flex: 0 0 auto; white-space: nowrap; }}
.filter-controls {{ display: grid; grid-template-columns: repeat(3, minmax(130px, 1fr)); gap: 8px; margin-top: 10px; align-items: end; }}
.filter-controls label {{ font-size: 14px; color: #475467; }}
.filter-controls input {{ display: block; margin-top: 3px; width: 100%; box-sizing: border-box; }}
.metric-controls {{ display: grid; grid-template-columns: repeat(2, minmax(250px, 1fr)); gap: 10px; margin-top: 10px; }}
.metric-control {{ display: grid; grid-template-columns: minmax(150px, 1fr) minmax(130px, .8fr); gap: 8px; align-items: center; padding: 8px; border: 1px solid #eaecf0; border-radius: 7px; background: #fff; }}
.metric-control .threshold-inputs {{ display: grid; gap: 5px; }}
.threshold-default {{ color: #667085; font-size: 14px; }}
.metric-control canvas {{ height: 70px; }}
.metric-histogram-panel {{ min-width: 180px; }}
.metric-histogram-panel summary {{ cursor: pointer; color: #175cd3; font-size: 14px; font-weight: 600; }}
.metric-histogram {{ height: 92px; min-width: 180px; margin-top: 4px; }}
.filter-subsection-title {{ margin-top: 10px; font-size: 15px; font-weight: 700; color: #344054; }}
.source-heading {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; }}
.filter-summary {{ color: #475467; font-size: 15px; }}
.dialog-actions {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; margin-top: 12px; }}
.dialog-header {{ display: flex; justify-content: space-between; gap: 10px; align-items: start; }}
.dialog-title {{ font-size: 18px; font-weight: 700; }}
.dialog-close {{ border: 0; background: transparent; color: #667085; font-size: 20px; line-height: 1; padding: 0 4px; }}
.dialog-close:hover {{ color: #111827; }}
.dialog-section {{ margin-top: 12px; padding-top: 12px; border-top: 1px solid #eaecf0; }}
.dialog-section:first-child {{ margin-top: 0; padding-top: 0; border-top: 0; }}
.dialog-section-title {{ font-size: 16px; font-weight: 700; margin-bottom: 6px; }}
.info-box {{ margin-top: 8px; padding: 8px 10px; background: #f8fafc; border: 1px solid #d0d5dd; border-radius: 6px; color: #475467; font-size: 14px; line-height: 1.35; }}
.info-box p {{ margin: 0 0 6px; }}
.info-box p:last-child {{ margin-bottom: 0; }}
.save-options {{ display: grid; gap: 10px; margin-top: 10px; }}
.bulk-label-controls {{ display: grid; gap: 8px; margin-top: 10px; justify-items: start; }}
.bulk-label-controls select {{ width: auto; min-width: 160px; }}
dialog {{ width: min(980px, calc(100vw - 40px)); border: 1px solid #d0d5dd; border-radius: 8px; padding: 14px; box-shadow: 0 24px 60px rgba(16,24,40,.24); }}
dialog::backdrop {{ background: rgba(15,23,42,.38); }}
button, input, select {{ font: inherit; }}
button {{ border: 1px solid #d0d5dd; background: #fff; border-radius: 6px; padding: 4px 7px; cursor: pointer; }}
button.good {{ border-color: #16a34a; color: #166534; }}
button.bad {{ border-color: #dc2626; color: #991b1b; }}
button.unsure {{ border-color: #d97706; color: #92400e; }}
button.unlabeled {{ border-color: #667085; color: #475467; }}
button.active {{ color: #fff; }}
button.good.active {{ background: #16a34a; }}
button.bad.active {{ background: #dc2626; }}
button.unsure.active {{ background: #d97706; }}
button.unlabeled.active {{ background: #667085; }}
input, select {{ border: 1px solid #d0d5dd; border-radius: 6px; padding: 5px 7px; width: 82px; }}
.filter-controls input {{ width: 100%; }}
.filter-controls select {{ width: 100%; }}
canvas {{ width: 100%; display: block; background: #fff; border: 1px solid #d0d5dd; box-sizing: border-box; }}
#stackCanvas {{ height: 560px; cursor: grab; }}
#stackCanvas.dragging {{ cursor: grabbing; }}
#traceCanvas {{ height: 220px; cursor: grab; }}
#traceCanvas.dragging {{ cursor: grabbing; }}
#motionDriftCanvas {{ height: 470px; }}
#motionDistributionCanvas {{ height: 390px; }}
.oasis-diagnostics {{ width: 100%; max-width: 600px; }}
.oasis-diagnostics-header {{ display: flex; gap: 8px; align-items: center; justify-content: space-between; }}
.oasis-diagnostics-actions {{ display: flex; gap: 6px; align-items: center; }}
.oasis-diagnostics.minimized {{ display: none; }}
.oasis-diagnostics .note {{ margin-bottom: 6px; }}
#oasisTransientCanvas {{ height: 220px; }}
#oasisGaussianCanvas {{ height: 190px; margin-top: 6px; }}
.oasis-panel {{ margin-top: 10px; }}
.oasis-toggle {{ display: flex; gap: 6px; align-items: center; font-weight: 700; }}
.oasis-toggle input {{ width: auto; }}
.oasis-threshold-row {{ display: flex; flex: 0 0 auto; gap: 6px; align-items: center; }}
.oasis-threshold-row input[type="range"] {{ width: 150px; padding: 0; }}
.oasis-threshold-row input[type="number"] {{ width: 78px; }}
.plots {{ display: grid; grid-template-columns: 1fr; gap: 8px; margin-top: 8px; }}
.trace-sort {{ display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: end; margin: 6px 0 10px; padding: 8px 10px; background: #f8fafc; border: 1px solid #d0d5dd; border-radius: 6px; }}
.trace-sort label {{ font-size: 14px; color: #475467; }}
.trace-sort select {{ display: block; margin-top: 3px; min-width: 150px; width: auto; }}
.sort-metric-list {{ display: grid; grid-template-columns: minmax(240px, 1fr); gap: 6px; align-items: start; }}
.sort-metric-list label {{ display: flex; gap: 6px; align-items: center; margin: 0; }}
.sort-metric-list input {{ width: auto; margin: 0; }}
.sort-metric-group {{ font-weight: 700; color: #344054; grid-column: 1 / -1; margin-top: 4px; }}
.sort-actions {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: end; }}
.metric-formula {{ flex: 1 1 320px; }}
.metric-formula summary {{ cursor: pointer; font-weight: 700; color: #344054; }}
.metric-table-wrap {{ max-height: 75vh; overflow: auto; border: 1px solid #d0d5dd; margin-top: 12px; }}
.metric-table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
.metric-table th, .metric-table td {{ border: 1px solid #e5e7eb; padding: 4px 7px; text-align: right; white-space: nowrap; }}
.metric-table th {{ position: sticky; top: 0; background: #f8fafc; z-index: 1; }}
.metric-table td:first-child, .metric-table td:nth-child(2), .metric-table td:last-child {{ text-align: left; }}
.metric-fail {{ background: rgba(248, 113, 113, .28); }}
.trace-loader {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 8px; }}
.trace-loader input {{ display: none; }}
.note {{ margin-top: 6px; color: #667085; font-size: 12px; }}
.docs-link {{ display: inline-block; color: #175cd3; font-size: 13px; font-weight: 600; text-decoration: none; }}
.docs-link:hover {{ text-decoration: underline; }}
@media (max-width: 1100px) {{ .review-main, .review-main.menu-collapsed, .fov-row, .fov-review, .grid, .controls {{ grid-template-columns: 1fr; }} .side-menu, .review-main.menu-collapsed .side-menu {{ position: static; width: auto; max-height: none; overflow: visible; }} .review-main.menu-collapsed .side-menu-toggle {{ writing-mode: horizontal-tb; min-height: 0; }} .head {{ display: block; }} .meta {{ text-align: left; }} }}
</style>
</head>
<body>
<div class="page">
  <div class="head"><h1>{session_name} preprocessing QC ({n_rois} ROIs)</h1><div class="meta" id="meta"></div></div>
  <dialog id="morphologyDialog">
    <div class="dialog-header">
      <div class="dialog-title">ROI QC Filters</div>
      <button id="closeMorphologyDialogTop" class="dialog-close" type="button" aria-label="Close">&times;</button>
    </div>
    <div class="dialog-section">
      <div id="targetStructureSummary" class="title"></div>
      <div class="filter-summary" id="filterSummary"></div>
      <div class="note">These preset thresholds preview pass/fail counts; labels change only when Apply Filters is clicked.</div>
      <div class="filter-controls metric-filter-controls">
        <label>Target structure <select id="filterPreset"></select></label>
        <button id="resetFilter">Restore selected QC thresholds</button>
        <input id="presetFile" type="file" accept=".json" style="display:none;">
        <button id="importPreset">Import QC thresholds JSON</button>
      </div>
      <div class="filter-subsection-title source-heading">
        <span>Morphology Metrics</span>
        <button class="info-button" type="button" data-info-target="morphologyMetricSources" aria-expanded="false">Read more</button>
        <button class="info-button" type="button" data-info-target="metricDistributionInfo" aria-expanded="false">Distribution Plot Info</button>
        <button class="info-button" type="button" data-info-target="metricSuggestionSources" aria-expanded="false">Suggested Thresholds Info</button>
      </div>
      <div id="morphologyMetricSources" class="info-box" hidden>
        <p><a class="docs-link" href="https://suite2p.readthedocs.io/en/latest/outputs/#statnpy-fields" target="_blank" rel="noopener noreferrer">Suite2p stat.npy field definitions</a>: <code>aspect_ratio</code> and <code>compact</code> are Suite2p ROI shape statistics.</p>
        <p><a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/preprocessing_summary.py#L390-L397" target="_blank" rel="noopener noreferrer">ROI area calculation code</a>: number of pixels in the Suite2p ROI mask field <code>xpix</code>.</p>
        <p><a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/roi_labels.py#L76-L91" target="_blank" rel="noopener noreferrer">Connectivity calculation code</a>: number of 4-connected components in each ROI pixel mask.</p>
      </div>
      <div id="metricDistributionInfo" class="info-box" hidden>
        <p>Distribution plots are hidden by default. When opened, each plot shows metric values across ROIs. With no threshold values entered, vertical lines reflect the active target-structure threshold, the mean for unthresholded Suite2p fields, or suggested min/max values for computed metrics. As threshold fields are updated the lines update to the new values.</p>
      </div>
      <div id="metricSuggestionSources" class="info-box" hidden>
        <p><strong>Suggested filter values are computed from the ROIs embedded in this HTML.</strong></p>
        <p><strong>Connectivity:</strong> suggested max is the 75th percentile of connectivity values.</p>
        <p><strong>ROI area:</strong> suggested min/max are the 25th and 75th percentiles across ROIs.</p>
        <p><strong>Cell type / indicator:</strong> no numeric threshold is suggested; choose a categorical label if cell-type labels are loaded or uploaded.</p>
        <p><strong>SNR: 95/50 percentile:</strong> suggested min is the mean SNR value across ROIs.</p>
        <p><strong>SNR: CaImAn:</strong> suggested min is the mean large-transient score across ROIs.</p>
        <p><strong>Autocorrelation e-fold time:</strong> suggested min/max are the 25th and 75th percentiles across ROIs.</p>
        <p><strong>Inferred spike metrics:</strong> suggested SNR min is the mean value, suggested rise/decay tau min/max are the 25th and 75th percentiles, and suggested residual Gaussian-fit max is the 75th percentile.</p>
      </div>
      <div class="filter-controls">
        <label>Aspect min <input id="aspectMin" type="number" step="0.01" placeholder="Not Used"></label>
        <label>Aspect max <input id="aspectMax" type="number" step="0.01" placeholder="Not Used"></label>
        <details class="metric-histogram-panel"><summary>Show distribution</summary><canvas class="metric-histogram" data-metric="aspect" data-min="aspectMin" data-max="aspectMax" aria-label="Aspect ratio distribution"></canvas></details>
        <label>Compact min <input id="compactMin" type="number" step="0.01" placeholder="Not Used"></label>
        <label>Compact max <input id="compactMax" type="number" step="0.01" placeholder="Not Used"></label>
        <details class="metric-histogram-panel"><summary>Show distribution</summary><canvas class="metric-histogram" data-metric="compact" data-min="compactMin" data-max="compactMax" aria-label="Compactness distribution"></canvas></details>
        <label>Connectivity max <span class="threshold-default" id="maxConnectDefault"></span><input id="maxConnect" type="number" min="0" step="1" placeholder="Not Used"></label>
        <span></span>
        <details class="metric-histogram-panel"><summary>Show distribution</summary><canvas class="metric-histogram" data-metric="connectivity" data-max="maxConnect" aria-label="Connectivity distribution"></canvas></details>
        <label>ROI area min (px) <span class="threshold-default" id="roiAreaMinDefault"></span><input id="roiAreaMin" type="number" min="0" step="1" placeholder="Not Used"></label>
        <label>ROI area max (px) <span class="threshold-default" id="roiAreaMaxDefault"></span><input id="roiAreaMax" type="number" min="0" step="1" placeholder="Not Used"></label>
        <details class="metric-histogram-panel"><summary>Show distribution</summary><canvas class="metric-histogram" data-metric="roi_area" data-min="roiAreaMin" data-max="roiAreaMax" aria-label="ROI area distribution"></canvas></details>
      </div>
      <div class="filter-subsection-title source-heading">
        <span>Cell Type / Indicator Metrics</span>
        <button class="info-button" type="button" data-info-target="cellTypeMetricSources" aria-expanded="false">Read more</button>
      </div>
      <div id="cellTypeMetricSources" class="info-box" hidden>
        <p>Cell-type labels are loaded from pipeline outputs when available. Codes are <code>-1</code> excitatory/non-red, <code>0</code> unsure, and <code>1</code> inhibitory/red.</p>
        <p>To override or add labels, upload a CSV/TSV file with one row per original Suite2p ROI and columns <code>cell_type_code</code> and <code>cell_type_label</code>. If a <code>suite2p_index</code>, <code>roi</code>, or <code>index</code> column is present, rows are mapped by that Suite2p index; otherwise row order is used.</p>
      </div>
      <div class="filter-controls">
        <label>Cell type <select id="cellTypeFilter"><option value="">Not Used</option><option value="1">inhibitory/red</option><option value="-1">excitatory/non-red</option><option value="0">unsure</option><option value="not_loaded">not loaded</option></select></label>
        <input id="cellTypeFile" type="file" accept=".csv,.tsv,.txt" style="display:none;">
        <button id="loadCellTypeFile" type="button">Load cell-type file</button>
      </div>
      <div class="filter-subsection-title source-heading">
        <span>Fluorescence Trace Metrics</span>
        <button class="info-button" type="button" data-info-target="fluorescenceMetricSources" aria-expanded="false">Read more</button>
      </div>
      <div id="fluorescenceMetricSources" class="info-box" hidden>
        <p><a class="docs-link" href="https://suite2p.readthedocs.io/en/latest/outputs/#statnpy-fields" target="_blank" rel="noopener noreferrer">Suite2p stat.npy field definitions</a>: <code>skew</code> and <code>footprint</code> are Suite2p activity/detection fields, not ROI shape-only fields.</p>
        <p><a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/roi_labels.py#L122-L158" target="_blank" rel="noopener noreferrer">SNR: 95/50 percentile calculation code</a>: robust transient amplitude, <code>P95 - P50</code>, divided by residual noise.</p>
        <p><a class="docs-link" href="https://github.com/farznaj/imaging_decisionMaking_exc_inh/blob/master/imaging/evaluate_components.py" target="_blank" rel="noopener noreferrer">CaImAn-style large-transient score source code</a>: exceptional-event score where larger values indicate stronger large-transient structure.</p>
        <p><a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/roi_labels.py#L251-L291" target="_blank" rel="noopener noreferrer">Autocorrelation e-fold time calculation code</a>: dF/F persistence time where autocorrelation drops to <code>1/e</code>; not a fitted calcium decay constant.</p>
      </div>
      <div class="filter-controls">
        <label>Skew min <input id="skewMin" type="number" step="0.01" placeholder="Not Used"></label>
        <label>Skew max <input id="skewMax" type="number" step="0.01" placeholder="Not Used"></label>
        <details class="metric-histogram-panel"><summary>Show distribution</summary><canvas class="metric-histogram" data-metric="skew" data-min="skewMin" data-max="skewMax" aria-label="Skew distribution"></canvas></details>
        <label>Footprint min <input id="footprintMin" type="number" step="0.01" placeholder="Not Used"></label>
        <label>Footprint max <input id="footprintMax" type="number" step="0.01" placeholder="Not Used"></label>
        <details class="metric-histogram-panel"><summary>Show distribution</summary><canvas class="metric-histogram" data-metric="footprint" data-min="footprintMin" data-max="footprintMax" aria-label="Footprint distribution"></canvas></details>
        <label>SNR: 95/50 percentile min <span class="threshold-default" id="eventSnrDefault"></span><input id="eventSnrMin" type="number" step="0.01" placeholder="Not Used"></label>
        <span></span>
        <details class="metric-histogram-panel"><summary>Show distribution</summary><canvas class="metric-histogram" data-metric="snr_95_50" data-min="eventSnrMin" aria-label="SNR 95/50 percentile distribution"></canvas></details>
        <label>SNR: CaImAn (large-transient score) min <span class="threshold-default" id="andreaPostdocSnrDefault"></span><input id="andreaPostdocSnrMin" type="number" step="0.01" placeholder="Not Used"></label>
        <span></span>
        <details class="metric-histogram-panel"><summary>Show distribution</summary><canvas class="metric-histogram" data-metric="andrea_postdoc_snr" data-min="andreaPostdocSnrMin" aria-label="CaImAn SNR distribution"></canvas></details>
        <label>Autocorrelation e-fold time min (s) <span class="threshold-default" id="autocorrEfoldMinDefault"></span><input id="autocorrEfoldMin" type="number" step="0.01" placeholder="Not Used"></label>
        <label>Autocorrelation e-fold time max (s) <span class="threshold-default" id="autocorrEfoldMaxDefault"></span><input id="autocorrEfoldMax" type="number" step="0.01" placeholder="Not Used"></label>
        <details class="metric-histogram-panel"><summary>Show distribution</summary><canvas class="metric-histogram" data-metric="autocorr_efold_time_seconds" data-min="autocorrEfoldMin" data-max="autocorrEfoldMax" aria-label="Autocorrelation e-fold time distribution"></canvas></details>
      </div>
      <div class="filter-subsection-title source-heading">
        <span>Inferred Spike Metrics</span>
        <button class="info-button" type="button" data-info-target="inferredSpikeMetricSources" aria-expanded="false">Read more</button>
      </div>
      <div id="inferredSpikeMetricSources" class="info-box" hidden>
        <p><a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/preprocessing_summary.py#L470-L556" target="_blank" rel="noopener noreferrer">Inferred spike metric calculation code</a>: metrics are calculated from dF/F windows around thresholded inferred spike amplitudes.</p>
        <p>The current pipeline can generate inferred spike amplitudes with Suite2p/OASIS, but these reviewer metrics only require an ROI-by-frame inferred-spike amplitude array in the expected format.</p>
      </div>
      <div class="filter-controls">
        <label>Inferred spike SNR min <span class="threshold-default" id="oasisEventSnrDefault"></span><input id="oasisEventSnrMin" type="number" step="0.01" placeholder="Not Used"></label>
        <span></span>
        <details class="metric-histogram-panel"><summary>Show distribution</summary><canvas class="metric-histogram" data-metric="oasis_event_snr" data-min="oasisEventSnrMin" aria-label="Inferred spike SNR distribution"></canvas></details>
        <label>Inferred spike rise tau min (s) <span class="threshold-default" id="oasisRiseTauMinDefault"></span><input id="oasisRiseTauMin" type="number" step="0.01" placeholder="Not Used"></label>
        <label>Inferred spike rise tau max (s) <span class="threshold-default" id="oasisRiseTauMaxDefault"></span><input id="oasisRiseTauMax" type="number" step="0.01" placeholder="Not Used"></label>
        <details class="metric-histogram-panel"><summary>Show distribution</summary><canvas class="metric-histogram" data-metric="oasis_rise_tau_seconds" data-min="oasisRiseTauMin" data-max="oasisRiseTauMax" aria-label="Inferred spike rise tau distribution"></canvas></details>
        <label>Inferred spike decay tau min (s) <span class="threshold-default" id="oasisDecayTauMinDefault"></span><input id="oasisDecayTauMin" type="number" step="0.01" placeholder="Not Used"></label>
        <label>Inferred spike decay tau max (s) <span class="threshold-default" id="oasisDecayTauMaxDefault"></span><input id="oasisDecayTauMax" type="number" step="0.01" placeholder="Not Used"></label>
        <details class="metric-histogram-panel"><summary>Show distribution</summary><canvas class="metric-histogram" data-metric="oasis_decay_tau_seconds" data-min="oasisDecayTauMin" data-max="oasisDecayTauMax" aria-label="Inferred spike decay tau distribution"></canvas></details>
        <label>Inferred spike residual Gaussian-fit distance max <span class="threshold-default" id="oasisResidualKsDefault"></span><input id="oasisResidualKsMax" type="number" step="0.01" placeholder="Not Used"></label>
        <span></span>
        <details class="metric-histogram-panel"><summary>Show distribution</summary><canvas class="metric-histogram" data-metric="oasis_event_residual_ks" data-max="oasisResidualKsMax" aria-label="Inferred spike residual Gaussian-fit distance distribution"></canvas></details>
      </div>
      <div class="filter-subsection-title">Save QC Thresholds</div>
      <div class="source-heading">
        <span class="note">Save the current filter values for reuse or export/import them for another session.</span>
        <button class="info-button" type="button" data-info-target="saveFiltersHelp" aria-expanded="false">Read more</button>
      </div>
      <div id="saveFiltersHelp" class="info-box" hidden>
        <p><strong>Save new QC thresholds:</strong> saves the current threshold settings as a named filter inside the currently open reviewer session, so it appears in the target-structure dropdown during this browser session.</p>
        <p><strong>Save QC thresholds into HTML:</strong> downloads a new HTML copy with the current thresholds embedded, so reopening that HTML later preserves them.</p>
        <p><strong>Import QC thresholds JSON:</strong> loads a previously exported threshold JSON file into this reviewer.</p>
      </div>
      <div class="filter-controls">
        <label>Custom filter name <input id="presetName" type="text" placeholder="my filters"></label>
        <button id="savePreset">Save new QC thresholds</button>
        <button id="savePresetHtml">Save QC thresholds into HTML</button>
      </div>
    </div>
    <div class="dialog-actions">
      <button id="applyFilterToLabels">Apply Filters</button>
      <button id="closeMorphologyDialog" type="button">Close</button>
    </div>
  </dialog>
  <dialog id="sortDialog">
    <div class="dialog-header">
      <div class="dialog-title">Sort ROIs and dF/Fs</div>
      <button id="closeSortDialogTop" class="dialog-close" type="button" aria-label="Close">&times;</button>
    </div>
    <div class="source-heading">
      <button class="info-button" type="button" data-info-target="sortMetricSources" aria-expanded="false">Read more: metric sources</button>
    </div>
    <div id="sortMetricSources" class="info-box" hidden>
      <p><a class="docs-link" href="https://suite2p.readthedocs.io/en/latest/outputs/#statnpy-fields" target="_blank" rel="noopener noreferrer">Suite2p stat.npy field definitions</a>: Suite2p-provided fields include ROI shape metrics, detection/template fields, and trace activity fields.</p>
      <p><a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/roi_labels.py#L76-L91" target="_blank" rel="noopener noreferrer">Connectivity calculation code</a>: number of 4-connected components in each ROI pixel mask.</p>
      <p><a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/preprocessing_summary.py#L390-L397" target="_blank" rel="noopener noreferrer">ROI area calculation code</a>: number of pixels in the Suite2p ROI mask field <code>xpix</code>.</p>
      <p>Cell-type code comes from loaded anatomical-channel label outputs or an uploaded table: <code>-1</code> excitatory/non-red, <code>0</code> unsure, <code>1</code> inhibitory/red.</p>
      <p><a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/roi_labels.py#L122-L158" target="_blank" rel="noopener noreferrer">SNR: 95/50 percentile calculation code</a></p>
      <p><a class="docs-link" href="https://github.com/farznaj/imaging_decisionMaking_exc_inh/blob/master/imaging/evaluate_components.py" target="_blank" rel="noopener noreferrer">CaImAn-style large-transient score source code</a></p>
      <p><a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/roi_labels.py#L251-L291" target="_blank" rel="noopener noreferrer">Autocorrelation e-fold time calculation code</a></p>
      <p><a class="docs-link" href="https://github.com/najafi-laboratory/2p_imaging/blob/main/utils_2p/preprocessing_summary.py#L470-L556" target="_blank" rel="noopener noreferrer">Inferred spike metric calculation code</a>: metrics are calculated from thresholded inferred-spike amplitude windows.</p>
    </div>
    <div class="trace-sort">
      <div>
        <div class="note">Check one or more metrics. Multiple checked metrics are converted to 0-1 normalized values and summed with equal weight.</div>
        <div id="sortMetricList" class="sort-metric-list">
          <label><input type="checkbox" name="sortMetric" value="original" checked> Original Suite2p index</label>
          <div class="sort-metric-group">Morphology Metrics</div>
          <label><input type="checkbox" name="sortMetric" value="aspect"> Aspect ratio</label>
          <label><input type="checkbox" name="sortMetric" value="compact"> Compactness</label>
          <label><input type="checkbox" name="sortMetric" value="roi_area"> ROI area (px)</label>
          <label><input type="checkbox" name="sortMetric" value="connectivity"> Connectivity</label>
          <div class="sort-metric-group">Cell Type / Indicator Metrics</div>
          <label><input type="checkbox" name="sortMetric" value="cell_type_code"> Cell type code</label>
          <div class="sort-metric-group">Fluorescence Trace Metrics</div>
          <label><input type="checkbox" name="sortMetric" value="footprint"> Footprint</label>
          <label><input type="checkbox" name="sortMetric" value="skew"> Skew</label>
          <label><input type="checkbox" name="sortMetric" value="snr_95_50"> SNR: 95/50 percentile</label>
          <label><input type="checkbox" name="sortMetric" value="andrea_postdoc_snr"> SNR: CaImAn (large-transient score)</label>
          <label><input type="checkbox" name="sortMetric" value="autocorr_efold_time_seconds"> Autocorrelation e-fold time</label>
          <div class="sort-metric-group">Inferred Spike Metrics</div>
          <label><input type="checkbox" name="sortMetric" value="oasis_event_snr"> Inferred spike SNR</label>
          <label><input type="checkbox" name="sortMetric" value="oasis_rise_tau_seconds"> Inferred spike rise tau</label>
          <label><input type="checkbox" name="sortMetric" value="oasis_decay_tau_seconds"> Inferred spike decay tau</label>
          <label><input type="checkbox" name="sortMetric" value="oasis_event_residual_ks"> Inferred spike residual Gaussian-fit distance</label>
        </div>
      </div>
      <div class="sort-actions">
        <label>Sort order
          <select id="sortDirection">
            <option value="desc">Highest first</option>
            <option value="asc" selected>Lowest first</option>
          </select>
        </label>
        <button id="applySort">Apply sort</button>
      </div>
    </div>
    <div class="dialog-actions"><button id="closeSortDialog" type="button">Close</button></div>
  </dialog>
  <dialog id="saveLabelsDialog">
    <div class="dialog-header">
      <div class="dialog-title">Save Labels</div>
      <div>
        <button id="saveLabelsInfo" class="info-button" type="button" aria-expanded="false" aria-controls="saveLabelsHelp">Read more</button>
        <button id="closeSaveLabelsDialogTop" class="dialog-close" type="button" aria-label="Close">&times;</button>
      </div>
    </div>
    <div id="saveLabelsHelp" class="info-box" hidden>
      <p><strong>Save current state into HTML:</strong> downloads a reviewed HTML copy that preserves labels and custom filters inside the file.</p>
      <p><strong>Export metric spreadsheet CSV:</strong> saves one row per ROI with code-friendly column names, current labels, ROI metrics, filter failures, and exclusion reasons.</p>
      <p><strong>Export labels NPY:</strong> saves <code>roi_manual_labels.npy</code>, a one-dimensional NumPy label array with one row per original Suite2p ROI: <code>NaN</code> not labeled, <code>0</code> bad, <code>1</code> good, and <code>2</code> unsure.</p>
      <p><strong>Export QC thresholds JSON:</strong> saves the current ROI metric thresholds/filter settings for reuse or documentation.</p>
    </div>
    <div class="save-options">
      <div class="save-option">
        <button id="saveHtmlWithLabels">Save current state into HTML</button>
        <span class="note">Use this to save the current state (labels, QC thresholds) of the .html to return to after closing the browser.</span>
      </div>
      <div class="save-option">
        <button id="saveMetricSpreadsheet">Export metric spreadsheet CSV</button>
        <span class="note">Exports one row per ROI with current labels, metrics, current filter failures, and reasons.</span>
      </div>
      <div class="save-option">
        <button id="saveManualLabels">Export labels NPY (roi_manual_labels.npy)</button>
        <span class="note">Exports a one-dimensional label array indexed by original Suite2p ROI.</span>
      </div>
      <div class="save-option">
        <button id="saveCurrentFilters">Export QC thresholds JSON</button>
        <span class="note">Exports the current ROI metric thresholds so they can be reused or documented.</span>
      </div>
      <a class="docs-link" href="https://najafi-laboratory.github.io/2p_imaging/roi-reviewer-exports/#2-export-format-and-downstream-use" target="_blank" rel="noopener noreferrer">Output format details</a>
    </div>
    <div class="dialog-actions"><button id="closeSaveLabelsDialog" type="button">Close</button></div>
  </dialog>
  <dialog id="labelAllDialog">
    <div class="dialog-header">
      <div class="dialog-title">Label all visible ROIs as ...</div>
      <button id="closeLabelAllDialogTop" class="dialog-close" type="button" aria-label="Close">&times;</button>
    </div>
    <div class="note">This applies only to the ROIs currently visible in the reviewer.</div>
    <div class="bulk-label-controls">
      <label>Label
        <select id="labelAllValue">
          <option value="1">Good</option>
          <option value="0">Bad</option>
          <option value="2">Unsure</option>
          <option value="-1">Not labeled</option>
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
        <div class="fov-review oasis-collapsed">
          <div class="grid {fov_grid_class}">
            <div class="panel"><div class="panel-title-row"><div class="title">Functional mean</div></div><div class="imagewrap"><img id="green"><svg class="overlay" preserveAspectRatio="xMidYMid meet"></svg></div></div>
            {red_panel}
          </div>
          {oasis_diagnostics_panel}
        </div>
      </div>
      <div class="panel">
        <div class="trace-header-row">
          <div class="title" id="traceTitle">Selected ROI dF/F</div>
          <div class="trace-header-controls" id="traceOasisControls">
            <label class="oasis-toggle"><input id="showInferredSpikes" type="checkbox" checked> Show Inferred Spikes</label>
            <label>Amplitude threshold</label>
            <div class="oasis-threshold-row">
              <input id="oasisThresholdSlider" type="range" min="0" max="1" step="0.001" value="0.05">
              <input id="oasisThreshold" type="number" min="0" step="0.001" value="0.05">
            </div>
            <button id="resetOasisThreshold" type="button">Reset to ROI default</button>
            <button id="toggleOasisDiagnostics" type="button">Show inferred spike fit metrics</button>
            <button class="info-button" type="button" data-info-target="oasisThresholdInfo" aria-expanded="false">Read more</button>
          </div>
        </div>
        <div id="oasisThresholdInfo" class="info-box" hidden>
          <p>Inferred spikes are loaded as an ROI-by-frame amplitude array. The current pipeline can produce that array with Suite2p's OASIS deconvolution, but the viewer only requires the same array shape and semantics.</p>
          <p>The underlying values are inferred spike amplitudes, not spike probabilities. Larger values mean the inference method assigned a stronger spike-like transient at that frame.</p>
          <p>The amplitude threshold is a viewer-side cutoff: frames with inferred spike amplitude above the selected threshold are drawn as red dots on the dF/F trace. The default per ROI is the precomputed threshold whose inferred-spike-window residuals had the smallest distance to a fitted Gaussian among tested candidate thresholds.</p>
          <p>The sortable residual Gaussian-fit metric is that minimized distance. Lower values mean the inferred-spike-window residuals were closer to Gaussian under the best tested threshold.</p>
          <p><a class="docs-link" href="https://suite2p.readthedocs.io/en/latest/deconvolution/" target="_blank" rel="noopener noreferrer">Suite2p spike deconvolution / OASIS documentation</a></p>
          <p><a class="docs-link" href="https://doi.org/10.1371/journal.pcbi.1005423" target="_blank" rel="noopener noreferrer">Original OASIS deconvolution paper</a></p>
        </div>
        <div class="trace-loader" id="traceLoader" style="display:none;">
          <input id="dffFile" type="file" accept=".npy">
          <input id="oasisFile" type="file" accept=".npy">
          <button id="loadDffFile">Load dF/F file</button>
          <button id="loadOasisFile" type="button">Load inferred spikes file</button>
        </div>
        <div class="note" id="traceLoadNote"></div>
        <canvas id="traceCanvas"></canvas>
        <div class="note">Wheel or drag to zoom/pan time. Double-click to reset.</div>
        <div class="controls">
          <label>Start s <input id="timeStart" type="number" min="0" step="0.001" value="0"></label>
          <label>End s <input id="timeEnd" type="number" min="0" step="0.001" value="0"></label>
          <button id="reset">Reset zoom</button>
        </div>
      </div>
    </div>
    <aside class="side-menu">
      <button id="toggleSideMenu" class="side-menu-toggle" type="button" aria-expanded="true">Hide menu</button>
      <div class="control-column">
          <details class="panel morphology-card menu-card" open>
            <summary>Filter</summary>
            <div class="menu-card-content">
            <div class="qc-header"><strong>ROI QC Filters</strong><span id="targetStructureInline" class="qc-current"></span></div>
            <div id="filterSummaryInline" class="filter-summary"></div>
            <button id="openMorphologyDialog" type="button">Filter ROIs</button>
            </div>
          </details>
          <details class="panel sort-card menu-card" open>
            <summary>Sort</summary>
            <div class="menu-card-content">
              <div class="sort-header"><strong>Sorting</strong><span id="sortCurrent" class="sort-current"></span></div>
              <button id="openSortDialog" type="button">Sort filtered ROIs</button>
            </div>
          </details>
          <details class="panel label-controls menu-card" open>
            <summary>Labeler</summary>
            <div class="menu-card-content">
            <span id="labelCounts"></span>
            <span class="note">Keyboard: G/B/U/N label; left/right arrows select the previous/next visible ROI.</span>
            <strong>Manually label filtered ROIs</strong>
            <label>Selected ROI (Suite2p Index) <input id="roiInput" type="number" min="0" value="0"> <span id="selectedSortPosition" class="note"></span></label>
            <details class="roi-details">
              <summary id="roiDetailsSummary">Selected ROI Details</summary>
              <div id="readout"></div>
            </details>
            <div class="note" id="cellTypeLabelStatus"></div>
            <div class="button-row">
              <button id="markGood" class="good">Good (G)</button>
              <button id="markBad" class="bad">Bad (B)</button>
            </div>
            <div class="button-row">
              <button id="markUnsure" class="unsure">Unsure (U)</button>
              <button id="markUnlabeled" class="unlabeled">Not labeled (N)</button>
            </div>
            <button id="openLabelAllDialog" type="button">Label all as ...</button>
            <div class="nav-row">
              <button id="previousRoi" class="nav-button" title="Previous visible ROI (Left arrow)">&#8592; Previous</button>
              <button id="nextRoi" class="nav-button" title="Next visible ROI (Right arrow)">Next &#8594;</button>
            </div>
            <strong>Show ROIs</strong>
            <details class="show-roi-menu" id="showRoiMenu">
              <summary id="showRoiSummary">All filtered ROIs</summary>
              <div class="show-roi-options">
                <label><input type="checkbox" class="roi-display-checkbox" value="good" checked> Good</label>
                <label><input type="checkbox" class="roi-display-checkbox" value="bad" checked> Bad</label>
                <label><input type="checkbox" class="roi-display-checkbox" value="unsure" checked> Unsure</label>
                <label><input type="checkbox" class="roi-display-checkbox" value="unlabeled" checked> Not labeled</label>
              </div>
            </details>
            </div>
          </details>
          <details class="panel label-controls menu-card" open>
            <summary>Export</summary>
            <div class="menu-card-content">
            <button id="openExclusions">Open ROI metric spreadsheet</button>
            <button id="openSaveLabelsDialog" type="button">Save Labels</button>
            </div>
          </details>
        </div>
    </aside>
  </div>
  <div class="plots">
    <div class="panel">
      <div class="title" id="stackTitle">dF/F, stacked ROIs</div>
      <div class="controls">
        <strong>Stacked trace range</strong>
        <label>First ROI <input id="yStart" type="number" min="0" value="0"></label>
        <label>Last ROI <input id="yEnd" type="number" min="0" value="0"></label>
        <button id="showAllVisibleTraces" type="button">Show all visible traces</button>
      </div>
      <canvas id="stackCanvas"></canvas>
      <div class="note">Wheel to zoom time.</div>
    </div>
    <div class="panel">
      <div class="title">Motion correction drift per frame</div>
      <canvas id="motionDriftCanvas"></canvas>
      <div class="note">Rigid x/y drift per frame from Suite2p motion correction. These plots follow the current trace time window.</div>
      <div class="title" style="margin-top:8px;">Motion Correction pixel shift distributions</div>
      <canvas id="motionDistributionCanvas"></canvas>
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
document.getElementById("meta").textContent = `${{data.nRois}} ROIs | ${{data.nFrames.toLocaleString()}} frames | ${{data.frameRate.toFixed(3)}} Hz${{data.redAvailable ? "" : " | no red channel detected"}}`;
document.querySelectorAll(".overlay").forEach(svg => svg.setAttribute("viewBox", `0 0 ${{data.imageWidth}} ${{data.imageHeight}}`));
function targetStructureLabel(name) {{
  return name === "neuron" ? "soma" : name;
}}
document.getElementById("targetStructureSummary").textContent = `Pipeline target structure: ${{targetStructureLabel(data.targetStructure)}}; no ROI filter is applied on load.`;
document.getElementById("targetStructureInline").textContent = `Pipeline target structure: ${{targetStructureLabel(data.targetStructure)}}; default view includes all Suite2p ROIs.`;
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
const motionX = data.motionAvailable && data.xoff ? b64f32(data.xoff) : null;
const motionY = data.motionAvailable && data.yoff ? b64f32(data.yoff) : null;
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
let oasisSpikes = null;
let oasisEventThreshold = Number(data.oasisEventThreshold ?? 0.05);
let showInferredSpikes = true;
let oasisSpikeMin = 0;
let oasisSpikeMax = 1;
let oasisResidualCache = new Map();
let oasisDiagnosticsCache = new Map();
function configureOasisThresholdControls() {{
  const sp = spikeTrace(selected);
  if (sp) {{
    let minValue = Infinity;
    let maxValue = -Infinity;
    for (let i = 0; i < sp.length; i++) {{
      const value = sp[i];
      if (Number.isFinite(value)) {{
        minValue = Math.min(minValue, value);
        maxValue = Math.max(maxValue, value);
      }}
    }}
    if (!Number.isFinite(minValue)) minValue = 0;
    if (!Number.isFinite(maxValue) || maxValue <= minValue) {{
      maxValue = minValue + 0.001;
    }}
    oasisSpikeMin = minValue;
    oasisSpikeMax = Math.max(maxValue, oasisEventThreshold, minValue + 0.001);
  }}
  const slider = document.getElementById("oasisThresholdSlider");
  const box = document.getElementById("oasisThreshold");
  slider.min = oasisSpikeMin.toPrecision(6);
  slider.max = oasisSpikeMax.toPrecision(6);
  slider.value = oasisEventThreshold;
  box.min = oasisSpikeMin.toPrecision(6);
  box.max = oasisSpikeMax.toPrecision(6);
  box.value = oasisEventThreshold;
}}
function setOasisThreshold(value, redraw = true) {{
  const fallback = Number(data.oasisEventThreshold ?? 0.05);
  const next = Number.isFinite(value) ? value : fallback;
  oasisEventThreshold = Math.min(Math.max(next, oasisSpikeMin), oasisSpikeMax);
  configureOasisThresholdControls();
  if (redraw) drawSelectedTraceAndOasis();
}}
function selectedOasisDefaultThreshold() {{
  const value = dffMetric(selected, "oasis_optimal_threshold");
  return Number.isFinite(Number(value)) ? Number(value) : Number(data.oasisEventThreshold ?? 0.05);
}}
function setDffFromArrayBuffer(arrayBuffer, silent = false) {{
  const parsed = parseNpy(arrayBuffer);
  dff = parsed.array;
  if (parsed.shape.length >= 2) {{
    data.nRois = parsed.shape[0];
    data.nFrames = parsed.shape[1];
  }}
  oasisResidualCache.clear();
  oasisDiagnosticsCache.clear();
  if (!silent) {{
    document.getElementById("traceLoadNote").textContent = `Loaded dF/F file with ${{data.nRois}} ROIs x ${{data.nFrames}} frames.`;
  }} else {{
    updateSidecarLoadNote();
  }}
  draw();
}}
function updateSidecarLoadNote() {{
  const note = document.getElementById("traceLoadNote");
  const parts = [];
  if (data.dffStorageMode === "file") {{
    parts.push(dff ? `dF/F loaded (${{data.dffSidecarName}})` : `dF/F not loaded (${{data.dffSidecarName || "sidecar .npy"}})`);
  }}
  if (data.oasisAvailable && data.oasisStorageMode === "file") {{
    parts.push(oasisSpikes ? `inferred spikes loaded (${{data.oasisSidecarName}})` : `inferred spikes not loaded (${{data.oasisSidecarName || "sidecar .npy"}})`);
  }}
  if (parts.length) {{
    note.textContent = `${{parts.join("; ")}}. If auto-loading is blocked, use the buttons above to select the matching .npy file(s).`;
  }}
}}
async function loadSidecarByRelativeName(sidecarName, setter, label) {{
  if (!sidecarName) return false;
  try {{
    const response = await fetch(sidecarName);
    if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
    const arrayBuffer = await response.arrayBuffer();
    setter(arrayBuffer, true);
    return true;
  }} catch (error) {{
    console.warn(`Could not auto-load ${{label}} sidecar ${{sidecarName}}:`, error);
    updateSidecarLoadNote();
    return false;
  }}
}}
if (data.dffStorageMode === "embedded" && data.dff) {{
  dff = b64f32(data.dff);
}}
if (data.dffStorageMode === "file" || (data.oasisAvailable && data.oasisStorageMode === "file")) {{
  const loader = document.getElementById("traceLoader");
  const note = document.getElementById("traceLoadNote");
  loader.style.display = "";
  note.textContent = `This session uses sidecar .npy files for large arrays. Trying to load them from this folder.`;
}}
function setOasisFromArrayBuffer(arrayBuffer, silent = false) {{
  const parsed = parseNpy(arrayBuffer);
  oasisSpikes = parsed.array;
  oasisResidualCache.clear();
  oasisDiagnosticsCache.clear();
  configureOasisThresholdControls();
  if (!silent) updateSidecarLoadNote();
  draw();
}}
function updateCellTypeLabelStatus() {{
  const status = document.getElementById("cellTypeLabelStatus");
  if (!status) return;
  const loaded = Array.isArray(data.cellTypeLabels) && data.cellTypeLabels.some(value => value === -1 || value === 0 || value === 1);
  status.textContent = loaded
    ? `Cell-type labels loaded from ${{data.cellTypeLabelSource}}. Codes: -1 excitatory/non-red, 0 unsure, 1 inhibitory/red.`
    : data.cellTypeLabelSource;
}}
function parseDelimitedRows(text) {{
  const delimiter = text.includes("\\t") && !text.includes(",") ? "\\t" : ",";
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {{
    const ch = text[i];
    if (quoted) {{
      if (ch === '"' && text[i + 1] === '"') {{ field += '"'; i++; }}
      else if (ch === '"') quoted = false;
      else field += ch;
    }} else if (ch === '"') quoted = true;
    else if (ch === delimiter) {{ row.push(field); field = ""; }}
    else if (ch === "\\n" || ch === "\\r") {{
      if (ch === "\\r" && text[i + 1] === "\\n") i++;
      row.push(field); field = "";
      if (row.some(value => value.trim() !== "")) rows.push(row);
      row = [];
    }} else field += ch;
  }}
  row.push(field);
  if (row.some(value => value.trim() !== "")) rows.push(row);
  return rows;
}}
function parseCellTypeCode(code, label) {{
  const numeric = Number(String(code ?? "").trim());
  if (numeric === -1 || numeric === 0 || numeric === 1) return numeric;
  const text = String(label ?? code ?? "").trim().toLowerCase();
  if (["inhibitory", "inhibitory/red", "red", "inh"].includes(text)) return 1;
  if (["excitatory", "excitatory/non-red", "non-red", "nonred", "exc"].includes(text)) return -1;
  if (["unsure", "uncertain", "unknown"].includes(text)) return 0;
  return null;
}}
function loadCellTypeTableText(text, sourceName) {{
  const rows = parseDelimitedRows(text);
  if (rows.length < 2) throw new Error("Expected a header row and one row per original Suite2p ROI.");
  const headers = rows[0].map(value => value.trim().toLowerCase());
  const codeIndex = headers.indexOf("cell_type_code");
  const labelIndex = headers.indexOf("cell_type_label");
  let suite2pIndex = headers.indexOf("suite2p_index");
  if (suite2pIndex < 0) suite2pIndex = headers.indexOf("roi");
  if (suite2pIndex < 0) suite2pIndex = headers.indexOf("index");
  if (codeIndex < 0 || labelIndex < 0) throw new Error("Expected columns cell_type_code and cell_type_label.");
  const nextLabels = new Array(data.nRois).fill(null);
  if (suite2pIndex >= 0) {{
    const roiBySuite2p = new Map(data.suite2pIndices.map((value, roi) => [Number(value), roi]));
    for (const row of rows.slice(1)) {{
      const roi = roiBySuite2p.get(Number(row[suite2pIndex]));
      if (roi === undefined) continue;
      nextLabels[roi] = parseCellTypeCode(row[codeIndex], row[labelIndex]);
    }}
  }} else {{
    if (rows.length - 1 !== data.nRois) throw new Error(`Expected ${{data.nRois}} data rows, found ${{rows.length - 1}}.`);
    rows.slice(1).forEach((row, roi) => {{
      nextLabels[roi] = parseCellTypeCode(row[codeIndex], row[labelIndex]);
    }});
  }}
  data.cellTypeLabels = nextLabels;
  data.cellTypeLabelSource = sourceName;
  updateCellTypeLabelStatus();
  evaluateFilter();
  updateSortCurrent();
  draw();
}}
const labels = new Int8Array(data.nRois);
for (let roi = 0; roi < data.nRois; roi++) {{
  labels[roi] = -1;
}}
if (Array.isArray(data.initialLabels) && data.initialLabels.length === data.nRois) {{
  for (let roi = 0; roi < data.nRois; roi++) {{
    const label = Number(data.initialLabels[roi]);
    if (label === -1 || label === 0 || label === 1 || label === 2) labels[roi] = label;
  }}
}}
const filterPass = new Uint8Array(data.nRois);
let customPresets = data.customMorphologyPresets && typeof data.customMorphologyPresets === "object" ? {{...data.customMorphologyPresets}} : {{}};
const defaultFilter = data.morphologyPresets.all_rois;
let selected = 0, x0 = 0, x1 = data.nFrames - 1, y0 = 0, y1 = 0, visibleRois = [];
let appliedSortMetrics = ["original"];
let appliedSortDirection = "asc";

function fit(canvas) {{
  const r = window.devicePixelRatio || 1, box = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(box.width * r));
  canvas.height = Math.max(1, Math.round(box.height * r));
}}
function syncControlColumnHeight() {{
  return;
}}
function trace(roi) {{
  if (!dff) return null;
  return dff.subarray(roi * data.nFrames, (roi + 1) * data.nFrames);
}}
function spikeTrace(roi) {{
  if (!oasisSpikes) return null;
  return oasisSpikes.subarray(roi * data.nFrames, (roi + 1) * data.nFrames);
}}
if (data.oasisAvailable) {{
  document.getElementById("traceOasisControls").classList.add("visible");
  if (data.oasisStorageMode === "embedded" && data.oasisSpikes) {{
    oasisSpikes = b64f32(data.oasisSpikes);
    configureOasisThresholdControls();
  }} else {{
    configureOasisThresholdControls();
  }}
}}
function val(roi, frame) {{
  if (!dff) return NaN;
  return dff[roi * data.nFrames + frame];
}}
function dffMetric(roi, key, fallbackKey = null) {{
  const metrics = data.dffMetrics[roi] || {{}};
  return dffMetricValue(metrics, key, fallbackKey);
}}
function dffMetricValue(metrics, key, fallbackKey = null) {{
  const value = metrics[key];
  if (value !== null && value !== undefined) return value;
  return fallbackKey ? metrics[fallbackKey] : undefined;
}}
function metricValue(roi, metric) {{
  if (metric === "snr_95_50" || metric === "event_snr") return dffMetric(roi, "snr_95_50", "event_snr");
  if (metric === "andrea_postdoc_snr") return dffMetric(roi, "andrea_postdoc_snr");
  if (metric === "autocorr_efold_time_seconds") return dffMetric(roi, "autocorr_efold_time_seconds", "decay_tau_seconds");
  if (metric === "oasis_event_residual_ks") return dffMetric(roi, "oasis_event_residual_ks");
  if (metric === "oasis_event_snr") return dffMetric(roi, "oasis_event_snr");
  if (metric === "oasis_rise_tau_seconds") return dffMetric(roi, "oasis_rise_tau_seconds");
  if (metric === "oasis_decay_tau_seconds") return dffMetric(roi, "oasis_decay_tau_seconds");
  if (metric === "cell_type_code") return data.cellTypeLabels?.[roi] ?? NaN;
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
  if (metric === "snr_95_50" || metric === "event_snr") return "SNR: 95/50 percentile";
  if (metric === "andrea_postdoc_snr") return "SNR: CaImAn (large-transient score)";
  if (metric === "autocorr_efold_time_seconds") return "Autocorrelation e-fold time";
  if (metric === "oasis_event_residual_ks") return "inferred spike residual Gaussian-fit distance";
  if (metric === "oasis_event_snr") return "inferred spike SNR";
  if (metric === "oasis_rise_tau_seconds") return "inferred spike rise tau";
  if (metric === "oasis_decay_tau_seconds") return "inferred spike decay tau";
  if (metric === "cell_type_code") return "cell type code";
  if (metric === "roi_area") return "ROI area (px)";
  if (metric === "connectivity") return "Connectivity";
  if (metric === "skew") return "Skew";
  if (metric === "aspect") return "Aspect ratio";
  if (metric === "compact") return "Compactness";
  if (metric === "footprint") return "Footprint";
  if (metric === "original" || metric === "suite2p_index") return "original Suite2p index";
  return metric.replace("_", " ");
}}
function sortMetricText(metrics = appliedSortMetrics) {{
  if (!metrics.length) return metricLabel("original");
  if (metrics.length === 1) return metricLabel(metrics[0]);
  return metrics.map(metricLabel).join(" + ");
}}
function finiteMetricValues(metric) {{
  const values = [];
  for (let roi = 0; roi < data.nRois; roi++) {{
    const value = Number(metricValue(roi, metric));
    if (Number.isFinite(value)) values.push(value);
  }}
  return values;
}}
function percentile(values, q) {{
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}}
function mean(values) {{
  const finite = values.filter(Number.isFinite);
  return finite.length ? finite.reduce((total, value) => total + value, 0) / finite.length : NaN;
}}
const suggestedThresholds = {{}};
function setSuggestedThreshold(inputId, value) {{
  suggestedThresholds[inputId] = Number.isFinite(value) ? value : NaN;
}}
function updateMetricDefaults() {{
  const skew = finiteMetricValues("skew");
  const aspect = finiteMetricValues("aspect");
  const footprint = finiteMetricValues("footprint");
  const compact = finiteMetricValues("compact");
  const connect = finiteMetricValues("connectivity");
  const roiArea = finiteMetricValues("roi_area");
  const snr9550 = finiteMetricValues("snr_95_50");
  const caiman = finiteMetricValues("andrea_postdoc_snr");
  const efold = finiteMetricValues("autocorr_efold_time_seconds");
  const oasisSnr = finiteMetricValues("oasis_event_snr");
  const oasisRiseTau = finiteMetricValues("oasis_rise_tau_seconds");
  const oasisDecayTau = finiteMetricValues("oasis_decay_tau_seconds");
  const oasisResidualKs = finiteMetricValues("oasis_event_residual_ks");
  const selectedPreset = data.morphologyPresets[data.targetStructure] || data.morphologyPresets.all_rois || {{}};
  setSuggestedThreshold("skewMin", selectedPreset.skewMin ?? mean(skew));
  setSuggestedThreshold("skewMax", selectedPreset.skewMax ?? mean(skew));
  setSuggestedThreshold("aspectMin", selectedPreset.aspectMin ?? mean(aspect));
  setSuggestedThreshold("aspectMax", selectedPreset.aspectMax ?? mean(aspect));
  setSuggestedThreshold("footprintMin", selectedPreset.footprintMin ?? mean(footprint));
  setSuggestedThreshold("footprintMax", selectedPreset.footprintMax ?? mean(footprint));
  setSuggestedThreshold("compactMin", selectedPreset.compactMin ?? mean(compact));
  setSuggestedThreshold("compactMax", selectedPreset.compactMax ?? mean(compact));
  setSuggestedThreshold("maxConnect", percentile(connect, 0.75));
  setSuggestedThreshold("roiAreaMin", percentile(roiArea, 0.25));
  setSuggestedThreshold("roiAreaMax", percentile(roiArea, 0.75));
  setSuggestedThreshold("eventSnrMin", mean(snr9550));
  setSuggestedThreshold("andreaPostdocSnrMin", mean(caiman));
  setSuggestedThreshold("autocorrEfoldMin", percentile(efold, 0.25));
  setSuggestedThreshold("autocorrEfoldMax", percentile(efold, 0.75));
  setSuggestedThreshold("oasisEventSnrMin", mean(oasisSnr));
  setSuggestedThreshold("oasisRiseTauMin", percentile(oasisRiseTau, 0.25));
  setSuggestedThreshold("oasisRiseTauMax", percentile(oasisRiseTau, 0.75));
  setSuggestedThreshold("oasisDecayTauMin", percentile(oasisDecayTau, 0.25));
  setSuggestedThreshold("oasisDecayTauMax", percentile(oasisDecayTau, 0.75));
  setSuggestedThreshold("oasisResidualKsMax", percentile(oasisResidualKs, 0.75));
  document.getElementById("maxConnectDefault").textContent = `(suggested max: ${{fmt(suggestedThresholds.maxConnect)}})`;
  document.getElementById("roiAreaMinDefault").textContent = `(suggested min: ${{fmt(suggestedThresholds.roiAreaMin)}})`;
  document.getElementById("roiAreaMaxDefault").textContent = `(suggested max: ${{fmt(suggestedThresholds.roiAreaMax)}})`;
  document.getElementById("eventSnrDefault").textContent = `(suggested min: ${{fmt(suggestedThresholds.eventSnrMin)}})`;
  document.getElementById("andreaPostdocSnrDefault").textContent = `(suggested min: ${{fmt(suggestedThresholds.andreaPostdocSnrMin)}})`;
  document.getElementById("autocorrEfoldMinDefault").textContent = `(suggested min: ${{fmt(suggestedThresholds.autocorrEfoldMin)}})`;
  document.getElementById("autocorrEfoldMaxDefault").textContent = `(suggested max: ${{fmt(suggestedThresholds.autocorrEfoldMax)}})`;
  document.getElementById("oasisEventSnrDefault").textContent = oasisSnr.length ? `(suggested min: ${{fmt(suggestedThresholds.oasisEventSnrMin)}})` : "";
  document.getElementById("oasisRiseTauMinDefault").textContent = oasisRiseTau.length ? `(suggested min: ${{fmt(suggestedThresholds.oasisRiseTauMin)}})` : "";
  document.getElementById("oasisRiseTauMaxDefault").textContent = oasisRiseTau.length ? `(suggested max: ${{fmt(suggestedThresholds.oasisRiseTauMax)}})` : "";
  document.getElementById("oasisDecayTauMinDefault").textContent = oasisDecayTau.length ? `(suggested min: ${{fmt(suggestedThresholds.oasisDecayTauMin)}})` : "";
  document.getElementById("oasisDecayTauMaxDefault").textContent = oasisDecayTau.length ? `(suggested max: ${{fmt(suggestedThresholds.oasisDecayTauMax)}})` : "";
  document.getElementById("oasisResidualKsDefault").textContent = oasisResidualKs.length ? `(suggested max: ${{fmt(suggestedThresholds.oasisResidualKsMax)}})` : "";
  configureOasisFilterControls();
}}
function configureOasisFilterControls() {{
  const oasisMetricIds = [
    ["oasisEventSnrMin", "oasis_event_snr"],
    ["oasisRiseTauMin", "oasis_rise_tau_seconds"],
    ["oasisRiseTauMax", "oasis_rise_tau_seconds"],
    ["oasisDecayTauMin", "oasis_decay_tau_seconds"],
    ["oasisDecayTauMax", "oasis_decay_tau_seconds"],
    ["oasisResidualKsMax", "oasis_event_residual_ks"],
  ];
  for (const [inputId, metric] of oasisMetricIds) {{
    const input = document.getElementById(inputId);
    if (!input) continue;
    const available = finiteMetricValues(metric).length > 0;
    input.disabled = !available;
    input.placeholder = available ? "Not Used" : "No inferred spikes file detected";
    if (!available) input.value = "";
  }}
}}
function updateSuite2pSuggestedThresholds(filter) {{
  const fallback = {{
    skewMin: mean(finiteMetricValues("skew")),
    skewMax: mean(finiteMetricValues("skew")),
    aspectMin: mean(finiteMetricValues("aspect")),
    aspectMax: mean(finiteMetricValues("aspect")),
    footprintMin: mean(finiteMetricValues("footprint")),
    footprintMax: mean(finiteMetricValues("footprint")),
    compactMin: mean(finiteMetricValues("compact")),
    compactMax: mean(finiteMetricValues("compact")),
  }};
  for (const id of ["skewMin","skewMax","aspectMin","aspectMax","footprintMin","footprintMax","compactMin","compactMax"]) {{
    setSuggestedThreshold(id, filter && filter[id] !== null && filter[id] !== undefined ? filter[id] : fallback[id]);
  }}
}}
function drawMetricHistogram(canvas) {{
  fit(canvas);
  const ctx = canvas.getContext("2d");
  const metric = canvas.dataset.metric;
  const values = finiteMetricValues(metric);
  const w = canvas.width, h = canvas.height, pad = 24;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#d0d5dd"; ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  if (!values.length) return;
  const r = window.devicePixelRatio || 1;
  ctx.fillStyle = "#344054"; ctx.font = `${{11 * r}}px Arial`; ctx.textAlign = "left";
  ctx.fillText(metricLabel(metric), pad, 12 * r);
  let lo = percentile(values, 0.01), hi = percentile(values, 0.99);
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) {{ lo = Math.min(...values); hi = Math.max(...values); }}
  if (hi <= lo) hi = lo + 1;
  const bins = 28, counts = Array(bins).fill(0);
  values.forEach(value => {{
    const bin = Math.max(0, Math.min(bins - 1, Math.floor((value - lo) / (hi - lo) * bins)));
    counts[bin]++;
  }});
  const maxCount = Math.max(...counts, 1);
  const plotW = w - pad * 2, plotTop = 32 * r, plotBottom = h - pad * 1.45;
  const plotH = Math.max(1, plotBottom - plotTop);
  ctx.fillStyle = "#dbeafe";
  counts.forEach((count, index) => {{
    const x = pad + index / bins * plotW;
    const bw = Math.max(1, plotW / bins - 1);
    const bh = count / maxCount * plotH;
    ctx.fillRect(x, plotBottom - bh, bw, bh);
  }});
  function drawLine(value, color, label) {{
    if (!Number.isFinite(value)) return;
    const x = pad + (value - lo) / (hi - lo) * plotW;
    if (x < pad || x > pad + plotW) return;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = Math.max(2, 1.5 * r);
    ctx.beginPath(); ctx.moveTo(x, plotTop); ctx.lineTo(x, plotBottom + 2); ctx.stroke();
    ctx.fillStyle = color; ctx.font = `${{10 * r}}px Arial`; ctx.textAlign = "center";
    ctx.fillText(`${{label}} ${{fmt(value)}}`, x, plotTop - 3 * r);
    ctx.restore();
  }}
  function thresholdInfo(inputId, label) {{
    if (!inputId) return;
    const activeValue = filterValue(inputId);
    const suggestedValue = suggestedThresholds[inputId];
    const value = Number.isFinite(activeValue) ? activeValue : suggestedValue;
    return {{value, label, active: Number.isFinite(activeValue)}};
  }}
  const minThreshold = thresholdInfo(canvas.dataset.min, "min");
  const maxThreshold = thresholdInfo(canvas.dataset.max, "max");
  if (
    minThreshold && maxThreshold &&
    !minThreshold.active && !maxThreshold.active &&
    Number.isFinite(minThreshold.value) && Number.isFinite(maxThreshold.value) &&
    Math.abs(minThreshold.value - maxThreshold.value) <= Math.max(1e-9, Math.abs(minThreshold.value) * 1e-9)
  ) {{
    drawLine(minThreshold.value, "#475467", "mean");
  }} else {{
    if (minThreshold) drawLine(minThreshold.value, "#16a34a", minThreshold.label);
    if (maxThreshold) drawLine(maxThreshold.value, "#dc2626", maxThreshold.label);
  }}
  const binSize = (hi - lo) / bins;
  const yText = h - 7 * (window.devicePixelRatio || 1);
  ctx.fillStyle = "#475467"; ctx.font = `${{10 * (window.devicePixelRatio || 1)}}px Arial`;
  ctx.textAlign = "left"; ctx.fillText(fmt(lo), pad, yText);
  ctx.textAlign = "center"; ctx.fillText(`bin ${{fmt(binSize)}}`, pad + plotW / 2, yText);
  ctx.textAlign = "right"; ctx.fillText(fmt(hi), pad + plotW, yText);
}}
function drawMetricHistograms() {{
  document.querySelectorAll(".metric-histogram-panel[open] .metric-histogram").forEach(drawMetricHistogram);
}}
function sortVisibleRois(rois) {{
  const metrics = appliedSortMetrics.length ? appliedSortMetrics : ["original"];
  const direction = appliedSortDirection;
  const ranges = {{}};
  for (const metric of metrics) {{
    const values = rois.map(roi => Number(metricValue(roi, metric))).filter(Number.isFinite);
    const minValue = values.length ? Math.min(...values) : NaN;
    const maxValue = values.length ? Math.max(...values) : NaN;
    ranges[metric] = {{min: minValue, max: maxValue}};
  }}
  function normalizedMetricValue(roi, metric) {{
    const value = Number(metricValue(roi, metric));
    if (!Number.isFinite(value)) return NaN;
    const range = ranges[metric];
    if (!range || !Number.isFinite(range.min) || !Number.isFinite(range.max)) return NaN;
    if (range.max === range.min) return 0.5;
    return (value - range.min) / (range.max - range.min);
  }}
  function sortScore(roi) {{
    let total = 0, count = 0;
    for (const metric of metrics) {{
      const value = normalizedMetricValue(roi, metric);
      if (Number.isFinite(value)) {{ total += value; count++; }}
    }}
    return count ? total / count : NaN;
  }}
  return rois.slice().sort((a, b) => {{
    const av = sortScore(a);
    const bv = sortScore(b);
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
  return `${{position}}/${{total}} by ${{sortMetricText()}} ${{appliedSortDirection}}`;
}}
function updateSortCurrent() {{
  document.getElementById("sortCurrent").textContent = `Current order: ${{sortMetricText()}}, ${{appliedSortDirection}}`;
  document.getElementById("selectedSortPosition").textContent = currentSortPositionText();
}}
function syncStackInputs() {{
  document.getElementById("yStart").value = y0;
  document.getElementById("yEnd").value = y1;
}}
function currentStackWindowSize() {{
  const start = Number(document.getElementById("yStart").value);
  const end = Number(document.getElementById("yEnd").value);
  if (Number.isFinite(start) && Number.isFinite(end) && end >= start) return Math.max(1, Math.round(end - start + 1));
  return Math.max(1, Math.round(y1 - y0 + 1));
}}
function ensureSelectedInStack() {{
  if (!visibleRois.length) return;
  const position = visibleRois.indexOf(selected);
  if (position < 0) return;
  const windowSize = Math.max(1, Math.min(visibleRois.length, currentStackWindowSize()));
  if (position >= y0 && position <= y1 && (y1 - y0 + 1) >= Math.min(windowSize, visibleRois.length)) {{
    syncStackInputs();
    return;
  }}
  let start = position;
  if (position > y1) start = position - windowSize + 1;
  start = Math.max(0, Math.min(start, visibleRois.length - windowSize));
  y0 = start;
  y1 = start + windowSize - 1;
  syncStackInputs();
}}
function selectedSortMetrics() {{
  const checked = Array.from(document.querySelectorAll('input[name="sortMetric"]:checked')).map(input => input.value);
  return checked.length ? checked : ["original"];
}}
function applySort() {{
  appliedSortMetrics = selectedSortMetrics();
  appliedSortDirection = document.getElementById("sortDirection").value;
  updateVisibleRois(true);
  updateSortCurrent();
  if (visibleRois.length) setSelected(visibleRois[0]);
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
  ensureSelectedInStack();
  const metrics = data.morphology[selected];
  const dffMetrics = data.dffMetrics[selected];
  const snr9550 = dffMetric(selected, "snr_95_50", "event_snr");
  const postdocSnr = dffMetric(selected, "andrea_postdoc_snr");
  const suite2pRoi = data.suite2pIndices[selected];
  const cellType = cellTypeLabelName(data.cellTypeLabels?.[selected]);
  if (data.oasisAvailable) setOasisThreshold(selectedOasisDefaultThreshold(), false);
  document.getElementById("roiInput").value = suite2pRoi;
  document.getElementById("roiDetailsSummary").textContent = "Selected ROI Details";
  document.getElementById("readout").textContent = `cell type ${{cellType}} | area ${{fmt(dffMetrics.roi_area)}} px | skew ${{fmt(metrics.skew)}} connect ${{metrics.connect}} aspect ${{fmt(metrics.aspect)}} compact ${{fmt(metrics.compact)}} footprint ${{fmt(metrics.footprint)}} | SNR: 95/50 percentile ${{fmt(snr9550)}} | SNR: CaImAn (large-transient score) ${{fmt(postdocSnr)}} | autocorrelation e-fold time ${{fmt(dffMetricValue(dffMetrics, "autocorr_efold_time_seconds", "decay_tau_seconds"))}} s | inferred spike residual Gaussian-fit distance ${{fmt(dffMetricValue(dffMetrics, "oasis_event_residual_ks"))}}`;
  document.getElementById("traceTitle").textContent = `Selected ROI - Suite2p Original Index ${{suite2pRoi}}/${{data.nRois}}, Current Sort ${{currentSortPositionText()}}`;
  document.querySelectorAll(".roi").forEach(c => c.classList.toggle("selected", Number(c.dataset.roi) === selected));
  updateLabelControls();
  updateSortCurrent();
  draw();
}}
function roiMatchesDisplayMode(roi) {{
  const active = selectedRoiDisplayLabels();
  if (active.has("all")) return true;
  if (labels[roi] === 1) return active.has("good");
  if (labels[roi] === 0) return active.has("bad");
  if (labels[roi] === 2) return active.has("unsure");
  return active.has("unlabeled");
}}
function selectedRoiDisplayLabels() {{
  const checked = Array.from(document.querySelectorAll(".roi-display-checkbox:checked")).map(input => input.value);
  return new Set(checked.length ? checked : ["all"]);
}}
function updateRoiDisplaySummary() {{
  const summary = document.getElementById("showRoiSummary");
  const boxes = Array.from(document.querySelectorAll(".roi-display-checkbox"));
  const checked = boxes.filter(input => input.checked).map(input => input.value);
  if (!checked.length || checked.length === boxes.length) {{
    summary.textContent = "All filtered ROIs";
    return;
  }}
  const active = new Set(checked);
  const labelsText = [];
  if (active.has("good")) labelsText.push("good");
  if (active.has("bad")) labelsText.push("bad");
  if (active.has("unsure")) labelsText.push("unsure");
  if (active.has("unlabeled")) labelsText.push("not labeled");
  summary.textContent = labelsText.length ? labelsText.join(", ") : "All filtered ROIs";
}}
function functionalRoiVisible(roi) {{
  return Boolean(filterPass[roi] && roiMatchesDisplayMode(roi));
}}
function updateVisibleRois(resetRange = false) {{
  const rois = [];
  for (let roi = 0; roi < data.nRois; roi++) if (functionalRoiVisible(roi)) rois.push(roi);
  visibleRois = sortVisibleRois(rois);
  document.getElementById("yStart").max = Math.max(0, visibleRois.length - 1);
  document.getElementById("yEnd").max = Math.max(0, visibleRois.length - 1);
  if (resetRange) {{
    y0 = 0;
    y1 = Math.max(0, Math.min(19, visibleRois.length - 1));
  }} else {{
    y0 = Math.max(0, Math.min(y0, Math.max(0, visibleRois.length - 1)));
    y1 = Math.max(y0, Math.min(y1 || Math.min(19, Math.max(0, visibleRois.length - 1)), Math.max(0, visibleRois.length - 1)));
  }}
  syncStackInputs();
  document.querySelectorAll(".roi, .roi-hit").forEach(path => {{
    const roi = Number(path.dataset.roi);
    path.style.display = functionalRoiVisible(roi) ? "" : "none";
  }});
  const redOverlayMode = document.getElementById("redOverlayMode");
  if (redOverlayMode) setRedOverlayMode(redOverlayMode.value);
  if (!visibleRois.length) draw();
  else if (!visibleRois.includes(selected)) setSelected(visibleRois[0]);
  else {{ ensureSelectedInStack(); draw(); }}
}}
function updateLabelControls() {{
  const label = labels[selected];
  document.getElementById("markGood").classList.toggle("active", label === 1);
  document.getElementById("markBad").classList.toggle("active", label === 0);
  document.getElementById("markUnsure").classList.toggle("active", label === 2);
  document.getElementById("markUnlabeled").classList.toggle("active", label === -1);
  let good = 0, bad = 0, unsure = 0, unlabeled = 0;
  for (const value of labels) {{ if (value === 1) good++; else if (value === 0) bad++; else if (value === 2) unsure++; else unlabeled++; }}
  document.getElementById("labelCounts").textContent = `${{good}} good | ${{bad}} bad | ${{unsure}} unsure | ${{unlabeled}} not labeled`;
}}
function labelName(label) {{
  if (label === 1) return "good";
  if (label === 0) return "bad";
  if (label === 2) return "unsure";
  return "not labeled";
}}
function cellTypeLabelName(label) {{
  if (label === 1) return "inhibitory/red";
  if (label === -1) return "excitatory/non-red";
  if (label === 0) return "unsure";
  return "not loaded";
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
function filterString(id) {{
  const element = document.getElementById(id);
  if (!element) return "";
  return String(element.value || "").trim();
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
    roiAreaMin: filterValue("roiAreaMin"), roiAreaMax: filterValue("roiAreaMax"),
    eventSnrMin: filterValue("eventSnrMin"),
    andreaPostdocSnrMin: filterValue("andreaPostdocSnrMin"),
    autocorrEfoldMin: filterValue("autocorrEfoldMin"), autocorrEfoldMax: filterValue("autocorrEfoldMax"),
    oasisEventSnrMin: filterValue("oasisEventSnrMin"),
    oasisRiseTauMin: filterValue("oasisRiseTauMin"), oasisRiseTauMax: filterValue("oasisRiseTauMax"),
    oasisDecayTauMin: filterValue("oasisDecayTauMin"), oasisDecayTauMax: filterValue("oasisDecayTauMax"),
    oasisResidualKsMax: filterValue("oasisResidualKsMax"),
    cellTypeFilter: filterString("cellTypeFilter"),
  }};
}}
function normalizeFilter(filter) {{
  if (filter.decayTauMin !== undefined && filter.autocorrEfoldMin === undefined) filter.autocorrEfoldMin = filter.decayTauMin;
  if (filter.decayTauMax !== undefined && filter.autocorrEfoldMax === undefined) filter.autocorrEfoldMax = filter.decayTauMax;
  const normalized = {{}};
  for (const key of ["skewMin","skewMax","maxConnect","aspectMin","aspectMax","footprintMin","footprintMax","compactMin","compactMax"]) {{
    if (filter[key] === null || filter[key] === undefined || String(filter[key]).trim() === "") {{
      normalized[key] = null;
      continue;
    }}
    const value = Number(filter[key]);
    normalized[key] = Number.isFinite(value) ? value : null;
  }}
  for (const key of ["eventSnrMin","eventSnrMax","andreaPostdocSnrMin","andreaPostdocSnrMax","roiAreaMin","roiAreaMax","autocorrEfoldMin","autocorrEfoldMax","oasisEventSnrMin","oasisRiseTauMin","oasisRiseTauMax","oasisDecayTauMin","oasisDecayTauMax","oasisResidualKsMax"]) {{
    if (filter[key] === null || filter[key] === undefined || String(filter[key]).trim() === "") {{
      normalized[key] = null;
      continue;
    }}
    const value = Number(filter[key]);
    normalized[key] = Number.isFinite(value) ? value : null;
    if ((key.endsWith("Max")) && normalized[key] === 0) normalized[key] = null;
  }}
  const cellTypeFilter = filter.cellTypeFilter === null || filter.cellTypeFilter === undefined ? "" : String(filter.cellTypeFilter).trim();
  normalized.cellTypeFilter = cellTypeFilter || null;
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
    const option = document.createElement("option"); option.value = `built-in:${{name}}`; option.textContent = targetStructureLabel(name); select.appendChild(option);
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
  if (preset) {{
    updateSuite2pSuggestedThresholds(preset);
    writeFilter(preset);
  }}
}}
function currentPresetName() {{
  return document.getElementById("presetName").value.trim();
}}
function saveCurrentPresetToPage() {{
  const name = currentPresetName();
  if (!name) {{ alert("Enter a custom filter name first."); return null; }}
  try {{
    customPresets[name] = normalizeFilter(readFilter());
  }} catch (error) {{
    alert(`Could not save QC thresholds: ${{error.message}}`);
    return null;
  }}
  populatePresetSelect(name);
  document.getElementById("filterPreset").value = `custom:${{name}}`;
  return name;
}}
function passesFilter(roi, metrics, filter) {{
  const dffMetrics = data.dffMetrics[roi];
  const snr9550 = dffMetric(roi, "snr_95_50", "event_snr");
  const postdocSnr = dffMetric(roi, "andrea_postdoc_snr");
  const oasisEventSnr = dffMetricValue(dffMetrics, "oasis_event_snr");
  const oasisRiseTau = dffMetricValue(dffMetrics, "oasis_rise_tau_seconds");
  const oasisDecayTau = dffMetricValue(dffMetrics, "oasis_decay_tau_seconds");
  const oasisResidualKs = dffMetricValue(dffMetrics, "oasis_event_residual_ks");
  const cellTypeFilter = filter.cellTypeFilter;
  const cellTypeValue = data.cellTypeLabels?.[roi];
  const cellTypePass = !cellTypeFilter || (
    cellTypeFilter === "not_loaded" ? (cellTypeValue === null || cellTypeValue === undefined) : Number(cellTypeValue) === Number(cellTypeFilter)
  );
  return (
    passesLower(metrics.footprint, filter.footprintMin) && passesUpper(metrics.footprint, filter.footprintMax) &&
    passesLower(metrics.skew, filter.skewMin) && passesUpper(metrics.skew, filter.skewMax) &&
    passesLower(metrics.aspect, filter.aspectMin) && passesUpper(metrics.aspect, filter.aspectMax) &&
    passesLower(metrics.compact, filter.compactMin) && passesUpper(metrics.compact, filter.compactMax) &&
    passesUpper(metrics.connect, filter.maxConnect) &&
    passesLower(dffMetrics.roi_area, filter.roiAreaMin) &&
    passesUpper(dffMetrics.roi_area, filter.roiAreaMax) &&
    passesLower(snr9550, filter.eventSnrMin) &&
    passesLower(postdocSnr, filter.andreaPostdocSnrMin) &&
    passesLower(dffMetricValue(dffMetrics, "autocorr_efold_time_seconds", "decay_tau_seconds"), filter.autocorrEfoldMin) &&
    passesUpper(dffMetricValue(dffMetrics, "autocorr_efold_time_seconds", "decay_tau_seconds"), filter.autocorrEfoldMax) &&
    passesLower(oasisEventSnr, filter.oasisEventSnrMin) &&
    passesLower(oasisRiseTau, filter.oasisRiseTauMin) &&
    passesUpper(oasisRiseTau, filter.oasisRiseTauMax) &&
    passesLower(oasisDecayTau, filter.oasisDecayTauMin) &&
    passesUpper(oasisDecayTau, filter.oasisDecayTauMax) &&
    passesUpper(oasisResidualKs, filter.oasisResidualKsMax) &&
    cellTypePass
  );
}}
function morphologyReasons(metrics, dffMetrics, filter, roi = null) {{
  const reasons = [];
  const snr9550 = dffMetrics.snr_95_50 ?? dffMetrics.event_snr;
  const postdocSnr = dffMetrics.andrea_postdoc_snr;
  const oasisEventSnr = dffMetricValue(dffMetrics, "oasis_event_snr");
  const oasisRiseTau = dffMetricValue(dffMetrics, "oasis_rise_tau_seconds");
  const oasisDecayTau = dffMetricValue(dffMetrics, "oasis_decay_tau_seconds");
  const oasisResidualKs = dffMetricValue(dffMetrics, "oasis_event_residual_ks");
  const cellTypeFilter = filter.cellTypeFilter;
  if (roi === null || roi === undefined) roi = data.dffMetrics.indexOf(dffMetrics);
  if (!passesLower(metrics.footprint, filter.footprintMin)) reasons.push(`footprint ${{fmt(metrics.footprint)}} below ${{filter.footprintMin}}`);
  if (!passesUpper(metrics.footprint, filter.footprintMax)) reasons.push(`footprint ${{fmt(metrics.footprint)}} above ${{filter.footprintMax}}`);
  if (!passesLower(metrics.skew, filter.skewMin)) reasons.push(`skew ${{fmt(metrics.skew)}} below ${{filter.skewMin}}`);
  if (!passesUpper(metrics.skew, filter.skewMax)) reasons.push(`skew ${{fmt(metrics.skew)}} above ${{filter.skewMax}}`);
  if (!passesLower(metrics.aspect, filter.aspectMin)) reasons.push(`aspect_ratio ${{fmt(metrics.aspect)}} below ${{filter.aspectMin}}`);
  if (!passesUpper(metrics.aspect, filter.aspectMax)) reasons.push(`aspect_ratio ${{fmt(metrics.aspect)}} above ${{filter.aspectMax}}`);
  if (!passesLower(metrics.compact, filter.compactMin)) reasons.push(`compact ${{fmt(metrics.compact)}} below ${{filter.compactMin}}`);
  if (!passesUpper(metrics.compact, filter.compactMax)) reasons.push(`compact ${{fmt(metrics.compact)}} above ${{filter.compactMax}}`);
  if (!passesUpper(metrics.connect, filter.maxConnect)) reasons.push(`connectivity ${{metrics.connect}} exceeds ${{filter.maxConnect}}`);
  if (!passesLower(dffMetrics.roi_area, filter.roiAreaMin)) reasons.push(`ROI area ${{fmt(dffMetrics.roi_area)}} below ${{filter.roiAreaMin}}`);
  if (!passesUpper(dffMetrics.roi_area, filter.roiAreaMax)) reasons.push(`ROI area ${{fmt(dffMetrics.roi_area)}} above ${{filter.roiAreaMax}}`);
  if (!passesLower(snr9550, filter.eventSnrMin)) reasons.push(`SNR: 95/50 percentile ${{fmt(snr9550)}} below ${{filter.eventSnrMin}}`);
  if (!passesLower(postdocSnr, filter.andreaPostdocSnrMin)) reasons.push(`SNR: CaImAn (large-transient score) ${{fmt(postdocSnr)}} below ${{filter.andreaPostdocSnrMin}}`);
  if (!passesLower(dffMetricValue(dffMetrics, "autocorr_efold_time_seconds", "decay_tau_seconds"), filter.autocorrEfoldMin)) reasons.push(`autocorrelation e-fold time ${{fmt(dffMetricValue(dffMetrics, "autocorr_efold_time_seconds", "decay_tau_seconds"))}} below ${{filter.autocorrEfoldMin}}`);
  if (!passesUpper(dffMetricValue(dffMetrics, "autocorr_efold_time_seconds", "decay_tau_seconds"), filter.autocorrEfoldMax)) reasons.push(`autocorrelation e-fold time ${{fmt(dffMetricValue(dffMetrics, "autocorr_efold_time_seconds", "decay_tau_seconds"))}} above ${{filter.autocorrEfoldMax}}`);
  if (!passesLower(oasisEventSnr, filter.oasisEventSnrMin)) reasons.push(`inferred spike SNR ${{fmt(oasisEventSnr)}} below ${{filter.oasisEventSnrMin}}`);
  if (!passesLower(oasisRiseTau, filter.oasisRiseTauMin)) reasons.push(`inferred spike rise tau ${{fmt(oasisRiseTau)}} below ${{filter.oasisRiseTauMin}}`);
  if (!passesUpper(oasisRiseTau, filter.oasisRiseTauMax)) reasons.push(`inferred spike rise tau ${{fmt(oasisRiseTau)}} above ${{filter.oasisRiseTauMax}}`);
  if (!passesLower(oasisDecayTau, filter.oasisDecayTauMin)) reasons.push(`inferred spike decay tau ${{fmt(oasisDecayTau)}} below ${{filter.oasisDecayTauMin}}`);
  if (!passesUpper(oasisDecayTau, filter.oasisDecayTauMax)) reasons.push(`inferred spike decay tau ${{fmt(oasisDecayTau)}} above ${{filter.oasisDecayTauMax}}`);
  if (!passesUpper(oasisResidualKs, filter.oasisResidualKsMax)) reasons.push(`inferred spike residual Gaussian-fit distance ${{fmt(oasisResidualKs)}} above ${{filter.oasisResidualKsMax}}`);
  if (cellTypeFilter) {{
    const cellTypeValue = data.cellTypeLabels?.[roi];
    const passesCellType = cellTypeFilter === "not_loaded" ? (cellTypeValue === null || cellTypeValue === undefined) : Number(cellTypeValue) === Number(cellTypeFilter);
    if (!passesCellType) reasons.push(`cell type ${{cellTypeLabelName(cellTypeValue)}} does not match ${{cellTypeFilter === "not_loaded" ? "not loaded" : cellTypeLabelName(Number(cellTypeFilter))}}`);
  }}
  return reasons;
}}
function evaluateFilter() {{
  const filter = readFilter();
  let pass = 0;
  for (let roi = 0; roi < data.nRois; roi++) {{
    filterPass[roi] = passesFilter(roi, data.morphology[roi], filter) ? 1 : 0;
    pass += filterPass[roi];
  }}
  const summary = `${{pass}} / ${{data.nRois}} original Suite2p ROIs pass the current QC metric filters.`;
  document.getElementById("filterSummary").textContent = summary;
  document.getElementById("filterSummaryInline").textContent = summary;
  drawMetricHistograms();
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
  updateVisibleRois(true);
  if (visibleRois.includes(current)) setSelected(current);
}}
function setLabel(label) {{
  labels[selected] = label;
  updateLabelControls();
  updateVisibleRois();
}}
function moveVisible(direction) {{
  if (!visibleRois.length) return;
  const currentIndex = visibleRois.indexOf(selected);
  const origin = currentIndex >= 0 ? currentIndex : 0;
  const nextIndex = (origin + direction + visibleRois.length) % visibleRois.length;
  setSelected(visibleRois[nextIndex]);
}}
function appendFunctionalRois(svg) {{
  data.rois.forEach(r => {{
    const hitPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    hitPath.setAttribute("d", r.hitPath || r.path);
    hitPath.dataset.roi = r.roi; hitPath.classList.add("roi-hit", "functional-overlay-item"); hitPath.addEventListener("click", () => setSelected(r.roi)); svg.appendChild(hitPath);
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", r.path);
    path.dataset.roi = r.roi; path.classList.add("roi", "functional-overlay-item"); svg.appendChild(path);
  }});
}}
function appendCellposeRois(svg) {{
  (data.cellposeRois || []).forEach(r => {{
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", r.path);
    path.dataset.cellposeRoi = r.label;
    path.classList.add("cellpose-roi", "cellpose-overlay-item");
    path.style.display = "none";
    svg.appendChild(path);
  }});
}}
function setRedOverlayMode(mode) {{
  const redOverlay = document.querySelector(".red-overlay");
  if (!redOverlay) return;
  const showCellpose = mode === "cellpose" && data.cellposeMaskAvailable;
  redOverlay.querySelectorAll(".functional-overlay-item").forEach(path => {{
    const roi = Number(path.dataset.roi);
    path.style.display = (!showCellpose && functionalRoiVisible(roi)) ? "" : "none";
  }});
  redOverlay.querySelectorAll(".cellpose-overlay-item").forEach(path => {{ path.style.display = showCellpose ? "" : "none"; }});
}}
function makeOverlays() {{
  document.querySelectorAll(".overlay").forEach(svg => {{
    appendFunctionalRois(svg);
    if (svg.classList.contains("red-overlay")) appendCellposeRois(svg);
  }});
  const redOverlayMode = document.getElementById("redOverlayMode");
  if (redOverlayMode) {{
    redOverlayMode.disabled = !data.cellposeMaskAvailable;
    redOverlayMode.title = data.cellposeMaskAvailable ? "" : "Cellpose anatomical masks were not available in the generated HTML.";
    redOverlayMode.addEventListener("change", event => setRedOverlayMode(event.target.value));
    setRedOverlayMode(redOverlayMode.value);
  }}
}}
const fovZoom = {{ scale: 1, x: 0, y: 0, dragging: false, lastX: 0, lastY: 0 }};
function applyFovTransform() {{
  const transform = `translate(${{fovZoom.x}}px, ${{fovZoom.y}}px) scale(${{fovZoom.scale}})`;
  document.querySelectorAll(".fov-review .imagewrap img, .fov-review .imagewrap svg").forEach(el => {{
    el.style.transform = transform;
  }});
}}
function clampFovPan() {{
  const wrap = document.querySelector(".fov-review .imagewrap");
  if (!wrap) return;
  const rect = wrap.getBoundingClientRect();
  const minX = rect.width * (1 - fovZoom.scale);
  const minY = rect.height * (1 - fovZoom.scale);
  fovZoom.x = Math.min(0, Math.max(minX, fovZoom.x));
  fovZoom.y = Math.min(0, Math.max(minY, fovZoom.y));
}}
function resetFovZoom() {{
  fovZoom.scale = 1; fovZoom.x = 0; fovZoom.y = 0; applyFovTransform();
}}
function setupFovZoom() {{
  document.querySelectorAll(".fov-review .imagewrap").forEach(wrap => {{
    wrap.addEventListener("wheel", event => {{
      event.preventDefault();
      const rect = wrap.getBoundingClientRect();
      const oldScale = fovZoom.scale;
      const nextScale = Math.min(8, Math.max(1, oldScale * (event.deltaY < 0 ? 1.18 : 1 / 1.18)));
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      fovZoom.x = px - (px - fovZoom.x) * (nextScale / oldScale);
      fovZoom.y = py - (py - fovZoom.y) * (nextScale / oldScale);
      fovZoom.scale = nextScale;
      clampFovPan();
      applyFovTransform();
    }}, {{passive: false}});
    wrap.addEventListener("mousedown", event => {{
      if (event.button !== 0) return;
      fovZoom.dragging = true; fovZoom.lastX = event.clientX; fovZoom.lastY = event.clientY;
      wrap.classList.add("panning");
    }});
    wrap.addEventListener("dblclick", resetFovZoom);
  }});
  window.addEventListener("mousemove", event => {{
    if (!fovZoom.dragging) return;
    fovZoom.x += event.clientX - fovZoom.lastX;
    fovZoom.y += event.clientY - fovZoom.lastY;
    fovZoom.lastX = event.clientX; fovZoom.lastY = event.clientY;
    clampFovPan();
    applyFovTransform();
  }});
  window.addEventListener("mouseup", () => {{
    if (!fovZoom.dragging) return;
    fovZoom.dragging = false;
    document.querySelectorAll(".fov-review .imagewrap").forEach(wrap => wrap.classList.remove("panning"));
  }});
}}
setupFovZoom();
function drawAxes(ctx, w, h, l, t, pw, ph, xLabel, yLabel, options = {{}}) {{
  ctx.strokeStyle = "#d0d5dd"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(l,t); ctx.lineTo(l,t+ph); ctx.lineTo(l+pw,t+ph); ctx.stroke();
  const xLabelY = options.xLabelY ?? h - 8;
  const yLabelX = options.yLabelX ?? 14;
  ctx.fillStyle = "#475467"; ctx.font = `${{12 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.fillText(xLabel, l + pw / 2, xLabelY);
  ctx.save(); ctx.translate(yLabelX, t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore();
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
function numericTickStep(span, target = 5) {{
  if (!Number.isFinite(span) || span <= 0) return 1;
  const raw = span / Math.max(1, target);
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const scaled = raw / pow;
  const mult = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return mult * pow;
}}
function numericLabel(value, step) {{
  if (!Number.isFinite(value)) return "";
  const decimals = Math.max(0, Math.min(4, Math.ceil(-Math.log10(Math.abs(step || 1))) + 1));
  return Math.abs(value) >= 1000 ? value.toExponential(1) : value.toFixed(decimals);
}}
function drawNumericTicks(ctx, l, t, pw, ph, xMin, xMax, yMin, yMax, xTicks = 5, yTicks = 4) {{
  ctx.save();
  ctx.strokeStyle = "#e5e7eb";
  ctx.fillStyle = "#475467";
  ctx.font = `${{10 * (window.devicePixelRatio || 1)}}px Arial`;
  ctx.lineWidth = 1;
  ctx.textBaseline = "top";
  ctx.textAlign = "center";
  const xStep = numericTickStep(xMax - xMin, xTicks);
  for (let value = Math.ceil(xMin / xStep) * xStep; value <= xMax + xStep * 0.5; value += xStep) {{
    const x = l + (value - xMin) / (xMax - xMin) * pw;
    ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke();
    ctx.fillText(numericLabel(value, xStep), x, t + ph + 8);
  }}
  const yStep = numericTickStep(yMax - yMin, yTicks);
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let value = Math.ceil(yMin / yStep) * yStep; value <= yMax + yStep * 0.5; value += yStep) {{
    const y = t + (1 - (value - yMin) / (yMax - yMin)) * ph;
    ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(l + pw, y); ctx.stroke();
    ctx.fillText(numericLabel(value, yStep), l - 8, y);
  }}
  ctx.restore();
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
  document.getElementById("stackTitle").textContent = `stacked raw dF/F, selected is ${{selectedPosition}}/${{visibleRois.length}} sorted by ${{sortMetricText()}} ${{appliedSortDirection}}`;
  const canvas = document.getElementById("stackCanvas"); fit(canvas); const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, l = 62, r = 16, t = 14, b = 56, pw = w-l-r, ph = h-t-b;
  ctx.clearRect(0,0,w,h); ctx.fillStyle = "#fff"; ctx.fillRect(0,0,w,h); drawAxes(ctx,w,h,l,t,pw,ph,"time (s)","Current Sort Order");
  if (!visibleRois.length) {{
    ctx.fillStyle = "#475467"; ctx.font = `${{14 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.fillText("No ROIs match the current metric and label display filters.", l + pw / 2, t + ph / 2);
    return;
  }}
  if (!dff) {{
    ctx.fillStyle = "#475467"; ctx.font = `${{14 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.fillText("Load the dF/F file to enable stacked traces.", l + pw / 2, t + ph / 2);
    return;
  }}
  const ys = Math.max(0, Math.floor(y0)), ye = Math.min(visibleRois.length - 1, Math.ceil(y1));
  const xs = Math.max(0, Math.floor(x0)), xe = Math.min(data.nFrames - 1, Math.ceil(x1));
  drawTimeGrid(ctx, l, t, pw, ph);
  const count = Math.max(1, ye - ys + 1), rowH = ph / (count + 1), pixelCount = Math.max(1, Math.floor(pw));
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
  const rowLabels = [];
  ctx.save(); ctx.beginPath(); ctx.rect(l, t, pw, ph); ctx.clip();
  for (let rowIndex = ys; rowIndex <= ye; rowIndex++) {{
    const roi = visibleRois[rowIndex], tr = trace(roi), row = rowIndex - ys, baseline = t + rowH * (row + 1), color = colorForRoi(roi);
    rowLabels.push({{roi, rowIndex, baseline, color}});
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
  }}
  ctx.restore();
  if (count <= 80) {{
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.font = `${{11 * (window.devicePixelRatio || 1)}}px Arial`;
    for (const rowLabel of rowLabels) {{
      ctx.fillStyle = rowLabel.roi === selected ? "#111827" : rowLabel.color;
      ctx.fillText(String(rowLabel.rowIndex + 1), l - 8, rowLabel.baseline);
    }}
  }}
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
  const sp = showInferredSpikes ? spikeTrace(selected) : null;
  if (sp) {{
    ctx.save();
    ctx.fillStyle = "rgba(220,38,38,.9)";
    const radius = Math.max(2.4, 2.4 * (window.devicePixelRatio || 1));
    for (let f=xs; f<=xe; f++) {{
      const s = sp[f];
      if (!Number.isFinite(s) || s <= oasisEventThreshold) continue;
      const x = xOf(f);
      const y = yOf(tr[f]);
      if (!Number.isFinite(y)) continue;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }}
    ctx.restore();
  }}
  ctx.fillStyle="#475467"; ctx.textAlign="right"; ctx.textBaseline="middle"; for (let i=0; i<=4; i++) {{ const v=lo+i/4*(hi-lo); ctx.fillText(v.toFixed(2), l-8, yOf(v)); }}
}}
function motionFiniteValues(values) {{
  if (!values) return [];
  const out = [];
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) out.push(values[i]);
  return out;
}}
function drawMotionDrift() {{
  const canvas = document.getElementById("motionDriftCanvas");
  fit(canvas);
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, l = 62, r = 18, t = 18, b = 56;
  const gap = 30 * (window.devicePixelRatio || 1);
  const panelH = (h - t - b - gap) / 2;
  const pw = w - l - r;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);
  if (!motionX || !motionY) {{
    ctx.fillStyle = "#475467"; ctx.font = `${{14 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center";
    ctx.fillText("Motion offsets not available for this session.", l + pw / 2, h / 2);
    return;
  }}
  const n = Math.min(motionX.length, motionY.length, data.nFrames);
  const xs = Math.max(0, Math.min(n - 1, Math.floor(x0)));
  const xe = Math.max(xs, Math.min(n - 1, Math.ceil(x1)));
  function drawPanel(values, top, title, color) {{
    const ph = panelH;
    let lo = Infinity, hi = -Infinity;
    for (let f = xs; f <= xe; f++) {{
      const value = values[f];
      if (Number.isFinite(value)) {{ lo = Math.min(lo, value); hi = Math.max(hi, value); }}
    }}
    if (!Number.isFinite(lo) || hi <= lo) {{ lo = -1; hi = 1; }}
    const pad = Math.max((hi - lo) * 0.08, 0.5); lo -= pad; hi += pad;
    const xOf = f => l + (f - x0) / (x1 - x0) * pw;
    const yOf = v => top + (1 - (v - lo) / (hi - lo)) * ph;
    drawAxes(ctx, w, h, l, top, pw, ph, "time (s)", "shift (px)");
    drawTimeGrid(ctx, l, top, pw, ph);
    const zeroY = yOf(0);
    if (zeroY >= top && zeroY <= top + ph) {{
      ctx.strokeStyle = "#9ca3af"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(l, zeroY); ctx.lineTo(l + pw, zeroY); ctx.stroke();
    }}
    ctx.save();
    ctx.beginPath(); ctx.rect(l, top, pw, ph); ctx.clip();
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1, window.devicePixelRatio || 1);
    ctx.beginPath();
    let first = true;
    const pixelCount = Math.max(1, Math.floor(pw));
    const framesPerPixel = (xe - xs + 1) / pixelCount;
    if (framesPerPixel <= 1.2) {{
      for (let f = xs; f <= xe; f++) {{
        const value = values[f];
        if (!Number.isFinite(value)) continue;
        const x = xOf(f), y = yOf(value);
        if (first) {{ ctx.moveTo(x, y); first = false; }} else ctx.lineTo(x, y);
      }}
    }} else {{
      const meanPoints = [];
      for (let px = 0; px < pixelCount; px++) {{
        const f0 = Math.max(xs, Math.floor(xs + px * framesPerPixel));
        const f1 = Math.min(xe, Math.floor(xs + (px + 1) * framesPerPixel));
        let minV = Infinity, maxV = -Infinity, sumV = 0, nV = 0;
        for (let f = f0; f <= f1; f++) {{
          const value = values[f];
          if (Number.isFinite(value)) {{ minV = Math.min(minV, value); maxV = Math.max(maxV, value); sumV += value; nV++; }}
        }}
        if (!Number.isFinite(minV)) continue;
        const x = l + px;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.18;
        ctx.beginPath();
        ctx.moveTo(x, yOf(minV)); ctx.lineTo(x, yOf(maxV));
        ctx.stroke();
        ctx.restore();
        meanPoints.push([x, yOf(sumV / nV)]);
      }}
      ctx.beginPath();
      for (let i = 0; i < meanPoints.length; i++) {{
        const [x, y] = meanPoints[i];
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
    }}
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = color; ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.font = `${{12 * (window.devicePixelRatio || 1)}}px Arial`;
    ctx.fillText(title, l + 8, top + 8);
    ctx.fillStyle = "#475467"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let i = 0; i <= 3; i++) {{
      const value = lo + i / 3 * (hi - lo);
      ctx.fillText(value.toFixed(1), l - 8, yOf(value));
    }}
  }}
  drawPanel(motionX, t, "X shift per frame", "#2563eb");
  drawPanel(motionY, t + panelH + gap, "Y shift per frame", "#dc2626");
}}
function drawMotionDistribution() {{
  const canvas = document.getElementById("motionDistributionCanvas");
  fit(canvas);
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, l = 74, r = 18, t = 30, b = 74, pw = w - l - r, ph = h - t - b;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);
  const xValues = motionFiniteValues(motionX);
  const yValues = motionFiniteValues(motionY);
  if (!xValues.length || !yValues.length) {{
    ctx.fillStyle = "#475467"; ctx.font = `${{14 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center";
    ctx.fillText("Motion offsets not available for this session.", l + pw / 2, t + ph / 2);
    return;
  }}
  const gap = 46 * (window.devicePixelRatio || 1);
  const panelW = (pw - 2 * gap) / 3;
  const absXValues = xValues.map(Math.abs);
  const absYValues = yValues.map(Math.abs);
  function drawPanelAxes(left, width, xLabel, yLabel) {{
    ctx.strokeStyle = "#d0d5dd";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, t);
    ctx.lineTo(left, t + ph);
    ctx.lineTo(left + width, t + ph);
    ctx.stroke();
    ctx.fillStyle = "#475467";
    ctx.font = `${{11 * (window.devicePixelRatio || 1)}}px Arial`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(xLabel, left + width / 2, h - 26);
    ctx.save();
    ctx.translate(left - 41, t + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }}
  function panelLimits(valuesA, valuesB, absolute = false) {{
    const all = valuesA.concat(valuesB).filter(Number.isFinite).sort((a, b) => a - b);
    if (!all.length) return absolute ? [0, 1] : [-1, 1];
    let lo = absolute ? 0 : all[0];
    let hi = all[all.length - 1];
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return absolute ? [0, 1] : [-1, 1];
    if (hi <= lo) hi = lo + 1;
    const pad = Math.max((hi - lo) * 0.08, 0.5);
    if (absolute) {{
      hi += pad;
    }} else {{
      lo -= pad;
      hi += pad;
    }}
    return [lo, hi];
  }}
  function histogram(values, lo, hi, bins) {{
    const counts = new Array(bins).fill(0);
    const step = (hi - lo) / bins;
    for (const value of values) {{
      if (!Number.isFinite(value) || value < lo || value > hi) continue;
      const index = Math.min(bins - 1, Math.max(0, Math.floor((value - lo) / step)));
      counts[index] += 1;
    }}
    const total = counts.reduce((sum, count) => sum + count, 0);
    return counts.map(count => total ? count / total : 0);
  }}
  function drawHistogramPanel(left, width, title, valuesA, valuesB, labelA, labelB) {{
    const [lo, hi] = panelLimits(valuesA, valuesB, false);
    const bins = 64;
    const histA = histogram(valuesA, lo, hi, bins);
    const histB = histogram(valuesB, lo, hi, bins);
    const yMax = Math.max(...histA, ...histB, 1e-9) * 1.12;
    const xOf = value => left + (value - lo) / (hi - lo) * width;
    const yOf = value => t + (1 - value / yMax) * ph;
    drawPanelAxes(left, width, "shift (px)", "fraction / bin");
    drawNumericTicks(ctx, left, t, width, ph, lo, hi, 0, yMax, 7, 4);
    ctx.fillStyle = "#475467"; ctx.font = `${{11 * (window.devicePixelRatio || 1)}}px Arial`; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(title, left + width / 2, 5);
    function drawBars(hist, color, label, yLabelOffset, offsetFraction) {{
      const binWidth = width / bins;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.34;
      for (let i = 0; i < hist.length; i++) {{
        const x0b = xOf(lo + i / bins * (hi - lo));
        const y = yOf(hist[i]);
        ctx.fillRect(x0b + offsetFraction * binWidth, y, binWidth * 0.48, t + ph - y);
      }}
      ctx.globalAlpha = 1;
      ctx.fillStyle = color; ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(label, left + 8, t + yLabelOffset);
    }}
    drawBars(histA, "#2563eb", labelA, 24, 0.02);
    drawBars(histB, "#dc2626", labelB, 42, 0.50);
  }}
  function drawCdfPanel(left, width, title, valuesA, valuesB, labelA, labelB, absolute = false) {{
    const [lo, hi] = panelLimits(valuesA, valuesB, absolute);
    const xOf = value => left + (value - lo) / (hi - lo) * width;
    const yOf = value => t + (1 - value) * ph;
    drawPanelAxes(left, width, "shift (px)", "cumulative fraction");
    drawNumericTicks(ctx, left, t, width, ph, lo, hi, 0, 1, 7, 4);
    ctx.fillStyle = "#475467"; ctx.font = `${{11 * (window.devicePixelRatio || 1)}}px Arial`;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(title, left + width / 2, 5);
    function drawCdf(values, color, label, yLabelOffset) {{
      const sorted = values.slice().filter(value => Number.isFinite(value) && value >= lo && value <= hi).sort((a, b) => a - b);
      if (!sorted.length) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(1.4, 1.4 * (window.devicePixelRatio || 1));
      ctx.beginPath();
      for (let i = 0; i < sorted.length; i++) {{
        const fraction = sorted.length === 1 ? 1 : i / (sorted.length - 1);
        const x = xOf(sorted[i]), y = yOf(fraction);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
      ctx.stroke();
      ctx.fillStyle = color; ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(label, left + 8, t + yLabelOffset);
    }}
    drawCdf(valuesA, "#2563eb", labelA, 24);
    drawCdf(valuesB, "#dc2626", labelB, 42);
  }}
  drawHistogramPanel(l, panelW, "Offset distribution", xValues, yValues, "x shift", "y shift");
  drawCdfPanel(l + panelW + gap, panelW, "Signed shift CDF", xValues, yValues, "x shift", "y shift", false);
  drawCdfPanel(l + 2 * (panelW + gap), panelW, "Absolute shift CDF", absXValues, absYValues, "|x shift|", "|y shift|", true);
}}
function gaussianPdf(x, mean, sd) {{
  if (!Number.isFinite(sd) || sd <= 0) return 0;
  const z = (x - mean) / sd;
  return Math.exp(-0.5 * z * z) / (sd * Math.sqrt(2 * Math.PI));
}}
function erfApprox(x) {{
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}}
function normalCdf(x, mean, sd) {{
  if (!Number.isFinite(sd) || sd <= 0) return NaN;
  return 0.5 * (1 + erfApprox((x - mean) / (sd * Math.SQRT2)));
}}
function diagnosticMean(values) {{
  let sum = 0, n = 0;
  for (const value of values) if (Number.isFinite(value)) {{ sum += value; n++; }}
  return n ? sum / n : NaN;
}}
function diagnosticSd(values, center) {{
  let sum = 0, n = 0;
  for (const value of values) if (Number.isFinite(value)) {{ const d = value - center; sum += d * d; n++; }}
  return n > 1 ? Math.sqrt(sum / (n - 1)) : NaN;
}}
function diagnosticMedian(values) {{
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return NaN;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}}
function oasisResidualSummary(roi, threshold) {{
  const key = `${{roi}}:${{Number(threshold).toPrecision(8)}}`;
  if (oasisResidualCache.has(key)) return oasisResidualCache.get(key);
  const tr = trace(roi), sp = spikeTrace(roi);
  if (!tr || !sp) return null;
  const eventFrames = new Uint8Array(data.nFrames);
  let eventCount = 0;
  for (let f=0; f<data.nFrames; f++) {{
    if (Number.isFinite(sp[f]) && sp[f] > threshold) {{
      eventCount++;
      const start = Math.max(0, f - 3), stop = Math.min(data.nFrames - 1, f + 3);
      for (let g=start; g<=stop; g++) eventFrames[g] = 1;
    }}
  }}
  const eventResiduals = [];
  const radius = Math.max(2, Math.round(data.frameRate * 0.25));
  for (let f=0; f<data.nFrames; f++) {{
    if (!eventFrames[f]) continue;
    let sum = 0, n = 0;
    const start = Math.max(0, f - radius), stop = Math.min(data.nFrames - 1, f + radius);
    for (let g=start; g<=stop; g++) {{ const v=tr[g]; if (Number.isFinite(v)) {{ sum += v; n++; }} }}
    if (!n || !Number.isFinite(tr[f])) continue;
    eventResiduals.push(tr[f] - sum / n);
  }}
  const summary = {{eventCount, eventResiduals}};
  oasisResidualCache.set(key, summary);
  return summary;
}}
function selectedOasisDiagnostics(roi, threshold) {{
  const key = `${{roi}}:${{Number(threshold).toPrecision(8)}}`;
  if (oasisDiagnosticsCache.has(key)) return oasisDiagnosticsCache.get(key);
  const tr = trace(roi), sp = spikeTrace(roi);
  if (!tr || !sp) return null;
  const preFrames = Math.max(3, Math.round(data.frameRate * 0.5));
  const postFrames = Math.max(6, Math.round(data.frameRate * 2.0));
  const minSeparation = Math.max(1, Math.round(data.frameRate * 0.25));
  const events = [];
  let last = -Infinity;
  for (let f = preFrames; f < data.nFrames - postFrames; f++) {{
    if (!Number.isFinite(sp[f]) || sp[f] <= threshold) continue;
    if (f - last < minSeparation) continue;
    events.push(f);
    last = f;
  }}
  if (!events.length) {{
    const empty = {{events, waveform: [], times: [], expModel: [], expCdf: [], obsCdf: [], residuals: [], gaussianCdf: [], residualCdf: [], expKs: NaN, gaussianKs: NaN, tau: NaN}};
    oasisDiagnosticsCache.set(key, empty);
    return empty;
  }}
  const len = preFrames + postFrames + 1;
  const sums = new Float64Array(len);
  const counts = new Uint32Array(len);
  for (const eventFrame of events) {{
    const baselineValues = [];
    for (let f = eventFrame - preFrames; f < eventFrame; f++) {{
      const value = tr[f];
      if (Number.isFinite(value)) baselineValues.push(value);
    }}
    const baseline = Number.isFinite(diagnosticMedian(baselineValues)) ? diagnosticMedian(baselineValues) : 0;
    for (let offset = -preFrames; offset <= postFrames; offset++) {{
      const value = tr[eventFrame + offset];
      if (!Number.isFinite(value)) continue;
      const index = offset + preFrames;
      sums[index] += value - baseline;
      counts[index] += 1;
    }}
  }}
  const waveform = [];
  const times = [];
  for (let i = 0; i < len; i++) {{
    waveform.push(counts[i] ? sums[i] / counts[i] : NaN);
    times.push((i - preFrames) / data.frameRate);
  }}
  let peakIndex = preFrames;
  let peak = -Infinity;
  for (let i = preFrames; i < waveform.length; i++) {{
    const value = waveform[i];
    if (Number.isFinite(value) && value > peak) {{ peak = value; peakIndex = i; }}
  }}
  const baselineTail = diagnosticMedian(waveform.slice(Math.max(0, waveform.length - Math.max(3, Math.round(data.frameRate * 0.25)))));
  const amplitude = peak - (Number.isFinite(baselineTail) ? baselineTail : 0);
  let tau = NaN;
  if (Number.isFinite(amplitude) && amplitude > 0) {{
    const target = (Number.isFinite(baselineTail) ? baselineTail : 0) + amplitude / Math.E;
    for (let i = peakIndex; i < waveform.length; i++) {{
      const value = waveform[i];
      if (!Number.isFinite(value)) continue;
      if (value <= target) {{
        tau = Math.max(0, times[i] - times[peakIndex]);
        break;
      }}
    }}
  }}
  const expModel = waveform.map((_, i) => {{
    if (!Number.isFinite(amplitude) || amplitude <= 0 || i < peakIndex) return NaN;
    if (!Number.isFinite(tau) || tau <= 0) return NaN;
    return (Number.isFinite(baselineTail) ? baselineTail : 0) + amplitude * Math.exp(-(times[i] - times[peakIndex]) / tau);
  }});
  const obsCdf = [];
  const expCdf = [];
  let expKs = NaN;
  if (Number.isFinite(amplitude) && amplitude > 0 && Number.isFinite(tau) && tau > 0) {{
    let maxDiff = 0;
    for (let i = peakIndex; i < waveform.length; i++) {{
      const observed = Math.min(1, Math.max(0, (peak - waveform[i]) / amplitude));
      const model = Math.min(1, Math.max(0, 1 - Math.exp(-(times[i] - times[peakIndex]) / tau)));
      obsCdf.push([times[i] - times[peakIndex], observed]);
      expCdf.push([times[i] - times[peakIndex], model]);
      if (Number.isFinite(observed) && Number.isFinite(model)) maxDiff = Math.max(maxDiff, Math.abs(observed - model));
    }}
    expKs = maxDiff;
  }}
  const residualSummary = oasisResidualSummary(roi, threshold);
  const residuals = residualSummary ? residualSummary.eventResiduals.filter(Number.isFinite).sort((a, b) => a - b) : [];
  const residualMean = diagnosticMean(residuals);
  const residualSd = diagnosticSd(residuals, residualMean);
  const residualCdf = [];
  const gaussianCdf = [];
  let gaussianKs = NaN;
  if (residuals.length >= 4 && Number.isFinite(residualSd) && residualSd > 0) {{
    let maxDiff = 0;
    for (let i = 0; i < residuals.length; i++) {{
      const empirical = residuals.length === 1 ? 1 : i / (residuals.length - 1);
      const model = normalCdf(residuals[i], residualMean, residualSd);
      residualCdf.push([residuals[i], empirical]);
      gaussianCdf.push([residuals[i], model]);
      if (Number.isFinite(model)) maxDiff = Math.max(maxDiff, Math.abs(empirical - model));
    }}
    gaussianKs = maxDiff;
  }}
  const diagnostics = {{events, waveform, times, expModel, expCdf, obsCdf, residuals, gaussianCdf, residualCdf, expKs, gaussianKs, tau}};
  oasisDiagnosticsCache.set(key, diagnostics);
  return diagnostics;
}}
function drawLineSeries(ctx, series, xOf, yOf, color, lineWidth = 1.5) {{
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(lineWidth, lineWidth * (window.devicePixelRatio || 1));
  ctx.beginPath();
  let first = true;
  for (const point of series) {{
    const x = Array.isArray(point) ? point[0] : point.x;
    const y = Array.isArray(point) ? point[1] : point.y;
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (first) {{ ctx.moveTo(xOf(x), yOf(y)); first = false; }} else ctx.lineTo(xOf(x), yOf(y));
  }}
  ctx.stroke();
}}
function drawOasisDiagnostics() {{
  const summary = document.getElementById("oasisDiagnosticsSummary");
  const transientCanvas = document.getElementById("oasisTransientCanvas");
  const gaussianCanvas = document.getElementById("oasisGaussianCanvas");
  if (!summary || !transientCanvas || !gaussianCanvas) return;
  fit(transientCanvas); fit(gaussianCanvas);
  const transientCtx = transientCanvas.getContext("2d");
  const gaussianCtx = gaussianCanvas.getContext("2d");
  transientCtx.clearRect(0, 0, transientCanvas.width, transientCanvas.height);
  transientCtx.fillStyle = "#fff"; transientCtx.fillRect(0, 0, transientCanvas.width, transientCanvas.height);
  gaussianCtx.clearRect(0, 0, gaussianCanvas.width, gaussianCanvas.height);
  gaussianCtx.fillStyle = "#fff"; gaussianCtx.fillRect(0, 0, gaussianCanvas.width, gaussianCanvas.height);
  if (!data.oasisAvailable || !dff || !oasisSpikes) {{
    summary.textContent = "Inferred spikes and dF/F must be loaded to show selected ROI fit metrics.";
    return;
  }}
  const diagnostics = selectedOasisDiagnostics(selected, oasisEventThreshold);
  if (!diagnostics || !diagnostics.events.length) {{
    summary.textContent = `No inferred spikes pass amplitude threshold ${{fmt(oasisEventThreshold)}} for this ROI.`;
    return;
  }}
  summary.textContent = `${{diagnostics.events.length}} inferred spikes at amplitude threshold ${{fmt(oasisEventThreshold)}} | empirical e-fold tau ${{fmt(diagnostics.tau)}} s | Gaussian residual KS ${{fmt(diagnostics.gaussianKs)}}`;

  let w = transientCanvas.width, h = transientCanvas.height, l = 64, r = 14, t = 34, b = 50, pw = w - l - r, ph = h - t - b;
  const finiteWave = diagnostics.waveform.filter(Number.isFinite);
  const finiteModel = diagnostics.expModel.filter(Number.isFinite);
  let xMin = diagnostics.times[0], xMax = diagnostics.times[diagnostics.times.length - 1];
  let yMin = Math.min(...finiteWave, ...finiteModel), yMax = Math.max(...finiteWave, ...finiteModel);
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMax <= yMin) {{ yMin = -1; yMax = 1; }}
  const pad = (yMax - yMin) * 0.12 || 1; yMin -= pad; yMax += pad;
  const xOf = x => l + (x - xMin) / (xMax - xMin) * pw;
  const yOf = y => t + (1 - (y - yMin) / (yMax - yMin)) * ph;
  drawAxes(transientCtx, w, h, l, t, pw, ph, "time from inferred spike (s)", "dF/F", {{ xLabelY: h - 6, yLabelX: 12 }});
  drawNumericTicks(transientCtx, l, t, pw, ph, xMin, xMax, yMin, yMax, 6, 5);
  transientCtx.fillStyle = "#475467"; transientCtx.textAlign = "left"; transientCtx.textBaseline = "top"; transientCtx.font = `${{11 * (window.devicePixelRatio || 1)}}px Arial`;
  transientCtx.fillText("Average transient vs exponential decay model", l + 4, 5);
  drawLineSeries(transientCtx, diagnostics.times.map((time, i) => [time, diagnostics.waveform[i]]), xOf, yOf, "#1d4ed8", 1.6);
  drawLineSeries(transientCtx, diagnostics.times.map((time, i) => [time, diagnostics.expModel[i]]), xOf, yOf, "#dc2626", 1.4);
  transientCtx.fillStyle = "#1d4ed8"; transientCtx.fillText("avg transient", l + 12, t - 18);
  transientCtx.fillStyle = "#dc2626"; transientCtx.fillText(`exp tau=${{fmt(diagnostics.tau)}}s`, l + 12, t - 4);

  w = gaussianCanvas.width; h = gaussianCanvas.height; l = 82; r = 18; t = 30; b = 56; pw = w - l - r; ph = h - t - b;
  gaussianCtx.fillStyle = "#475467"; gaussianCtx.textAlign = "left"; gaussianCtx.textBaseline = "top"; gaussianCtx.font = `${{11 * (window.devicePixelRatio || 1)}}px Arial`;
  gaussianCtx.fillText("Event-window residual CDF vs fitted Gaussian", l + 4, 5);
  if (diagnostics.residualCdf.length && diagnostics.gaussianCdf.length) {{
    const xs = diagnostics.residuals;
    let lo = xs[0], hi = xs[xs.length - 1];
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) {{ lo = -1; hi = 1; }}
    const pad = Math.max((hi - lo) * 0.08, 0.01);
    lo -= pad; hi += pad;
    const xOfResidual = x => l + (x - lo) / (hi - lo) * pw;
    const yOfCdf = y => t + (1 - y) * ph;
    drawAxes(gaussianCtx, w, h, l, t, pw, ph, "residual dF/F", "cumulative fraction", {{ xLabelY: h - 18, yLabelX: 16 }});
    drawNumericTicks(gaussianCtx, l, t, pw, ph, lo, hi, 0, 1, 6, 4);
    drawLineSeries(gaussianCtx, diagnostics.residualCdf, xOfResidual, yOfCdf, "#059669", 1.4);
    drawLineSeries(gaussianCtx, diagnostics.gaussianCdf, xOfResidual, yOfCdf, "#7c3aed", 1.4);
    gaussianCtx.fillStyle = "#059669"; gaussianCtx.fillText("residuals", l + 28, t + 8);
    gaussianCtx.fillStyle = "#7c3aed"; gaussianCtx.fillText(`Gaussian fit, KS=${{fmt(diagnostics.gaussianKs)}}`, l + 28, t + 24);
  }} else {{
    gaussianCtx.fillStyle = "#475467"; gaussianCtx.textAlign = "center"; gaussianCtx.textBaseline = "middle";
    gaussianCtx.fillText("Not enough residual samples for Gaussian KS CDF.", l + pw / 2, t + ph / 2);
  }}
}}
function drawSelectedTraceAndOasis() {{ drawTrace(); drawOasisDiagnostics(); }}
function draw() {{ drawSelectedTraceAndOasis(); drawStack(); drawMotionDrift(); drawMotionDistribution(); }}
function reset() {{ x0=0; x1=data.nFrames-1; y0=0; y1=Math.min(19, visibleRois.length-1); document.getElementById("yStart").value=0; document.getElementById("yEnd").value=y1; syncTimeInputs(); draw(); }}
function showAllVisibleTraces() {{
  y0 = 0;
  y1 = Math.max(0, visibleRois.length - 1);
  document.getElementById("yStart").value = y0;
  document.getElementById("yEnd").value = y1;
  draw();
}}
function closeDialog(dialog) {{
  if (typeof dialog.close === "function") dialog.close();
  else dialog.removeAttribute("open");
}}
document.getElementById("roiInput").addEventListener("change", e => setSelected(roiFromSuite2pIndex(e.target.value)));
document.getElementById("timeStart").addEventListener("change", () => setTimeWindow(document.getElementById("timeStart").value, document.getElementById("timeEnd").value));
document.getElementById("timeEnd").addEventListener("change", () => setTimeWindow(document.getElementById("timeStart").value, document.getElementById("timeEnd").value));
document.getElementById("yStart").addEventListener("change", e => {{ y0=Number(e.target.value); draw(); }});
document.getElementById("yEnd").addEventListener("change", e => {{ y1=Number(e.target.value); draw(); }});
document.getElementById("showAllVisibleTraces").addEventListener("click", showAllVisibleTraces);
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
  closeDialog(labelAllDialog);
}});
document.getElementById("closeLabelAllDialogTop").addEventListener("click", () => closeDialog(labelAllDialog));
document.getElementById("applyLabelAll").addEventListener("click", () => {{
  const label = Number(document.getElementById("labelAllValue").value);
  for (const roi of visibleRois) {{
    labels[roi] = label;
  }}
  updateLabelControls();
  updateVisibleRois();
  closeDialog(labelAllDialog);
}});
document.getElementById("previousRoi").addEventListener("click", () => moveVisible(-1));
document.getElementById("nextRoi").addEventListener("click", () => moveVisible(1));
document.querySelectorAll(".roi-display-checkbox").forEach(input => {{
  input.addEventListener("change", () => {{
    updateRoiDisplaySummary();
    updateVisibleRois(true);
  }});
}});
document.getElementById("applySort").addEventListener("click", applySort);
document.querySelectorAll('input[name="sortMetric"]').forEach(input => {{
  input.addEventListener("change", event => {{
    const original = document.querySelector('input[name="sortMetric"][value="original"]');
    const metrics = Array.from(document.querySelectorAll('input[name="sortMetric"]'));
    if (event.target.value === "original" && event.target.checked) {{
      metrics.forEach(metric => {{ if (metric.value !== "original") metric.checked = false; }});
    }} else if (event.target.checked && original) {{
      original.checked = false;
    }}
    if (!metrics.some(metric => metric.checked) && original) original.checked = true;
  }});
}});
const sortDialog = document.getElementById("sortDialog");
document.getElementById("openSortDialog").addEventListener("click", () => {{
  if (typeof sortDialog.showModal === "function") sortDialog.showModal();
  else sortDialog.setAttribute("open", "");
}});
document.getElementById("closeSortDialog").addEventListener("click", () => {{
  closeDialog(sortDialog);
}});
document.getElementById("closeSortDialogTop").addEventListener("click", () => closeDialog(sortDialog));
const dffFileInput = document.getElementById("dffFile");
const oasisFileInput = document.getElementById("oasisFile");
const cellTypeFileInput = document.getElementById("cellTypeFile");
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
document.getElementById("loadOasisFile").addEventListener("click", () => oasisFileInput.click());
oasisFileInput.addEventListener("change", () => {{
  const file = oasisFileInput.files && oasisFileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {{
    try {{
      setOasisFromArrayBuffer(reader.result);
    }} catch (error) {{
      document.getElementById("traceLoadNote").textContent = `Could not load inferred spikes file: ${{error.message}}`;
    }}
  }};
  reader.readAsArrayBuffer(file);
}});
document.getElementById("loadCellTypeFile").addEventListener("click", () => cellTypeFileInput.click());
cellTypeFileInput.addEventListener("change", () => {{
  const file = cellTypeFileInput.files && cellTypeFileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {{
    try {{
      loadCellTypeTableText(String(reader.result || ""), file.name);
    }} catch (error) {{
      alert(`Could not load cell-type file: ${{error.message}}`);
    }}
  }};
  reader.readAsText(file);
}});
document.getElementById("showInferredSpikes").addEventListener("change", event => {{
  showInferredSpikes = event.target.checked;
  draw();
}});
document.getElementById("oasisThresholdSlider").addEventListener("input", event => {{
  setOasisThreshold(Number(event.target.value));
}});
document.getElementById("oasisThreshold").addEventListener("input", event => {{
  setOasisThreshold(Number(event.target.value));
}});
document.getElementById("resetOasisThreshold").addEventListener("click", () => {{
  setOasisThreshold(selectedOasisDefaultThreshold());
}});
function csvEscape(value) {{
  const text = String(value ?? "");
  return /[",\\n]/.test(text) ? `"${{text.replaceAll('"', '""')}}"` : text;
}}
function metricSpreadsheetRows() {{
  const filter = readFilter();
  return data.morphology.map((metrics, roi) => {{
    const dffMetrics = data.dffMetrics[roi];
    const snr9550 = dffMetric(roi, "snr_95_50", "event_snr");
    const postdocSnr = dffMetric(roi, "andrea_postdoc_snr");
    const autocorrEfold = dffMetricValue(dffMetrics, "autocorr_efold_time_seconds", "decay_tau_seconds");
    const oasisEventSnr = dffMetricValue(dffMetrics, "oasis_event_snr");
    const oasisRiseTau = dffMetricValue(dffMetrics, "oasis_rise_tau_seconds");
    const oasisDecayTau = dffMetricValue(dffMetrics, "oasis_decay_tau_seconds");
    const oasisResidualKs = dffMetricValue(dffMetrics, "oasis_event_residual_ks");
    const cellTypeCode = data.cellTypeLabels?.[roi] ?? null;
    const reasons = morphologyReasons(metrics, dffMetrics, filter, roi);
    if (labels[roi] === 0) reasons.push("manual/current label: bad");
    else if (labels[roi] === 2) reasons.push("manual/current label: unsure");
    else if (labels[roi] === -1 && reasons.length === 0) reasons.push("not labeled");
    return {{
      suite2p: data.suite2pIndices[roi],
      label: labelName(labels[roi]),
      cellTypeLabel: cellTypeLabelName(cellTypeCode),
      cellTypeCode,
      footprint: metrics.footprint,
      skew: metrics.skew,
      aspect: metrics.aspect,
      compact: metrics.compact,
      connectivity: metrics.connect,
      roiArea: dffMetrics.roi_area,
      snr9550,
      postdocSnr,
      autocorrEfold,
      oasisEventSnr,
      oasisRiseTau,
      oasisDecayTau,
      oasisResidualKs,
      fail: {{
        footprint: !(passesLower(metrics.footprint, filter.footprintMin) && passesUpper(metrics.footprint, filter.footprintMax)),
        skew: !(passesLower(metrics.skew, filter.skewMin) && passesUpper(metrics.skew, filter.skewMax)),
        aspect: !(passesLower(metrics.aspect, filter.aspectMin) && passesUpper(metrics.aspect, filter.aspectMax)),
        compact: !(passesLower(metrics.compact, filter.compactMin) && passesUpper(metrics.compact, filter.compactMax)),
        connectivity: !passesUpper(metrics.connect, filter.maxConnect),
        roiArea: !(passesLower(dffMetrics.roi_area, filter.roiAreaMin) && passesUpper(dffMetrics.roi_area, filter.roiAreaMax)),
        snr9550: !passesLower(snr9550, filter.eventSnrMin),
        postdocSnr: !passesLower(postdocSnr, filter.andreaPostdocSnrMin),
        autocorrEfold: !(passesLower(autocorrEfold, filter.autocorrEfoldMin) && passesUpper(autocorrEfold, filter.autocorrEfoldMax)),
        oasisEventSnr: !passesLower(oasisEventSnr, filter.oasisEventSnrMin),
        oasisRiseTau: !(passesLower(oasisRiseTau, filter.oasisRiseTauMin) && passesUpper(oasisRiseTau, filter.oasisRiseTauMax)),
        oasisDecayTau: !(passesLower(oasisDecayTau, filter.oasisDecayTauMin) && passesUpper(oasisDecayTau, filter.oasisDecayTauMax)),
        oasisResidualKs: !passesUpper(oasisResidualKs, filter.oasisResidualKsMax),
      }},
      reason: reasons.join("; ") || "included",
    }};
  }});
}}
function metricSpreadsheetCsv(rowsData = metricSpreadsheetRows()) {{
  const csvHeader = ["suite2p_index","manual_label","cell_type_label","cell_type_code","footprint","skew","aspect_ratio","compact","connectivity","roi_area_px","snr_95_50","caiman_exceptional_event_snr","autocorr_efold_time_seconds","inferred_spike_snr","inferred_spike_rise_tau_seconds","inferred_spike_decay_tau_seconds","inferred_spike_residual_gaussian_ks","reason"];
  const csvRows = rowsData.map(row => [
    row.suite2p, row.label, row.cellTypeLabel, row.cellTypeCode ?? "", fmt(row.footprint), fmt(row.skew), fmt(row.aspect), fmt(row.compact),
    row.connectivity, fmt(row.roiArea), fmt(row.snr9550), fmt(row.postdocSnr), fmt(row.autocorrEfold),
    fmt(row.oasisEventSnr), fmt(row.oasisRiseTau), fmt(row.oasisDecayTau), fmt(row.oasisResidualKs), row.reason,
  ].map(csvEscape).join(","));
  return [csvHeader.join(","), ...csvRows].join("\\n") + "\\n";
}}
function openMetricSpreadsheet() {{
  function td(value, failed = false) {{
    return `<td${{failed ? ' class="metric-fail"' : ""}}>${{value}}</td>`;
  }}
  function labelTd(label) {{
    const cls = label === "good" ? "label-good" : label === "bad" ? "label-bad" : label === "unsure" ? "label-unsure" : "";
    return `<td${{cls ? ` class="${{cls}}"` : ""}}>${{label}}</td>`;
  }}
  const rowsData = metricSpreadsheetRows();
  const rows = rowsData.map(row => `<tr>${{
    td(row.suite2p) +
    labelTd(row.label) +
    td(row.cellTypeLabel) +
    td(row.cellTypeCode ?? "") +
    td(fmt(row.footprint), row.fail.footprint) +
    td(fmt(row.skew), row.fail.skew) +
    td(fmt(row.aspect), row.fail.aspect) +
    td(fmt(row.compact), row.fail.compact) +
    td(row.connectivity, row.fail.connectivity) +
    td(fmt(row.roiArea), row.fail.roiArea) +
    td(fmt(row.snr9550), row.fail.snr9550) +
    td(fmt(row.postdocSnr), row.fail.postdocSnr) +
    td(fmt(row.autocorrEfold), row.fail.autocorrEfold) +
    td(fmt(row.oasisEventSnr), row.fail.oasisEventSnr) +
    td(fmt(row.oasisRiseTau), row.fail.oasisRiseTau) +
    td(fmt(row.oasisDecayTau), row.fail.oasisDecayTau) +
    td(fmt(row.oasisResidualKs), row.fail.oasisResidualKs) +
    td(row.reason)
  }}</tr>`).join("");
  const csv = metricSpreadsheetCsv(rowsData);
  const win = window.open("", "_blank");
  win.document.write(`<!doctype html><title>${{data.session}} ROI metrics</title><style>body{{font-family:Arial,sans-serif;margin:20px}}button{{margin:8px 0 12px;padding:6px 10px}}.metric-table-wrap{{max-height:80vh;overflow:auto;border:1px solid #d0d5dd}}.metric-table{{border-collapse:collapse;width:100%;font-size:12px}}.metric-table th,.metric-table td{{border:1px solid #e5e7eb;padding:4px 7px;text-align:right;white-space:nowrap}}.metric-table th{{position:sticky;top:0;background:#f8fafc;z-index:1}}.metric-table td:nth-child(1),.metric-table td:nth-child(2),.metric-table td:nth-child(3),.metric-table td:last-child{{text-align:left}}.metric-fail,.label-bad{{background:rgba(248,113,113,.28)}}.label-good{{background:rgba(34,197,94,.28)}}.label-unsure{{background:rgba(250,204,21,.28)}}</style><h1>${{data.session}} ROI metric spreadsheet</h1><p>target_structure: ${{data.targetStructure}}</p><p>cell_type_label_source: ${{data.cellTypeLabelSource || "not loaded"}}</p><button id="downloadCsv">Download CSV</button><div class="metric-table-wrap"><table class="metric-table"><thead><tr><th>suite2p_index</th><th>manual_label</th><th>cell_type_label</th><th>cell_type_code</th><th>footprint</th><th>skew</th><th>aspect_ratio</th><th>compact</th><th>connectivity</th><th>roi_area_px</th><th>snr_95_50</th><th>caiman_exceptional_event_snr</th><th>autocorr_efold_time_seconds</th><th>inferred_spike_snr</th><th>inferred_spike_rise_tau_seconds</th><th>inferred_spike_decay_tau_seconds</th><th>inferred_spike_residual_gaussian_ks</th><th>reason</th></tr></thead><tbody>${{rows}}</tbody></table></div><script>const csv = ${{JSON.stringify(csv)}}; document.getElementById("downloadCsv").addEventListener("click", () => {{ const blob = new Blob([csv], {{type: "text/csv"}}); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "${{data.session}}_roi_metric_spreadsheet.csv"; a.click(); URL.revokeObjectURL(a.href); }});<\\/script>`);
  win.document.close();
}}
function saveMetricSpreadsheet() {{
  downloadBlob(new Blob([metricSpreadsheetCsv()], {{type: "text/csv"}}), `${{data.session}}_roi_metric_spreadsheet.csv`);
}}
function saveCurrentFilters() {{
  const payload = {{
    format: "utils_2p_roi_metric_filter_v1",
    session: data.session,
    target_structure: data.targetStructure,
    saved_at: new Date().toISOString(),
    filter: normalizeFilter(readFilter()),
  }};
  downloadBlob(new Blob([JSON.stringify(payload, null, 2) + "\\n"], {{type: "application/json"}}), `${{data.session}}_roi_metric_filters.json`);
}}
document.getElementById("openExclusions").addEventListener("click", openMetricSpreadsheet);
function npyBlob(values, rows) {{
  const encoder = new TextEncoder();
  let header = `{{'descr': '<f8', 'fortran_order': False, 'shape': (${{rows}},), }}`;
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
function reviewedLabelArray() {{
  const rows = data.suite2pRoiCount;
  const values = new Float64Array(rows);
  values.fill(NaN);
  for (let roi = 0; roi < data.nRois; roi++) {{
    const suite2pRoi = data.suite2pIndices[roi];
    const label = labels[roi];
    values[suite2pRoi] = label === -1 ? NaN : label;
  }}
  return values;
}}
async function saveManualLabels() {{
  const reviewedLabels = reviewedLabelArray();
  const blob = npyBlob(reviewedLabels, data.suite2pRoiCount);
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
function importPresetObject(payload) {{
  const name = String(payload.name || payload.preset_name || "").trim();
  const filter = payload.filter || payload.morphology_filter || payload;
  if (!name) throw new Error("QC thresholds JSON must include a name.");
  customPresets[name] = normalizeFilter(filter);
  populatePresetSelect(name);
  document.getElementById("filterPreset").value = `custom:${{name}}`;
  document.getElementById("presetName").value = name;
  writeFilter(customPresets[name]);
}}
document.getElementById("saveManualLabels").addEventListener("click", saveManualLabels);
document.getElementById("saveHtmlWithLabels").addEventListener("click", saveHtmlWithLabels);
document.getElementById("saveMetricSpreadsheet").addEventListener("click", saveMetricSpreadsheet);
document.getElementById("saveCurrentFilters").addEventListener("click", saveCurrentFilters);
const saveLabelsDialog = document.getElementById("saveLabelsDialog");
const saveLabelsHelp = document.getElementById("saveLabelsHelp");
const saveLabelsInfo = document.getElementById("saveLabelsInfo");
document.getElementById("openSaveLabelsDialog").addEventListener("click", () => {{
  if (typeof saveLabelsDialog.showModal === "function") saveLabelsDialog.showModal();
  else saveLabelsDialog.setAttribute("open", "");
}});
document.getElementById("closeSaveLabelsDialog").addEventListener("click", () => {{
  closeDialog(saveLabelsDialog);
}});
document.getElementById("closeSaveLabelsDialogTop").addEventListener("click", () => closeDialog(saveLabelsDialog));
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
      alert(`Could not import QC thresholds: ${{error.message}}`);
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
  requestAnimationFrame(drawMetricHistograms);
}});
document.getElementById("closeMorphologyDialog").addEventListener("click", () => {{
  closeDialog(morphologyDialog);
}});
document.getElementById("closeMorphologyDialogTop").addEventListener("click", () => closeDialog(morphologyDialog));
document.getElementById("filterPreset").addEventListener("change", loadSelectedPreset);
document.getElementById("savePreset").addEventListener("click", () => {{
  saveCurrentPresetToPage();
}});
document.getElementById("savePresetHtml").addEventListener("click", () => {{
  if (saveCurrentPresetToPage()) saveHtmlWithLabels();
}});
document.getElementById("toggleSideMenu").addEventListener("click", () => {{
  const review = document.querySelector(".review-main");
  const collapsed = !review.classList.contains("menu-collapsed");
  review.classList.toggle("menu-collapsed", collapsed);
  const button = document.getElementById("toggleSideMenu");
  button.textContent = collapsed ? "Show menu" : "Hide menu";
  button.setAttribute("aria-expanded", String(!collapsed));
  requestAnimationFrame(draw);
}});
const oasisDiagnosticsPanel = document.querySelector(".oasis-diagnostics");
const toggleOasisDiagnostics = document.getElementById("toggleOasisDiagnostics");
if (oasisDiagnosticsPanel && toggleOasisDiagnostics) {{
  toggleOasisDiagnostics.addEventListener("click", () => {{
    const shouldShow = oasisDiagnosticsPanel.classList.contains("minimized");
    oasisDiagnosticsPanel.classList.toggle("minimized", !shouldShow);
    document.querySelector(".fov-review").classList.toggle("oasis-collapsed", !shouldShow);
    toggleOasisDiagnostics.textContent = shouldShow ? "Hide inferred spike fit metrics" : "Show inferred spike fit metrics";
    requestAnimationFrame(draw);
  }});
}}
["skewMin","skewMax","maxConnect","aspectMin","aspectMax","footprintMin","footprintMax","compactMin","compactMax","roiAreaMin","roiAreaMax","eventSnrMin","andreaPostdocSnrMin","autocorrEfoldMin","autocorrEfoldMax","oasisEventSnrMin","oasisRiseTauMin","oasisRiseTauMax","oasisDecayTauMin","oasisDecayTauMax","oasisResidualKsMax","cellTypeFilter"].forEach(id => {{
  document.getElementById(id).addEventListener("change", evaluateFilter);
  document.getElementById(id).addEventListener("input", drawMetricHistograms);
}});
document.querySelectorAll(".metric-histogram-panel").forEach(panel => {{
  panel.addEventListener("toggle", () => {{
    if (!panel.open) return;
    requestAnimationFrame(() => {{
      const canvas = panel.querySelector(".metric-histogram");
      if (canvas) drawMetricHistogram(canvas);
    }});
  }});
}});
window.addEventListener("keydown", event => {{
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if (event.key.toLowerCase() === "g") setLabel(1);
  else if (event.key.toLowerCase() === "b") setLabel(0);
  else if (event.key.toLowerCase() === "u") setLabel(2);
  else if (event.key.toLowerCase() === "n") setLabel(-1);
  else if (event.key === "ArrowLeft") moveVisible(-1);
  else if (event.key === "ArrowRight") moveVisible(1);
}});
document.getElementById("stackCanvas").addEventListener("wheel", e => {{
  e.preventDefault(); const rect=e.target.getBoundingClientRect(), xf=(e.clientX-rect.left)/rect.width, c=x0+xf*(x1-x0), s=(e.deltaY<0?.78:1.28)*(x1-x0);
  setFrameWindow(Math.max(0,c-xf*s), Math.min(data.nFrames-1, Math.max(0,c-xf*s)+s));
}}, {{passive:false}});
let dragging=false, sx=0, start0=0, start1=0;
let stackDragging=false, stackSx=0, stackStart0=0, stackStart1=0;
document.getElementById("traceCanvas").addEventListener("wheel", e => {{
  e.preventDefault(); const rect=e.target.getBoundingClientRect(), xf=(e.clientX-rect.left)/rect.width, c=x0+xf*(x1-x0), s=(e.deltaY<0?.78:1.28)*(x1-x0);
  setFrameWindow(Math.max(0,c-xf*s), Math.min(data.nFrames-1, Math.max(0,c-xf*s)+s));
}}, {{passive:false}});
document.getElementById("traceCanvas").addEventListener("mousedown", e => {{ dragging=true; sx=e.clientX; start0=x0; start1=x1; e.target.classList.add("dragging"); }});
window.addEventListener("mousemove", e => {{ if (!dragging) return; const rect=document.getElementById("traceCanvas").getBoundingClientRect(), shift=-(e.clientX-sx)/rect.width*(start1-start0); setFrameWindow(start0+shift, start1+shift); }});
window.addEventListener("mouseup", () => {{ dragging=false; document.getElementById("traceCanvas").classList.remove("dragging"); }});
document.getElementById("traceCanvas").addEventListener("dblclick", reset);
document.getElementById("stackCanvas").addEventListener("mousedown", e => {{ stackDragging=true; stackSx=e.clientX; stackStart0=x0; stackStart1=x1; e.target.classList.add("dragging"); }});
window.addEventListener("mousemove", e => {{ if (!stackDragging) return; const rect=document.getElementById("stackCanvas").getBoundingClientRect(), shift=-(e.clientX-stackSx)/rect.width*(stackStart1-stackStart0); setFrameWindow(stackStart0+shift, stackStart1+shift); }});
window.addEventListener("mouseup", () => {{ stackDragging=false; document.getElementById("stackCanvas").classList.remove("dragging"); }});
document.getElementById("stackCanvas").addEventListener("dblclick", reset);
let motionDragging=false, motionSx=0, motionStart0=0, motionStart1=0;
document.getElementById("motionDriftCanvas").addEventListener("wheel", e => {{
  e.preventDefault(); const rect=e.target.getBoundingClientRect(), xf=(e.clientX-rect.left)/rect.width, c=x0+xf*(x1-x0), s=(e.deltaY<0?.78:1.28)*(x1-x0);
  setFrameWindow(Math.max(0,c-xf*s), Math.min(data.nFrames-1, Math.max(0,c-xf*s)+s));
}}, {{passive:false}});
document.getElementById("motionDriftCanvas").addEventListener("mousedown", e => {{ motionDragging=true; motionSx=e.clientX; motionStart0=x0; motionStart1=x1; e.target.classList.add("dragging"); }});
window.addEventListener("mousemove", e => {{ if (!motionDragging) return; const rect=document.getElementById("motionDriftCanvas").getBoundingClientRect(), shift=-(e.clientX-motionSx)/rect.width*(motionStart1-motionStart0); setFrameWindow(motionStart0+shift, motionStart1+shift); }});
window.addEventListener("mouseup", () => {{ motionDragging=false; document.getElementById("motionDriftCanvas").classList.remove("dragging"); }});
document.getElementById("motionDriftCanvas").addEventListener("dblclick", reset);
window.addEventListener("resize", () => {{ syncControlColumnHeight(); draw(); }});
makeOverlays(); syncTimeInputs(); updateMetricDefaults(); updateCellTypeLabelStatus(); updateRoiDisplaySummary(); populatePresetSelect("all_rois"); resetFilter();
if (data.initialMorphologyFilter) writeFilter(data.initialMorphologyFilter);
applySort(); syncControlColumnHeight(); setSelected(visibleRois[0]);
if (data.dffStorageMode === "file" && data.dffSidecarName) {{
  loadSidecarByRelativeName(data.dffSidecarName, setDffFromArrayBuffer, "dF/F");
}}
if (data.oasisAvailable && data.oasisStorageMode === "file" && data.oasisSidecarName) {{
  loadSidecarByRelativeName(data.oasisSidecarName, setOasisFromArrayBuffer, "inferred spikes");
}}
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
    input_layout: str = "auto",
) -> tuple[Path, Path]:
    input_path = Path(os.path.abspath(os.path.expanduser(os.fspath(session_data_path))))
    session_dir, resolved_layout = _resolve_summary_session_and_layout(input_path, input_layout)
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else session_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ops = _load_ops(session_dir)
    _log_summary_input_layout(resolved_layout, input_layout)
    suite2p_stat = np.load(resolved_layout.stat_path, allow_pickle=True)
    qc_parameters = _load_qc_parameters(session_dir)
    pipeline_parameters = _load_pipeline_parameters(session_dir)
    xoff, yoff = _load_offsets(session_dir, ops)

    mean_green = _load_masks_h5_image(session_dir, "mean_func")
    if mean_green is None:
        mean_green = np.asarray(ops.get("meanImg"))
    elif "meanImg" in ops and np.asarray(mean_green).shape[:2] != np.asarray(ops.get("meanImg")).shape[:2]:
        mean_green = np.asarray(ops.get("meanImg"))
    max_green = _load_masks_h5_image(session_dir, "max_func")
    if max_green is None:
        max_green = np.asarray(ops.get("max_proj", ops.get("maxImg", mean_green)))
    elif np.asarray(max_green).shape[:2] != np.asarray(mean_green).shape[:2]:
        max_green = np.asarray(ops.get("max_proj", ops.get("maxImg", mean_green)))
        if np.asarray(max_green).shape[:2] != np.asarray(mean_green).shape[:2]:
            max_green = mean_green
    mean_red = _load_masks_h5_image(session_dir, "mean_anat")
    red_available = mean_red is not None
    if (
        (mean_red is None or np.asarray(mean_red).shape[:2] != np.asarray(mean_green).shape[:2])
        and ops.get("nchannels", 1) != 1
        and "meanImg_chan2" in ops
    ):
        mean_red = np.asarray(ops.get("meanImg_chan2"))
        red_available = True
    max_red = _load_masks_h5_image(session_dir, "max_anat")
    if max_red is None and red_available:
        max_red = np.asarray(ops.get("meanImg_chan2_corrected", mean_red))
    elif max_red is not None and red_available and np.asarray(max_red).shape[:2] != np.asarray(mean_green).shape[:2]:
        max_red = np.asarray(ops.get("meanImg_chan2_corrected", mean_red))
        if np.asarray(max_red).shape[:2] != np.asarray(mean_green).shape[:2]:
            max_red = mean_red
    if not red_available:
        mean_red = None
        max_red = None
    if mean_green is None:
        raise KeyError("Could not find green functional mean image in masks.h5 or ops.npy")
    cellpose_mask = (
        _load_masks_h5_label_mask(session_dir, "masks_anat", np.asarray(mean_green).shape[:2], ops)
        if red_available
        else None
    )

    frame_rate = float(ops.get("fs", 30.0))
    dff_label = resolved_layout.dff_label
    fluo_probe = np.load(resolved_layout.fluo_path, allow_pickle=False, mmap_mode="r")
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
        resolved_layout,
        n_rois,
        n_frames,
        frame_rate,
        dff_storage_mode,
        dff_sidecar_path,
    )
    dff_metrics = _add_roi_area_metrics(dff_metrics, stat)
    oasis_spikes, oasis_attrs = _load_oasis_spikes(session_dir, n_rois, n_frames, resolved_layout)
    oasis_storage_mode = "none"
    oasis_sidecar_name = None
    if oasis_spikes is not None:
        dff_metrics = _add_oasis_residual_metrics(
            dff_metrics,
            dff,
            dff_sidecar_path,
            oasis_spikes,
            float(oasis_attrs.get("event_threshold", 0.05)),
            frame_rate,
        )
        oasis_storage_mode = "embedded" if estimated_embedded_dff_bytes <= EMBEDDED_DFF_BYTE_LIMIT else "file"
        if oasis_storage_mode == "file":
            oasis_sidecar_name = f"{session_dir.name}_oasis_spikes.npy"
            np.save(out_dir / oasis_sidecar_name, oasis_spikes)
    mask = _stat_to_mask(stat, np.asarray(mean_green).shape[:2])
    iscell_path = resolved_layout.iscell_path
    iscell = load_iscell(iscell_path, n_rois)
    if not iscell_path.exists():
        iscell[:, :] = 1.0
    suite2p_fingerprint = suite2p_stat_fingerprint(stat)
    morphology_metrics = roi_morphology_metrics(stat)
    cell_type_labels, cell_type_label_source = _load_cell_type_labels(session_dir, n_rois, stat)
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
        cellpose_mask=cellpose_mask,
        mask=mask,
        stat=stat,
        suite2p_indices=suite2p_indices,
        iscell=iscell,
        suite2p_fingerprint=suite2p_fingerprint,
        morphology_metrics=morphology_metrics,
        cell_type_labels=cell_type_labels,
        cell_type_label_source=cell_type_label_source,
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
        xoff=xoff,
        yoff=yoff,
        oasis_spikes=oasis_spikes,
        oasis_attrs=oasis_attrs,
        oasis_storage_mode=oasis_storage_mode,
        oasis_sidecar_name=oasis_sidecar_name,
    )
    return pdf_path, html_path


def run(ops: dict[str, Any], output_dir: str | os.PathLike[str] | None = None) -> tuple[Path, Path]:
    return create_preprocessing_summary(ops["save_path0"], output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create preprocessing QC PDF and interactive HTML outputs.")
    parser.add_argument(
        "session_data_path",
        help=(
            "Processed session root or a concrete data directory such as "
            "suite2p/plane0, suite2p, qc_results, or manual_qc_results."
        ),
    )
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: resolved session root.")
    parser.add_argument("--pdf-name", default=None)
    parser.add_argument("--html-name", default=None)
    parser.add_argument(
        "--input-layout",
        choices=["auto", "suite2p", "qc_results", "manual_qc_results"],
        default="auto",
        help=(
            "Input file layout for ROI/stat/trace arrays. 'suite2p' uses "
            "suite2p/plane0/stat.npy, F.npy, Fneu.npy, and iscell.npy. "
            "'qc_results' and 'manual_qc_results' map stat.npy, fluo.npy, "
            "neuropil.npy, and optional iscell.npy from those filtered-output "
            "directories. If session_data_path already points to a concrete "
            "layout directory, that directory is used directly. Default: "
            "auto-detect filtered layouts before Suite2p from a session root."
        ),
    )
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
        input_layout=args.input_layout,
    )
    print(f"Saved preprocessing PDF: {pdf_path}")
    print(f"Saved interactive HTML: {html_path}")


if __name__ == "__main__":
    main()
