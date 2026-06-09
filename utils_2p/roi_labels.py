#!/usr/bin/env python3
"""Map and apply manual ROI labels without replacing Suite2p ``iscell.npy``."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter


def _roi_key(entry: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    ypix = np.asarray(entry.get("ypix", []), dtype=int)
    xpix = np.asarray(entry.get("xpix", []), dtype=int)
    if ypix.shape != xpix.shape:
        raise ValueError("ROI xpix and ypix arrays have different shapes")
    return tuple(sorted(zip(ypix.tolist(), xpix.tolist())))


def map_qc_to_suite2p_rois(
    qc_stat: Sequence[dict[str, Any]],
    suite2p_stat: Sequence[dict[str, Any]],
) -> np.ndarray:
    """Return the original Suite2p row for each post-QC ROI."""

    by_pixels: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for index, entry in enumerate(suite2p_stat):
        by_pixels.setdefault(_roi_key(entry), []).append(index)

    mapped: list[int] = []
    for qc_index, entry in enumerate(qc_stat):
        matches = by_pixels.get(_roi_key(entry), [])
        if len(matches) != 1:
            raise ValueError(
                f"QC ROI {qc_index} maps to {len(matches)} Suite2p ROIs; "
                "manual labels cannot be exported safely"
            )
        mapped.append(matches[0])
    return np.asarray(mapped, dtype=np.int64)


def suite2p_stat_fingerprint(stat: Sequence[dict[str, Any]]) -> str:
    """Return a stable fingerprint for ROI pixel coordinates and row order."""

    digest = hashlib.sha256()
    for index, entry in enumerate(stat):
        digest.update(f"{index}:".encode("ascii"))
        for ypix, xpix in _roi_key(entry):
            digest.update(f"{ypix},{xpix};".encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _connected_components_4(pixels: set[tuple[int, int]]) -> int:
    components = 0
    remaining = set(pixels)
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            ypix, xpix = stack.pop()
            for neighbor in ((ypix - 1, xpix), (ypix + 1, xpix), (ypix, xpix - 1), (ypix, xpix + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def roi_morphology_metrics(stat: Sequence[dict[str, Any]]) -> list[dict[str, float | int]]:
    """Return the morphology fields used by ``QualControlDataIO.thres_stat``."""

    metrics: list[dict[str, float | int]] = []
    for entry in stat:
        pixels = set(_roi_key(entry))
        metrics.append(
            {
                "skew": float(entry.get("skew", np.nan)),
                "connect": int(_connected_components_4(pixels)) if pixels else 0,
                "aspect": float(entry.get("aspect_ratio", np.nan)),
                "compact": float(entry.get("compact", np.nan)),
                "footprint": float(entry.get("footprint", np.nan)),
            }
        )
    return metrics


def morphology_exclusion_reasons(
    metrics: Sequence[dict[str, float | int]],
    parameters: dict[str, Any],
) -> list[list[str]]:
    """Return the QC threshold violations for every ROI."""

    skew_min, skew_max = parameters["range_skew"]
    aspect_min, aspect_max = parameters["range_aspect"]
    compact_min, compact_max = parameters["range_compact"]
    footprint_min, footprint_max = parameters["range_footprint"]
    max_connect = parameters["max_connect"]
    reasons: list[list[str]] = []
    for roi in metrics:
        failed: list[str] = []
        if not footprint_min <= roi["footprint"] <= footprint_max:
            failed.append(f"footprint {roi['footprint']:.3f} outside [{footprint_min}, {footprint_max}]")
        if not skew_min <= roi["skew"] <= skew_max:
            failed.append(f"skew {roi['skew']:.3f} outside [{skew_min}, {skew_max}]")
        if not aspect_min <= roi["aspect"] <= aspect_max:
            failed.append(f"aspect_ratio {roi['aspect']:.3f} outside [{aspect_min}, {aspect_max}]")
        if not compact_min <= roi["compact"] <= compact_max:
            failed.append(f"compact {roi['compact']:.3f} outside [{compact_min}, {compact_max}]")
        if roi["connect"] > max_connect:
            failed.append(f"connectivity {roi['connect']} exceeds {max_connect}")
        reasons.append(failed)
    return reasons


def load_iscell(path: str | Path, n_rois: int) -> np.ndarray:
    """Load and validate a Suite2p two-column ``iscell.npy`` array."""

    iscell_path = Path(path)
    if iscell_path.exists():
        iscell = np.asarray(np.load(iscell_path, allow_pickle=False), dtype=np.float64)
    else:
        iscell = np.zeros((n_rois, 2), dtype=np.float64)
    if iscell.shape != (n_rois, 2):
        raise ValueError(f"Expected iscell shape {(n_rois, 2)}, got {iscell.shape}")
    return iscell


def _find_label_export(session_dir: Path, label_path: str | Path | None) -> Path:
    if label_path is not None:
        path = Path(label_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"ROI label export does not exist: {path}")
        return path

    iscell_path = session_dir / "suite2p" / "plane0" / "iscell_qc.npy"
    if not iscell_path.exists():
        raise FileNotFoundError(
            f"No suite2p/plane0/iscell_qc.npy found under {session_dir}; "
            "pass label_path for a JSON export or QC file stored elsewhere"
        )
    return iscell_path


def _load_export_labels(label_path: Path, stat: Sequence[dict[str, Any]]) -> np.ndarray:
    if label_path.suffix.lower() == ".json":
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        labels = payload.get("labels")
        if not isinstance(labels, list):
            raise ValueError("ROI label export does not contain a labels list")
        expected_count = payload.get("suite2p_roi_count")
        if expected_count is not None and int(expected_count) != len(stat):
            raise ValueError(f"Label export expects {expected_count} Suite2p ROIs, found {len(stat)}")
        expected_fingerprint = payload.get("suite2p_stat_fingerprint")
        if expected_fingerprint is not None and expected_fingerprint != suite2p_stat_fingerprint(stat):
            raise ValueError("Label export does not match this Suite2p stat.npy")

        values = np.full(len(stat), -1, dtype=np.int8)
        seen: set[int] = set()
        for item in labels:
            index = int(item["suite2p_roi"])
            if index < 0 or index >= len(stat):
                raise ValueError(f"Suite2p ROI index {index} is outside 0..{len(stat) - 1}")
            if index in seen:
                raise ValueError(f"Suite2p ROI index {index} appears more than once")
            seen.add(index)
            raw_label = item.get("label")
            if raw_label is None:
                continue
            label = int(raw_label)
            if label not in (0, 1):
                raise ValueError(f"ROI label must be 0 or 1, got {label}")
            values[index] = label
        return values

    iscell = load_iscell(label_path, len(stat))
    return (iscell[:, 0] > 0.5).astype(np.int8)


def load_reviewed_dff(
    session_dir: str | Path,
    *,
    label_path: str | Path | None = None,
    policy: str = "good_only",
    baseline_sigma: float = 600.0,
) -> dict[str, Any]:
    """Load reviewer-selected Suite2p ROIs and calculate their dF/F traces.

    ``label_path`` may point to an interactive reviewer JSON export or an
    ``iscell_qc.npy`` file stored elsewhere. When omitted,
    ``suite2p/plane0/iscell_qc.npy`` is required. Suite2p's original
    ``iscell.npy`` is never selected implicitly.

    ``policy="good_only"`` keeps only label 1. ``policy="not_bad"`` also keeps
    JSON rows whose label is null/unlabeled.
    """

    session_path = Path(session_dir).expanduser().resolve()
    plane_dir = session_path / "suite2p" / "plane0"
    required = [plane_dir / name for name in ("ops.npy", "stat.npy", "F.npy", "Fneu.npy")]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Suite2p files:\n" + "\n".join(str(path) for path in missing))

    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
    resolved_label_path = _find_label_export(session_path, label_path)
    labels = _load_export_labels(resolved_label_path, stat)

    if policy == "good_only":
        keep = labels == 1
    elif policy == "not_bad":
        keep = labels != 0
    else:
        raise ValueError("policy must be 'good_only' or 'not_bad'")

    fluo = np.load(plane_dir / "F.npy", mmap_mode="r")
    neuropil = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
    if fluo.shape != neuropil.shape or fluo.shape[0] != len(stat):
        raise ValueError("F.npy, Fneu.npy, and stat.npy do not share the same ROI axis")

    roi_indices = np.flatnonzero(keep)
    signal = (
        np.asarray(fluo[keep], dtype=np.float32)
        - float(ops.get("neucoeff", 0.7)) * np.asarray(neuropil[keep], dtype=np.float32)
    )
    if signal.shape[0]:
        baseline = gaussian_filter(signal, sigma=[0.0, float(baseline_sigma)])
        with np.errstate(divide="ignore", invalid="ignore"):
            dff = (signal - baseline) / baseline
        dff[~np.isfinite(dff)] = np.nan
    else:
        dff = signal

    return {
        "dff": dff.astype(np.float32, copy=False),
        "roi_indices": roi_indices,
        "stat": stat[keep],
        "labels": labels,
        "ops": ops,
        "label_path": resolved_label_path,
        "policy": policy,
    }


def apply_label_export(
    export_path: str | Path,
    suite2p_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    backup: bool = True,
) -> Path:
    """Apply an HTML JSON export to a separate ``iscell_qc.npy`` file."""

    payload = json.loads(Path(export_path).read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not isinstance(labels, list):
        raise ValueError("ROI label export does not contain a labels list")

    suite2p_path = Path(suite2p_dir).expanduser().resolve()
    stat = np.load(suite2p_path / "stat.npy", allow_pickle=True)
    expected_count = payload.get("suite2p_roi_count")
    if expected_count is not None and int(expected_count) != len(stat):
        raise ValueError(f"Label export expects {expected_count} Suite2p ROIs, found {len(stat)}")
    expected_fingerprint = payload.get("suite2p_stat_fingerprint")
    if expected_fingerprint is not None and expected_fingerprint != suite2p_stat_fingerprint(stat):
        raise ValueError("Label export does not match this Suite2p stat.npy")
    original_iscell_path = suite2p_path / "iscell.npy"
    iscell = load_iscell(original_iscell_path, len(stat))

    seen: set[int] = set()
    for item in labels:
        index = int(item["suite2p_roi"])
        raw_label = item.get("label")
        if raw_label is None:
            continue
        label = int(raw_label)
        if index < 0 or index >= len(stat):
            raise ValueError(f"Suite2p ROI index {index} is outside 0..{len(stat) - 1}")
        if label not in (0, 1):
            raise ValueError(f"ROI label must be 0 or 1, got {label}")
        if index in seen:
            raise ValueError(f"Suite2p ROI index {index} appears more than once")
        seen.add(index)
        iscell[index, 0] = label
        iscell[index, 1] = float(label)

    qc_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else suite2p_path / "iscell_qc.npy"
    )
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    if backup and qc_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(qc_path, qc_path.with_name(f"{qc_path.name}.bak_{stamp}"))
    np.save(qc_path, iscell)
    return qc_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_json", help="JSON file exported by the interactive QC HTML.")
    parser.add_argument("suite2p_dir", help="Suite2p plane directory containing stat.npy.")
    parser.add_argument("--output", default=None, help="Output path. Default: suite2p_dir/iscell_qc.npy.")
    parser.add_argument("--no-backup", action="store_true", help="Do not back up an existing iscell_qc.npy.")
    args = parser.parse_args()
    output = apply_label_export(
        args.export_json,
        args.suite2p_dir,
        output_path=args.output,
        backup=not args.no_backup,
    )
    print(f"Saved manual ROI labels: {output}")


if __name__ == "__main__":
    main()
