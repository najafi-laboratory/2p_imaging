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
from scipy.ndimage import binary_dilation, gaussian_filter, gaussian_filter1d
from scipy.stats import norm


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


def robust_event_snr(
    dff: Sequence[float] | np.ndarray,
    *,
    sigma: float = 3.0,
    event_percentile: float = 80.0,
    dilation: int = 5,
) -> dict[str, float]:
    """Return a robust event SNR estimate for one dF/F trace.

    The signal term is ``P95(dF/F) - P50(dF/F)``. Noise is estimated from the
    median absolute deviation of the high-frequency residual after smoothing
    the trace with a Gaussian kernel. Event-heavy frames are masked before the
    noise estimate is computed.
    """

    trace = np.asarray(dff, dtype=float).ravel()
    trace = trace[np.isfinite(trace)]
    if trace.size < 3:
        return {"snr_95_50": np.nan, "signal_amp": np.nan, "noise_sd": np.nan}

    signal_amp = float(np.nanpercentile(trace, 95) - np.nanpercentile(trace, 50))
    smooth = gaussian_filter1d(trace, sigma=float(sigma))
    resid = trace - smooth

    baseline_resid = resid
    if 0.0 <= event_percentile <= 100.0 and trace.size:
        event_frames = trace > np.nanpercentile(trace, event_percentile)
        if np.any(event_frames):
            event_frames = binary_dilation(event_frames, iterations=max(0, int(dilation)))
            baseline_resid = resid[~event_frames]
            if baseline_resid.size == 0:
                baseline_resid = resid

    centered = baseline_resid - np.nanmedian(baseline_resid)
    noise_sd = float(1.4826 * np.nanmedian(np.abs(centered)))
    snr = float(signal_amp / noise_sd) if noise_sd > 0 else np.nan
    return {"snr_95_50": snr, "signal_amp": signal_amp, "noise_sd": noise_sd}


def half_sample_mode(values: Sequence[float] | np.ndarray) -> float:
    """Return a robust half-sample mode estimate."""

    data = np.sort(np.asarray(values, dtype=float).ravel())
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.nan

    def _hsm(sorted_data: np.ndarray) -> float:
        size = sorted_data.size
        if size == 1:
            return float(sorted_data[0])
        if size == 2:
            return float(sorted_data.mean())
        if size == 3:
            left = sorted_data[1] - sorted_data[0]
            right = sorted_data[2] - sorted_data[1]
            if left < right:
                return float(sorted_data[:2].mean())
            if right < left:
                return float(sorted_data[1:].mean())
            return float(sorted_data[1])

        half = size // 2 + size % 2
        widths = sorted_data[half - 1 :] - sorted_data[: size - half + 1]
        start = int(np.nanargmin(widths))
        return _hsm(sorted_data[start : start + half])

    return _hsm(data)


def andrea_postdoc_snr(
    dff: Sequence[float] | np.ndarray,
    *,
    consecutive_events: int = 5,
    robust_std: bool = False,
) -> float:
    """Return the Andrea G/postdoc exceptional-event SNR-like score.

    This follows ``evaluate_components.py`` from Andrea G with Farzaneh Najafi's
    modifications. The original ``fitness`` is the minimum summed log tail
    probability across ``N`` consecutive samples; more negative values indicate
    more exceptional events. This function returns ``-fitness`` so larger values
    sort as stronger event-like traces in the reviewer.
    """

    trace = np.asarray(dff, dtype=float).ravel()
    trace = trace[np.isfinite(trace)]
    if trace.size < max(1, int(consecutive_events)):
        return np.nan

    mode = half_sample_mode(trace)
    if not np.isfinite(mode):
        return np.nan

    below_mode = -(trace - mode) * (trace < mode)
    positive = below_mode[below_mode > 0]
    if positive.size == 0:
        return np.nan

    if robust_std:
        sorted_positive = np.sort(positive)
        index = max(0, int(round(sorted_positive.size * 0.5)) - 1)
        noise_sd = float(2.0 * sorted_positive[index] / 1.349)
    else:
        noise_sd = float(np.sqrt(np.sum(below_mode**2) / positive.size))
    if not np.isfinite(noise_sd) or noise_sd <= 0:
        return np.nan

    z = (trace - mode) / (3.0 * noise_sd)
    tail_probability = np.clip(1.0 - norm.cdf(z), np.finfo(float).tiny, 1.0)
    log_probability = np.log(tail_probability)
    window = np.ones(max(1, int(consecutive_events)), dtype=float)
    summed = np.convolve(log_probability, window, mode="full")[: trace.size]
    fitness = float(np.nanmin(summed))
    return float(-fitness) if np.isfinite(fitness) else np.nan


def temporal_smoothness_snr(dff: Sequence[float] | np.ndarray) -> float:
    """Return a Suite2p-like temporal smoothness score for a dF/F trace."""

    trace = np.asarray(dff, dtype=float).ravel()
    trace = trace[np.isfinite(trace)]
    if trace.size < 3:
        return np.nan
    trace_var = float(np.nanvar(trace))
    if not np.isfinite(trace_var) or trace_var <= 0:
        return np.nan
    diff_var = float(np.nanvar(np.diff(trace)))
    if not np.isfinite(diff_var):
        return np.nan
    return float(1.0 - diff_var / (2.0 * trace_var))


def autocorrelation_efold_time(
    dff: Sequence[float] | np.ndarray,
    *,
    frame_rate: float = 1.0,
    max_lag_seconds: float = 10.0,
) -> float:
    """Return the dF/F autocorrelation e-folding time in seconds."""

    trace = np.asarray(dff, dtype=float).ravel()
    trace = trace[np.isfinite(trace)]
    if trace.size < 3 or frame_rate <= 0:
        return np.nan
    trace = trace - np.nanmedian(trace)
    variance = float(np.nanvar(trace))
    if not np.isfinite(variance) or variance <= 0:
        return np.nan

    max_lag = min(trace.size - 1, max(1, int(round(float(max_lag_seconds) * float(frame_rate)))))
    fft_size = 1 << int(np.ceil(np.log2(2 * trace.size - 1)))
    spectrum = np.fft.rfft(trace, n=fft_size)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[: max_lag + 1]
    if autocorr[0] <= 0 or not np.isfinite(autocorr[0]):
        return np.nan
    autocorr = autocorr / autocorr[0]

    threshold = 1.0 / np.e
    below = np.flatnonzero(autocorr[1:] <= threshold)
    if below.size == 0:
        return np.nan
    lag = int(below[0] + 1)
    if lag == 1:
        crossing = float(lag)
    else:
        x0, x1 = float(lag - 1), float(lag)
        y0, y1 = float(autocorr[lag - 1]), float(autocorr[lag])
        crossing = x1 if y1 == y0 else x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    return float(crossing / float(frame_rate)) if np.isfinite(crossing) else np.nan


def autocorrelation_decay_tau(
    dff: Sequence[float] | np.ndarray,
    *,
    frame_rate: float = 1.0,
    max_lag_seconds: float = 10.0,
) -> float:
    """Backward-compatible alias for :func:`autocorrelation_efold_time`."""

    return autocorrelation_efold_time(dff, frame_rate=frame_rate, max_lag_seconds=max_lag_seconds)


def dff_qc_metrics(dff: np.ndarray, *, frame_rate: float = 1.0) -> list[dict[str, float]]:
    """Return QC metrics for every dF/F trace in ``dff``."""

    metrics: list[dict[str, float]] = []
    for trace in np.asarray(dff, dtype=float):
        event = robust_event_snr(trace)
        temporal = temporal_smoothness_snr(trace)
        autocorr_efold = autocorrelation_efold_time(trace, frame_rate=frame_rate)
        postdoc_snr = andrea_postdoc_snr(trace)
        metrics.append(
            {
                "snr_95_50": event["snr_95_50"],
                "event_signal_amp": event["signal_amp"],
                "event_noise_sd": event["noise_sd"],
                "andrea_postdoc_snr": float(postdoc_snr),
                "temporal_snr": float(temporal),
                "autocorr_efold_time_seconds": float(autocorr_efold),
            }
        )
    return metrics


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

    label_path = session_dir / "suite2p" / "plane0" / "roi_manual_labels.npy"
    if not label_path.exists():
        raise FileNotFoundError(
            f"No suite2p/plane0/roi_manual_labels.npy found under {session_dir}; "
            "pass label_path for a JSON export or manual label file stored elsewhere"
        )
    return label_path


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
            if label not in (0, 1, 2):
                raise ValueError(f"ROI label must be 0, 1, or 2, got {label}")
            values[index] = label
        return values

    masks = np.asarray(np.load(label_path, allow_pickle=False), dtype=np.float64)
    if masks.shape != (len(stat), 3):
        raise ValueError(f"Expected roi_manual_labels shape {(len(stat), 3)}, got {masks.shape}")
    return masks


def load_reviewed_dff(
    session_dir: str | Path,
    *,
    label_path: str | Path | None = None,
    policy: str = "good_only",
    baseline_sigma: float = 600.0,
) -> dict[str, Any]:
    """Load reviewer-selected Suite2p ROIs and calculate their dF/F traces.

    ``label_path`` may point to an interactive reviewer JSON export or a
    ``roi_manual_labels.npy`` file stored elsewhere. When omitted,
    ``suite2p/plane0/roi_manual_labels.npy`` is required. Suite2p's original
    ``iscell.npy`` is never selected implicitly.

    For JSON labels, ``policy="good_only"`` keeps label 1, ``policy="not_bad"``
    keeps all labels except 0, and ``policy="good_or_unsure"`` keeps labels 1
    and 2. For ``roi_manual_labels.npy``, ``good_only`` selects column 1
    (morphology-filtered good) and ``good_or_unsure`` selects column 2
    (morphology-filtered good or unsure).
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

    if labels.ndim == 2:
        if policy in {"good_only", "morphology_good"}:
            keep = labels[:, 1] == 1
        elif policy in {"good_or_unsure", "morphology_good_or_unsure", "not_bad"}:
            keep = labels[:, 2] == 1
        elif policy == "full_suite2p_good":
            keep = labels[:, 0] == 1
        else:
            raise ValueError(
                "policy must be 'good_only', 'good_or_unsure', 'not_bad', "
                "'morphology_good', 'morphology_good_or_unsure', or 'full_suite2p_good'"
            )
    else:
        if policy == "good_only":
            keep = labels == 1
        elif policy == "not_bad":
            keep = labels != 0
        elif policy == "good_or_unsure":
            keep = (labels == 1) | (labels == 2)
        else:
            raise ValueError("policy must be 'good_only', 'good_or_unsure', or 'not_bad'")

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
    """Apply an HTML JSON export to a ``roi_manual_labels.npy`` file."""

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
    masks = np.zeros((len(stat), 3), dtype=np.float64)

    seen: set[int] = set()
    for item in labels:
        index = int(item["suite2p_roi"])
        raw_label = item.get("label")
        if index < 0 or index >= len(stat):
            raise ValueError(f"Suite2p ROI index {index} is outside 0..{len(stat) - 1}")
        if index in seen:
            raise ValueError(f"Suite2p ROI index {index} appears more than once")
        seen.add(index)
        morphology_pass = bool(item.get("morphology_pass", True))
        if not morphology_pass:
            masks[index, 1:] = np.nan
        if raw_label is None:
            continue
        label = int(raw_label)
        if label not in (0, 1, 2):
            raise ValueError(f"ROI label must be 0, 1, or 2, got {label}")
        masks[index, 0] = 1.0 if label == 1 else 0.0
        if morphology_pass:
            masks[index, 1] = 1.0 if label == 1 else 0.0
            masks[index, 2] = 1.0 if label in (1, 2) else 0.0

    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else suite2p_path / "roi_manual_labels.npy"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if backup and output.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(output, output.with_name(f"{output.name}.bak_{stamp}"))
    np.save(output, masks)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_json", help="JSON file exported by the interactive QC HTML.")
    parser.add_argument("suite2p_dir", help="Suite2p plane directory containing stat.npy.")
    parser.add_argument("--output", default=None, help="Output path. Default: suite2p_dir/roi_manual_labels.npy.")
    parser.add_argument("--no-backup", action="store_true", help="Do not back up an existing roi_manual_labels.npy.")
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
