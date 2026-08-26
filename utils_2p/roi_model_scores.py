#!/usr/bin/env python3
"""Run trained ROI model scoring on Suite2p outputs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


MODEL_PATH_ENV_VAR = "TWO_P_ROI_MODEL_PATH"
MODEL_REGISTRY_ENV_VAR = "TWO_P_ROI_MODEL_REGISTRY"
LEGACY_MODEL_PATH_ENV_VAR = "TWO_P_ROI_QC_MODEL_PATH"
LEGACY_MODEL_REGISTRY_ENV_VAR = "TWO_P_ROI_QC_MODEL_REGISTRY"
DEFAULT_GOOD_THRESHOLD = 0.8
DEFAULT_BAD_THRESHOLD = 0.2
DEFAULT_PATCH_SIZE = 64
DEFAULT_BATCH_SIZE = 128
TORCH_THREADS_ENV_VAR = "TWO_P_ROI_MODEL_TORCH_THREADS"
LEGACY_TORCH_THREADS_ENV_VAR = "TWO_P_ROI_QC_TORCH_THREADS"
TARGET_ALIASES = {
    "neuron": "soma",
    "cerebellum_lax": "dendrite_relaxed",
}


@dataclass(frozen=True)
class RoiModelScorePrediction:
    """One model prediction for an original Suite2p ROI row."""

    summary_roi: int
    suite2p_roi: int | None
    probability: float
    state: str


@dataclass(frozen=True)
class RoiModelScoreSelection:
    """Resolved checkpoint for a target-specific ROI model score."""

    model_path: Path
    target_structure: str
    source: str


def normalize_target_structure(target_structure: str | None) -> str:
    """Normalize user-facing target names to registry keys."""

    target = (target_structure or "").strip()
    if not target:
        return ""
    return TARGET_ALIASES.get(target, target)


def _parse_registry_entries(entries: Sequence[str] | str | None) -> dict[str, Path]:
    registry: dict[str, Path] = {}
    if isinstance(entries, str):
        entries = (entries,)
    for entry in entries or ():
        if "=" not in entry:
            raise ValueError(f"ROI model target model entries must use target=/path/to/model.pt: {entry}")
        target, path = entry.split("=", 1)
        target = normalize_target_structure(target)
        if not target:
            raise ValueError(f"ROI model target model entry has an empty target: {entry}")
        registry[target] = Path(path).expanduser().resolve()
    return registry


def load_model_registry(registry_path: Path | str | None = None) -> dict[str, Path]:
    """Load a target-structure-to-checkpoint mapping from JSON.

    Accepted JSON formats are either ``{"dendrite": "/path/model.pt"}`` or
    ``{"models": {"dendrite": "/path/model.pt"}}``.
    """

    configured = registry_path or os.environ.get(MODEL_REGISTRY_ENV_VAR) or os.environ.get(LEGACY_MODEL_REGISTRY_ENV_VAR)
    if not configured:
        return {}
    path = Path(configured).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    models = raw.get("models", raw) if isinstance(raw, dict) else raw
    if not isinstance(models, dict):
        raise ValueError(f"ROI model score registry must be a JSON object: {path}")
    return {
        normalize_target_structure(str(target)): Path(model_path).expanduser().resolve()
        for target, model_path in models.items()
        if normalize_target_structure(str(target))
    }


def _roi_binary_mask(masks: np.ndarray, roi_id: int) -> np.ndarray:
    return masks == roi_id


def _roi_centroid(masks: np.ndarray, roi_id: int) -> tuple[float, float]:
    binary = _roi_binary_mask(masks, roi_id)
    if not np.any(binary):
        raise ValueError(f"ROI {roi_id} not found in mask.")
    ypix, xpix = np.nonzero(binary)
    return float(np.mean(ypix)), float(np.mean(xpix))


def _crop_with_padding(
    image: np.ndarray,
    cy: float,
    cx: float,
    patch_size: int,
    *,
    pad_value: float = 0,
) -> np.ndarray:
    half = patch_size // 2
    cy = int(round(cy))
    cx = int(round(cx))
    y0 = cy - half
    y1 = y0 + patch_size
    x0 = cx - half
    x1 = x0 + patch_size
    out = np.full((patch_size, patch_size), pad_value, dtype=image.dtype)
    src_y0 = max(0, y0)
    src_y1 = min(image.shape[0], y1)
    src_x0 = max(0, x0)
    src_x1 = min(image.shape[1], x1)
    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def _normalize_patch(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    image = image.astype(np.float32)
    return (image - float(image.mean())) / (float(image.std()) + eps)


def make_two_channel_patch(
    mean_img: np.ndarray,
    masks: np.ndarray,
    roi_id: int,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> np.ndarray:
    """Return the model input patch with channels ``mean image`` and ``ROI mask``."""

    cy, cx = _roi_centroid(masks, roi_id)
    image_patch = _crop_with_padding(
        mean_img,
        cy,
        cx,
        patch_size,
        pad_value=float(np.nanmin(mean_img)),
    )
    mask_patch = _crop_with_padding(
        _roi_binary_mask(masks, roi_id).astype(np.uint8),
        cy,
        cx,
        patch_size,
        pad_value=0,
    )
    return np.stack((_normalize_patch(image_patch), mask_patch.astype(np.float32)), axis=0).astype(np.float32)


def _stat_binary_mask(stat_row: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    ypix = np.asarray(stat_row["ypix"], dtype=np.int64)
    xpix = np.asarray(stat_row["xpix"], dtype=np.int64)
    valid = (ypix >= 0) & (ypix < shape[0]) & (xpix >= 0) & (xpix < shape[1])
    mask[ypix[valid], xpix[valid]] = True
    return mask


def make_stat_two_channel_patch(
    mean_img: np.ndarray,
    stat_row: dict[str, Any],
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> np.ndarray:
    """Return the model input patch from one Suite2p ``stat.npy`` row."""

    ypix = np.asarray(stat_row["ypix"], dtype=np.float64)
    xpix = np.asarray(stat_row["xpix"], dtype=np.float64)
    if ypix.size == 0 or xpix.size == 0:
        raise ValueError("Suite2p ROI has no pixels.")
    cy = float(stat_row.get("med", [np.mean(ypix), np.mean(xpix)])[0])
    cx = float(stat_row.get("med", [np.mean(ypix), np.mean(xpix)])[1])
    image_patch = _crop_with_padding(
        mean_img,
        cy,
        cx,
        patch_size,
        pad_value=float(np.nanmin(mean_img)),
    )
    mask_patch = _crop_with_padding(
        _stat_binary_mask(stat_row, mean_img.shape).astype(np.uint8),
        cy,
        cx,
        patch_size,
        pad_value=0,
    )
    return np.stack((_normalize_patch(image_patch), mask_patch.astype(np.float32)), axis=0).astype(np.float32)


def default_model_path() -> Path:
    """Return the configured ROI model checkpoint path.

    The predictor code is packaged with ``utils_2p``. Model weights are kept
    outside the package so labs can update or swap checkpoints without changing
    the Python package.
    """

    configured = os.environ.get(MODEL_PATH_ENV_VAR) or os.environ.get(LEGACY_MODEL_PATH_ENV_VAR)
    if not configured:
        raise FileNotFoundError(
            "No ROI model score checkpoint was supplied. Pass --roi-model-path "
            f"or set {MODEL_PATH_ENV_VAR}=/path/to/best_model.pt."
        )
    return Path(configured).expanduser().resolve()


def _resolve_model_path(model_path: Path | str | None) -> Path:
    return default_model_path() if model_path in (None, "") else Path(model_path).expanduser().resolve()


def select_model(
    *,
    model_path: Path | str | None = None,
    target_structure: str | None = None,
    model_registry_path: Path | str | None = None,
    target_models: Sequence[str] | None = None,
) -> RoiModelScoreSelection:
    """Resolve the checkpoint to use for a target structure.

    Priority:
    1. explicit ``--roi-model-path`` fallback,
    2. repeated ``target=/path.pt`` command-line entries,
    3. JSON registry path or ``TWO_P_ROI_MODEL_REGISTRY``,
    4. ``TWO_P_ROI_MODEL_<TARGET>`` environment variable,
    5. ``TWO_P_ROI_MODEL_PATH`` / legacy ``TWO_P_ROI_QC_MODEL_PATH`` fallback.
    """

    target = normalize_target_structure(target_structure)
    if model_path not in (None, ""):
        return RoiModelScoreSelection(
            model_path=Path(model_path).expanduser().resolve(),
            target_structure=target,
            source="explicit --roi-model-path",
        )
    registry = load_model_registry(model_registry_path)
    registry.update(_parse_registry_entries(target_models))
    if target and target in registry:
        return RoiModelScoreSelection(
            model_path=registry[target],
            target_structure=target,
            source="target model registry",
        )
    if target:
        env_name = f"TWO_P_ROI_MODEL_{target.upper().replace('-', '_')}"
        legacy_env_name = f"TWO_P_ROI_QC_MODEL_{target.upper().replace('-', '_')}"
        configured = os.environ.get(env_name) or os.environ.get(legacy_env_name)
        if configured:
            return RoiModelScoreSelection(
                model_path=Path(configured).expanduser().resolve(),
                target_structure=target,
                source=env_name if os.environ.get(env_name) else legacy_env_name,
            )
    return RoiModelScoreSelection(
        model_path=default_model_path(),
        target_structure=target,
        source=MODEL_PATH_ENV_VAR,
    )


def _load_model(model_path: Path):
    import torch
    import torch.nn as nn
    import torchvision.models as models

    torch.set_num_threads(int(os.environ.get(TORCH_THREADS_ENV_VAR, os.environ.get(LEGACY_TORCH_THREADS_ENV_VAR, "1"))))

    class ROICNN(nn.Module):
        def __init__(self, in_channels: int = 2):
            super().__init__()
            self.backbone = models.resnet18(weights=None)
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

        def forward(self, x):
            return self.backbone(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ROICNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device


def _predict_probability(model: Any, device: str, patch: np.ndarray) -> float:
    import torch

    with torch.no_grad():
        x = torch.tensor(patch / 255.0, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(x)
        return float(torch.sigmoid(logits)[0, 0].detach().cpu().numpy())


def _predict_probabilities(model: Any, device: str, patches: np.ndarray, *, batch_size: int) -> np.ndarray:
    import torch

    probabilities = []
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.tensor(patches[start : start + batch_size] / 255.0, dtype=torch.float32, device=device)
            logits = model(batch)
            probabilities.append(torch.sigmoid(logits).reshape(-1).detach().cpu().numpy())
    if not probabilities:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(probabilities).astype(np.float32, copy=False)


def _prediction_state(probability: float, *, good_threshold: float, bad_threshold: float) -> str:
    if probability >= good_threshold:
        return "good"
    if probability <= bad_threshold:
        return "bad"
    return "gray"


def load_session_inputs(session_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Suite2p ``meanImg`` and original Suite2p ``stat.npy`` rows."""

    ops_path = session_dir / "suite2p" / "plane0" / "ops.npy"
    stat_path = session_dir / "suite2p" / "plane0" / "stat.npy"
    if not ops_path.exists():
        raise FileNotFoundError(f"Missing Suite2p ops file: {ops_path}")
    if not stat_path.exists():
        raise FileNotFoundError(f"Missing Suite2p stat file: {stat_path}")
    ops = np.load(ops_path, allow_pickle=True).item()
    stat = np.load(stat_path, allow_pickle=True)
    return np.asarray(ops["meanImg"]), stat


def predict_session(
    session_dir: Path | str,
    *,
    model_path: Path | str | None = None,
    target_structure: str | None = None,
    model_registry_path: Path | str | None = None,
    target_models: Sequence[str] | None = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    good_threshold: float = DEFAULT_GOOD_THRESHOLD,
    bad_threshold: float = DEFAULT_BAD_THRESHOLD,
) -> list[RoiModelScorePrediction]:
    """Score all original Suite2p ROIs in a processed session with a trained model."""

    session_dir = Path(session_dir).expanduser().resolve()
    selection = select_model(
        model_path=model_path,
        target_structure=target_structure,
        model_registry_path=model_registry_path,
        target_models=target_models,
    )
    if not selection.model_path.exists():
        raise FileNotFoundError(f"ROI model score checkpoint does not exist: {selection.model_path}")
    mean_img, stat = load_session_inputs(session_dir)
    model, device = _load_model(selection.model_path)
    predictions: list[RoiModelScorePrediction] = []
    roi_ids = np.arange(len(stat), dtype=np.int64)
    patches = np.stack(
        [make_stat_two_channel_patch(mean_img, stat[int(roi_id)], patch_size=patch_size) for roi_id in roi_ids],
        axis=0,
    )
    probabilities = _predict_probabilities(model, device, patches, batch_size=batch_size)
    for roi_id, probability in zip(roi_ids, probabilities):
        suite2p_roi = int(roi_id)
        predictions.append(
            RoiModelScorePrediction(
                summary_roi=suite2p_roi,
                suite2p_roi=suite2p_roi,
                probability=float(probability),
                state=_prediction_state(
                    probability,
                    good_threshold=good_threshold,
                    bad_threshold=bad_threshold,
                ),
            )
        )
    return predictions


def save_predictions(
    session_dir: Path | str,
    predictions: list[RoiModelScorePrediction],
    *,
    output_name: str = "roi_model_scores.h5",
    model_path: Path | str | None = None,
    target_structure: str | None = None,
    model_source: str = "",
    good_threshold: float = DEFAULT_GOOD_THRESHOLD,
    bad_threshold: float = DEFAULT_BAD_THRESHOLD,
    patch_size: int = DEFAULT_PATCH_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Path:
    """Save model predictions in the processed session directory."""

    session_dir = Path(session_dir).expanduser().resolve()
    output = session_dir / output_name
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output, "w") as h5:
        h5.create_dataset("summary_roi", data=np.asarray([p.summary_roi for p in predictions], dtype=np.int32))
        h5.create_dataset(
            "suite2p_roi",
            data=np.asarray([-1 if p.suite2p_roi is None else p.suite2p_roi for p in predictions], dtype=np.int32),
        )
        h5.create_dataset("probability", data=np.asarray([p.probability for p in predictions], dtype=np.float32))
        h5.create_dataset("state", data=np.asarray([p.state for p in predictions], dtype=dt))
        h5.attrs["index_space"] = "suite2p_original"
        h5.attrs["roi_source"] = "suite2p/plane0/stat.npy"
        h5.attrs["model_path"] = str(model_path or "")
        h5.attrs["target_structure"] = normalize_target_structure(target_structure)
        h5.attrs["model_source"] = model_source
        h5.attrs["good_threshold"] = float(good_threshold)
        h5.attrs["bad_threshold"] = float(bad_threshold)
        h5.attrs["patch_size"] = int(patch_size)
        h5.attrs["batch_size"] = int(batch_size)
    return output


def update_roi_label_h5(session_dir: Path | str, predictions: list[RoiModelScorePrediction]) -> Path:
    """Update legacy ``ROI_label.h5`` while preserving existing good/bad labels."""

    session_dir = Path(session_dir).expanduser().resolve()
    label_path = session_dir / "ROI_label.h5"
    good_roi: set[int] = set()
    bad_roi: set[int] = set()
    if label_path.exists():
        with h5py.File(label_path, "r") as h5:
            if "good_roi" in h5:
                good_roi.update(int(v) for v in h5["good_roi"][:])
            if "bad_roi" in h5:
                bad_roi.update(int(v) for v in h5["bad_roi"][:])

    for prediction in predictions:
        roi = prediction.suite2p_roi
        if roi is None or roi < 0 or roi in good_roi or roi in bad_roi:
            continue
        if prediction.state == "good":
            good_roi.add(int(roi))
        elif prediction.state == "bad":
            bad_roi.add(int(roi))

    with h5py.File(label_path, "w") as h5:
        h5.create_dataset("good_roi", data=np.asarray(sorted(good_roi), dtype=np.int32))
        h5.create_dataset("bad_roi", data=np.asarray(sorted(bad_roi), dtype=np.int32))
        h5.attrs["index_space"] = "suite2p_original"
        h5.attrs["source"] = "utils_2p.roi_model_scores"
    return label_path


def run(
    session_dir: Path | str,
    *,
    model_path: Path | str | None = None,
    target_structure: str | None = None,
    model_registry_path: Path | str | None = None,
    target_models: Sequence[str] | None = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    good_threshold: float = DEFAULT_GOOD_THRESHOLD,
    bad_threshold: float = DEFAULT_BAD_THRESHOLD,
    force: bool = False,
) -> tuple[Path, Path]:
    """Run ROI model score inference and update session-level label files."""

    session_dir = Path(session_dir).expanduser().resolve()
    prediction_path = session_dir / "roi_model_scores.h5"
    legacy_prediction_path = session_dir / "roi_qc_predictions.h5"
    label_path = session_dir / "ROI_label.h5"
    if prediction_path.exists() and not force:
        print(f"Using existing ROI model scores: {prediction_path}")
        print("Pass --force to regenerate ROI model score predictions.")
        return prediction_path, label_path
    if legacy_prediction_path.exists() and not force:
        print(f"Using existing legacy ROI model scores: {legacy_prediction_path}")
        print("Pass --force to regenerate ROI model score predictions as roi_model_scores.h5.")
        return legacy_prediction_path, label_path

    selection = select_model(
        model_path=model_path,
        target_structure=target_structure,
        model_registry_path=model_registry_path,
        target_models=target_models,
    )
    print("ROI model scores warning: the currently available trained model is intended for cerebellar dendrite ROIs only.")
    predictions = predict_session(
        session_dir,
        model_path=selection.model_path,
        target_structure=selection.target_structure,
        patch_size=patch_size,
        batch_size=batch_size,
        good_threshold=good_threshold,
        bad_threshold=bad_threshold,
    )
    prediction_path = save_predictions(
        session_dir,
        predictions,
        model_path=selection.model_path,
        target_structure=selection.target_structure,
        model_source=selection.source,
        good_threshold=good_threshold,
        bad_threshold=bad_threshold,
        patch_size=patch_size,
        batch_size=batch_size,
    )
    label_path = update_roi_label_h5(session_dir, predictions)
    counts = {state: sum(1 for prediction in predictions if prediction.state == state) for state in ("good", "bad", "gray")}
    print(f"Saved ROI model scores: {prediction_path}")
    print(f"Updated ROI labels: {label_path}")
    print(f"ROI model score target: {selection.target_structure or 'unspecified'} ({selection.model_path})")
    print(f"Good: {counts['good']} Bad: {counts['bad']} Gray: {counts['gray']}")
    return prediction_path, label_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"{__doc__} The currently available trained model is intended for "
            "cerebellar dendrite ROIs only."
        )
    )
    parser.add_argument("session", type=Path, help="Processed session directory.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=f"Path to model checkpoint. If omitted, uses the target registry or ${MODEL_PATH_ENV_VAR}.",
    )
    parser.add_argument("--target-structure", default=None, help="Target anatomical structure for model selection.")
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=None,
        help=f"JSON mapping target structures to checkpoints. If omitted, uses ${MODEL_REGISTRY_ENV_VAR} when set.",
    )
    parser.add_argument(
        "--target-model",
        action="append",
        default=None,
        help="Register one target-specific model as target=/path/to/checkpoint.pt. Repeat as needed.",
    )
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--good-threshold", type=float, default=DEFAULT_GOOD_THRESHOLD)
    parser.add_argument("--bad-threshold", type=float, default=DEFAULT_BAD_THRESHOLD)
    parser.add_argument("--force", action="store_true", help="Regenerate predictions even if roi_model_scores.h5 exists.")
    args = parser.parse_args()
    run(
        args.session,
        model_path=args.model_path,
        target_structure=args.target_structure,
        model_registry_path=args.model_registry,
        target_models=args.target_model,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        good_threshold=args.good_threshold,
        bad_threshold=args.bad_threshold,
        force=args.force,
    )


if __name__ == "__main__":
    main()
