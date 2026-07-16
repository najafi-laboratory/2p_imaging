#!/usr/bin/env python3
"""Utilities for Suite2p manual ROI editing workflows."""

from __future__ import annotations

import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROW_ALIGNED_FILES = (
    "iscell.npy",
    "F.npy",
    "Fneu.npy",
    "spks.npy",
    "F_chan2.npy",
    "Fneu_chan2.npy",
    "redcell.npy",
)

QC_TO_SUITE2P_FILES = {
    "stat.npy": "stat.npy",
    "fluo.npy": "F.npy",
    "neuropil.npy": "Fneu.npy",
}

SUITE2P_TO_QC_FILES = {suite2p: qc for qc, suite2p in QC_TO_SUITE2P_FILES.items()}

DERIVED_QC_FILES = (
    "dff.h5",
    "denoised_dff.h5",
    "spikes.h5",
)


def _roi_key(entry: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    ypix = np.asarray(entry.get("ypix", []), dtype=int)
    xpix = np.asarray(entry.get("xpix", []), dtype=int)
    if ypix.shape != xpix.shape:
        raise ValueError("ROI xpix and ypix arrays have different shapes")
    return tuple(sorted(zip(ypix.tolist(), xpix.tolist())))


def _stat_fingerprint(stat: Sequence[dict[str, Any]]) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(_roi_key(entry) for entry in stat)


def _backup_path(path: Path, timestamp: str) -> Path:
    return path.with_name(f"{path.name}.manual_roi_backup_{timestamp}")


def _copy_or_link(source: Path, destination: Path, *, symlink: bool) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing workspace file: {destination}")
    if symlink:
        destination.symlink_to(source)
    else:
        shutil.copy2(source, destination)


def _write_generated_iscell(path: Path, n_rois: int) -> None:
    iscell = np.ones((n_rois, 2), dtype=float)
    np.save(path, iscell)


def _write_generated_spks(path: Path, f_path: Path) -> None:
    f = np.load(f_path, mmap_mode="r")
    np.save(path, np.zeros(f.shape, dtype=np.float32))


def _copy_spikes_h5_to_spks_npy(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    import h5py

    with h5py.File(source, "r") as h5:
        if "spikes" not in h5:
            return False
        np.save(destination, np.asarray(h5["spikes"], dtype=np.float32))
    return True


def _write_h5_dataset(path: Path, name: str, data: np.ndarray) -> None:
    import h5py

    with h5py.File(path, "w") as h5:
        h5.create_dataset(name, data=data)


def _compute_dff(fluo: np.ndarray, neuropil: np.ndarray, ops: dict[str, Any]) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    signal = fluo.astype(np.float64, copy=False) - float(ops.get("neucoeff", 0.7)) * neuropil.astype(
        np.float64, copy=False
    )
    sigma = float(ops.get("sig_baseline", 600.0))
    baseline = gaussian_filter(signal, [0.0, sigma])
    with np.errstate(divide="ignore", invalid="ignore"):
        return (signal - baseline) / baseline


def _stat_to_masks(stat: Sequence[dict[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    masks = np.zeros(shape, dtype=float)
    for index, entry in enumerate(stat):
        ypix = np.asarray(entry["ypix"], dtype=int)
        xpix = np.asarray(entry["xpix"], dtype=int)
        masks[ypix, xpix] = index + 1
    return masks


def _mask_shape(workspace_dir: Path, qc_dir: Path) -> tuple[int, int]:
    masks_path = qc_dir / "masks.npy"
    if masks_path.exists():
        masks = np.load(masks_path, mmap_mode="r")
        if masks.ndim != 2:
            raise ValueError(f"{masks_path} must be 2D, found shape {masks.shape}")
        return int(masks.shape[0]), int(masks.shape[1])

    ops_path = workspace_dir / "ops.npy"
    if ops_path.exists():
        ops = np.load(ops_path, allow_pickle=True).item()
        if "Ly" in ops and "Lx" in ops:
            return int(ops["Ly"]), int(ops["Lx"])

    raise FileNotFoundError(
        f"Cannot infer mask shape; expected {masks_path} or ops.npy with Ly/Lx in {workspace_dir}"
    )


def _load_ops(workspace_dir: Path, qc_dir: Path) -> dict[str, Any]:
    for path in (workspace_dir / "ops.npy", qc_dir.parent / "ops.npy"):
        if path.exists():
            return np.load(path, allow_pickle=True).item()
    return {}


def _regenerate_derived_outputs(
    workspace_dir: Path,
    qc_dir: Path,
    *,
    existing_only: bool,
) -> list[Path]:
    targets = [qc_dir / name for name in DERIVED_QC_FILES]
    if existing_only:
        targets = [path for path in targets if path.exists()]
    if not targets:
        return []

    target_names = {path.name for path in targets}
    fluo = np.load(qc_dir / "fluo.npy", allow_pickle=False)
    neuropil = np.load(qc_dir / "neuropil.npy", allow_pickle=False)
    ops = _load_ops(workspace_dir, qc_dir)
    written: list[Path] = []

    dff = None
    if "dff.h5" in target_names or "denoised_dff.h5" in target_names:
        dff = _compute_dff(fluo, neuropil, ops)
        if "dff.h5" in target_names:
            _write_h5_dataset(qc_dir / "dff.h5", "dff", dff)
            written.append(qc_dir / "dff.h5")

    if "denoised_dff.h5" in target_names:
        from scipy.ndimage import gaussian_filter1d

        if dff is None:
            dff = _compute_dff(fluo, neuropil, ops)
        denoised = gaussian_filter1d(dff, sigma=2.0, axis=1, mode="nearest")
        _write_h5_dataset(qc_dir / "denoised_dff.h5", "denoised_dff", denoised)
        written.append(qc_dir / "denoised_dff.h5")

    if "spikes.h5" in target_names:
        spks_path = workspace_dir / "spks.npy"
        if not spks_path.exists():
            raise FileNotFoundError(f"Cannot regenerate spike-derived outputs without {spks_path}")
        spikes = np.load(spks_path, allow_pickle=False)
        if spikes.shape != fluo.shape:
            raise ValueError(f"{spks_path} shape {spikes.shape} does not match fluo shape {fluo.shape}")
        _write_h5_dataset(qc_dir / "spikes.h5", "spikes", spikes.astype(np.float32, copy=False))
        written.append(qc_dir / "spikes.h5")

    return written


def _detect_manual_roi_rows(stat_orig: np.ndarray, stat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(remove_indices, keep_indices)`` for manual rows in ``stat``."""

    if len(stat) < len(stat_orig):
        raise ValueError(
            f"Current stat has fewer ROIs ({len(stat)}) than stat_orig ({len(stat_orig)})"
        )

    original_fingerprint = _stat_fingerprint(stat_orig)
    manual_flag_indices = np.asarray(
        [index for index, entry in enumerate(stat) if bool(entry.get("manual", 0))],
        dtype=np.int64,
    )
    if manual_flag_indices.size:
        keep_indices = np.asarray(
            [index for index in range(len(stat)) if index not in set(manual_flag_indices.tolist())],
            dtype=np.int64,
        )
        if len(keep_indices) == len(stat_orig) and _stat_fingerprint(stat[keep_indices]) == original_fingerprint:
            return manual_flag_indices, keep_indices

    remaining = Counter(_roi_key(entry) for entry in stat_orig)
    keep: list[int] = []
    remove: list[int] = []
    for index, entry in enumerate(stat):
        key = _roi_key(entry)
        if remaining[key] > 0:
            keep.append(index)
            remaining[key] -= 1
        else:
            remove.append(index)

    missing = sum(remaining.values())
    if missing:
        raise ValueError(f"Current stat is missing {missing} ROI(s) found in stat_orig")

    keep_indices = np.asarray(keep, dtype=np.int64)
    if _stat_fingerprint(stat[keep_indices]) != original_fingerprint:
        raise ValueError(
            "Removing detected extra rows would not restore stat_orig row order; "
            "refusing to modify row-aligned arrays"
        )
    return np.asarray(remove, dtype=np.int64), keep_indices


def remove_all_manual_rois(
    stat_orig_path: str | Path,
    stat_path: str | Path,
    *,
    row_aligned_files: Sequence[str] = ROW_ALIGNED_FILES,
    backup: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove all manual ROI rows added after ``stat_orig.npy`` was saved.

    Parameters
    ----------
    stat_orig_path
        Path to the backup stat file written by Suite2p's manual ROI GUI.
    stat_path
        Path to the edited active ``stat.npy`` file.
    row_aligned_files
        File names in the same plane directory whose first axis corresponds to
        rows in ``stat.npy``.
    backup
        If True, copy each file before overwriting it.
    dry_run
        If True, report the rows/files that would change without writing files.

    Returns
    -------
    dict
        Summary containing removed row indices, kept row indices, changed files,
        and backup paths.
    """

    stat_orig_path = Path(stat_orig_path)
    stat_path = Path(stat_path)
    plane_dir = stat_path.parent

    stat_orig = np.load(stat_orig_path, allow_pickle=True)
    stat = np.load(stat_path, allow_pickle=True)
    remove_indices, keep_indices = _detect_manual_roi_rows(stat_orig, stat)

    changed_files: list[str] = []
    backups: dict[str, str] = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if remove_indices.size == 0:
        return {
            "stat_orig": str(stat_orig_path.resolve()),
            "stat": str(stat_path.resolve()),
            "removed_indices": [],
            "kept_indices": keep_indices.tolist(),
            "changed_files": [],
            "backups": {},
            "dry_run": dry_run,
        }

    files_to_update: list[Path] = [stat_path]
    row_files: list[Path] = []
    for name in row_aligned_files:
        path = plane_dir / name
        if path.exists():
            arr = np.load(path, allow_pickle=True, mmap_mode="r")
            if arr.shape[0] != len(stat):
                raise ValueError(
                    f"{path} has {arr.shape[0]} rows, expected {len(stat)} to match {stat_path}"
                )
            row_files.append(path)
            files_to_update.append(path)

    if not dry_run and backup:
        for path in files_to_update:
            backup_file = _backup_path(path, timestamp)
            shutil.copy2(path, backup_file)
            backups[str(path)] = str(backup_file)

    if not dry_run:
        np.save(stat_path, stat_orig)
        changed_files.append(str(stat_path))
        for path in row_files:
            arr = np.load(path, allow_pickle=True)
            np.save(path, arr[keep_indices])
            changed_files.append(str(path))

    return {
        "stat_orig": str(stat_orig_path.resolve()),
        "stat": str(stat_path.resolve()),
        "removed_indices": remove_indices.tolist(),
        "kept_indices": keep_indices.tolist(),
        "changed_files": changed_files,
        "backups": backups,
        "dry_run": dry_run,
    }


def create_manual_roi_workspace(
    qc_dir: str | Path,
    suite2p_plane_dir: str | Path,
    workspace_dir: str | Path,
    *,
    copy_binary: bool = False,
) -> dict[str, Any]:
    """Create a Suite2p-compatible workspace from a QC-style ROI folder.

    ``qc_dir`` is expected to contain ``stat.npy``, ``fluo.npy``, and
    ``neuropil.npy``. The workspace receives Suite2p's expected file names:
    ``stat.npy``, ``F.npy``, ``Fneu.npy``, ``spks.npy``, ``iscell.npy``,
    ``ops.npy``, and ``data.bin``. Trace arrays are copied so Suite2p GUI saves
    cannot accidentally overwrite the source QC arrays while editing.
    """

    qc_dir = Path(qc_dir)
    suite2p_plane_dir = Path(suite2p_plane_dir)
    workspace_dir = Path(workspace_dir)
    if workspace_dir.exists() and any(workspace_dir.iterdir()):
        raise FileExistsError(f"Workspace directory is not empty: {workspace_dir}")
    workspace_dir.mkdir(parents=True, exist_ok=True)

    created: dict[str, str] = {}
    for qc_name, suite2p_name in QC_TO_SUITE2P_FILES.items():
        source = qc_dir / qc_name
        if not source.exists():
            raise FileNotFoundError(f"Missing required QC file: {source}")
        destination = workspace_dir / suite2p_name
        shutil.copy2(source, destination)
        created[suite2p_name] = str(destination)

    stat = np.load(workspace_dir / "stat.npy", allow_pickle=True)
    f = np.load(workspace_dir / "F.npy", mmap_mode="r")
    fneu = np.load(workspace_dir / "Fneu.npy", mmap_mode="r")
    if f.shape != fneu.shape:
        raise ValueError(f"F/Fneu shape mismatch: {f.shape} vs {fneu.shape}")
    if f.shape[0] != len(stat):
        raise ValueError(f"Trace rows ({f.shape[0]}) do not match stat rows ({len(stat)})")

    source_iscell = qc_dir / "iscell.npy"
    if source_iscell.exists():
        shutil.copy2(source_iscell, workspace_dir / "iscell.npy")
        iscell_source = "copied"
    else:
        _write_generated_iscell(workspace_dir / "iscell.npy", len(stat))
        iscell_source = "generated_all_cells"
    created["iscell.npy"] = str(workspace_dir / "iscell.npy")

    source_spks = qc_dir / "spks.npy"
    if source_spks.exists():
        shutil.copy2(source_spks, workspace_dir / "spks.npy")
        spks_source = "copied"
    elif _copy_spikes_h5_to_spks_npy(qc_dir / "spikes.h5", workspace_dir / "spks.npy"):
        spks_source = "copied_from_spikes_h5"
    else:
        _write_generated_spks(workspace_dir / "spks.npy", workspace_dir / "F.npy")
        spks_source = "generated_zeros"
    created["spks.npy"] = str(workspace_dir / "spks.npy")

    for name in ("ops.npy", "settings.npy", "db.npy"):
        source = suite2p_plane_dir / name
        if source.exists():
            shutil.copy2(source, workspace_dir / name)
            created[name] = str(workspace_dir / name)

    data_bin = suite2p_plane_dir / "data.bin"
    if data_bin.exists() or data_bin.is_symlink():
        _copy_or_link(data_bin.resolve(), workspace_dir / "data.bin", symlink=not copy_binary)
        created["data.bin"] = str(workspace_dir / "data.bin")
    else:
        raise FileNotFoundError(f"Missing Suite2p binary file: {data_bin}")

    manifest = workspace_dir / "manual_roi_workspace_source.txt"
    manifest.write_text(
        "\n".join(
            [
                f"qc_dir={qc_dir.resolve()}",
                f"suite2p_plane_dir={suite2p_plane_dir.resolve()}",
                f"workspace_dir={workspace_dir.resolve()}",
                f"iscell={iscell_source}",
                f"spks={spks_source}",
                "aliases=stat.npy->stat.npy, fluo.npy->F.npy, neuropil.npy->Fneu.npy",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    created[manifest.name] = str(manifest)

    return {
        "qc_dir": str(qc_dir.resolve()),
        "suite2p_plane_dir": str(suite2p_plane_dir.resolve()),
        "workspace_dir": str(workspace_dir.resolve()),
        "created_files": created,
        "iscell_source": iscell_source,
        "spks_source": spks_source,
    }


def export_manual_roi_workspace(
    workspace_dir: str | Path,
    qc_dir: str | Path,
    *,
    backup: bool = True,
    stale_policy: str = "manifest",
    export_iscell: bool = False,
    export_spks: bool = False,
    update_masks: bool = True,
    regenerate_derived: bool = False,
    derived_existing_only: bool = True,
    cleanup_workspace: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Export edited Suite2p-compatible workspace files back to QC names.

    ``stat.npy``, ``F.npy``, and ``Fneu.npy`` are exported to ``stat.npy``,
    ``fluo.npy``, and ``neuropil.npy``. ``masks.npy`` is regenerated by
    default. Non-Suite2p derived outputs are left untouched unless
    ``regenerate_derived=True`` is passed explicitly.
    """

    workspace_dir = Path(workspace_dir)
    qc_dir = Path(qc_dir)
    if stale_policy not in {"manifest", "ignore"}:
        raise ValueError("stale_policy must be 'manifest' or 'ignore'")

    stat = np.load(workspace_dir / "stat.npy", allow_pickle=True)
    f = np.load(workspace_dir / "F.npy", mmap_mode="r")
    fneu = np.load(workspace_dir / "Fneu.npy", mmap_mode="r")
    if f.shape != fneu.shape:
        raise ValueError(f"F/Fneu shape mismatch: {f.shape} vs {fneu.shape}")
    if f.shape[0] != len(stat):
        raise ValueError(f"Trace rows ({f.shape[0]}) do not match stat rows ({len(stat)})")

    exports = dict(SUITE2P_TO_QC_FILES)
    if export_iscell and (workspace_dir / "iscell.npy").exists():
        exports["iscell.npy"] = "iscell.npy"
    if export_spks and (workspace_dir / "spks.npy").exists():
        exports["spks.npy"] = "spks.npy"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups: dict[str, str] = {}
    changed_files: list[str] = []

    for suite2p_name, qc_name in exports.items():
        source = workspace_dir / suite2p_name
        destination = qc_dir / qc_name
        if not source.exists():
            raise FileNotFoundError(f"Missing workspace file: {source}")
        if destination.exists() and backup and not dry_run:
            backup_file = _backup_path(destination, timestamp)
            shutil.copy2(destination, backup_file)
            backups[str(destination)] = str(backup_file)
        if not dry_run:
            shutil.copy2(source, destination)
            changed_files.append(str(destination))

    masks_path = qc_dir / "masks.npy"
    if update_masks:
        mask_shape = _mask_shape(workspace_dir, qc_dir)
        if masks_path.exists() and backup and not dry_run:
            backup_file = _backup_path(masks_path, timestamp)
            shutil.copy2(masks_path, backup_file)
            backups[str(masks_path)] = str(backup_file)
        if not dry_run:
            np.save(masks_path, _stat_to_masks(stat, mask_shape))
            changed_files.append(str(masks_path))

    derived_files = [qc_dir / name for name in DERIVED_QC_FILES if (qc_dir / name).exists()]
    if backup and regenerate_derived and not dry_run:
        for path in derived_files:
            backup_file = _backup_path(path, timestamp)
            shutil.copy2(path, backup_file)
            backups[str(path)] = str(backup_file)

    regenerated_derived: list[str] = []
    if regenerate_derived and not dry_run:
        regenerated_derived = [
            str(path)
            for path in _regenerate_derived_outputs(
                workspace_dir,
                qc_dir,
                existing_only=derived_existing_only,
            )
        ]
        changed_files.extend(regenerated_derived)

    stale_files = [
        str(path)
        for path in derived_files
        if str(path) not in set(regenerated_derived)
    ]
    stale_manifest = None
    if stale_policy == "manifest" and stale_files and not dry_run:
        path = qc_dir / f"manual_roi_stale_outputs_{timestamp}.txt"
        path.write_text(
            "Manual ROI edits changed stat/fluo/neuropil. These derived files were not regenerated:\n"
            + "\n".join(stale_files)
            + "\n",
            encoding="ascii",
        )
        stale_manifest = str(path)
        changed_files.append(str(path))

    workspace_removed = False
    if cleanup_workspace and not dry_run:
        manifest = workspace_dir / "manual_roi_workspace_source.txt"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Refusing to remove workspace without manual_roi_workspace_source.txt: {workspace_dir}"
            )
        shutil.rmtree(workspace_dir)
        workspace_removed = True

    return {
        "workspace_dir": str(workspace_dir.resolve()),
        "qc_dir": str(qc_dir.resolve()),
        "changed_files": changed_files,
        "backups": backups,
        "stale_files": stale_files,
        "regenerated_derived": regenerated_derived,
        "stale_manifest": stale_manifest,
        "workspace_removed": workspace_removed,
        "dry_run": dry_run,
    }
