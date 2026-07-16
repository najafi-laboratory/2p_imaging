#!/usr/bin/env python3
"""Prebuild Suite2p 1.x data.bin from OME-TIFF files in parallel.

This is an experimental helper for single-plane, single-channel sessions.  It
creates the same minimal plane0 binary/db/settings files that Suite2p uses to
skip TIFF-to-binary conversion and continue with registration/detection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils_2p.preprocessing_qc_pipeline import _processing_ops, _suite2p_v1_settings_db


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _first_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return Path(str(value[0]))
    return Path(str(value))


def _old_ops_path(processed_session: Path) -> Path:
    candidates = [
        processed_session / "suite2p" / "plane0" / "ops.npy",
        processed_session / "plane0" / "ops.npy",
        processed_session / "ops.npy",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No Suite2p ops.npy found under {processed_session}; expected suite2p/plane0/ops.npy"
    )


def _raw_path_from_ops(ops: dict[str, Any]) -> Path:
    raw_path = _first_path(ops.get("data_path"))
    if raw_path is not None:
        return raw_path
    file_list = ops.get("filelist", ops.get("file_list"))
    first_file = _first_path(file_list)
    if first_file is not None:
        return first_file.parent
    raise ValueError("Could not infer raw TIFF path from old ops.npy; pass raw_path explicitly")


def _create_manifest_from_old_suite2p(
    processed_session: Path | str,
    output_root: Path | str,
    *,
    manifest_path: Path | str | None = None,
    raw_path: Path | str | None = None,
    target_structure: str = "dendrite",
    session_name: str | None = None,
    processing_root: Path | str | None = None,
    postprocess_root: Path | str | None = None,
    python_bin: Path | str | None = None,
    keep_suite2p_bin: bool = True,
) -> Path:
    """Create an internal prebuild manifest from an older processed Suite2p session."""

    processed = Path(processed_session).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    ops = np.load(_old_ops_path(processed), allow_pickle=True).item()
    source_raw = Path(raw_path).expanduser().resolve() if raw_path is not None else _raw_path_from_ops(ops)
    name = session_name or processed.name
    output_path = output / name

    overrides: dict[str, Any] = {}
    for key in (
        "input_format",
        "fs",
        "tau",
        "diameter",
        "sparse_mode",
        "threshold_scaling",
        "max_iterations",
        "high_pass",
        "spatial_hp_detect",
    ):
        if key in ops:
            overrides[key] = _jsonable(ops[key])

    pipeline = {
        "repo_root": str(REPO_ROOT),
        "processing_root": str(Path(processing_root).expanduser().resolve() if processing_root else REPO_ROOT / "2p_processing_pipeline_202401"),
        "postprocess_root": str(Path(postprocess_root).expanduser().resolve() if postprocess_root else REPO_ROOT / "2p_post_process_module_202404"),
        "python_bin": str(
            Path(python_bin).expanduser().resolve()
            if python_bin
            else Path("/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_suite2p_1x/bin/python")
        ),
        "suite2p_version": "1.x",
        "account": "gts-fnajafi3",
        "username": "grubin6",
        "mail_user": None,
        "numba_cache_dir": str(output / ".cache" / "numba"),
        "fast_disk": "tmp",
        "qos_cpu": "embers",
        "qos_gpu": "embers",
        "partition_cpu": None,
        "partition_gpu": None,
        "suite2p_gpu": True,
        "suite2p_batch_size": None,
        "suite2p_binary_batch_size": 5000,
        "suite2p_registration_batch_size": 500,
        "suite2p_extraction_batch_size": 500,
        "suite2p_detection_nbins": None,
        "suite2p_diameter": None,
        "keep_suite2p_bin": bool(keep_suite2p_bin),
    }
    session = {
        "raw_path": str(source_raw),
        "name": name,
        "target_structure": target_structure,
        "nchannels": int(ops.get("nchannels", 1)),
        "functional_chan": int(ops.get("functional_chan", 1)),
        "denoise": _jsonable(ops.get("denoise")),
        "spatial_scale": _jsonable(ops.get("spatial_scale")),
        "bpod_mat_path": None,
        "run_label": False,
        "stages": ["suite2p"],
        "output_path": str(output_path),
        "suite2p_ops_overrides": overrides,
        "qc_parameters": {
            "range_skew": [0.0, 2.0],
            "max_connect": 2,
            "range_aspect": [1.2, 5.0],
            "range_footprint": [1.0, 2.0],
            "range_compact": [1.06, 5.0],
            "diameter": 6,
            "source": f"generated from {processed / 'suite2p' / 'plane0' / 'ops.npy'}",
        },
    }
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output),
        "pipeline": pipeline,
        "resources": {
            "cpu_cpus": 4,
            "cpu_mem": "32G",
            "cpu_time": "04:00:00",
            "suite2p_cpus": 8,
            "suite2p_mem": "192G",
            "suite2p_time": "08:00:00",
            "gpu_cpus": 8,
            "gpu_mem": "192G",
            "gpu_time": "08:00:00",
            "gpu_gres": "gpu:1",
            "summary_mem": "24G",
            "summary_time": "02:00:00",
        },
        "sessions": [session],
    }

    destination = Path(manifest_path).expanduser().resolve() if manifest_path else processed / "manual_roi_prebuild_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    return destination


def _read_file_to_chunk(args: tuple[str, str, str | None, int, int, int]) -> dict[str, Any]:
    source, chunk_path, chunk_chan2_path, batch_size, nchannels, functional_chan = args
    from suite2p.io.tiff import open_tiff, read_tiff

    source_path = Path(source)
    chunk = Path(chunk_path)
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk_chan2 = Path(chunk_chan2_path) if chunk_chan2_path else None
    if chunk_chan2 is not None:
        chunk_chan2.parent.mkdir(parents=True, exist_ok=True)

    if nchannels not in (1, 2):
        raise ValueError(f"Parallel prebuild helper supports nchannels 1 or 2, got {nchannels}")
    if functional_chan not in (1, 2):
        raise ValueError(f"functional_chan must be 1 or 2, got {functional_chan}")
    if nchannels == 1 and functional_chan != 1:
        functional_chan = 1
    if functional_chan > nchannels:
        raise ValueError(f"functional_chan {functional_chan} is not available with nchannels={nchannels}")
    functional_index = functional_chan - 1
    alternate_index = 1 - functional_index
    effective_batch_size = int(nchannels * np.ceil(batch_size / nchannels))

    tif, n_pages = open_tiff(str(source_path), True)
    nframes = 0
    raw_frames = 0
    ly = lx = None
    mean_sum = None
    mean_chan2_sum = None
    try:
        with chunk.open("wb") as handle, (
            chunk_chan2.open("wb") if chunk_chan2 is not None else open(os.devnull, "wb")
        ) as handle_chan2:
            ix = 0
            while True:
                frames = read_tiff(str(source_path), tif, n_pages, ix, effective_batch_size, True)
                if frames is None:
                    break
                if frames.ndim != 3:
                    raise ValueError(f"{source_path} produced frames with shape {frames.shape}; expected 3-D")
                if ly is None:
                    ly, lx = int(frames.shape[1]), int(frames.shape[2])
                    mean_sum = np.zeros((ly, lx), dtype=np.float64)
                    if nchannels > 1:
                        mean_chan2_sum = np.zeros((ly, lx), dtype=np.float64)
                elif frames.shape[1:] != (ly, lx):
                    raise ValueError(f"{source_path} changed frame shape from {(ly, lx)} to {frames.shape[1:]}")

                if nchannels == 1:
                    functional_frames = frames
                    alternate_frames = None
                else:
                    functional_frames = frames[functional_index::nchannels]
                    alternate_frames = frames[alternate_index::nchannels]
                    if functional_frames.shape[0] != alternate_frames.shape[0]:
                        raise ValueError(
                            f"{source_path} has unmatched two-channel frames in batch starting at frame {ix}: "
                            f"functional={functional_frames.shape[0]}, alternate={alternate_frames.shape[0]}"
                        )

                handle.write(np.ascontiguousarray(functional_frames, dtype=np.int16).tobytes())
                mean_sum += functional_frames.sum(axis=0, dtype=np.float64)
                nframes += int(functional_frames.shape[0])
                if alternate_frames is not None and mean_chan2_sum is not None:
                    handle_chan2.write(np.ascontiguousarray(alternate_frames, dtype=np.int16).tobytes())
                    mean_chan2_sum += alternate_frames.sum(axis=0, dtype=np.float64)
                raw_frames += int(frames.shape[0])
                ix += int(frames.shape[0])
    finally:
        close = getattr(tif, "close", None)
        if callable(close):
            close()

    if ly is None or lx is None or mean_sum is None or nframes == 0:
        raise ValueError(f"No frames read from {source_path}")
    return {
        "source": str(source_path),
        "chunk": str(chunk),
        "chunk_chan2": str(chunk_chan2) if chunk_chan2 is not None else None,
        "nframes": nframes,
        "raw_frames": raw_frames,
        "Ly": ly,
        "Lx": lx,
        "mean_sum": mean_sum,
        "mean_chan2_sum": mean_chan2_sum,
    }


def _write_single_channel_tiff_to_chunk(path: Path, chunk: Path, batch_size: int) -> tuple[int, int, int, np.ndarray]:
    from suite2p.io.tiff import open_tiff, read_tiff

    tif, n_pages = open_tiff(str(path), True)
    chunk.parent.mkdir(parents=True, exist_ok=True)
    nframes = 0
    ly = lx = None
    mean_sum = None
    try:
        with chunk.open("wb") as handle:
            ix = 0
            while True:
                frames = read_tiff(str(path), tif, n_pages, ix, batch_size, True)
                if frames is None:
                    break
                if frames.ndim != 3:
                    raise ValueError(f"{path} produced frames with shape {frames.shape}; expected 3-D")
                if ly is None:
                    ly, lx = int(frames.shape[1]), int(frames.shape[2])
                    mean_sum = np.zeros((ly, lx), dtype=np.float64)
                elif frames.shape[1:] != (ly, lx):
                    raise ValueError(f"{path} changed frame shape from {(ly, lx)} to {frames.shape[1:]}")
                frames = np.ascontiguousarray(frames, dtype=np.int16)
                handle.write(frames.tobytes())
                mean_sum += frames.sum(axis=0, dtype=np.float64)
                nframes += int(frames.shape[0])
                ix += int(frames.shape[0])
    finally:
        close = getattr(tif, "close", None)
        if callable(close):
            close()

    if ly is None or lx is None or mean_sum is None or nframes == 0:
        raise ValueError(f"No frames read from {path}")
    return nframes, ly, lx, mean_sum


def _read_bruker_pair_to_chunk(args: tuple[str, str | None, str, str | None, int]) -> dict[str, Any]:
    functional_source, alternate_source, chunk_path, chunk_chan2_path, batch_size = args
    functional_path = Path(functional_source)
    alternate_path = Path(alternate_source) if alternate_source else None
    chunk = Path(chunk_path)
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk_chan2 = Path(chunk_chan2_path) if chunk_chan2_path else None
    if chunk_chan2 is not None:
        chunk_chan2.parent.mkdir(parents=True, exist_ok=True)

    nframes, ly, lx, mean_sum = _write_single_channel_tiff_to_chunk(functional_path, chunk, batch_size)
    mean_chan2_sum = None

    if alternate_path is not None and chunk_chan2 is not None:
        alt_nframes, alt_ly, alt_lx, mean_chan2_sum = _write_single_channel_tiff_to_chunk(
            alternate_path, chunk_chan2, batch_size
        )
        if (alt_nframes, alt_ly, alt_lx) != (nframes, ly, lx):
            raise ValueError(
                f"Channel pair shape mismatch: {functional_path} has {(nframes, ly, lx)}, "
                f"{alternate_path} has {(alt_nframes, alt_ly, alt_lx)}"
            )

    return {
        "source": str(functional_path),
        "chunk": str(chunk),
        "chunk_chan2": str(chunk_chan2) if chunk_chan2 is not None else None,
        "nframes": nframes,
        "raw_frames": nframes,
        "Ly": ly,
        "Lx": lx,
        "mean_sum": mean_sum,
        "mean_chan2_sum": mean_chan2_sum,
    }


def _load_manifest(path: Path, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="ascii"))
    return data, data["sessions"][index]


def _session_matches_processed_output(session: dict[str, Any], processed_session: Path) -> bool:
    candidates = [
        session.get("output_path"),
        Path(session["output_path"]).name if session.get("output_path") else None,
        session.get("name"),
    ]
    target = processed_session.resolve()
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            if Path(str(candidate)).expanduser().resolve() == target:
                return True
        except OSError:
            pass
        if str(candidate) == processed_session.name:
            return True
    return False


def _find_existing_manifest(processed_session: Path, output_root: Path | None) -> tuple[Path, int] | None:
    search_roots: list[Path] = []
    if output_root is not None:
        search_roots.append(output_root / ".preprocessing_qc_jobs")
    search_roots.extend(
        [
            processed_session / ".preprocessing_qc_jobs",
            processed_session.parent / ".preprocessing_qc_jobs",
        ]
    )
    for root in search_roots:
        if not root.exists():
            continue
        for manifest in sorted(root.glob("**/manifest.json")):
            try:
                data = json.loads(manifest.read_text(encoding="ascii"))
            except (OSError, json.JSONDecodeError):
                continue
            for index, session in enumerate(data.get("sessions", [])):
                if _session_matches_processed_output(session, processed_session):
                    return manifest, index
    return None


def _manifest_for_processed_session(
    processed_session: Path | str,
    output_root: Path | str | None,
    *,
    raw_path: Path | str | None,
    target_structure: str,
    session_name: str | None,
) -> tuple[Path, int]:
    processed = Path(processed_session).expanduser().resolve()
    output = Path(output_root).expanduser().resolve() if output_root is not None else None
    existing = _find_existing_manifest(processed, output)
    if existing is not None:
        return existing
    if output is None:
        output = processed.parent
    name = session_name or processed.name
    manifest = _create_manifest_from_old_suite2p(
        processed,
        output,
        manifest_path=output / ".prebuild_manifests" / f"{name}_prebuild_manifest.json",
        raw_path=raw_path,
        target_structure=target_structure,
        session_name=name,
    )
    return manifest, 0


def _staged_tiffs(raw_path: Path) -> list[Path]:
    files = sorted(raw_path.glob("*.ome.tif")) + sorted(raw_path.glob("*.ome.TIF"))
    if not files:
        files = sorted(raw_path.glob("*.tif")) + sorted(raw_path.glob("*.tiff"))
    if not files:
        raise FileNotFoundError(f"No TIFF files found directly under {raw_path}")
    return files


def _bruker_channel_files(files: list[Path], functional_chan: int) -> tuple[list[Path], list[Path]]:
    ch1 = sorted(path for path in files if "Ch1" in path.name)
    ch2 = sorted(path for path in files if "Ch2" in path.name)
    if not ch1 and not ch2:
        return [], []
    if functional_chan == 1:
        functional, alternate = ch1, ch2
    else:
        functional, alternate = ch2, ch1
    if not functional:
        raise FileNotFoundError(f"No Bruker Ch{functional_chan} TIFF files found")
    if alternate and len(functional) != len(alternate):
        raise ValueError(
            f"Bruker channel file count mismatch: functional={len(functional)}, alternate={len(alternate)}"
        )
    return functional, alternate


def _prebuild_from_manifest(
    manifest: Path,
    index: int,
    raw_path: Path,
    workers: int,
    batch_size: int,
    force: bool,
) -> Path:
    data, session = _load_manifest(manifest, index)
    session = dict(session)
    session["raw_path"] = str(raw_path)

    ops, db = _processing_ops(data, session)
    settings, new_db = _suite2p_v1_settings_db(ops, db)
    if int(new_db.get("nplanes", 1)) != 1:
        raise ValueError("Parallel prebuild helper currently supports nplanes=1 only")
    nchannels = int(new_db.get("nchannels", 1))
    if nchannels not in (1, 2):
        raise ValueError(f"Parallel prebuild helper supports nchannels 1 or 2, got {nchannels}")
    functional_chan = int(new_db.get("functional_chan", 1))
    if functional_chan > nchannels:
        raise ValueError(f"functional_chan {functional_chan} is not available with nchannels={nchannels}")

    output_path = Path(session["output_path"])
    suite2p_root = output_path / "suite2p"
    plane0 = suite2p_root / "plane0"
    data_bin = plane0 / "data.bin"
    data_chan2_bin = plane0 / "data_chan2.bin" if nchannels > 1 else None
    if data_bin.exists() and not force:
        raise FileExistsError(f"{data_bin} already exists; pass --force to replace it")
    if data_chan2_bin is not None and data_chan2_bin.exists() and not force:
        raise FileExistsError(f"{data_chan2_bin} already exists; pass --force to replace it")

    if force:
        for path in (data_bin, data_chan2_bin, plane0 / "db.npy", plane0 / "settings.npy"):
            if path is None:
                continue
            if path.exists():
                path.unlink()
        chunk_dir = output_path / ".parallel_tiff_chunks"
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)

    files = _staged_tiffs(raw_path)
    bruker_functional_files: list[Path] = []
    bruker_alternate_files: list[Path] = []
    if str(new_db.get("input_format", "")).lower() == "bruker" or any(
        "Ch1" in path.name or "Ch2" in path.name for path in files
    ):
        bruker_functional_files, bruker_alternate_files = _bruker_channel_files(files, functional_chan)
        if nchannels > 1 and not bruker_alternate_files:
            nchannels = 1
            new_db["nchannels"] = 1
            data_chan2_bin = None
    new_db["data_path"] = [str(raw_path)]
    new_db["file_list"] = [str(path) for path in files]
    frame_count_files = bruker_functional_files if bruker_functional_files else files
    new_db["first_files"] = np.asarray([True] + [False] * (len(frame_count_files) - 1), dtype=bool)
    new_db["fast_disk"] = str(output_path)
    new_db["save_path0"] = str(output_path)
    new_db["save_folder"] = "suite2p"
    new_db["keep_movie_raw"] = False

    from suite2p import io

    dbs = io.init_dbs(new_db)
    if len(dbs) != 1:
        raise RuntimeError(f"Expected one Suite2p db, got {len(dbs)}")
    db0 = dbs[0]
    plane0.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_path / ".parallel_tiff_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if bruker_functional_files:
        tasks = [
            (
                str(path),
                str(bruker_alternate_files[i]) if nchannels > 1 and bruker_alternate_files else None,
                str(chunk_dir / f"{i:04d}_{path.stem}.bin"),
                str(chunk_dir / f"{i:04d}_{path.stem}_chan2.bin") if nchannels > 1 else None,
                int(batch_size),
            )
            for i, path in enumerate(bruker_functional_files)
        ]
        worker = _read_bruker_pair_to_chunk
    else:
        tasks = [
            (
                str(path),
                str(chunk_dir / f"{i:04d}_{path.stem}.bin"),
                str(chunk_dir / f"{i:04d}_{path.stem}_chan2.bin") if nchannels > 1 else None,
                int(batch_size),
                nchannels,
                functional_chan,
            )
            for i, path in enumerate(files)
        ]
        worker = _read_file_to_chunk
    channel_note = f" and {data_chan2_bin}" if data_chan2_bin is not None else ""
    print(f"Prebuilding {data_bin}{channel_note} from {len(files)} TIFFs with {workers} workers")
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"  read {Path(result['source']).name}: {result['nframes']} frames per Suite2p channel")
            results.append(result)

    by_source = {result["source"]: result for result in results}
    ordered = [by_source[str(path)] for path in frame_count_files]
    ly_values = {result["Ly"] for result in ordered}
    lx_values = {result["Lx"] for result in ordered}
    if len(ly_values) != 1 or len(lx_values) != 1:
        raise ValueError(f"TIFF shapes differ: Ly={sorted(ly_values)}, Lx={sorted(lx_values)}")

    total_frames = int(sum(result["nframes"] for result in ordered))
    mean_sum = np.zeros((ordered[0]["Ly"], ordered[0]["Lx"]), dtype=np.float64)
    mean_chan2_sum = np.zeros((ordered[0]["Ly"], ordered[0]["Lx"]), dtype=np.float64) if nchannels > 1 else None
    with data_bin.open("wb") as output:
        for result in ordered:
            mean_sum += result["mean_sum"]
            with Path(result["chunk"]).open("rb") as chunk:
                shutil.copyfileobj(chunk, output, length=1024 * 1024 * 64)
    if data_chan2_bin is not None and mean_chan2_sum is not None:
        with data_chan2_bin.open("wb") as output:
            for result in ordered:
                if not result["chunk_chan2"]:
                    raise RuntimeError(f"Missing channel-2 chunk for {result['source']}")
                mean_chan2_sum += result["mean_chan2_sum"]
                with Path(result["chunk_chan2"]).open("rb") as chunk:
                    shutil.copyfileobj(chunk, output, length=1024 * 1024 * 64)

    db0["Ly"] = int(ordered[0]["Ly"])
    db0["Lx"] = int(ordered[0]["Lx"])
    db0["nframes"] = total_frames
    db0["frames_per_file"] = np.asarray([result["nframes"] for result in ordered], dtype=int)
    db0["frames_per_folder"] = np.asarray([total_frames], dtype=int)
    db0["meanImg"] = mean_sum / total_frames
    db0["reg_file"] = str(data_bin)
    db0["data_path"] = [str(raw_path)]
    db0["file_list"] = [str(path) for path in files]
    db0["first_files"] = new_db["first_files"]
    db0["fast_disk"] = str(plane0)
    db0["save_path"] = str(plane0)
    db0["save_path0"] = str(output_path)
    db0["db_path"] = str(plane0 / "db.npy")
    db0["settings_path"] = str(plane0 / "settings.npy")
    if data_chan2_bin is not None and mean_chan2_sum is not None:
        db0["reg_file_chan2"] = str(data_chan2_bin)
        db0["meanImg_chan2"] = mean_chan2_sum / total_frames

    np.save(plane0 / "db.npy", db0)
    np.save(plane0 / "settings.npy", settings)
    np.save(suite2p_root / "db.npy", new_db)
    np.save(suite2p_root / "settings.npy", settings)
    print(f"Wrote {total_frames} frames to {data_bin}")
    print(f"Wrote {plane0 / 'db.npy'}")
    return data_bin


def prebuild(
    processed_session: Path | str,
    raw_path: Path | str,
    output_root: Path | str | None = None,
    *,
    workers: int = 8,
    batch_size: int = 5000,
    force: bool = True,
    target_structure: str = "dendrite",
    session_name: str | None = None,
) -> Path:
    """Prebuild Suite2p ``data.bin`` without requiring callers to manage manifests.

    If ``processed_session`` belongs to a newer pipeline run, the existing
    pipeline metadata is reused. Otherwise temporary metadata is generated from
    ``suite2p/plane0/ops.npy`` and used internally.
    """

    manifest, index = _manifest_for_processed_session(
        processed_session,
        output_root,
        raw_path=raw_path,
        target_structure=target_structure,
        session_name=session_name,
    )
    return _prebuild_from_manifest(
        manifest=manifest,
        index=index,
        raw_path=Path(raw_path).expanduser().resolve(),
        workers=workers,
        batch_size=batch_size,
        force=force,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-session",
        required=True,
        type=Path,
        help="Processed Suite2p session directory. New pipeline metadata is reused; old outputs use suite2p/plane0/ops.npy.",
    )
    parser.add_argument("--raw-path", required=True, type=Path, help="Staged raw TIFF directory on scratch/local disk.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root for data.bin. If omitted, uses the existing pipeline output or the processed session parent.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--target-structure", default="dendrite", help="Target structure config name for old-output metadata.")
    parser.add_argument("--session-name", default=None, help="Output session name. Default: processed session directory name.")
    args = parser.parse_args()
    data_bin = prebuild(
        args.processed_session,
        args.raw_path,
        args.output_root,
        workers=args.workers,
        batch_size=args.batch_size,
        force=args.force,
        target_structure=args.target_structure,
        session_name=args.session_name,
    )
    print(data_bin)


if __name__ == "__main__":
    main()
