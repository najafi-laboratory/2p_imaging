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
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from utils_2p.preprocessing_qc_pipeline import _processing_ops, _suite2p_v1_settings_db


def _read_file_to_chunk(args: tuple[str, str, int]) -> dict[str, Any]:
    source, chunk_path, batch_size = args
    from suite2p.io.tiff import open_tiff, read_tiff

    source_path = Path(source)
    chunk = Path(chunk_path)
    chunk.parent.mkdir(parents=True, exist_ok=True)

    tif, n_pages = open_tiff(str(source_path), True)
    nframes = 0
    ly = lx = None
    mean_sum = None
    try:
        with chunk.open("wb") as handle:
            ix = 0
            while True:
                frames = read_tiff(str(source_path), tif, n_pages, ix, batch_size, True)
                if frames is None:
                    break
                if frames.ndim != 3:
                    raise ValueError(f"{source_path} produced frames with shape {frames.shape}; expected 3-D")
                if ly is None:
                    ly, lx = int(frames.shape[1]), int(frames.shape[2])
                    mean_sum = np.zeros((ly, lx), dtype=np.float64)
                elif frames.shape[1:] != (ly, lx):
                    raise ValueError(f"{source_path} changed frame shape from {(ly, lx)} to {frames.shape[1:]}")
                handle.write(np.ascontiguousarray(frames, dtype=np.int16).tobytes())
                mean_sum += frames.sum(axis=0, dtype=np.float64)
                nframes += int(frames.shape[0])
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
        "nframes": nframes,
        "Ly": ly,
        "Lx": lx,
        "mean_sum": mean_sum,
    }


def _load_manifest(path: Path, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="ascii"))
    return data, data["sessions"][index]


def _staged_tiffs(raw_path: Path) -> list[Path]:
    files = sorted(raw_path.glob("*.ome.tif")) + sorted(raw_path.glob("*.ome.TIF"))
    if not files:
        files = sorted(raw_path.glob("*.tif")) + sorted(raw_path.glob("*.tiff"))
    if not files:
        raise FileNotFoundError(f"No TIFF files found directly under {raw_path}")
    return files


def prebuild(
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

    if int(session["nchannels"]) != 1:
        raise ValueError("Parallel prebuild helper currently supports nchannels=1 only")

    ops, db = _processing_ops(data, session)
    settings, new_db = _suite2p_v1_settings_db(ops, db)
    if int(new_db.get("nplanes", 1)) != 1:
        raise ValueError("Parallel prebuild helper currently supports nplanes=1 only")
    if int(new_db.get("nchannels", 1)) != 1:
        raise ValueError("Parallel prebuild helper currently supports nchannels=1 only")

    output_path = Path(session["output_path"])
    suite2p_root = output_path / "suite2p"
    plane0 = suite2p_root / "plane0"
    data_bin = plane0 / "data.bin"
    if data_bin.exists() and not force:
        raise FileExistsError(f"{data_bin} already exists; pass --force to replace it")

    if force:
        for path in (data_bin, plane0 / "db.npy", plane0 / "settings.npy"):
            if path.exists():
                path.unlink()
        chunk_dir = output_path / ".parallel_tiff_chunks"
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)

    files = _staged_tiffs(raw_path)
    new_db["data_path"] = [str(raw_path)]
    new_db["file_list"] = [str(path) for path in files]
    new_db["first_files"] = np.asarray([True] + [False] * (len(files) - 1), dtype=bool)
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

    tasks = [
        (str(path), str(chunk_dir / f"{i:04d}_{path.stem}.bin"), int(batch_size))
        for i, path in enumerate(files)
    ]
    print(f"Prebuilding {data_bin} from {len(files)} TIFFs with {workers} workers")
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_read_file_to_chunk, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"  read {Path(result['source']).name}: {result['nframes']} frames")
            results.append(result)

    by_source = {result["source"]: result for result in results}
    ordered = [by_source[str(path)] for path in files]
    ly_values = {result["Ly"] for result in ordered}
    lx_values = {result["Lx"] for result in ordered}
    if len(ly_values) != 1 or len(lx_values) != 1:
        raise ValueError(f"TIFF shapes differ: Ly={sorted(ly_values)}, Lx={sorted(lx_values)}")

    total_frames = int(sum(result["nframes"] for result in ordered))
    mean_sum = np.zeros((ordered[0]["Ly"], ordered[0]["Lx"]), dtype=np.float64)
    with data_bin.open("wb") as output:
        for result in ordered:
            mean_sum += result["mean_sum"]
            with Path(result["chunk"]).open("rb") as chunk:
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

    np.save(plane0 / "db.npy", db0)
    np.save(plane0 / "settings.npy", settings)
    np.save(suite2p_root / "db.npy", new_db)
    np.save(suite2p_root / "settings.npy", settings)
    print(f"Wrote {total_frames} frames to {data_bin}")
    print(f"Wrote {plane0 / 'db.npy'}")
    return data_bin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--raw-path", required=True, type=Path, help="Staged raw TIFF directory on scratch/local disk.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prebuild(args.manifest, args.index, args.raw_path, args.workers, args.batch_size, args.force)


if __name__ == "__main__":
    main()
