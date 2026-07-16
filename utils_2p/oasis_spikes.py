"""Optional OASIS spike inference for preprocessing QC outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from utils_2p.dff_traces import compute_dff


def _load_suite2p_dff(result_dir: Path, ops: dict[str, Any]) -> tuple[np.ndarray, str]:
    suite2p_dir = result_dir / "suite2p" / "plane0"
    fluo_path = suite2p_dir / "F.npy"
    neuropil_path = suite2p_dir / "Fneu.npy"
    if fluo_path.exists() and neuropil_path.exists():
        fluo = np.load(fluo_path, allow_pickle=False)
        neuropil = np.load(neuropil_path, allow_pickle=False)
        signal = fluo.astype(np.float32, copy=False) - float(ops.get("neucoeff", 0.7)) * neuropil.astype(
            np.float32, copy=False
        )
        return compute_dff(signal, normalize=False).astype(np.float32, copy=False), "suite2p/plane0/F.npy,Fneu.npy"

    for path in (result_dir / "dff.h5", result_dir / "qc_results" / "dff.h5"):
        if path.exists():
            with h5py.File(path, "r") as h5:
                if "dff" not in h5:
                    raise KeyError(f"{path} does not contain dataset 'dff'")
                return np.asarray(h5["dff"], dtype=np.float32), str(path.relative_to(result_dir))
    raise FileNotFoundError(
        f"Could not find Suite2p F.npy/Fneu.npy or dff.h5 in {result_dir}"
    )


def run(
    ops: dict[str, Any],
    *,
    tau: float | None = None,
    fs: float | None = None,
    batch_size: int | None = None,
    event_threshold: float = 0.05,
) -> Path:
    """Run Suite2p's OASIS deconvolution on saved dF/F traces.

    The historical postprocessing code in this repository calls
    ``suite2p.extraction.dcnv.oasis(F=dff, batch_size=..., tau=..., fs=...)``.
    This wrapper keeps that convention and writes a compact HDF5 file that the
    HTML reviewer can load for event overlays and residual-noise inspection.
    """

    from suite2p.extraction.dcnv import oasis

    result_dir = Path(os.fspath(ops["save_path0"]))
    dff, source_dff = _load_suite2p_dff(result_dir, ops)
    oasis_tau = float(ops.get("tau", 0.25) if tau is None else tau)
    frame_rate = float(ops.get("fs", 30.0) if fs is None else fs)
    oasis_batch_size = int(ops.get("batch_size", 500) if batch_size is None else batch_size)

    print("===================================================")
    print("============== OASIS Spike Inference ==============")
    print("===================================================")
    print(f"dF/F shape: {dff.shape}")
    print(f"tau={oasis_tau}, fs={frame_rate}, batch_size={oasis_batch_size}, event_threshold={event_threshold}")

    spikes = oasis(F=dff, batch_size=oasis_batch_size, tau=oasis_tau, fs=frame_rate)
    spikes = np.asarray(spikes, dtype=np.float32)
    event_mask = spikes > float(event_threshold)

    output_path = result_dir / "spikes.h5"
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("spikes", data=spikes, compression="gzip", shuffle=True)
        h5.create_dataset("event_mask", data=event_mask.astype(np.uint8), compression="gzip", shuffle=True)
        h5.attrs["method"] = "suite2p.extraction.dcnv.oasis"
        h5.attrs["tau"] = oasis_tau
        h5.attrs["fs"] = frame_rate
        h5.attrs["batch_size"] = oasis_batch_size
        h5.attrs["event_threshold"] = float(event_threshold)
        h5.attrs["source_dff"] = source_dff
    print(f"Saved OASIS spike inference: {output_path}")
    return output_path
