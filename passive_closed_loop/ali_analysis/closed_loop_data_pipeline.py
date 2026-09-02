"""
Data-loading pipeline for closed-loop two-photon sessions.

This module provides a clean first step for deeper analysis. It reads the
session directory and organizes the data into typed containers:

* stimulus/task CSVs
* Bonsai logger CSVs
* Suite2p NumPy arrays
* QC dF/F, smoothed, and spike HDF5 traces
* behavior/motion offset HDF5 data
* raw voltage HDF5 streams

Large arrays are handled carefully:

* ``.npy`` arrays are memory-mapped when possible.
* HDF5 datasets are represented as ``H5DatasetRef`` objects by default.
  Use ``ref.read()`` or ``ref.read(selection)`` to load the full array or a
  slice only when needed.
* The very large Prairie voltage CSV is not loaded by default. The pipeline
  records its path and header; use ``read_voltage_csv_sample`` for quick checks.

Typical notebook usage:

    from pathlib import Path
    import sys

    code_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/Code/2p")
    data_dir = Path("/home/ihsan/Desktop/data/Georgia_Tech/Closed_loop/2p_Data/VW01_20260520_Closed_Loop_test-1556")

    sys.path.insert(0, str(code_dir))
    from ali_analysis.closed_loop_data_pipeline import load_closed_loop_session

    session = load_closed_loop_session(data_dir)
    session.summary()

    dff = session.two_photon.qc_h5["dff"].read()
    voltage_time = session.voltages.raw_h5["raw/vol_time"].read(slice(0, 1000))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd


TASK_CSV = "sensorimotor_mismatch_example.csv"
BONSAI_OUTPUT_CSV = "orientations_orientations0.csv"
LOGGER_CSV = "orientations_logger.csv"
MOVE_OFFSET_H5 = "move_offset.h5"
RAW_VOLTAGES_H5 = "raw_voltages.h5"
QC_DIR = "qc_results"
SUITE2P_PLANE0_DIR = "suite2p/plane0"


STIMULUS_RENAME_TO_CANONICAL = {
    "BlockNumber": "Block_Number",
    "BlockLabel": "Block_Label",
    "TrialNumber": "Trial_Number",
    "SequenceNumber": "Sequence_Number",
    "TrialInSequence": "Trial_In_Sequence",
    "SpatialFrequency": "Spatial_Frequency",
    "TemporalFrequency": "Temporal_Frequency",
    "TrialType": "Trial_Type",
    "BlockType": "Block_Type",
}

STIMULUS_NUMERIC_COLS = [
    "Block_Number",
    "Block_Duration_Minutes",
    "Trial_Number",
    "Sequence_Number",
    "Trial_In_Sequence",
    "Contrast",
    "Delay",
    "DiameterX",
    "DiameterY",
    "Duration",
    "Orientation",
    "Spatial_Frequency",
    "Temporal_Frequency",
    "X",
    "Y",
]


@dataclass(frozen=True)
class H5DatasetRef:
    """Reference to one HDF5 dataset.

    The file is opened only inside ``read``. This keeps the session object safe
    to pass around without holding many open HDF5 file handles.
    """

    path: Path
    key: str
    shape: tuple[int, ...]
    dtype: str
    chunks: Optional[tuple[int, ...]] = None
    compression: Optional[str] = None

    def read(self, selection: Any = None) -> np.ndarray:
        """Read the full dataset or a slice.

        Parameters
        ----------
        selection
            Any valid h5py selection. Examples:
            ``slice(0, 1000)``, ``np.s_[0:10, :]``, or ``np.s_[:, 0:1000]``.
            If omitted, the full dataset is loaded into memory.
        """
        with h5py.File(self.path, "r") as h5:
            dataset = h5[self.key]
            if selection is None:
                return dataset[()]
            return dataset[selection]

    def head(self, n: int = 10) -> np.ndarray:
        """Read the first ``n`` samples along the first axis."""
        if not self.shape:
            return self.read()
        selection = (slice(0, n),) + tuple(slice(None) for _ in self.shape[1:])
        return self.read(selection)


@dataclass(frozen=True)
class NpyArrayRef:
    """Loaded or memory-mapped NumPy array with basic metadata."""

    path: Path
    array: np.ndarray
    mmap_mode: Optional[str]
    loaded_with_pickle: bool

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    def item_if_scalar_object(self) -> Any:
        """Return ``array.item()`` for scalar object arrays such as Suite2p ops."""
        if self.array.dtype == object and self.array.shape == ():
            return self.array.item()
        return self.array


@dataclass
class StimulusData:
    """Stimulus CSVs and Bonsai event logger tables."""

    task: Optional[pd.DataFrame] = None
    bonsai_output: Optional[pd.DataFrame] = None
    logger: Optional[pd.DataFrame] = None
    logger_timing: Optional[pd.DataFrame] = None


@dataclass
class TwoPhotonData:
    """Two-photon traces, Suite2p outputs, and QC outputs."""

    qc_h5: dict[str, H5DatasetRef] = field(default_factory=dict)
    qc_npy: dict[str, NpyArrayRef] = field(default_factory=dict)
    suite2p: dict[str, NpyArrayRef] = field(default_factory=dict)
    root_npy: dict[str, NpyArrayRef] = field(default_factory=dict)

    @property
    def dff(self) -> Optional[H5DatasetRef]:
        """Shortcut for the QC dF/F HDF5 dataset."""
        return self.qc_h5.get("dff")

    @property
    def fluo(self) -> Optional[np.ndarray]:
        """Shortcut for QC fluorescence traces, usually memory-mapped."""
        ref = self.qc_npy.get("fluo")
        return None if ref is None else ref.array


@dataclass
class BehaviorData:
    """Behavior-related logs and motion offsets."""

    move_offset: dict[str, H5DatasetRef] = field(default_factory=dict)
    logger: Optional[pd.DataFrame] = None
    logger_timing: Optional[pd.DataFrame] = None


@dataclass
class VoltageData:
    """Raw voltage streams and the large Prairie voltage CSV metadata."""

    raw_h5: dict[str, H5DatasetRef] = field(default_factory=dict)
    voltage_csv_path: Optional[Path] = None
    voltage_csv_columns: list[str] = field(default_factory=list)
    voltage_csv_sample: Optional[pd.DataFrame] = None


@dataclass
class ClosedLoopSession:
    """All loaded session data organized by analysis domain."""

    data_dir: Path
    manifest: pd.DataFrame
    stimulus: StimulusData
    two_photon: TwoPhotonData
    behavior: BehaviorData
    voltages: VoltageData

    def summary(self) -> pd.DataFrame:
        """Return a compact summary of the loaded session components."""
        rows = [
            {
                "component": "task_csv",
                "loaded": self.stimulus.task is not None,
                "shape_or_count": None if self.stimulus.task is None else str(self.stimulus.task.shape),
            },
            {
                "component": "bonsai_output_csv",
                "loaded": self.stimulus.bonsai_output is not None,
                "shape_or_count": None
                if self.stimulus.bonsai_output is None
                else str(self.stimulus.bonsai_output.shape),
            },
            {
                "component": "logger_csv",
                "loaded": self.stimulus.logger is not None,
                "shape_or_count": None if self.stimulus.logger is None else str(self.stimulus.logger.shape),
            },
            {
                "component": "qc_h5_datasets",
                "loaded": bool(self.two_photon.qc_h5),
                "shape_or_count": len(self.two_photon.qc_h5),
            },
            {
                "component": "qc_npy_arrays",
                "loaded": bool(self.two_photon.qc_npy),
                "shape_or_count": len(self.two_photon.qc_npy),
            },
            {
                "component": "suite2p_arrays",
                "loaded": bool(self.two_photon.suite2p),
                "shape_or_count": len(self.two_photon.suite2p),
            },
            {
                "component": "move_offset_h5",
                "loaded": bool(self.behavior.move_offset),
                "shape_or_count": len(self.behavior.move_offset),
            },
            {
                "component": "raw_voltage_h5",
                "loaded": bool(self.voltages.raw_h5),
                "shape_or_count": len(self.voltages.raw_h5),
            },
            {
                "component": "voltage_csv",
                "loaded": self.voltages.voltage_csv_path is not None,
                "shape_or_count": len(self.voltages.voltage_csv_columns),
            },
        ]
        return pd.DataFrame(rows)


def build_file_manifest(data_dir: str | Path) -> pd.DataFrame:
    """List files in the session directory with relative paths and sizes."""
    data_dir = Path(data_dir).expanduser().resolve()
    rows = []
    for path in sorted(p for p in data_dir.rglob("*") if p.is_file()):
        rows.append(
            {
                "relative_path": str(path.relative_to(data_dir)),
                "path": path,
                "size_bytes": path.stat().st_size,
                "size_mb": path.stat().st_size / 1024**2,
            }
        )
    return pd.DataFrame(rows)


def discover_h5_datasets(path: str | Path) -> dict[str, H5DatasetRef]:
    """Return references to all datasets inside one HDF5 file."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {}

    refs: dict[str, H5DatasetRef] = {}
    with h5py.File(path, "r") as h5:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                refs[name] = H5DatasetRef(
                    path=path,
                    key=name,
                    shape=tuple(obj.shape),
                    dtype=str(obj.dtype),
                    chunks=obj.chunks,
                    compression=obj.compression,
                )

        h5.visititems(visitor)
    return refs


def load_npy_array(path: str | Path, mmap_mode: Optional[str] = "r") -> Optional[NpyArrayRef]:
    """Load a NumPy array, memory-mapping when possible.

    Object arrays cannot be memory-mapped. When that happens, the array is
    loaded with ``allow_pickle=True``. This is required for Suite2p ``ops.npy``
    and ``stat.npy`` files.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return None

    try:
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        return NpyArrayRef(path=path, array=array, mmap_mode=mmap_mode, loaded_with_pickle=False)
    except ValueError:
        array = np.load(path, allow_pickle=True)
        return NpyArrayRef(path=path, array=array, mmap_mode=None, loaded_with_pickle=True)


def standardize_stimulus_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize task/Bonsai CSV column names and add timing/orientation columns."""
    df = df.rename(columns=STIMULUS_RENAME_TO_CANONICAL).copy()

    for col in STIMULUS_NUMERIC_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Block_Label", "Trial_Type", "Block_Type"]:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].astype(str)

    df["Duration_sec"] = df["Duration"].fillna(0)
    df["Csv_Time_sec"] = df["Duration_sec"].shift(fill_value=0).cumsum()
    df["Csv_End_sec"] = df["Csv_Time_sec"] + df["Duration_sec"]

    finite_oris = df["Orientation"].dropna().abs()
    orientation_in_radians = finite_oris.max() <= (2 * np.pi + 1e-6) if not finite_oris.empty else False
    df["Orientation_Deg"] = np.rad2deg(df["Orientation"]) if orientation_in_radians else df["Orientation"]
    df["Orientation_Deg"] = df["Orientation_Deg"].round(6)
    df["Orientation_Units_Source"] = "radians converted to degrees" if orientation_in_radians else "degrees"

    return df


def load_logger_timing(logger: pd.DataFrame) -> pd.DataFrame:
    """Extract StimStart/StimEnd timing from the Bonsai logger table."""
    values = logger["Value"].astype(str)

    starts = logger.loc[values.str.startswith("StimStart-"), ["Frame", "Timestamp", "Value"]].copy()
    starts["Id"] = starts["Value"].str.replace("StimStart-", "", regex=False)
    starts = starts.rename(columns={"Frame": "Logger_Start_Frame", "Timestamp": "Logger_Start_sec_abs"})

    ends = logger.loc[values.str.startswith("StimEnd-"), ["Frame", "Timestamp", "Value"]].copy()
    ends["Id"] = ends["Value"].str.replace("StimEnd-", "", regex=False)
    ends = ends.rename(columns={"Frame": "Logger_End_Frame", "Timestamp": "Logger_End_sec_abs"})

    timing = starts[["Id", "Logger_Start_Frame", "Logger_Start_sec_abs"]].merge(
        ends[["Id", "Logger_End_Frame", "Logger_End_sec_abs"]], on="Id", how="outer"
    )
    timing["Logger_Duration_sec"] = timing["Logger_End_sec_abs"] - timing["Logger_Start_sec_abs"]
    return timing


def attach_logger_timing(stimulus_df: pd.DataFrame, logger_timing: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Attach logger timing to a stimulus table when stimulus IDs are available."""
    if logger_timing is None or "Id" not in stimulus_df.columns:
        return stimulus_df

    df = stimulus_df.merge(logger_timing, on="Id", how="left")
    has_logger = df["Logger_Start_sec_abs"].notna() & df["Logger_End_sec_abs"].notna()
    if has_logger.any():
        t0 = df.loc[has_logger, "Logger_Start_sec_abs"].min()
        df["Logger_Time_sec"] = np.nan
        df["Logger_End_sec"] = np.nan
        df.loc[has_logger, "Logger_Time_sec"] = df.loc[has_logger, "Logger_Start_sec_abs"] - t0
        df.loc[has_logger, "Logger_End_sec"] = df.loc[has_logger, "Logger_End_sec_abs"] - t0
    return df


def load_stimulus_data(data_dir: str | Path, load_logger: bool = True) -> StimulusData:
    """Read task CSV, Bonsai output CSV, and Bonsai logger CSV."""
    data_dir = Path(data_dir).expanduser().resolve()

    task_path = data_dir / TASK_CSV
    bonsai_path = data_dir / BONSAI_OUTPUT_CSV
    logger_path = data_dir / LOGGER_CSV

    logger = pd.read_csv(logger_path) if load_logger and logger_path.exists() else None
    logger_timing = load_logger_timing(logger) if logger is not None else None

    task = standardize_stimulus_table(pd.read_csv(task_path)) if task_path.exists() else None
    bonsai_output = standardize_stimulus_table(pd.read_csv(bonsai_path)) if bonsai_path.exists() else None
    bonsai_output = attach_logger_timing(bonsai_output, logger_timing) if bonsai_output is not None else None

    return StimulusData(
        task=task,
        bonsai_output=bonsai_output,
        logger=logger,
        logger_timing=logger_timing,
    )


def load_two_photon_data(
    data_dir: str | Path,
    include_qc: bool = True,
    include_suite2p: bool = True,
    npy_mmap_mode: Optional[str] = "r",
) -> TwoPhotonData:
    """Read two-photon QC outputs and Suite2p arrays.

    HDF5 trace files are returned as ``H5DatasetRef`` objects. NumPy arrays are
    memory-mapped where possible.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    data = TwoPhotonData()

    for rel in ["ops.npy", "masks.npy"]:
        ref = load_npy_array(data_dir / rel, mmap_mode=npy_mmap_mode)
        if ref is not None:
            data.root_npy[Path(rel).stem] = ref

    if include_qc:
        qc_dir = data_dir / QC_DIR
        for rel in ["dff.h5", "smoothed.h5", "spikes.h5"]:
            for key, ref in discover_h5_datasets(qc_dir / rel).items():
                data.qc_h5[key] = ref

        for rel in ["fluo.npy", "neuropil.npy", "masks.npy", "stat.npy"]:
            ref = load_npy_array(qc_dir / rel, mmap_mode=npy_mmap_mode)
            if ref is not None:
                data.qc_npy[Path(rel).stem] = ref

    if include_suite2p:
        suite2p_dir = data_dir / SUITE2P_PLANE0_DIR
        for path in sorted(suite2p_dir.glob("*.npy")):
            ref = load_npy_array(path, mmap_mode=npy_mmap_mode)
            if ref is not None:
                data.suite2p[path.stem] = ref

    return data


def load_behavior_data(data_dir: str | Path, stimulus: Optional[StimulusData] = None) -> BehaviorData:
    """Read behavior-related data: movement offsets and logger-derived events."""
    data_dir = Path(data_dir).expanduser().resolve()
    move_offset = discover_h5_datasets(data_dir / MOVE_OFFSET_H5)

    return BehaviorData(
        move_offset=move_offset,
        logger=None if stimulus is None else stimulus.logger,
        logger_timing=None if stimulus is None else stimulus.logger_timing,
    )


def find_voltage_csv(data_dir: str | Path) -> Optional[Path]:
    """Find the Prairie voltage recording CSV in the session directory."""
    data_dir = Path(data_dir).expanduser().resolve()
    matches = sorted(data_dir.glob("*VoltageRecording*.csv"))
    return matches[0] if matches else None


def read_voltage_csv_header(path: str | Path) -> list[str]:
    """Read only the header of the large voltage CSV."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return []
    return pd.read_csv(path, nrows=0).columns.tolist()


def read_voltage_csv_sample(path: str | Path, nrows: int = 1000) -> pd.DataFrame:
    """Read a small sample from the large voltage CSV for inspection."""
    path = Path(path).expanduser().resolve()
    return pd.read_csv(path, nrows=nrows)


def load_voltage_data(
    data_dir: str | Path,
    load_csv_sample_rows: int = 0,
) -> VoltageData:
    """Read raw voltage HDF5 references and optional voltage CSV metadata/sample."""
    data_dir = Path(data_dir).expanduser().resolve()
    raw_h5 = discover_h5_datasets(data_dir / RAW_VOLTAGES_H5)

    voltage_csv_path = find_voltage_csv(data_dir)
    voltage_csv_columns = read_voltage_csv_header(voltage_csv_path) if voltage_csv_path is not None else []
    voltage_csv_sample = (
        read_voltage_csv_sample(voltage_csv_path, nrows=load_csv_sample_rows)
        if voltage_csv_path is not None and load_csv_sample_rows > 0
        else None
    )

    return VoltageData(
        raw_h5=raw_h5,
        voltage_csv_path=voltage_csv_path,
        voltage_csv_columns=voltage_csv_columns,
        voltage_csv_sample=voltage_csv_sample,
    )


def h5_dataset_summary(refs: dict[str, H5DatasetRef]) -> pd.DataFrame:
    """Summarize a dictionary of HDF5 dataset references."""
    rows = []
    for name, ref in refs.items():
        rows.append(
            {
                "name": name,
                "path": ref.path,
                "key": ref.key,
                "shape": ref.shape,
                "dtype": ref.dtype,
                "chunks": ref.chunks,
                "compression": ref.compression,
            }
        )
    return pd.DataFrame(rows)


def npy_array_summary(refs: dict[str, NpyArrayRef]) -> pd.DataFrame:
    """Summarize a dictionary of NumPy array references."""
    rows = []
    for name, ref in refs.items():
        rows.append(
            {
                "name": name,
                "path": ref.path,
                "shape": ref.shape,
                "dtype": str(ref.dtype),
                "mmap_mode": ref.mmap_mode,
                "loaded_with_pickle": ref.loaded_with_pickle,
            }
        )
    return pd.DataFrame(rows)


def load_closed_loop_session(
    data_dir: str | Path,
    load_logger: bool = True,
    include_qc: bool = True,
    include_suite2p: bool = True,
    include_voltages: bool = True,
    npy_mmap_mode: Optional[str] = "r",
    voltage_csv_sample_rows: int = 0,
) -> ClosedLoopSession:
    """Load a closed-loop session into organized analysis containers.

    Parameters
    ----------
    data_dir
        Session directory containing CSV, HDF5, and NumPy files.
    load_logger
        Read ``orientations_logger.csv`` and extract StimStart/StimEnd timing.
    include_qc
        Include QC trace files from ``qc_results``.
    include_suite2p
        Include Suite2p ``plane0`` arrays.
    include_voltages
        Include raw voltage HDF5 references and voltage CSV metadata.
    npy_mmap_mode
        Mode passed to ``np.load`` for normal numeric arrays. Use ``"r"`` for
        memory mapping, or ``None`` to load arrays into memory.
    voltage_csv_sample_rows
        Number of rows to read from the large voltage CSV. Default is 0, which
        reads only the header to avoid loading the 1.4 GB CSV.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Session directory does not exist: {data_dir}")

    manifest = build_file_manifest(data_dir)
    stimulus = load_stimulus_data(data_dir, load_logger=load_logger)
    two_photon = load_two_photon_data(
        data_dir,
        include_qc=include_qc,
        include_suite2p=include_suite2p,
        npy_mmap_mode=npy_mmap_mode,
    )
    behavior = load_behavior_data(data_dir, stimulus=stimulus)
    voltages = load_voltage_data(data_dir, load_csv_sample_rows=voltage_csv_sample_rows) if include_voltages else VoltageData()

    return ClosedLoopSession(
        data_dir=data_dir,
        manifest=manifest,
        stimulus=stimulus,
        two_photon=two_photon,
        behavior=behavior,
        voltages=voltages,
    )
