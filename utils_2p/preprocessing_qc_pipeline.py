#!/usr/bin/env python3
"""Generate and submit linked Slurm preprocessing/QC jobs for 2p sessions.

The default chain is:

    prep (CPU) -> suite2p (high-memory CPU) -> qc (CPU) -> label (GPU, two-channel only)
    -> dff (CPU) -> summary (CPU)

Job files and provenance are written beneath the requested output root. Source
raw sessions are read in place; processed session outputs are not written back
to raw storage.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


QC_PRESETS: dict[str, dict[str, Any]] = {
    "neuron": {
        "range_skew": [-5.0, 5.0],
        "max_connect": 1,
        "range_aspect": [0.0, 5.0],
        "range_footprint": [1.0, 2.0],
        "range_compact": [0.0, 1.06],
        "diameter": 6,
        "source": "2p_post_process_module_202404/run_postprocess.py neurons preset",
    },
    "dendrite": {
        "range_skew": [0.0, 2.0],
        "max_connect": 2,
        "range_aspect": [1.2, 5.0],
        "range_footprint": [1.0, 2.0],
        "range_compact": [1.06, 5.0],
        "diameter": 6,
        "source": "2p_post_process_module_202404/run_postprocess.py dendrites preset",
    },
}

STAGE_ORDER = ("prep", "suite2p", "qc", "label", "dff", "summary")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path | str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(os.environ.get(name, str(default))))).resolve()


@dataclass(frozen=True)
class SessionSpec:
    """Inputs and modality settings for a raw imaging session."""

    raw_path: Path | str
    name: str | None = None
    target_structure: str = "neuron"
    nchannels: int = 2
    functional_chan: int = 2
    denoise: int | None = None
    spatial_scale: int | None = None
    bpod_mat_path: Path | str | None = None
    run_label: bool | None = None
    stages: Sequence[str] | str | None = None

    def normalized(self) -> "SessionSpec":
        raw_path = Path(self.raw_path).expanduser().resolve()
        if not raw_path.is_dir():
            raise FileNotFoundError(f"Raw session directory does not exist: {raw_path}")
        if self.target_structure not in QC_PRESETS:
            raise ValueError(f"target_structure must be one of {sorted(QC_PRESETS)}")
        if self.nchannels not in (1, 2):
            raise ValueError("nchannels must be 1 or 2")
        if self.functional_chan < 1 or self.functional_chan > self.nchannels:
            raise ValueError("functional_chan must identify an available channel")
        bpod = None if self.bpod_mat_path is None else Path(self.bpod_mat_path).expanduser().resolve()
        if bpod is not None and not bpod.is_file():
            raise FileNotFoundError(f"Bpod .mat file does not exist: {bpod}")
        name = (self.name or raw_path.name).rstrip()
        if not name:
            raise ValueError(f"Cannot derive an output session name from {raw_path}")
        run_label = self.run_label if self.run_label is not None else self.nchannels == 2
        stages = _normalize_stages(self.stages, run_label=run_label)
        return SessionSpec(
            raw_path=raw_path,
            name=name,
            target_structure=self.target_structure,
            nchannels=self.nchannels,
            functional_chan=self.functional_chan,
            denoise=self.denoise,
            spatial_scale=self.spatial_scale,
            bpod_mat_path=bpod,
            run_label=run_label,
            stages=stages,
        )


@dataclass(frozen=True)
class SlurmResources:
    """Resource requests for the generated job stages."""

    cpu_cpus: int = 4
    cpu_mem: str = "32G"
    cpu_time: str = "04:00:00"
    suite2p_cpus: int = 8
    suite2p_mem: str = "192G"
    suite2p_time: str = "08:00:00"
    gpu_cpus: int = 8
    gpu_mem: str = "192G"
    gpu_time: str = "08:00:00"
    gpu_gres: str = "gpu:1"
    summary_mem: str = "24G"
    summary_time: str = "02:00:00"


@dataclass(frozen=True)
class PipelineConfig:
    """Configurable software, user, and Slurm settings."""

    repo_root: Path | str = ""
    processing_root: Path | str = ""
    postprocess_root: Path | str = ""
    python_bin: Path | str = ""
    account: str = ""
    username: str = ""
    mail_user: str | None = None
    numba_cache_dir: Path | str = ""
    qos_cpu: str = ""
    qos_gpu: str = ""
    partition_cpu: str | None = None
    partition_gpu: str | None = None

    def normalized(self, output_root: Path) -> "PipelineConfig":
        repo = Path(self.repo_root).expanduser().resolve() if self.repo_root else _repo_root()
        processing = (
            Path(self.processing_root).expanduser().resolve()
            if self.processing_root
            else repo / "2p_processing_pipeline_202401"
        )
        postprocess = (
            Path(self.postprocess_root).expanduser().resolve()
            if self.postprocess_root
            else repo / "2p_post_process_module_202404"
        )
        configured_python = self.python_bin or os.environ.get("TWO_P_PYTHON")
        if configured_python:
            python_bin = Path(configured_python).expanduser().resolve()
        elif importlib.util.find_spec("suite2p") is not None:
            python_bin = Path(sys.executable).resolve()
        else:
            raise ValueError(
                "No Suite2p job environment configured. Activate a Suite2p environment, "
                "set TWO_P_PYTHON, or supply PipelineConfig(python_bin=...) / --python-bin."
            )
        if not python_bin.exists():
            raise FileNotFoundError(f"Configured Python executable does not exist: {python_bin}")
        account = self.account or os.environ.get("TWO_P_SLURM_ACCOUNT", "gts-fnajafi3")
        username = self.username or os.environ.get("USER", "unknown")
        mail_user = self.mail_user or os.environ.get("TWO_P_SLURM_MAIL_USER") or None
        qos = os.environ.get("TWO_P_SLURM_QOS", "embers")
        qos_cpu = self.qos_cpu or os.environ.get("TWO_P_SLURM_QOS_CPU") or qos
        qos_gpu = self.qos_gpu or os.environ.get("TWO_P_SLURM_QOS_GPU") or qos
        cache = (
            Path(self.numba_cache_dir).expanduser().resolve()
            if self.numba_cache_dir
            else _env_path("TWO_P_NUMBA_CACHE_DIR", output_root / ".cache" / "numba")
        )
        for required in (
            processing / "config_neuron.json",
            processing / "config_neuron_1chan.json",
            processing / "config_dendrite.json",
            postprocess / "modules" / "QualControlDataIO.py",
            postprocess / "modules" / "LabelExcInh.py",
        ):
            if not required.exists():
                raise FileNotFoundError(f"Required pipeline code is missing: {required}")
        return PipelineConfig(
            repo_root=repo,
            processing_root=processing,
            postprocess_root=postprocess,
            python_bin=python_bin,
            account=account,
            username=username,
            mail_user=mail_user,
            numba_cache_dir=cache,
            qos_cpu=qos_cpu,
            qos_gpu=qos_gpu,
            partition_cpu=self.partition_cpu,
            partition_gpu=self.partition_gpu,
        )


@dataclass(frozen=True)
class GeneratedRun:
    run_dir: Path
    manifest: Path
    submit_script: Path
    stage_scripts: Mapping[str, Path]
    commands: tuple[str, ...]


def _quote(value: Path | str) -> str:
    return shlex.quote(str(value))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_jsonable(value) for value in obj]
    return obj


def _safe_id(name: str, index: int) -> str:
    return f"s{index + 1}_{re.sub(r'[^A-Za-z0-9_]+', '_', name).strip('_') or 'session'}"


def _normalize_stages(stages: Sequence[str] | str | None, *, run_label: bool) -> tuple[str, ...]:
    if stages is None:
        requested = STAGE_ORDER if run_label else tuple(stage for stage in STAGE_ORDER if stage != "label")
    elif isinstance(stages, str):
        requested = tuple(part.strip() for part in stages.split(",") if part.strip())
    else:
        requested = tuple(stages)
    unknown = set(requested) - set(STAGE_ORDER)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}; valid stages are {list(STAGE_ORDER)}")
    if len(set(requested)) != len(requested):
        raise ValueError("Each processing stage may be selected only once")
    if "label" in requested and not run_label:
        raise ValueError("The label stage requires a two-channel session unless run_label=True is specified")
    return tuple(stage for stage in STAGE_ORDER if stage in requested)


def _stage_sequence(session: SessionSpec) -> tuple[str, ...]:
    return tuple(session.stages or ())


def _sbatch_text(
    stage: str,
    run_dir: Path,
    manifest: Path,
    config: PipelineConfig,
    resources: SlurmResources,
) -> str:
    is_gpu = stage == "label"
    if stage == "summary":
        cpus, mem, walltime = resources.cpu_cpus, resources.summary_mem, resources.summary_time
    elif stage == "suite2p":
        cpus, mem, walltime = resources.suite2p_cpus, resources.suite2p_mem, resources.suite2p_time
    elif is_gpu:
        cpus, mem, walltime = resources.gpu_cpus, resources.gpu_mem, resources.gpu_time
    else:
        cpus, mem, walltime = resources.cpu_cpus, resources.cpu_mem, resources.cpu_time
    partition = config.partition_gpu if is_gpu else config.partition_cpu
    qos = config.qos_gpu if is_gpu else config.qos_cpu
    directives = [
        "#!/bin/bash",
        f"#SBATCH --job-name=2p-{stage}",
        f"#SBATCH --account={config.account}",
        f"#SBATCH --qos={qos}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --output={run_dir}/logs/%x_%j.out",
        f"#SBATCH --error={run_dir}/logs/%x_%j.err",
    ]
    if partition:
        directives.append(f"#SBATCH --partition={partition}")
    if is_gpu:
        directives.append(f"#SBATCH --gres={resources.gpu_gres}")
    if config.mail_user:
        directives.extend((f"#SBATCH --mail-user={config.mail_user}", "#SBATCH --mail-type=FAIL"))
    matplotlib_cache_dir = Path(config.numba_cache_dir).parent / "matplotlib"
    body = [
        "",
        "set -euo pipefail",
        "unset LD_PRELOAD || true",
        'export PYTHONNOUSERSITE="1"',
        f"export NUMBA_CACHE_DIR={_quote(config.numba_cache_dir)}",
        f"export MPLCONFIGDIR={_quote(matplotlib_cache_dir)}",
        'export MPLBACKEND="Agg"',
        f"export PYTHONPATH={_quote(config.repo_root)}${{PYTHONPATH:+:${{PYTHONPATH}}}}",
        ': "${SESSION_INDEX:?SESSION_INDEX is required}"',
        f"mkdir -p {_quote(run_dir / 'logs')} {_quote(config.numba_cache_dir)} {_quote(matplotlib_cache_dir)}",
        f"cd {_quote(config.repo_root)}",
        (
            f"{_quote(config.python_bin)} -m utils_2p.preprocessing_qc_pipeline run-stage "
            f"--manifest {_quote(manifest)} --index \"${{SESSION_INDEX}}\" --stage {stage}"
        ),
        "",
    ]
    return "\n".join(directives + body)


def _write_submit_script(
    path: Path, sessions: Sequence[SessionSpec], stage_scripts: Mapping[str, Path]
) -> tuple[str, ...]:
    lines = ["#!/bin/bash", "set -euo pipefail", ""]
    commands: list[str] = []
    for index, session in enumerate(sessions):
        job_id = _safe_id(session.name or "", index)
        prior_var = None
        lines.append(f"# {session.name}")
        for stage in _stage_sequence(session):
            var = f"{stage}_{job_id}"
            dependency = "" if prior_var is None else f"--dependency=afterok:${{{prior_var}}} "
            command = (
                f"{var}=$(sbatch --parsable {dependency}"
                f"--export=ALL,SESSION_INDEX={index} {_quote(stage_scripts[stage])})"
            )
            lines.append(command)
            lines.append(f'echo "{session.name}: {stage} -> ${{{var}}}"')
            commands.append(command)
            prior_var = var
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    path.chmod(0o750)
    return tuple(commands)


def generate_preprocessing_qc_jobs(
    sessions: Sequence[SessionSpec | Path | str],
    output_root: Path | str,
    *,
    config: PipelineConfig | None = None,
    resources: SlurmResources | None = None,
    run_name: str | None = None,
) -> GeneratedRun:
    """Write Slurm job files and a submission script without submitting jobs.

    Bare paths use the default `SessionSpec`: two-channel neuronal recording,
    functional channel 2. Use `SessionSpec` for dendrite or single-channel
    sessions so the QC preset is explicit.
    """

    if not sessions:
        raise ValueError("At least one session is required")
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    normalized_sessions = tuple(
        (session if isinstance(session, SessionSpec) else SessionSpec(session)).normalized()
        for session in sessions
    )
    settings = (config or PipelineConfig()).normalized(output)
    requested = resources or SlurmResources()
    run_label = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output / ".preprocessing_qc_jobs" / f"{run_label}_{settings.username}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()

    session_records = []
    for session in normalized_sessions:
        record = _jsonable(asdict(session))
        record["output_path"] = str(output / (session.name or "session"))
        record["stages"] = list(_stage_sequence(session))
        record["qc_parameters"] = QC_PRESETS[session.target_structure]
        session_records.append(record)
    manifest_data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output),
        "pipeline": _jsonable(asdict(settings)),
        "resources": asdict(requested),
        "sessions": session_records,
    }
    manifest = run_dir / "manifest.json"
    manifest.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="ascii")

    stage_scripts: dict[str, Path] = {}
    selected_stages = tuple(stage for stage in STAGE_ORDER if any(stage in record["stages"] for record in session_records))
    for stage in selected_stages:
        script = run_dir / f"{stage}.sbatch"
        script.write_text(_sbatch_text(stage, run_dir, manifest, settings, requested), encoding="ascii")
        script.chmod(0o750)
        stage_scripts[stage] = script
    submit_script = run_dir / "submit_jobs.sh"
    commands = _write_submit_script(submit_script, normalized_sessions, stage_scripts)
    return GeneratedRun(run_dir, manifest, submit_script, stage_scripts, commands)


def submit_preprocessing_qc_jobs(
    sessions: Sequence[SessionSpec | Path | str],
    output_root: Path | str,
    *,
    config: PipelineConfig | None = None,
    resources: SlurmResources | None = None,
    run_name: str | None = None,
) -> dict[str, dict[str, str]]:
    """Generate and immediately submit isolated dependency chains per session."""

    generated = generate_preprocessing_qc_jobs(
        sessions, output_root, config=config, resources=resources, run_name=run_name
    )
    manifest_data = json.loads(generated.manifest.read_text(encoding="ascii"))
    submitted: dict[str, dict[str, str]] = {}
    for index, session in enumerate(manifest_data["sessions"]):
        jobs: dict[str, str] = {}
        dependency = None
        for stage in session["stages"]:
            command = ["sbatch", "--parsable"]
            if dependency:
                command.append(f"--dependency=afterok:{dependency}")
            command.extend((f"--export=ALL,SESSION_INDEX={index}", str(generated.stage_scripts[stage])))
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            dependency = result.stdout.strip().split(";", 1)[0]
            jobs[stage] = dependency
        submitted[session["name"]] = jobs
    submission_record = generated.run_dir / "submitted_jobs.json"
    submission_record.write_text(json.dumps(submitted, indent=2) + "\n", encoding="ascii")
    return submitted


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _session_runtime(manifest: Path, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads(manifest.read_text(encoding="ascii"))
    sessions = data["sessions"]
    if index < 0 or index >= len(sessions):
        raise IndexError(f"Session index {index} outside manifest range 0..{len(sessions) - 1}")
    return data, sessions[index]


def _processing_ops(data: dict[str, Any], session: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    processing_root = Path(data["pipeline"]["processing_root"])
    target = session["target_structure"]
    config_name = "config_neuron_1chan.json" if target == "neuron" and session["nchannels"] == 1 else f"config_{target}.json"
    config_path = processing_root / config_name
    with config_path.open("r", encoding="utf-8") as handle:
        nested = json.load(handle)
    ops: dict[str, Any] = {}
    for section in nested.values():
        ops.update(section)
    ops.update(
        {
            "data_path": session["raw_path"],
            "save_path0": session["output_path"],
            "nchannels": session["nchannels"],
            "functional_chan": session["functional_chan"],
            "align_by_chan": (
                session["functional_chan"] if session["nchannels"] == 1 else 3 - session["functional_chan"]
            ),
        }
    )
    if session.get("denoise") is not None:
        ops["denoise"] = session["denoise"]
    if session.get("spatial_scale") is not None:
        ops["spatial_scale"] = session["spatial_scale"]
    db = {"data_path": [session["raw_path"]], "save_path0": session["output_path"]}
    return ops, db


def _read_ops(session: dict[str, Any]):
    import numpy as np

    path = Path(session["output_path"]) / "suite2p" / "plane0" / "ops.npy"
    ops = np.load(path, allow_pickle=True).item()
    ops["save_path0"] = session["output_path"]
    return ops


def _write_raw_voltages(raw_path: Path, output_path: Path) -> None:
    import h5py
    import numpy as np
    import pandas as pd

    candidates = sorted(raw_path.glob("*VoltageRecording*.csv"))
    if not candidates:
        print("Valid voltage recordings csv file not found")
        return
    frame = pd.read_csv(candidates[0], engine="python")
    time = frame["Time(ms)"].to_numpy()
    channels = {
        "vol_start": (" Input 0", 1.0),
        "vol_stim_vis": (" Input 1", 1.0),
        "vol_hifi": (" Input 2", 0.5),
        "vol_img": (" Input 3", 1.0),
        "vol_stim_aud": (" Input 4", None),
        "vol_flir": (" Input 5", 1.0),
        "vol_pmt": (" Input 6", 1.0),
        "vol_led": (" Input 7", 1.0),
        "vol_2p_stim": (" Input 8", None),
    }
    with h5py.File(output_path / "raw_voltages.h5", "w") as output:
        raw = output.create_group("raw")
        raw["vol_time"] = time
        for name, (column, threshold) in channels.items():
            values = frame[column].to_numpy() if column in frame.columns else np.zeros_like(time)
            if threshold is not None:
                values = (values > threshold).astype(values.dtype)
            raw[name] = values


def _run_prep(data: dict[str, Any], session: dict[str, Any]) -> None:
    output = Path(session["output_path"])
    output.mkdir(parents=True, exist_ok=True)
    _write_raw_voltages(Path(session["raw_path"]), output)
    candidates = sorted(Path(session["raw_path"]).glob("*.mat"))
    bpod_source = Path(session["bpod_mat_path"]) if session.get("bpod_mat_path") else None
    if bpod_source is None and len(candidates) == 1:
        bpod_source = candidates[0]
    if bpod_source and bpod_source.is_file():
        shutil.copyfile(bpod_source, output / "bpod_session_data.mat")
        print(f"Copied Bpod file from {bpod_source}")
    else:
        print("No unambiguous Bpod .mat file found; continuing without bpod_session_data.mat")
    provenance = output / "preprocessing_pipeline_parameters.json"
    provenance.write_text(json.dumps(session, indent=2) + "\n", encoding="ascii")


def _run_suite2p(data: dict[str, Any], session: dict[str, Any]) -> None:
    from suite2p import run_s2p

    ops, db = _processing_ops(data, session)
    run_s2p(ops=ops, db=db)


def _run_qc(data: dict[str, Any], session: dict[str, Any]) -> None:
    import numpy as np

    ops = _read_ops(session)
    qc = _load_module(
        "quality_control",
        Path(data["pipeline"]["postprocess_root"]) / "modules" / "QualControlDataIO.py",
    )
    params = session["qc_parameters"]
    qc.run(
        ops,
        np.asarray(params["range_skew"], dtype="float32"),
        np.asarray(params["max_connect"], dtype="float32"),
        np.asarray(params["range_aspect"], dtype="float32"),
        np.asarray(params["range_compact"], dtype="float32"),
        np.asarray(params["range_footprint"], dtype="float32"),
    )
    path = Path(session["output_path"]) / "qc_results" / "qc_parameters.json"
    path.write_text(json.dumps(params, indent=2) + "\n", encoding="ascii")


def _patch_cellpose_api() -> None:
    import inspect
    from cellpose import io, models

    original = io.masks_flows_to_seg
    accepted = set(inspect.signature(original).parameters)

    def compatible_seg(*args, **kwargs):
        return original(*args, **{key: value for key, value in kwargs.items() if key in accepted})

    io.masks_flows_to_seg = compatible_seg
    if not hasattr(models, "Cellpose") and hasattr(models, "CellposeModel"):
        class CompatibleCellpose:
            def __init__(self, model_type=None, gpu=False, **kwargs):
                self.model = models.CellposeModel(gpu=gpu, model_type=model_type, **kwargs)

            def eval(self, *args, **kwargs):
                result = self.model.eval(*args, **kwargs)
                if isinstance(result, tuple) and len(result) == 3:
                    masks, flows, styles = result
                    return masks, flows, styles, kwargs.get("diameter")
                return result

        models.Cellpose = CompatibleCellpose


def _run_label(data: dict[str, Any], session: dict[str, Any]) -> None:
    _patch_cellpose_api()
    ops = _read_ops(session)
    label = _load_module(
        "label_exc_inh",
        Path(data["pipeline"]["postprocess_root"]) / "modules" / "LabelExcInh.py",
    )
    cellpose_constructor = label.models.Cellpose

    def gpu_cellpose(*args, **kwargs):
        kwargs.setdefault("gpu", True)
        return cellpose_constructor(*args, **kwargs)

    label.models.Cellpose = gpu_cellpose
    label.run(ops, int(session["qc_parameters"]["diameter"]))


def _run_dff(data: dict[str, Any], session: dict[str, Any]) -> None:
    from utils_2p import dff_traces

    ops = _read_ops(session)
    dff_traces.run(ops, normalize=False, correct_pmt=False)


def _run_summary(data: dict[str, Any], session: dict[str, Any]) -> None:
    from utils_2p import preprocessing_summary as summary

    pdf, html = summary.create_preprocessing_summary(session["output_path"])
    print(f"Saved preprocessing PDF: {pdf}")
    print(f"Saved interactive HTML: {html}")


def run_stage(manifest: Path | str, index: int, stage: str) -> None:
    data, session = _session_runtime(Path(manifest), index)
    os.environ.setdefault("MPLBACKEND", "Agg")
    cache_dir = Path(data["pipeline"]["numba_cache_dir"])
    matplotlib_cache_dir = cache_dir.parent / "matplotlib"
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    if stage not in session["stages"]:
        print(f"Stage {stage} is disabled for {session['name']}; skipping.")
        return
    handlers = {
        "prep": _run_prep,
        "suite2p": _run_suite2p,
        "qc": _run_qc,
        "label": _run_label,
        "dff": _run_dff,
        "summary": _run_summary,
    }
    handlers[stage](data, session)


def _specs_from_args(args: argparse.Namespace) -> list[SessionSpec]:
    raw_paths: list[str] = list(args.session or [])
    if args.sessions_file:
        raw_paths.extend(
            line.strip()
            for line in Path(args.sessions_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return [
        SessionSpec(
            path,
            target_structure=args.target_structure,
            nchannels=args.nchannels,
            functional_chan=(
                args.functional_chan
                if args.functional_chan is not None
                else (1 if args.nchannels == 1 else 2)
            ),
            denoise=args.denoise,
            spatial_scale=args.spatial_scale,
            run_label=False if args.no_label else None,
            stages=args.stages,
        )
        for path in raw_paths
    ]


def _pipeline_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        repo_root=args.repo_root or "",
        processing_root=args.processing_root or "",
        postprocess_root=args.postprocess_root or "",
        python_bin=args.python_bin or "",
        account=args.account or "",
        username=args.username or "",
        mail_user=args.mail_user,
        numba_cache_dir=args.numba_cache_dir or "",
        qos_cpu=args.qos_cpu or args.qos or "",
        qos_gpu=args.qos_gpu or args.qos or "",
        partition_cpu=args.partition_cpu,
        partition_gpu=args.partition_gpu,
    )


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session", action="append", help="Raw session directory. Repeat for multiple sessions.")
    parser.add_argument("--sessions-file", help="Text file containing one raw session directory per line.")
    parser.add_argument("--output-root", required=True, help="Directory where processed sessions will be written.")
    parser.add_argument("--target-structure", choices=sorted(QC_PRESETS), default="neuron")
    parser.add_argument("--nchannels", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--functional-chan",
        type=int,
        default=None,
        help="Functional channel index. Default: 2 for two-channel or 1 for one-channel sessions.",
    )
    parser.add_argument("--denoise", type=int, choices=(0, 1), default=None)
    parser.add_argument(
        "--spatial-scale",
        type=int,
        default=None,
        help="Override Suite2p spatial scale. Default: preserve target config value.",
    )
    parser.add_argument("--no-label", action="store_true", help="Skip anatomical Cellpose labeling stage.")
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stages to submit in dependency order. Default: full pipeline.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--processing-root", default=None)
    parser.add_argument("--postprocess-root", default=None)
    parser.add_argument("--python-bin", default=None, help="Python executable in a Suite2p-capable environment.")
    parser.add_argument("--account", default=None, help="Slurm account; defaults to TWO_P_SLURM_ACCOUNT or gts-fnajafi3.")
    parser.add_argument("--username", default=None)
    parser.add_argument("--mail-user", default=None)
    parser.add_argument("--numba-cache-dir", default=None)
    parser.add_argument("--qos", default=None, help="Slurm QOS for all stages; defaults to embers.")
    parser.add_argument("--qos-cpu", default=None, help="Slurm QOS for CPU stages; overrides --qos.")
    parser.add_argument("--qos-gpu", default=None, help="Slurm QOS for GPU stages; overrides --qos.")
    parser.add_argument("--partition-cpu", default=None)
    parser.add_argument("--partition-gpu", default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "submit"):
        subparser = commands.add_parser(command)
        _add_generation_args(subparser)
    stage_parser = commands.add_parser("run-stage")
    stage_parser.add_argument("--manifest", required=True, type=Path)
    stage_parser.add_argument("--index", required=True, type=int)
    stage_parser.add_argument("--stage", required=True, choices=STAGE_ORDER)
    args = parser.parse_args()

    if args.command == "run-stage":
        run_stage(args.manifest, args.index, args.stage)
        return
    specs = _specs_from_args(args)
    if not specs:
        parser.error("Supply --session at least once or use --sessions-file")
    config = _pipeline_config_from_args(args)
    if args.command == "generate":
        generated = generate_preprocessing_qc_jobs(
            specs, args.output_root, config=config, run_name=args.run_name
        )
        print(f"Generated job directory: {generated.run_dir}")
        print(f"Submit with: bash {generated.submit_script}")
    else:
        jobs = submit_preprocessing_qc_jobs(
            specs, args.output_root, config=config, run_name=args.run_name
        )
        print(json.dumps(jobs, indent=2))


if __name__ == "__main__":
    main()
