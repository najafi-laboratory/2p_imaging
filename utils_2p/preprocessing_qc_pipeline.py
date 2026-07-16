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
import inspect
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
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
    "cerebellum_lax": {
        "range_skew": [0.4, 2.5],
        "max_connect": 30,
        "range_aspect": [0.0, 55.0],
        "range_footprint": [1.0, 2.0],
        "range_compact": [1.0, 5.0],
        "diameter": 6,
        "source": "Joystick/PostProcessing/config.py cerebellum dendrite tuning",
    },
}

SUITE2P_CONFIG_TARGETS = {
    "cerebellum_lax": "dendrite",
}

SUITE2P_SHARED_ENV_ROOT = Path("/storage/project/r-fnajafi3-0/grubin6/shared_envs")
SUITE2P_VERSIONED_PYTHONS = {
    "0.x": SUITE2P_SHARED_ENV_ROOT / "2p_preprocessing_qc_suite2p_0x" / "bin" / "python",
    "1.x": SUITE2P_SHARED_ENV_ROOT / "2p_preprocessing_qc_suite2p_1x" / "bin" / "python",
}

STAGE_ORDER = ("prep", "suite2p", "qc", "label", "dff", "summary")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _suite2p_python_path(version: str) -> Path:
    try:
        return SUITE2P_VERSIONED_PYTHONS[version]
    except KeyError as error:
        raise ValueError(f"suite2p_version must be one of {sorted(SUITE2P_VERSIONED_PYTHONS)}") from error


def _env_path(name: str, default: Path | str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(os.environ.get(name, str(default))))).resolve()


def _detect_imaging_channels(raw_path: Path) -> dict[str, int]:
    """Infer Suite2p channel settings from TIFF names.

    Bruker OME TIFFs in this project are usually split into Ch1 anatomical and
    Ch2 functional files. Some functional-only recordings still use Ch2 file
    names. Suite2p's Bruker reader needs ``functional_chan=2`` for those files
    even when ``nchannels=1`` so the single available channel becomes the
    primary binary stream.
    """

    ch1 = 0
    ch2 = 0
    unlabeled = 0
    for directory, dirnames, filenames in os.walk(raw_path):
        current = Path(directory)
        if len(current.relative_to(raw_path).parts) > 2:
            dirnames[:] = []
            continue
        for filename in filenames:
            if not filename.lower().endswith((".tif", ".tiff")):
                continue
            lower = filename.lower()
            if "ch1" in lower or re.search(r"channel[\s_-]*1", lower):
                ch1 += 1
            elif "ch2" in lower or re.search(r"channel[\s_-]*2", lower):
                ch2 += 1
            else:
                unlabeled += 1
    if ch1 and ch2:
        return {"nchannels": 2, "functional_chan": 2}
    if ch1:
        return {"nchannels": 1, "functional_chan": 1}
    if ch2 or unlabeled:
        return {"nchannels": 1, "functional_chan": 2}
    return {"nchannels": 2, "functional_chan": 2}


@dataclass(frozen=True)
class SessionSpec:
    """Inputs and modality settings for a raw imaging session."""

    raw_path: Path | str
    name: str | None = None
    target_structure: str = "neuron"
    nchannels: int | None = None
    functional_chan: int | None = None
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
        detected = _detect_imaging_channels(raw_path)
        nchannels = self.nchannels if self.nchannels is not None else detected["nchannels"]
        functional_chan = (
            self.functional_chan if self.functional_chan is not None else detected["functional_chan"]
        )
        if nchannels not in (1, 2):
            raise ValueError("nchannels must be 1 or 2")
        if functional_chan not in (1, 2):
            raise ValueError("functional_chan must be 1 or 2")
        if nchannels == 2 and functional_chan > nchannels:
            raise ValueError("functional_chan must identify an available channel")
        bpod = None if self.bpod_mat_path is None else Path(self.bpod_mat_path).expanduser().resolve()
        if bpod is not None and not bpod.is_file():
            raise FileNotFoundError(f"Bpod .mat file does not exist: {bpod}")
        name = (self.name or raw_path.name).rstrip()
        if not name:
            raise ValueError(f"Cannot derive an output session name from {raw_path}")
        run_label = self.run_label if self.run_label is not None else nchannels == 2
        stages = _normalize_stages(self.stages, run_label=run_label)
        return SessionSpec(
            raw_path=raw_path,
            name=name,
            target_structure=self.target_structure,
            nchannels=nchannels,
            functional_chan=functional_chan,
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
    suite2p_version: str = "1.x"
    account: str = ""
    username: str = ""
    mail_user: str | None = None
    numba_cache_dir: Path | str = ""
    fast_disk: Path | str = ""
    qos_cpu: str = ""
    qos_gpu: str = ""
    partition_cpu: str | None = None
    partition_gpu: str | None = None
    suite2p_gpu: bool | None = True
    suite2p_batch_size: int | None = None
    suite2p_binary_batch_size: int | None = 5000
    suite2p_registration_batch_size: int | None = 500
    suite2p_extraction_batch_size: int | None = 500

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
            python_bin = Path(configured_python).expanduser()
        else:
            python_bin = _suite2p_python_path(self.suite2p_version).expanduser()
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
        fast_disk = self.fast_disk or os.environ.get("TWO_P_SUITE2P_FAST_DISK") or "tmp"
        if str(fast_disk).lower() not in {"tmp", "local_tmp", "slurm_tmp"}:
            fast_disk = Path(fast_disk).expanduser()
        suite2p_binary_batch_size = (
            self.suite2p_binary_batch_size if self.suite2p_binary_batch_size is not None else 5000
        )
        suite2p_registration_batch_size = (
            self.suite2p_registration_batch_size
            if self.suite2p_registration_batch_size is not None
            else 500
        )
        suite2p_extraction_batch_size = (
            self.suite2p_extraction_batch_size
            if self.suite2p_extraction_batch_size is not None
            else 500
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
            suite2p_version=self.suite2p_version,
            account=account,
            username=username,
            mail_user=mail_user,
            numba_cache_dir=cache,
            fast_disk=fast_disk or "",
            qos_cpu=qos_cpu,
            qos_gpu=qos_gpu,
            partition_cpu=self.partition_cpu,
            partition_gpu=self.partition_gpu,
            suite2p_gpu=True if self.suite2p_gpu is None else self.suite2p_gpu,
            suite2p_batch_size=self.suite2p_batch_size,
            suite2p_binary_batch_size=suite2p_binary_batch_size,
            suite2p_registration_batch_size=suite2p_registration_batch_size,
            suite2p_extraction_batch_size=suite2p_extraction_batch_size,
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
    is_gpu = stage == "label" or (stage == "suite2p" and config.suite2p_gpu)
    if stage == "summary":
        cpus, mem, walltime = resources.cpu_cpus, resources.summary_mem, resources.summary_time
    elif stage == "suite2p":
        cpus, mem, walltime = resources.suite2p_cpus, resources.suite2p_mem, resources.suite2p_time
    elif is_gpu:
        cpus, mem, walltime = resources.gpu_cpus, resources.gpu_mem, resources.gpu_time
    else:
        cpus, mem, walltime = resources.cpu_cpus, resources.cpu_mem, resources.cpu_time
    partition = config.partition_cpu
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
    ]
    if stage == "suite2p" and config.suite2p_gpu:
        body.append('export TWO_P_SUITE2P_TORCH_DEVICE="cuda"')
    if stage == "suite2p" and str(config.fast_disk).lower() in {"tmp", "local_tmp", "slurm_tmp"}:
        body.append('export TWO_P_SUITE2P_FAST_DISK="${TMPDIR:-/tmp}/suite2p_fast_disk_${SLURM_JOB_ID:-manual}"')
    if stage == "suite2p":
        body.append(f"export HOME={_quote(run_dir / 'home')}")
    if stage == "suite2p" and config.suite2p_gpu:
        gpu_log = run_dir / "logs" / "gpu_${SLURM_JOB_ID:-manual}.log"
        body.extend(
            [
                f'GPU_LOG="{gpu_log}"',
                'if command -v nvidia-smi >/dev/null 2>&1; then',
                '  nvidia-smi > "${GPU_LOG}" || true',
                '  (while true; do date; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits; sleep 60; done >> "${GPU_LOG}" 2>&1) &',
                "  GPU_MONITOR_PID=$!",
                '  trap \'kill "${GPU_MONITOR_PID}" 2>/dev/null || true\' EXIT',
                "fi",
            ]
        )
    body.extend(
        [
            ': "${SESSION_INDEX:?SESSION_INDEX is required}"',
            f"mkdir -p {_quote(run_dir / 'logs')} {_quote(run_dir / 'home')} {_quote(config.numba_cache_dir)} {_quote(matplotlib_cache_dir)}",
            f"cd {_quote(config.repo_root)}",
            (
                f"{_quote(config.python_bin)} -m utils_2p.preprocessing_qc_pipeline run-stage "
                f"--manifest {_quote(manifest)} --index \"${{SESSION_INDEX}}\" --stage {stage}"
            ),
            "",
        ]
    )
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

    Bare paths use the default `SessionSpec`: channel count and functional
    channel are inferred from TIFF filenames. Use `SessionSpec` when the target
    structure, channel settings, or stage list need explicit overrides.
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
    config_target = SUITE2P_CONFIG_TARGETS.get(target, target)
    config_name = (
        "config_neuron_1chan.json"
        if config_target == "neuron" and session["nchannels"] == 1
        else f"config_{config_target}.json"
    )
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
    batch_size = data["pipeline"].get("suite2p_batch_size")
    if batch_size is not None:
        ops["batch_size"] = int(batch_size)
    binary_batch_size = data["pipeline"].get("suite2p_binary_batch_size")
    if binary_batch_size is not None:
        ops["binary_batch_size"] = int(binary_batch_size)
    registration_batch_size = data["pipeline"].get("suite2p_registration_batch_size")
    if registration_batch_size is not None:
        ops["registration_batch_size"] = int(registration_batch_size)
    extraction_batch_size = data["pipeline"].get("suite2p_extraction_batch_size")
    if extraction_batch_size is not None:
        ops["extraction_batch_size"] = int(extraction_batch_size)
    for key, value in session.get("suite2p_ops_overrides", {}).items():
        if value is not None:
            ops[key] = value
    ops.setdefault("delete_bin", True)
    torch_device = os.environ.get("TWO_P_SUITE2P_TORCH_DEVICE")
    if torch_device:
        ops["torch_device"] = torch_device
    fast_disk = os.environ.get("TWO_P_SUITE2P_FAST_DISK") or data["pipeline"].get("fast_disk")
    if str(fast_disk).lower() in {"tmp", "local_tmp", "slurm_tmp"}:
        job_id = os.environ.get("SLURM_JOB_ID", "manual")
        fast_disk = str(Path(os.environ.get("TMPDIR", "/tmp")) / f"suite2p_fast_disk_{job_id}")
    if fast_disk:
        ops["fast_disk"] = fast_disk
    else:
        source_disk = Path(session["raw_path"]) / ".suite2p_fast_disk"
        fallback_disk = Path(session["output_path"]) / ".suite2p_fast_disk"
        ops["fast_disk"] = str(source_disk if os.access(session["raw_path"], os.W_OK) else fallback_disk)
    db = {"data_path": [session["raw_path"]], "save_path0": session["output_path"]}
    return ops, db


def _suite2p_file_list(raw_path: Path, input_format: str) -> list[str]:
    """Return explicit input filenames for Suite2p 1.x db["file_list"]."""

    patterns = ["*.ome.tif", "*.ome.TIF"] if input_format == "bruker" else ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(raw_path.glob(pattern))
    return [path.name for path in sorted(files)]


def _suite2p_v1_diameter(ops: Mapping[str, Any]) -> float | list[float]:
    """Return a finite diameter for Suite2p 1.x ROI stats.

    Suite2p 0.14 allowed ``diameter=0`` with forced ``spatial_scale``. Suite2p
    1.x passes diameter directly into ROI-stat normalization, where zero causes
    divide-by-zero and NaNs. Match sparse detection's scale-to-pixels estimate
    when the legacy config leaves diameter unset.
    """

    diameter = ops.get("diameter")
    if isinstance(diameter, (list, tuple)):
        values = [float(value) for value in diameter]
        if values and all(value > 0 for value in values):
            return values
    elif diameter is not None:
        value = float(diameter)
        if value > 0:
            return value
    spatial_scale = int(ops.get("spatial_scale", 1))
    estimated = float(3 * (2**max(spatial_scale, 0)))
    return [estimated, estimated]


def _suite2p_v1_settings_db(ops: dict[str, Any], db: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate the legacy flat Suite2p 0.14 ops into Suite2p 1.x settings/db."""

    from suite2p.parameters import default_db, default_settings

    settings = default_settings()
    new_db = default_db()

    new_db.update(
        {
            "data_path": db["data_path"],
            "save_path0": db["save_path0"],
            "look_one_level_down": bool(ops.get("look_one_level_down", False)),
            "input_format": ops.get("input_format", "tif"),
            "keep_movie_raw": bool(ops.get("keep_movie_raw", False)),
            "nplanes": int(ops.get("nplanes", 1)),
            "nchannels": int(ops.get("nchannels", 1)),
            "functional_chan": int(ops.get("functional_chan", 1)),
            "align_by_chan": int(ops.get("align_by_chan", ops.get("functional_chan", 1))),
            "subfolders": ops.get("subfolders") or None,
            "save_folder": ops.get("save_folder") or "suite2p",
            "fast_disk": ops.get("fast_disk") or None,
            "h5py_key": ops.get("h5py_key", "data"),
            "nwb_driver": ops.get("nwb_driver", ""),
            "nwb_series": ops.get("nwb_series", ""),
            "ignore_flyback": ops.get("ignore_flyback") or None,
            "force_sktiff": bool(ops.get("force_sktiff", False)),
            "bruker_bidirectional": bool(ops.get("bruker_bidirectional", False)),
            "batch_size": int(ops.get("binary_batch_size", ops.get("batch_size", new_db.get("batch_size", 500)))),
        }
    )
    file_list = _suite2p_file_list(Path(db["data_path"][0]), str(new_db["input_format"]))
    if file_list:
        new_db["file_list"] = file_list

    if ops.get("torch_device"):
        settings["device"] = ops["torch_device"]
        settings["torch_device"] = ops["torch_device"]
    settings["tau"] = float(ops.get("tau", settings["tau"]))
    settings["fs"] = float(ops.get("fs", settings["fs"]))
    settings["diameter"] = _suite2p_v1_diameter(ops)

    settings["run"].update(
        {
            "do_registration": int(ops.get("do_registration", settings["run"]["do_registration"])),
            "do_regmetrics": bool(ops.get("do_regmetrics", True)),
            "do_detection": bool(ops.get("roidetect", settings["run"]["do_detection"])),
            "do_deconvolution": bool(ops.get("spikedetect", settings["run"]["do_deconvolution"])),
            "multiplane_parallel": bool(ops.get("multiplane_parallel", False)),
        }
    )
    align_by_chan = int(ops.get("align_by_chan", 1))
    settings["registration"].update(
        {
            "align_by_chan2": align_by_chan == 2 and int(ops.get("nchannels", 1)) > 1,
            "nimg_init": int(ops.get("nimg_init", settings["registration"]["nimg_init"])),
            "maxregshift": float(ops.get("maxregshift", settings["registration"]["maxregshift"])),
            "do_bidiphase": bool(ops.get("do_bidiphase", settings["registration"]["do_bidiphase"])),
            "bidiphase": float(ops.get("bidiphase", settings["registration"]["bidiphase"])),
            "batch_size": int(
                ops.get("registration_batch_size", ops.get("batch_size", settings["registration"]["batch_size"]))
            ),
            "nonrigid": bool(ops.get("nonrigid", settings["registration"]["nonrigid"])),
            "maxregshiftNR": float(ops.get("maxregshiftNR", settings["registration"]["maxregshiftNR"])),
            "block_size": tuple(ops.get("block_size", settings["registration"]["block_size"])),
            "smooth_sigma_time": float(ops.get("smooth_sigma_time", settings["registration"]["smooth_sigma_time"])),
            "smooth_sigma": float(ops.get("smooth_sigma", settings["registration"]["smooth_sigma"])),
            "spatial_taper": float(ops.get("spatial_taper", settings["registration"]["spatial_taper"])),
            "th_badframes": float(ops.get("th_badframes", settings["registration"]["th_badframes"])),
            "norm_frames": bool(ops.get("norm_frames", settings["registration"]["norm_frames"])),
            "snr_thresh": float(ops.get("snr_thresh", settings["registration"]["snr_thresh"])),
            "subpixel": int(ops.get("subpixel", settings["registration"]["subpixel"])),
            "two_step_registration": bool(
                ops.get("two_step_registration", settings["registration"]["two_step_registration"])
            ),
            "reg_tif": bool(ops.get("reg_tif", settings["registration"]["reg_tif"])),
            "reg_tif_chan2": bool(ops.get("reg_tif_chan2", settings["registration"]["reg_tif_chan2"])),
        }
    )
    settings["detection"].update(
        {
            "denoise": bool(ops.get("denoise", settings["detection"]["denoise"])),
            "block_size": tuple(ops.get("block_size", settings["detection"]["block_size"])),
            "highpass_time": int(ops.get("high_pass", settings["detection"]["highpass_time"])),
            "threshold_scaling": float(
                ops.get("threshold_scaling", settings["detection"]["threshold_scaling"])
            ),
            "max_overlap": float(ops.get("max_overlap", settings["detection"]["max_overlap"])),
            "soma_crop": bool(ops.get("soma_crop", settings["detection"]["soma_crop"])),
            "chan2_threshold": float(ops.get("chan2_thres", settings["detection"]["chan2_threshold"])),
        }
    )
    settings["detection"]["sparsery_settings"].update(
        {
            "highpass_neuropil": int(
                ops.get("spatial_hp_detect", settings["detection"]["sparsery_settings"]["highpass_neuropil"])
            ),
            "spatial_scale": int(ops.get("spatial_scale", settings["detection"]["sparsery_settings"]["spatial_scale"])),
        }
    )
    settings["detection"]["sourcery_settings"].update(
        {
            "connected": bool(ops.get("connected", settings["detection"]["sourcery_settings"]["connected"])),
            "max_iterations": int(
                ops.get("max_iterations", settings["detection"]["sourcery_settings"]["max_iterations"])
            ),
        }
    )
    settings["detection"]["cellpose_settings"].update(
        {
            "highpass_spatial": float(
                ops.get("spatial_hp_cp", settings["detection"]["cellpose_settings"]["highpass_spatial"])
            ),
            "flow_threshold": float(
                ops.get("flow_threshold", settings["detection"]["cellpose_settings"]["flow_threshold"])
            ),
            "cellprob_threshold": float(
                ops.get("cellprob_threshold", settings["detection"]["cellpose_settings"]["cellprob_threshold"])
            ),
        }
    )
    settings["classification"].update(
        {
            "classifier_path": ops.get("classifier_path") or None,
            "use_builtin_classifier": bool(
                ops.get("use_builtin_classifier", settings["classification"]["use_builtin_classifier"])
            ),
            "preclassify": float(ops.get("preclassify", settings["classification"]["preclassify"])),
        }
    )
    settings["extraction"].update(
        {
            "batch_size": int(
                ops.get("extraction_batch_size", ops.get("batch_size", settings["extraction"]["batch_size"]))
            ),
            "neuropil_extract": bool(ops.get("neuropil_extract", settings["extraction"]["neuropil_extract"])),
            "neuropil_coefficient": float(
                ops.get("neucoeff", settings["extraction"]["neuropil_coefficient"])
            ),
            "inner_neuropil_radius": int(
                ops.get("inner_neuropil_radius", settings["extraction"]["inner_neuropil_radius"])
            ),
            "min_neuropil_pixels": int(
                ops.get("min_neuropil_pixels", settings["extraction"]["min_neuropil_pixels"])
            ),
            "lam_percentile": float(ops.get("lam_percentile", settings["extraction"]["lam_percentile"])),
            "allow_overlap": bool(ops.get("allow_overlap", settings["extraction"]["allow_overlap"])),
        }
    )
    settings["dcnv_preprocess"].update(
        {
            "baseline": ops.get("baseline", settings["dcnv_preprocess"]["baseline"]),
            "win_baseline": float(ops.get("win_baseline", settings["dcnv_preprocess"]["win_baseline"])),
            "sig_baseline": float(ops.get("sig_baseline", settings["dcnv_preprocess"]["sig_baseline"])),
            "prctile_baseline": float(
                ops.get("prctile_baseline", settings["dcnv_preprocess"]["prctile_baseline"])
            ),
        }
    )
    settings["io"].update(
        {
            "combined": bool(ops.get("combined", settings["io"]["combined"])),
            "save_mat": bool(ops.get("save_mat", settings["io"]["save_mat"])),
            "save_NWB": bool(ops.get("save_NWB", settings["io"]["save_NWB"])),
            "delete_bin": bool(ops.get("delete_bin", settings["io"]["delete_bin"])),
            "move_bin": bool(ops.get("move_bin", settings["io"]["move_bin"])),
            "save_ops_orig": True,
        }
    )
    return settings, new_db


def _read_ops(session: dict[str, Any]):
    import numpy as np

    path = Path(session["output_path"]) / "suite2p" / "plane0" / "ops.npy"
    ops = np.load(path, allow_pickle=True).item()
    ops["save_path0"] = session["output_path"]
    if "neucoeff" not in ops and isinstance(ops.get("extraction"), dict):
        ops["neucoeff"] = ops["extraction"].get("neuropil_coefficient", 0.7)
    return ops


def _write_raw_voltages(raw_path: Path, output_path: Path) -> None:
    import h5py
    import numpy as np
    import pandas as pd

    candidates = sorted(raw_path.glob("*VoltageRecording*.csv"))
    if not candidates:
        print("Valid voltage recordings csv file not found")
        return
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
    csv_path = candidates[0]
    header = pd.read_csv(csv_path, nrows=0)
    available_columns = set(header.columns)
    usecols = ["Time(ms)"] + [column for column, _ in channels.values() if column in available_columns]
    chunksize = 250_000
    with h5py.File(output_path / "raw_voltages.h5", "w") as output:
        raw = output.create_group("raw")
        datasets: dict[str, Any] = {}
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
            time = chunk["Time(ms)"].to_numpy()
            if "vol_time" not in datasets:
                datasets["vol_time"] = raw.create_dataset(
                    "vol_time",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=time.dtype,
                    chunks=True,
                )
            time_ds = datasets["vol_time"]
            start = time_ds.shape[0]
            end = start + len(time)
            time_ds.resize((end,))
            time_ds[start:end] = time
            for name, (column, threshold) in channels.items():
                if name not in datasets:
                    template = chunk[column].to_numpy() if column in chunk.columns else np.zeros_like(time)
                    datasets[name] = raw.create_dataset(
                        name,
                        shape=(0,),
                        maxshape=(None,),
                        dtype=template.dtype,
                        chunks=True,
                    )
                values = chunk[column].to_numpy() if column in chunk.columns else np.zeros_like(time)
                if threshold is not None:
                    values = (values > threshold).astype(values.dtype)
                ds = datasets[name]
                start = ds.shape[0]
                end = start + len(values)
                ds.resize((end,))
                ds[start:end] = values


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


def _remove_empty_suite2p_binaries(output_path: Path, fast_disk: Path) -> None:
    """Remove failed-run binary placeholders before Suite2p sees them."""

    candidates = []
    for root in (output_path, fast_disk):
        plane = root / "suite2p" / "plane0"
        candidates.extend((plane / "data.bin", plane / "data_chan2.bin"))
    for path in candidates:
        if path.is_file() and path.stat().st_size == 0:
            path.unlink()
            print(f"Removed stale empty Suite2p binary: {path}")


def _run_suite2p(data: dict[str, Any], session: dict[str, Any]) -> None:
    from suite2p import run_s2p

    ops, db = _processing_ops(data, session)
    if ops.get("fast_disk"):
        Path(ops["fast_disk"]).mkdir(parents=True, exist_ok=True)
    _remove_empty_suite2p_binaries(Path(session["output_path"]), Path(ops["fast_disk"]))
    if "ops" in inspect.signature(run_s2p).parameters:
        run_s2p(ops=ops, db=db)
    else:
        try:
            from suite2p.run_s2p import logger_setup

            logger_setup(str(Path(session["output_path"]) / "suite2p"))
        except Exception as error:
            print(f"Suite2p logger setup skipped: {error}")
        settings, new_db = _suite2p_v1_settings_db(ops, db)
        run_s2p(db=new_db, settings=settings)


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
            functional_chan=args.functional_chan,
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
        suite2p_version=args.suite2p_version,
        account=args.account or "",
        username=args.username or "",
        mail_user=args.mail_user,
        numba_cache_dir=args.numba_cache_dir or "",
        fast_disk=args.fast_disk or "",
        qos_cpu=args.qos_cpu or args.qos or "",
        qos_gpu=args.qos_gpu or args.qos or "",
        partition_cpu=args.partition_cpu,
        partition_gpu=args.partition_gpu,
        suite2p_gpu=args.suite2p_gpu,
        suite2p_batch_size=args.suite2p_batch_size,
        suite2p_binary_batch_size=args.suite2p_binary_batch_size,
        suite2p_registration_batch_size=args.suite2p_registration_batch_size,
        suite2p_extraction_batch_size=args.suite2p_extraction_batch_size,
    )


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session", action="append", help="Raw session directory. Repeat for multiple sessions.")
    parser.add_argument("--sessions-file", help="Text file containing one raw session directory per line.")
    parser.add_argument("--output-root", required=True, help="Directory where processed sessions will be written.")
    parser.add_argument("--target-structure", choices=sorted(QC_PRESETS), default="neuron")
    parser.add_argument(
        "--nchannels",
        type=int,
        choices=(1, 2),
        default=None,
        help="Override channel count. Default: infer from TIFF filenames.",
    )
    parser.add_argument(
        "--functional-chan",
        type=int,
        default=None,
        choices=(1, 2),
        help=(
            "Override functional channel index. Default: infer from TIFF filenames; "
            "two-channel recordings use Ch2 functional/Ch1 anatomical, and "
            "single-channel recordings use the only detected channel as functional."
        ),
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
    parser.add_argument(
        "--suite2p-version",
        choices=sorted(SUITE2P_VERSIONED_PYTHONS),
        default="1.x",
        help="Default Suite2p environment to use when --python-bin is not supplied.",
    )
    parser.add_argument("--python-bin", default=None, help="Python executable in a Suite2p-capable environment.")
    parser.add_argument("--account", default=None, help="Slurm account; defaults to TWO_P_SLURM_ACCOUNT or gts-fnajafi3.")
    parser.add_argument("--username", default=None)
    parser.add_argument("--mail-user", default=None)
    parser.add_argument("--numba-cache-dir", default=None)
    parser.add_argument("--fast-disk", default=None, help="Suite2p fast_disk location for temporary binary files.")
    parser.add_argument("--qos", default=None, help="Slurm QOS for all stages; defaults to embers.")
    parser.add_argument("--qos-cpu", default=None, help="Slurm QOS for CPU stages; overrides --qos.")
    parser.add_argument("--qos-gpu", default=None, help="Slurm QOS for GPU stages; overrides --qos.")
    parser.add_argument("--partition-cpu", default=None)
    parser.add_argument("--partition-gpu", default=None)
    parser.add_argument(
        "--suite2p-gpu",
        action="store_true",
        default=None,
        help="Request a GPU for the suite2p stage using --gres=gpu:1.",
    )
    parser.add_argument(
        "--no-suite2p-gpu",
        action="store_false",
        dest="suite2p_gpu",
        help="Run the suite2p stage without requesting a GPU.",
    )
    parser.add_argument(
        "--suite2p-batch-size",
        type=int,
        default=None,
        help="Override Suite2p batch_size globally for benchmarking or memory tuning.",
    )
    parser.add_argument(
        "--suite2p-binary-batch-size",
        type=int,
        default=None,
        help="Override Suite2p 1.x TIFF-to-binary batch size. Default: 5000.",
    )
    parser.add_argument(
        "--suite2p-registration-batch-size",
        type=int,
        default=None,
        help="Override Suite2p 1.x registration batch size. Default: 500.",
    )
    parser.add_argument(
        "--suite2p-extraction-batch-size",
        type=int,
        default=None,
        help="Override Suite2p 1.x extraction/deconvolution batch size. Default: 500.",
    )


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
