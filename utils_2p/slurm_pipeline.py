"""Slurm helpers for staged 2p processing and post-processing jobs.

This module generalizes the scratch MC11 staged workflow:

prep CPU -> suite2p GPU -> qc CPU -> label GPU -> dff CPU

The launcher reads sessions from a CSV/TSV manifest instead of hard-coding
specific mice or session names.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_QC = {
    "range_skew": "-5,5",
    "max_connect": "1",
    "range_aspect": "0,5",
    "range_compact": "0,1.06",
    "range_footprint": "1,2",
    "diameter": "6",
}


@dataclass(frozen=True)
class Session:
    session_name: str
    data_path: str
    save_path: str
    bpod_mat_path: str = ""


@dataclass(frozen=True)
class StageSpec:
    key: str
    script_name: str
    runner: str
    cpus: int
    mem: str
    time: str
    gres: str = ""


STAGES = {
    "processing_cpu": StageSpec(
        key="processing_cpu",
        script_name="processing_cpu_stage.sbatch",
        runner="run-processing-stage",
        cpus=4,
        mem="32G",
        time="01:00:00",
    ),
    "processing_gpu": StageSpec(
        key="processing_gpu",
        script_name="processing_gpu_stage.sbatch",
        runner="run-processing-stage",
        cpus=8,
        mem="192G",
        time="08:00:00",
        gres="gpu:1",
    ),
    "postprocess_cpu": StageSpec(
        key="postprocess_cpu",
        script_name="postprocess_cpu_stage.sbatch",
        runner="run-postprocess-stage",
        cpus=4,
        mem="64G",
        time="04:00:00",
    ),
    "postprocess_gpu": StageSpec(
        key="postprocess_gpu",
        script_name="postprocess_gpu_stage.sbatch",
        runner="run-postprocess-stage",
        cpus=4,
        mem="64G",
        time="02:00:00",
        gres="gpu:1",
    ),
}


CHAIN = (
    ("prep", "processing_cpu"),
    ("suite2p", "processing_gpu"),
    ("qc", "postprocess_cpu"),
    ("label", "postprocess_gpu"),
    ("dff", "postprocess_cpu"),
)


STAGE_MARKERS = {
    "prep": (
        "raw_voltages.h5",
        "suite2p/plane0/ops.npy",
    ),
    "suite2p": (
        "suite2p/plane0/F.npy",
        "suite2p/plane0/Fneu.npy",
        "suite2p/plane0/spks.npy",
        "suite2p/plane0/stat.npy",
        "suite2p/plane0/iscell.npy",
        "suite2p/plane0/redcell.npy",
    ),
    "qc": (
        "qc_results/fluo.npy",
        "qc_results/neuropil.npy",
        "qc_results/stat.npy",
        "qc_results/masks.npy",
        "move_offset.h5",
        "ops.npy",
    ),
    "label": (
        "masks.h5",
    ),
    "dff": (
        "dff.h5",
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def split_numeric(value: str, dtype=float) -> np.ndarray:
    return np.array([item.strip() for item in value.split(",") if item.strip()], dtype=dtype)


def read_ops(session_data_path: str) -> dict:
    ops = np.load(
        os.path.join(session_data_path, "suite2p", "plane0", "ops.npy"),
        allow_pickle=True,
    ).item()
    ops["save_path0"] = session_data_path
    return ops


def read_sessions(path: str, output_root: str | None = None) -> list[Session]:
    manifest_path = Path(path)
    delimiter = "\t" if manifest_path.suffix.lower() in {".tsv", ".tab"} else ","
    sessions: list[Session] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {"session_name", "data_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns: session_name, data_path; "
                "optional columns: save_path, bpod_mat_path"
            )
        for row in reader:
            session_name = (row.get("session_name") or "").strip()
            data_path = (row.get("data_path") or "").strip()
            save_path = (row.get("save_path") or "").strip()
            if not session_name or not data_path:
                continue
            if not save_path:
                if output_root is None:
                    raise ValueError(
                        f"Session {session_name} has no save_path; pass --output-root or add save_path."
                    )
                save_path = str(Path(output_root) / session_name)
            sessions.append(
                Session(
                    session_name=session_name,
                    data_path=data_path,
                    save_path=save_path,
                    bpod_mat_path=(row.get("bpod_mat_path") or "").strip(),
                )
            )
    if not sessions:
        raise ValueError(f"No sessions found in {path}")
    return sessions


def infer_processing_status(session_data_path: str | Path) -> dict[str, object]:
    """Infer completed processing stages from expected output files.

    This is intentionally file-marker based. It does not prove output validity,
    but it gives a conservative answer about which pipeline stages appear to
    have produced their expected artifacts.
    """
    root = Path(session_data_path)
    stages: dict[str, dict[str, object]] = {}
    complete_through = ""
    next_stage = ""
    for stage, _script_key in CHAIN:
        markers = STAGE_MARKERS[stage]
        present = [marker for marker in markers if (root / marker).exists()]
        missing = [marker for marker in markers if not (root / marker).exists()]
        complete = not missing
        stages[stage] = {
            "complete": complete,
            "present": present,
            "missing": missing,
        }
        if complete and not next_stage:
            complete_through = stage
        elif not next_stage:
            next_stage = stage
    return {
        "session_data_path": str(root),
        "exists": root.exists(),
        "complete": all(stage["complete"] for stage in stages.values()),
        "complete_through": complete_through,
        "next_stage": next_stage,
        "stages": stages,
    }


def first_incomplete_stage(session_data_path: str | Path) -> str | None:
    status = infer_processing_status(session_data_path)
    next_stage = status["next_stage"]
    return str(next_stage) if next_stage else None


def quote_command(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def shell_arg(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        return f'"{value}"'
    return shlex.quote(str(value))


def shell_command(args: Sequence[str]) -> str:
    return " ".join(shell_arg(str(arg)) for arg in args)


def shell_option(name: str, value: str | int | float | bool) -> str:
    return f"{name}={shell_arg(str(value))}"


def stage_script_text(spec: StageSpec, args: argparse.Namespace) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --account={args.account}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={getattr(args, f'{spec.key}_cpus')}",
        f"#SBATCH --mem={getattr(args, f'{spec.key}_mem')}",
        f"#SBATCH --time={getattr(args, f'{spec.key}_time')}",
    ]
    gres = getattr(args, f"{spec.key}_gres")
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    if args.partition:
        lines.append(f"#SBATCH --partition={args.partition}")
    if args.qos:
        lines.append(f"#SBATCH --qos={args.qos}")
    if args.mail_user:
        lines.append(f"#SBATCH --mail-user={args.mail_user}")
        lines.append(f"#SBATCH --mail-type={args.mail_type}")
    lines.extend(
        [
            f"#SBATCH --output={args.sbatch_dir}/logs/%x_%j.out",
            "",
            "set -euo pipefail",
            "",
            "unset LD_PRELOAD || true",
            f"export NUMBA_CACHE_DIR={shlex.quote(args.numba_cache_dir)}",
            f"export PYTHONPATH={shlex.quote(str(repo_root()))}:${{PYTHONPATH:-}}",
            "mkdir -p \"${NUMBA_CACHE_DIR}\"",
            f"mkdir -p {shlex.quote(args.sbatch_dir)}/logs",
            "",
            ": \"${STAGE:?STAGE is required}\"",
            ": \"${SESSION_NAME:?SESSION_NAME is required}\"",
            ": \"${SAVE_PATH:?SAVE_PATH is required}\"",
        ]
    )

    base_cmd = [args.python_bin, "-m", "utils_2p.slurm_pipeline", spec.runner]
    if spec.runner == "run-processing-stage":
        lines.extend(
            [
                ": \"${DATA_PATH:?DATA_PATH is required}\"",
                f"cd {shlex.quote(args.processing_root)}",
                shell_command(base_cmd)
                + " "
                + shell_command(
                    [
                        "--stage",
                        "${STAGE}",
                        "--processing-root",
                        args.processing_root,
                        "--denoise",
                        str(args.denoise),
                        "--spatial-scale",
                        str(args.spatial_scale),
                        "--data-path",
                        "${DATA_PATH}",
                        "--save-path",
                        "${SAVE_PATH}",
                        "--nchannels",
                        str(args.nchannels),
                        "--functional-chan",
                        str(args.functional_chan),
                        "--target-structure",
                        args.target_structure,
                        "--bpod-mat-path",
                        "${BPOD_MAT_PATH:-}",
                    ]
                ),
            ]
        )
    else:
        lines.append(
            shell_command(base_cmd)
            + " "
            + " ".join(
                [
                    shell_option("--stage", "${STAGE}"),
                    shell_option("--postprocess-root", args.postprocess_root),
                    shell_option("--range-skew", args.range_skew),
                    shell_option("--max-connect", args.max_connect),
                    shell_option("--range-aspect", args.range_aspect),
                    shell_option("--range-compact", args.range_compact),
                    shell_option("--range-footprint", args.range_footprint),
                    shell_option("--diameter", args.diameter),
                    shell_option("--dff-norm", int(args.dff_norm)),
                    shell_option("--correct-pmt", int(args.correct_pmt)),
                    shell_arg("${SAVE_PATH}"),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def write_stage_scripts(args: argparse.Namespace) -> dict[str, Path]:
    sbatch_dir = Path(args.sbatch_dir)
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    (sbatch_dir / "logs").mkdir(exist_ok=True)
    scripts = {}
    for key, spec in STAGES.items():
        path = sbatch_dir / spec.script_name
        path.write_text(stage_script_text(spec, args), encoding="utf-8")
        path.chmod(0o755)
        scripts[key] = path
    return scripts


def sanitize_job_name(value: str) -> str:
    allowed = []
    for char in value:
        allowed.append(char if char.isalnum() or char in "._-" else "_")
    return "".join(allowed)[:120]


def sbatch_export(stage: str, session: Session) -> str:
    return (
        "ALL,"
        f"STAGE={stage},"
        f"SESSION_NAME={session.session_name},"
        f"DATA_PATH={session.data_path},"
        f"SAVE_PATH={session.save_path},"
        f"BPOD_MAT_PATH={session.bpod_mat_path}"
    )


def submit_stage(
    script_path: Path,
    job_name: str,
    export_value: str,
    dependency: str | None = None,
    dry_run: bool = False,
) -> str:
    cmd = ["sbatch", "--job-name", job_name, "--export", export_value]
    if dependency:
        cmd.extend(["--dependency", f"afterok:{dependency}"])
    cmd.append(str(script_path))
    if dry_run:
        print(quote_command(cmd))
        return f"DRYRUN_{job_name}"
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit {job_name}: {output}")
    parts = output.split()
    if len(parts) < 4 or parts[-2:] == ["batch", "job"]:
        raise RuntimeError(f"Unexpected sbatch output for {job_name}: {output}")
    print(output, file=sys.stderr)
    return parts[-1]


def chain_from_stage(start_stage: str) -> tuple[tuple[str, str], ...]:
    stages = [stage for stage, _script_key in CHAIN]
    if start_stage not in stages:
        raise ValueError(f"Unknown stage {start_stage!r}; expected one of {', '.join(stages)}")
    start_index = stages.index(start_stage)
    return CHAIN[start_index:]


def submit_chains(args: argparse.Namespace) -> None:
    scripts = write_stage_scripts(args)
    sessions = read_sessions(args.manifest, output_root=args.output_root)
    for session in sessions:
        if args.resume:
            start_stage = first_incomplete_stage(session.save_path)
            if start_stage is None:
                print(f"{session.session_name}: complete")
                continue
            chain = chain_from_stage(start_stage)
        else:
            chain = CHAIN
        dependency = None
        submitted: list[str] = []
        for stage, script_key in chain:
            job_name = sanitize_job_name(f"{stage}_{session.session_name}")
            job_id = submit_stage(
                scripts[script_key],
                job_name,
                sbatch_export(stage, session),
                dependency=dependency,
                dry_run=args.dry_run,
            )
            submitted.append(f"{stage}={job_id}")
            dependency = job_id
        print(f"{session.session_name}: {' '.join(submitted)}")


def print_status(args: argparse.Namespace) -> None:
    import json

    roots = args.session_data_path
    if args.children:
        roots = [
            str(path)
            for root in roots
            for path in sorted(Path(root).iterdir())
            if path.is_dir()
        ]
    statuses = [infer_processing_status(root) for root in roots]
    if args.json:
        print(json.dumps(statuses, indent=2))
        return
    for status in statuses:
        print(status["session_data_path"])
        print(f"  exists: {status['exists']}")
        print(f"  complete: {status['complete']}")
        print(f"  complete_through: {status['complete_through'] or 'none'}")
        print(f"  next_stage: {status['next_stage'] or 'none'}")
        for stage, details in status["stages"].items():
            state = "complete" if details["complete"] else "missing"
            print(f"  {stage}: {state}")
            if details["missing"]:
                print(f"    missing: {', '.join(details['missing'])}")


def run_processing_stage(args: argparse.Namespace) -> None:
    processing_root = os.path.abspath(args.processing_root)
    sys.path.insert(0, processing_root)
    import run_suite2p_pipeline as pipeline

    class PipelineArgs:
        pass

    pipeline_args = PipelineArgs()
    pipeline_args.denoise = args.denoise
    pipeline_args.spatial_scale = args.spatial_scale
    pipeline_args.data_path = args.data_path
    pipeline_args.save_path = args.save_path
    pipeline_args.nchannels = args.nchannels
    pipeline_args.functional_chan = args.functional_chan
    pipeline_args.target_structure = args.target_structure
    pipeline_args.bpod_mat_path = args.bpod_mat_path or None

    cwd = os.getcwd()
    os.makedirs(args.save_path, exist_ok=True)
    os.chdir(processing_root)
    try:
        pipeline.ops, db = pipeline.set_params(pipeline_args)
        if args.stage == "prep":
            pipeline.process_vol(pipeline_args)
            pipeline.move_bpod_mat(pipeline_args)
            return
        if args.stage == "suite2p":
            from suite2p import run_s2p

            run_s2p(ops=pipeline.ops, db=db)
            return
    finally:
        os.chdir(cwd)
    raise ValueError(f"Unsupported processing stage: {args.stage}")


def run_postprocess_stage(args: argparse.Namespace) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.stage == "label":
        os.environ.setdefault("CELLPOSE_USE_GPU", "1")
    postprocess_root = os.path.abspath(args.postprocess_root)
    sys.path.insert(0, postprocess_root)
    ops = read_ops(args.session_data_path)

    if args.stage == "qc":
        from modules import QualControlDataIO

        QualControlDataIO.run(
            ops,
            split_numeric(args.range_skew, dtype="float32"),
            np.array(args.max_connect, dtype="float32"),
            split_numeric(args.range_aspect, dtype="float32"),
            split_numeric(args.range_compact, dtype="float32"),
            split_numeric(args.range_footprint, dtype="float32"),
        )
        return
    if args.stage == "label":
        from modules import LabelExcInh

        LabelExcInh.run(ops, float(args.diameter))
        return
    if args.stage == "dff":
        from modules import DffTraces

        DffTraces.run(ops, norm=bool(args.dff_norm), correct_pmt=bool(args.correct_pmt))
        return
    raise ValueError(f"Unsupported postprocess stage: {args.stage}")


def add_resource_args(parser: argparse.ArgumentParser) -> None:
    for key, spec in STAGES.items():
        parser.add_argument(f"--{key.replace('_', '-')}-cpus", type=int, default=spec.cpus)
        parser.add_argument(f"--{key.replace('_', '-')}-mem", default=spec.mem)
        parser.add_argument(f"--{key.replace('_', '-')}-time", default=spec.time)
        parser.add_argument(f"--{key.replace('_', '-')}-gres", default=spec.gres)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="Submit staged Slurm chains from a session manifest.")
    submit.add_argument("--manifest", required=True, help="CSV/TSV with session_name,data_path and optional save_path,bpod_mat_path.")
    submit.add_argument("--output-root", help="Root used to build save_path when manifest omits save_path.")
    submit.add_argument("--processing-root", required=True, help="Directory containing run_suite2p_pipeline.py and configs.")
    submit.add_argument("--postprocess-root", required=True, help="Directory containing run_postprocess.py and modules/.")
    submit.add_argument("--sbatch-dir", required=True, help="Directory where generated sbatch files and logs are written.")
    submit.add_argument("--python-bin", default=sys.executable, help="Python executable used inside Slurm jobs.")
    submit.add_argument("--account", default="gts-fnajafi3")
    submit.add_argument("--partition", default="")
    submit.add_argument("--qos", default="")
    submit.add_argument("--mail-user", default="")
    submit.add_argument("--mail-type", default="END,FAIL")
    submit.add_argument("--numba-cache-dir", default="/storage/scratch1/3/grubin6/.numba_cache")
    submit.add_argument("--denoise", type=int, default=1)
    submit.add_argument("--spatial-scale", type=int, default=1)
    submit.add_argument("--nchannels", type=int, default=2)
    submit.add_argument("--functional-chan", type=int, default=2)
    submit.add_argument("--target-structure", default="neuron")
    submit.add_argument("--range-skew", default=DEFAULT_QC["range_skew"])
    submit.add_argument("--max-connect", default=DEFAULT_QC["max_connect"])
    submit.add_argument("--range-aspect", default=DEFAULT_QC["range_aspect"])
    submit.add_argument("--range-compact", default=DEFAULT_QC["range_compact"])
    submit.add_argument("--range-footprint", default=DEFAULT_QC["range_footprint"])
    submit.add_argument("--diameter", default=DEFAULT_QC["diameter"])
    submit.add_argument("--dff-norm", type=int, default=1)
    submit.add_argument("--correct-pmt", type=int, default=0)
    submit.add_argument("--resume", action="store_true", help="Start each session at the first stage whose output markers are missing.")
    submit.add_argument("--dry-run", action="store_true")
    add_resource_args(submit)
    submit.set_defaults(func=submit_chains)

    proc = subparsers.add_parser("run-processing-stage", help="Run one processing stage inside Slurm.")
    proc.add_argument("--stage", required=True, choices=["prep", "suite2p"])
    proc.add_argument("--processing-root", required=True)
    proc.add_argument("--denoise", required=True, type=int)
    proc.add_argument("--spatial-scale", required=True, type=int)
    proc.add_argument("--data-path", required=True)
    proc.add_argument("--save-path", required=True)
    proc.add_argument("--nchannels", required=True, type=int)
    proc.add_argument("--functional-chan", required=True, type=int)
    proc.add_argument("--target-structure", required=True)
    proc.add_argument("--bpod-mat-path", default="")
    proc.set_defaults(func=run_processing_stage)

    post = subparsers.add_parser("run-postprocess-stage", help="Run one postprocessing stage inside Slurm.")
    post.add_argument("--stage", required=True, choices=["qc", "label", "dff"])
    post.add_argument("--postprocess-root", required=True)
    post.add_argument("--range-skew", default=DEFAULT_QC["range_skew"])
    post.add_argument("--max-connect", default=DEFAULT_QC["max_connect"])
    post.add_argument("--range-aspect", default=DEFAULT_QC["range_aspect"])
    post.add_argument("--range-compact", default=DEFAULT_QC["range_compact"])
    post.add_argument("--range-footprint", default=DEFAULT_QC["range_footprint"])
    post.add_argument("--diameter", default=DEFAULT_QC["diameter"])
    post.add_argument("--dff-norm", type=int, default=1)
    post.add_argument("--correct-pmt", type=int, default=0)
    post.add_argument("session_data_path")
    post.set_defaults(func=run_postprocess_stage)

    status = subparsers.add_parser("status", help="Infer completed pipeline stages from files present on disk.")
    status.add_argument("session_data_path", nargs="+", help="Processed session directory, or roots when --children is set.")
    status.add_argument("--children", action="store_true", help="Inspect direct child directories of each supplied path.")
    status.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    status.set_defaults(func=print_status)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
