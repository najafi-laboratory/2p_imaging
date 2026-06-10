#!/usr/bin/env python3
"""Reproducible YH24 test for the staged preprocessing pipeline.

Edit the variables below, then run this file from the repository root:

    python utils_2p/scripts/run_yh24_preprocessing_pipeline_test.py

By default this only generates the Slurm scripts. Set MODE = "submit" or pass
--submit to actually submit the linked job chain.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils_2p.preprocessing_qc_pipeline import (
    PipelineConfig,
    SessionSpec,
    generate_preprocessing_qc_jobs,
    submit_preprocessing_qc_jobs,
)


# -----------------------------
# User-editable test parameters
# -----------------------------

MODE = "generate"  # "generate" writes sbatch files; "submit" submits them.

RAW_SESSION = Path(
    "/storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging/"
    "YH24LG_Processed/YH24LG_CRBL_lobulev_20250609_EBC-442"
)

OUTPUT_ROOT = Path(f"/storage/scratch1/3/{getpass.getuser()}/2p_pipeline_tests/yh24_preprocessing")

PYTHON_BIN = Path(
    "/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_v1/bin/python"
)

SLURM_ACCOUNT = "gts-fnajafi3"
SLURM_QOS = "embers"  # Use "inferno" for paid, non-preemptible jobs.
RUN_NAME = "yh24_lobulev_20250609_pipeline_test"

# This default YH24 test session has Ch2 TIFFs only, so it is configured as a
# functional-only dendrite/EBC recording and skips anatomical labeling.
SESSION = SessionSpec(
    RAW_SESSION,
    target_structure="dendrite",
    nchannels=1,
    functional_chan=1,
    run_label=False,
    stages=("prep", "suite2p", "qc", "dff", "summary"),
)

CONFIG = PipelineConfig(
    python_bin=PYTHON_BIN,
    account=SLURM_ACCOUNT,
    qos_cpu=SLURM_QOS,
    qos_gpu=SLURM_QOS,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="Generate sbatch files without submitting.")
    parser.add_argument("--submit", action="store_true", help="Submit the linked Slurm chain.")
    return parser.parse_args()


def _mode_from_args(args: argparse.Namespace) -> str:
    if args.generate and args.submit:
        raise SystemExit("Use only one of --generate or --submit.")
    if args.generate:
        return "generate"
    if args.submit:
        return "submit"
    if MODE not in {"generate", "submit"}:
        raise SystemExit('MODE must be "generate" or "submit".')
    return MODE


def _validate_inputs() -> None:
    if not RAW_SESSION.is_dir():
        raise SystemExit(f"Raw session directory does not exist: {RAW_SESSION}")
    if not PYTHON_BIN.exists():
        raise SystemExit(f"Suite2p Python executable does not exist: {PYTHON_BIN}")


def main() -> None:
    mode = _mode_from_args(_parse_args())
    _validate_inputs()

    print(f"Mode: {mode}")
    print(f"Raw session: {RAW_SESSION}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Python: {PYTHON_BIN}")
    print(f"Slurm account: {SLURM_ACCOUNT}")
    print(f"Slurm QOS: {SLURM_QOS}")

    if mode == "generate":
        generated = generate_preprocessing_qc_jobs(
            [SESSION],
            OUTPUT_ROOT,
            config=CONFIG,
            run_name=RUN_NAME,
        )
        print(f"Generated job directory: {generated.run_dir}")
        print(f"Submit with: bash {generated.submit_script}")
        return

    submitted = submit_preprocessing_qc_jobs(
        [SESSION],
        OUTPUT_ROOT,
        config=CONFIG,
        run_name=RUN_NAME,
    )
    print("Submitted jobs:")
    for session_name, jobs in submitted.items():
        print(f"  {session_name}: {jobs}")


if __name__ == "__main__":
    main()
