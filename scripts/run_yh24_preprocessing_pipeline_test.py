#!/usr/bin/env python3
"""Run a prefilled YH24 preprocessing-pipeline CLI test.

This script intentionally calls the same command-line interface a user would
run by hand:

    python -m utils_2p.preprocessing_qc_pipeline generate ...

Default mode is `generate`, which writes Slurm files but does not submit jobs.
Use `--submit` to submit the linked Slurm chain.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Prefilled test inputs. Edit these values when testing a different session.
PYTHON_BIN = Path("/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_v1/bin/python")
RAW_SESSION = Path(
    "/storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging/"
    "YH24LG_Processed/YH24LG_CRBL_lobulev_20250609_EBC-442"
)
OUTPUT_ROOT = Path(f"/storage/scratch1/3/{getpass.getuser()}/2p_pipeline_tests/yh24_preprocessing")
RUN_NAME = "yh24_lobulev_20250609_pipeline_cli_test"
SLURM_ACCOUNT = "gts-fnajafi3"
SLURM_QOS = "embers"

# This YH24 session has only a functional channel, so anatomical labeling is
# detected automatically and the label/GPU stage is omitted.
PREFILLED_PIPELINE_ARGS = [
    "--session",
    str(RAW_SESSION),
    "--output-root",
    str(OUTPUT_ROOT),
    "--run-name",
    RUN_NAME,
    "--target-structure",
    "dendrite",
    "--python-bin",
    str(PYTHON_BIN),
    "--account",
    SLURM_ACCOUNT,
    "--qos",
    SLURM_QOS,
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate", action="store_const", const="generate", dest="mode")
    mode.add_argument("--submit", action="store_const", const="submit", dest="mode")
    parser.set_defaults(mode="generate")
    return parser.parse_args()


def _validate_inputs() -> None:
    if not RAW_SESSION.is_dir():
        raise SystemExit(f"Raw session directory does not exist: {RAW_SESSION}")
    if not PYTHON_BIN.exists():
        raise SystemExit(f"Suite2p Python executable does not exist: {PYTHON_BIN}")


def main() -> None:
    args = _parse_args()
    _validate_inputs()

    command = [
        str(PYTHON_BIN),
        "-m",
        "utils_2p.preprocessing_qc_pipeline",
        args.mode,
        *PREFILLED_PIPELINE_ARGS,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(REPO_ROOT)

    print("Running:")
    print(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
