#!/usr/bin/env bash
set -euo pipefail

# Reproducible end-to-end test for the staged preprocessing pipeline.
# Default mode generates the Slurm scripts without submitting them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RAW_SESSION="${RAW_SESSION:-/storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging/YH24LG_Processed/YH24LG_CRBL_lobulev_20250609_EBC-442}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/storage/scratch1/3/${USER}/2p_pipeline_tests/yh24_preprocessing}"
PYTHON_BIN="${TWO_P_PYTHON:-/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_v1/bin/python}"
ACCOUNT="${TWO_P_SLURM_ACCOUNT:-gts-fnajafi3}"
QOS="${TWO_P_SLURM_QOS:-embers}"
RUN_NAME="${RUN_NAME:-yh24_lobulev_20250609_pipeline_test}"
MODE="generate"

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_yh24_preprocessing_pipeline_test.sh [--generate|--submit]

Environment overrides:
  RAW_SESSION       Raw YH24 session directory to process.
  OUTPUT_ROOT       Output root for processed files and generated Slurm scripts.
  TWO_P_PYTHON      Python executable for the shared Suite2p environment.
  TWO_P_SLURM_ACCOUNT
  TWO_P_SLURM_QOS   Defaults to embers; set to inferno when paid, non-preemptible jobs are needed.
  RUN_NAME          Name used for the generated Slurm job directory.

Default test session:
  /storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging/YH24LG_Processed/YH24LG_CRBL_lobulev_20250609_EBC-442

This session is treated as a functional-only YH24LG cerebellar dendrite/EBC
recording, so the full chain is:
  prep -> suite2p -> qc -> dff -> summary

The anatomical label stage is skipped because this default session has only one
functional imaging channel present.
USAGE
}

while (($#)); do
    case "$1" in
        --generate)
            MODE="generate"
            shift
            ;;
        --submit)
            MODE="submit"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -d "${RAW_SESSION}" ]]; then
    echo "Raw session directory does not exist: ${RAW_SESSION}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Suite2p Python executable is missing or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

cd "${REPO_ROOT}"

echo "Repository: ${REPO_ROOT}"
echo "Mode: ${MODE}"
echo "Raw session: ${RAW_SESSION}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Slurm account: ${ACCOUNT}"
echo "Slurm QOS: ${QOS}"

"${PYTHON_BIN}" -m utils_2p.preprocessing_qc_pipeline "${MODE}" \
    --session "${RAW_SESSION}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${RUN_NAME}" \
    --target-structure dendrite \
    --nchannels 1 \
    --functional-chan 1 \
    --no-label \
    --stages prep,suite2p,qc,dff,summary \
    --python-bin "${PYTHON_BIN}" \
    --account "${ACCOUNT}" \
    --qos "${QOS}"
