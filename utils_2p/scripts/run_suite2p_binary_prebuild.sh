#!/bin/bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage:" >&2
  echo "  $0 PROCESSED_SESSION RAW_PATH OUTPUT_ROOT WORKERS [BATCH_SIZE] [TARGET_STRUCTURE]" >&2
  exit 2
fi

processed_session=$1
raw_path=$2
output_root=$3
workers=$4
batch_size=${5:-5000}
target_structure=${6:-dendrite}

repo_root=/storage/home/hcoda1/3/grubin6/2p_imaging
python_bin=/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_suite2p_1x/bin/python

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" utils_2p/scripts/prebuild_suite2p_binary_parallel.py \
  --processed-session "$processed_session" \
  --raw-path "$raw_path" \
  --output-root "$output_root" \
  --workers "$workers" \
  --batch-size "$batch_size" \
  --target-structure "$target_structure" \
  --force
