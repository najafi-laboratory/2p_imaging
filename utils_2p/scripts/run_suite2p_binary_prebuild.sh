#!/bin/bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 MANIFEST INDEX RAW_PATH WORKERS [BATCH_SIZE]" >&2
  exit 2
fi

manifest=$1
index=$2
raw_path=$3
workers=$4
batch_size=${5:-5000}

repo_root=/storage/home/hcoda1/3/grubin6/2p_imaging
python_bin=/storage/project/r-fnajafi3-0/grubin6/shared_envs/2p_preprocessing_qc_suite2p_1x/bin/python

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" utils_2p/scripts/prebuild_suite2p_binary_parallel.py \
  --manifest "$manifest" \
  --index "$index" \
  --raw-path "$raw_path" \
  --workers "$workers" \
  --batch-size "$batch_size" \
  --force
