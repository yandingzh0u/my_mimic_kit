#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/y/my_mimic_mixgrpo/MimicKit"
python_bin="${MIMICKIT_PYTHON:-/home/y/miniconda3/envs/env_isaaclab/bin/python}"
arg_file="args/relative_logistic_aligned_add/roll_10k_4096_args.txt"
out_dir="output/relative_logistic_aligned_add/roll_10k_4096_seed0"
target_samples=1310720000

cd "$repo_dir"
mkdir -p "$out_dir"

if [[ -f "$out_dir/DONE" ]]; then
  exit 0
fi

resume_args=()
redirect_mode=">"
if [[ -f "$out_dir/checkpoint.pt" ]]; then
  resume_args=(--resume_file "$out_dir/checkpoint.pt")
  redirect_mode=">>"
fi

if [[ "$redirect_mode" == ">>" ]]; then
  PYTHONUNBUFFERED=1 "$python_bin" mimickit/run.py \
    --arg_file "$arg_file" "${resume_args[@]}" \
    >> "$out_dir/console.log" 2>&1
else
  PYTHONUNBUFFERED=1 "$python_bin" mimickit/run.py \
    --arg_file "$arg_file" \
    > "$out_dir/console.log" 2>&1
fi

last_samples="$($python_bin - "$out_dir/train_metrics.jsonl" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
print(int(rows[-1]["samples"]) if rows else 0)
PY
)"

if [[ "$last_samples" -ne "$target_samples" ]]; then
  echo "Training stopped at $last_samples / $target_samples samples" >&2
  exit 1
fi

touch "$out_dir/DONE"
