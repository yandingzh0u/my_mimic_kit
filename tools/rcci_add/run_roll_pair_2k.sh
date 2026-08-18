#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
python_bin="${MIMICKIT_PYTHON:-/home/y/miniconda3/envs/env_isaaclab/bin/python}"

if [[ ! -x "$python_bin" ]]; then
  printf 'ERROR: Python executable not found: %s\n' "$python_bin" >&2
  exit 2
fi

cd "$repo_dir"

run_one() {
  local name="$1"
  local arg_file="$2"
  local out_dir="output/$name"
  mkdir -p "$out_dir"
  PYTHONUNBUFFERED=1 "$python_bin" mimickit/run.py \
    --arg_file "$arg_file" > "$out_dir/console.log" 2>&1
}

run_one "rcci_absolute_roll_2k_8192_seed0" \
  "args/rcci_absolute_humanoid_roll_2k_8192_args.txt"
run_one "rcci_residual_roll_2k_8192_seed0" \
  "args/rcci_residual_humanoid_roll_2k_8192_args.txt"
