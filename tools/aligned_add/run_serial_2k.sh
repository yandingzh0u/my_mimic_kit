#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/y/my_mimic_mixgrpo/MimicKit"
python_bin="/home/y/miniconda3/envs/env_isaaclab/bin/python"

cd "$repo_dir"

roll_out="output/aligned_add_roll_2k_8192_seed0"
spinkick_out="output/aligned_add_spinkick_2k_8192_seed0"
mkdir -p "$roll_out" "$spinkick_out"

PYTHONUNBUFFERED=1 "$python_bin" mimickit/run.py \
  --arg_file args/aligned_add_humanoid_roll_2k_8192_args.txt \
  > "$roll_out/console.log" 2>&1

PYTHONUNBUFFERED=1 "$python_bin" mimickit/run.py \
  --arg_file args/aligned_add_humanoid_spinkick_2k_8192_args.txt \
  > "$spinkick_out/console.log" 2>&1
