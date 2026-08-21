#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/y/my_mimic_mixgrpo/MimicKit"
python_bin="${MIMICKIT_PYTHON:-/home/y/miniconda3/envs/env_isaaclab/bin/python}"
smoke_args="args/ray_ordinal_aligned_add/roll_smoke_args.txt"
formal_args="args/ray_ordinal_aligned_add/roll_2k_8192_args.txt"
smoke_dir="output/ray_ordinal_aligned_add/roll_smoke_seed0"
formal_dir="output/ray_ordinal_aligned_add/roll_2k_8192_seed0"
smoke_samples=10240
formal_samples=524288000

cd "$repo_dir"
mkdir -p "$smoke_dir" "$formal_dir"

validate_metrics() {
  local metrics_file="$1"
  local expected_samples="$2"
  "$python_bin" - "$metrics_file" "$expected_samples" <<'PY'
import json
import math
import sys

metrics_file, expected_samples = sys.argv[1], int(sys.argv[2])
with open(metrics_file) as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
if not rows:
    raise SystemExit("metrics file is empty")
row = rows[-1]
if int(row["samples"]) != expected_samples:
    raise SystemExit(
        f"unexpected sample count: {row['samples']} != {expected_samples}")
required = {
    "Disc_Ray_Objective",
    "Disc_Absolute_Pos_Loss",
    "Disc_Absolute_Neg_Loss",
    "Disc_Ordinal_Near_Loss",
    "Disc_Ordinal_Far_Loss",
    "Disc_Order_Anchor_Ray_Acc",
    "Disc_Order_Ray_Neg_Acc",
    "Disc_Order_Full_Acc",
    "Disc_Reward_Mean",
    "Disc_Reward_Std",
}
missing = sorted(required.difference(row))
if missing:
    raise SystemExit(f"missing metrics: {missing}")
bad = {
    key: value for key, value in row.items()
    if isinstance(value, float) and not math.isfinite(value)
}
if bad:
    raise SystemExit(f"non-finite metrics: {bad}")
for key in (
        "Disc_Order_Anchor_Ray_Acc",
        "Disc_Order_Ray_Neg_Acc",
        "Disc_Order_Full_Acc"):
    if not 0.0 <= row[key] <= 1.0:
        raise SystemExit(f"invalid accuracy {key}={row[key]}")
print(json.dumps({
    "iteration": row["iteration"],
    "samples": row["samples"],
    "disc_ray_objective": row["Disc_Ray_Objective"],
    "order_anchor_ray": row["Disc_Order_Anchor_Ray_Acc"],
    "order_ray_negative": row["Disc_Order_Ray_Neg_Acc"],
    "order_full": row["Disc_Order_Full_Acc"],
    "reward_mean": row["Disc_Reward_Mean"],
    "reward_std": row["Disc_Reward_Std"],
}, sort_keys=True))
PY
}

if [[ ! -f "$smoke_dir/SMOKE_DONE" ]]; then
  PYTHONUNBUFFERED=1 "$python_bin" mimickit/run.py \
    --arg_file "$smoke_args" \
    > "$smoke_dir/console.log" 2>&1
  validate_metrics "$smoke_dir/train_metrics.jsonl" "$smoke_samples" \
    | tee "$smoke_dir/validation.json"
  touch "$smoke_dir/SMOKE_DONE"
fi

if [[ -f "$formal_dir/DONE" ]]; then
  exit 0
fi

resume_args=()
redirect_mode=">"
if [[ -f "$formal_dir/checkpoint.pt" ]]; then
  resume_args=(--resume_file "$formal_dir/checkpoint.pt")
  redirect_mode=">>"
fi

if [[ "$redirect_mode" == ">>" ]]; then
  PYTHONUNBUFFERED=1 "$python_bin" mimickit/run.py \
    --arg_file "$formal_args" "${resume_args[@]}" \
    >> "$formal_dir/console.log" 2>&1
else
  PYTHONUNBUFFERED=1 "$python_bin" mimickit/run.py \
    --arg_file "$formal_args" \
    > "$formal_dir/console.log" 2>&1
fi

validate_metrics "$formal_dir/train_metrics.jsonl" "$formal_samples" \
  | tee "$formal_dir/validation.json"
touch "$formal_dir/DONE"
