#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${MIMICKIT_PYTHON:-/home/y/miniconda3/envs/env_isaaclab/bin/python}"
num_envs=4096
iterations=1000
steps_per_iter=32
target_samples=$((num_envs * iterations * steps_per_iter))
seed=0
root="output/gadd_ablation_run_1k_4096_seed${seed}"
mkdir -p "$root"

exec 9>"$root/.campaign.lock"
if ! flock -n 9; then
  printf 'Another G-ADD ablation launcher is already running.\n' >&2
  exit 3
fi

declare -A agents=(
  [01_add]="data/agents/add_humanoid_agent.yaml"
  [02_refconcat]="data/agents/gadd_refconcat_humanoid_agent.yaml"
  [03_global_metric]="data/agents/gadd_global_metric_humanoid_agent.yaml"
  [04_metric_raw_gp]="data/agents/gadd_metric_raw_gp_humanoid_agent.yaml"
  [05_metric_z_gp]="data/agents/gadd_metric_z_gp_humanoid_agent.yaml"
)

checkpoint_samples() {
  "$python_bin" - "$1" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint["trainer_state"]["sample_count"]))
PY
}

job_complete() {
  local out="$1"
  [[ -s "$out/checkpoint.pt" && -s "$out/model.pt" && -s "$out/log.txt" ]] || return 1
  local samples
  samples="$(checkpoint_samples "$out/checkpoint.pt")" || return 1
  [[ "$samples" -ge "$target_samples" ]]
}

run_job() {
  local name="$1"
  local out="$root/$name"
  local agent="${agents[$name]}"
  mkdir -p "$out"

  if job_complete "$out"; then
    printf '[%s] SKIP %s (complete)\n' "$(date --iso-8601=seconds)" "$name" | tee -a "$root/campaign.log"
    return 0
  fi

  local -a resume=()
  if [[ -s "$out/checkpoint.pt" ]]; then
    resume=(--resume_file "$out/checkpoint.pt")
  fi

  printf '[%s] START %s\n' "$(date --iso-8601=seconds)" "$name" | tee -a "$root/campaign.log"
  PYTHONUNBUFFERED=1 "$python_bin" mimickit/run.py \
    --mode train \
    --devices cuda:0 \
    --engine_config data/engines/isaac_lab_engine.yaml \
    --env_config data/envs/paper_benchmark/add_run_env.yaml \
    --agent_config "$agent" \
    --num_envs "$num_envs" \
    --max_samples "$target_samples" \
    --rand_seed "$seed" \
    --visualize false \
    --save_int_models false \
    --logger txt \
    --out_dir "$out" \
    "${resume[@]}" >> "$out/console.log" 2>&1 || true

  if ! job_complete "$out"; then
    printf '[%s] INCOMPLETE %s\n' "$(date --iso-8601=seconds)" "$name" | tee -a "$root/campaign.log"
    return 1
  fi

  touch "$out/DONE"
  printf '[%s] DONE %s\n' "$(date --iso-8601=seconds)" "$name" | tee -a "$root/campaign.log"
}

run_pair() {
  local first="$1" second="$2"
  set +e
  run_job "$first" & local first_pid=$!
  run_job "$second" & local second_pid=$!
  wait "$first_pid"; local first_rc=$?
  wait "$second_pid"; local second_rc=$?
  set -e

  # A single 16 GB GPU normally fits both 4096-env Run jobs.  If a driver or
  # allocator rejects the concurrent pair, resume only the incomplete jobs
  # serially rather than discarding their valid checkpoints.
  if [[ "$first_rc" -ne 0 ]]; then
    run_job "$first"
  fi
  if [[ "$second_rc" -ne 0 ]]; then
    run_job "$second"
  fi
}

printf 'target_samples=%s\n' "$target_samples" > "$root/plan.txt"
printf 'stages=01_add+02_refconcat,03_global_metric+04_metric_raw_gp,05_metric_z_gp\n' >> "$root/plan.txt"

run_pair 01_add 02_refconcat
run_pair 03_global_metric 04_metric_raw_gp
run_job 05_metric_z_gp

printf '[%s] CAMPAIGN DONE\n' "$(date --iso-8601=seconds)" | tee -a "$root/campaign.log"
