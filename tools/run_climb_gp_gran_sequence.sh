#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${MIMICKIT_PYTHON:-/home/y/miniconda3/envs/env_isaaclab/bin/python}"
num_envs=4096
iterations=2000
steps_per_iter=32
target_samples=$((num_envs * iterations * steps_per_iter))
seed=0
root="output/climb_gp_gran_2k_4096_seed${seed}"
mkdir -p "$root"

exec 9>"$root/.campaign.lock"
if ! flock -n 9; then
  echo "The Climb GP/GraN campaign is already running." >&2
  exit 3
fi

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
  [[ -s "$out/checkpoint.pt" && -s "$out/model.pt" ]] || return 1
  [[ "$(checkpoint_samples "$out/checkpoint.pt")" -ge "$target_samples" ]]
}

run_job() {
  local name="$1"
  local agent_config="$2"
  local out="$root/$name"
  mkdir -p "$out"

  if job_complete "$out"; then
    echo "[$(date --iso-8601=seconds)] SKIP $name (complete)" | tee -a "$root/campaign.log"
    return
  fi

  local -a resume=()
  if [[ -s "$out/checkpoint.pt" ]]; then
    resume=(--resume_file "$out/checkpoint.pt")
  fi

  echo "[$(date --iso-8601=seconds)] START $name" | tee -a "$root/campaign.log"
  PYTHONUNBUFFERED=1 "$python_bin" mimickit/run.py \
    --mode train \
    --devices cuda:0 \
    --engine_config data/engines/isaac_lab_engine.yaml \
    --env_config data/envs/paper_benchmark/add_climb_pt_env.yaml \
    --agent_config "$agent_config" \
    --num_envs "$num_envs" \
    --max_samples "$target_samples" \
    --rand_seed "$seed" \
    --visualize false \
    --save_int_models true \
    --logger txt \
    --out_dir "$out" \
    "${resume[@]}" >>"$out/console.log" 2>&1
  touch "$out/DONE"
  echo "[$(date --iso-8601=seconds)] DONE $name" | tee -a "$root/campaign.log"
}

echo "target_samples=$target_samples" >"$root/plan.txt"
echo "order=01_add_bothgp_gp01,02_add_bothgp_gp10,03_gran_add" >>"$root/plan.txt"

run_job 01_add_bothgp_gp01 data/agents/add_humanoid_climb_paper_agent.yaml
run_job 02_add_bothgp_gp10 data/agents/add_humanoid_climb_gp1_agent.yaml
run_job 03_gran_add data/agents/gran_add_humanoid_climb_agent.yaml

echo "[$(date --iso-8601=seconds)] CAMPAIGN DONE" | tee -a "$root/campaign.log"
