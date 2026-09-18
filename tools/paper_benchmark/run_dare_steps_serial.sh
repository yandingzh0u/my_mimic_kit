#!/usr/bin/env bash
set -Eeuo pipefail

# Clean v6 parent followed by the Step 3--7 variants.  Each job is independent
# and uses the same Climb/8192/seed-0 budget; jobs are deliberately serial so
# they never contend for Isaac Lab's single GPU.
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
python_bin="${MIMICKIT_PYTHON:-/home/y/miniconda3/envs/env_isaaclab/bin/python}"
arg_file="args/paper_benchmark/dare_2k_8192_args.txt"
env_config="data/envs/paper_benchmark/dare_climb_env.yaml"
engine_config="data/engines/isaac_lab_engine.yaml"
target_samples=786432000  # 3000 * 8192 * 32

training_complete() {
  local out_dir="$1"
  [[ -s "$out_dir/checkpoint.pt" && -s "$out_dir/model.pt" \
      && -s "$out_dir/train_metrics.jsonl" ]] || return 1
  "$python_bin" - "$out_dir/train_metrics.jsonl" "$target_samples" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    rows = [line for line in stream if line.strip()]
if not rows:
    raise SystemExit(1)
last = json.loads(rows[-1])
raise SystemExit(0 if int(last.get("samples", -1)) >= int(sys.argv[2]) else 1)
PY
}

run_one() {
  local name="$1" agent_config="$2"
  local out_dir="output/paper_benchmark/${name}"
  mkdir -p "$out_dir"
  if [[ -f "$out_dir/DONE" ]] && training_complete "$out_dir"; then
    echo "[$(date --iso-8601=seconds)] SKIP $name (DONE)"
    return
  fi
  echo "[$(date --iso-8601=seconds)] START $name"
  local attempt
  for attempt in 1 2 3; do
    if "$python_bin" mimickit/run.py \
      --arg_file "$arg_file" \
      --env_config "$env_config" \
      --agent_config "$agent_config" \
      --engine_config "$engine_config" \
      --out_dir "$out_dir" \
      --num_envs 8192 \
      --max_samples "$target_samples" \
      --rand_seed 0 \
      --save_int_models true \
      --logger txt >"$out_dir/console.log" 2>&1 \
      && training_complete "$out_dir"; then
      touch "$out_dir/DONE"
      echo "[$(date --iso-8601=seconds)] DONE $name"
      # Isaac Kit occasionally leaves its USD/database resources alive for a
      # few seconds after process exit.  Do not start the next stage on them.
      sleep 30
      return
    fi
    echo "[$(date --iso-8601=seconds)] RETRY $name ($attempt/3)"
    sleep 60
  done
  echo "[$(date --iso-8601=seconds)] FAILED $name after 3 attempts" >&2
  return 1
}

run_smoke() {
  local name="$1" agent_config="$2"
  local out_dir="output/paper_benchmark_smoke/${name}"
  if [[ -f "$out_dir/DONE" ]]; then return; fi
  mkdir -p "$out_dir"
  echo "[$(date --iso-8601=seconds)] SMOKE $name"
  "$python_bin" mimickit/run.py \
    --arg_file "$arg_file" \
    --env_config "$env_config" \
    --agent_config "$agent_config" \
    --engine_config "$engine_config" \
    --out_dir "$out_dir" \
    --num_envs 64 \
    --max_samples 4096 \
    --rand_seed 0 \
    --save_int_models false \
    --logger txt >"$out_dir/console.log" 2>&1
  touch "$out_dir/DONE"
}

declare -a jobs=(
  "dare_v6_clean_climb_3000_8192_seed0|data/agents/dare_humanoid_agent.yaml"
  "dare_step3_anchorroot_2080_climb_3000_8192_seed0|data/agents/step_variants/dare_step3_anchorroot_agent.yaml"
  "dare_step4_objective_target_climb_3000_8192_seed0|data/agents/step_variants/dare_step4_objective_agent.yaml"
  "dare_step5_fixedprobe_climb_3000_8192_seed0|data/agents/step_variants/dare_step5_adaptive_agent.yaml"
  "dare_step6_stablefreeze_climb_3000_8192_seed0|data/agents/step_variants/dare_step6_stablefreeze_agent.yaml"
  "dare_step7_nologitreg_climb_3000_8192_seed0|data/agents/step_variants/dare_step7_nologitreg_agent.yaml"
)

for job in "${jobs[@]}"; do
  IFS='|' read -r name agent_config <<<"$job"
  run_smoke "${name}_smoke" "$agent_config"
done
if [[ "${1:-}" == "--smoke-only" ]]; then
  echo "[$(date --iso-8601=seconds)] ALL DARE SMOKE JOBS COMPLETE"
  exit 0
fi
for job in "${jobs[@]}"; do
  IFS='|' read -r name agent_config <<<"$job"
  run_one "$name" "$agent_config"
done
echo "[$(date --iso-8601=seconds)] ALL DARE STEP JOBS COMPLETE"
