#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

python_bin="/home/y/miniconda3/envs/env_isaaclab/bin/python"
agent="data/agents/g1_real_deployment/dare_g1_sim2real_agent.yaml"
engine="data/engines/isaac_lab_engine.yaml"
suite_dir="output/g1_interaction_suite"
mkdir -p "$suite_dir"

motions=(climb_slope carry_box sit_sofa backflip)

last_iteration() {
  local log_file="$1"
  [ -f "$log_file" ] || return 0
  tr '\r' '\n' < "$log_file" | awk 'NF && $1 ~ /^[0-9]+$/ {last=$1} END {print last}'
}

run_motion() {
  local motion="$1"
  env_cfg="data/envs/g1_interaction_suite/dare_g1_${motion}_env.yaml"
  out_dir="$suite_dir/dare_g1_${motion}_2k_4096_seed0"
  mkdir -p "$out_dir"
  if [ "$(last_iteration "$out_dir/log.txt")" = "1999" ] \
      && ! rg -q 'Traceback|RuntimeError|Boost\.Python\.ArgumentError' "$out_dir/console.log" 2>/dev/null; then
    printf '%s\tSKIP_ALREADY_DONE\t%s\n' "$(date --iso-8601=seconds)" "$motion" >> "$suite_dir/events.tsv"
    return 0
  fi

  for attempt in 1 2 3; do
    if [ -f "$out_dir/console.log" ]; then
      mv "$out_dir/console.log" "$out_dir/console.failed_attempt${attempt}.$(date +%s).log"
    fi
    printf '%s\tSTART_ATTEMPT_%d\t%s\n' "$(date --iso-8601=seconds)" "$attempt" "$motion" >> "$suite_dir/events.tsv"
    set +e
    TERM=xterm "$python_bin" mimickit/run.py \
      --mode train \
      --num_envs 4096 \
      --devices cuda:0 \
      --engine_config "$engine" \
      --env_config "$env_cfg" \
      --agent_config "$agent" \
      --max_samples 262144000 \
      --rand_seed 0 \
      --visualize false \
      --save_int_models true \
      --logger txt \
      --out_dir "$out_dir" 2>&1 | tee "$out_dir/console.log"
    status=${PIPESTATUS[0]}
    set -e
    iter="$(last_iteration "$out_dir/log.txt")"
    if [ "$status" -eq 0 ] && [ "$iter" = "1999" ] \
        && ! rg -q 'Traceback|RuntimeError|Boost\.Python\.ArgumentError' "$out_dir/console.log" 2>/dev/null; then
      printf '%s\tDONE\t%s\n' "$(date --iso-8601=seconds)" "$motion" >> "$suite_dir/events.tsv"
      return 0
    fi
    printf '%s\tFAILED_ATTEMPT_%d_STATUS_%d_ITER_%s\t%s\n' \
      "$(date --iso-8601=seconds)" "$attempt" "$status" "${iter:-none}" "$motion" >> "$suite_dir/events.tsv"
    sleep 30
  done
  return 1
}

for motion in "${motions[@]}"; do
  run_motion "$motion" || exit $?
done
