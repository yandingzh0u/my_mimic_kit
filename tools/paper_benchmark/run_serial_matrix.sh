#!/usr/bin/env bash
set -Eeuo pipefail

# Serial paper benchmark campaign.
#
# Default behavior:
#   1. wait until any pre-existing MimicKit training process exits;
#   2. run every 64-env, two-iteration smoke test serially;
#   3. run one 8192-env iteration per method to validate scale/replay capacity;
#   4. only if every smoke passes, run every 8192-env, 2000-iteration job.
#
# Re-running this script skips jobs carrying a DONE marker and resumes an
# interrupted job from output/.../checkpoint.pt through --resume_file.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
python_bin="${MIMICKIT_PYTHON:-/home/y/miniconda3/envs/env_isaaclab/bin/python}"
wait_for_gpu=true
run_smoke=true
run_scale_smoke=true
run_formal=true

usage() {
  printf '%s\n' \
    "Usage: $0 [--smoke-only|--scale-smoke-only|--formal-only] [--no-wait] [--python PATH]" \
    "" \
    "Environment:" \
    "  MIMICKIT_PYTHON   Python executable (default: env_isaaclab Python)" \
    "  WAIT_POLL_SECONDS Poll interval while another run.py owns the GPU"
}

while (($# > 0)); do
  case "$1" in
    --smoke-only)
      run_smoke=true
      run_scale_smoke=true
      run_formal=false
      ;;
    --scale-smoke-only)
      run_smoke=false
      run_scale_smoke=true
      run_formal=false
      ;;
    --formal-only)
      run_smoke=false
      run_scale_smoke=false
      run_formal=true
      ;;
    --no-wait)
      wait_for_gpu=false
      ;;
    --python)
      shift
      if (($# == 0)); then
        printf 'ERROR: --python requires a path\n' >&2
        exit 2
      fi
      python_bin="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'ERROR: unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

cd "$repo_dir"

if [[ ! -x "$python_bin" ]]; then
  printf 'ERROR: Python executable not found: %s\n' "$python_bin" >&2
  exit 2
fi

methods=(deepmimic amp add residual)
motions=(run backflip crawl getup_facedown spinkick climb)

declare -A arg_files=(
  [deepmimic]="args/paper_benchmark/deepmimic_2k_8192_args.txt"
  [amp]="args/paper_benchmark/amp_2k_8192_args.txt"
  [add]="args/paper_benchmark/add_2k_8192_args.txt"
  [residual]="args/paper_benchmark/residual_2k_8192_args.txt"
)

declare -A agent_files=(
  [deepmimic]="data/agents/deepmimic_humanoid_ppo_agent.yaml"
  [amp]="data/agents/amp_humanoid_agent.yaml"
  [add]="data/agents/add_humanoid_agent.yaml"
  [residual]="data/agents/rcci_add_humanoid_agent.yaml"
)

smoke_envs=64
steps_per_iter=32
smoke_iters=2
scale_smoke_envs=8192
scale_smoke_iters=1
formal_envs=8192
formal_iters=2000
smoke_samples=$((smoke_envs * steps_per_iter * smoke_iters))
scale_smoke_samples=$((scale_smoke_envs * steps_per_iter * scale_smoke_iters))
formal_samples=$((formal_envs * steps_per_iter * formal_iters))
seed=0

campaign_root="output/paper_benchmark"
smoke_root="output/paper_benchmark_smoke"
scale_smoke_root="output/paper_benchmark_scale_smoke"
mkdir -p "$campaign_root" "$smoke_root" "$scale_smoke_root"

# Prevent two copies of this campaign from controlling the same output tree.
exec 9>"$campaign_root/.serial_matrix.lock"
if ! flock -n 9; then
  printf 'ERROR: another paper benchmark matrix launcher holds %s\n' \
    "$campaign_root/.serial_matrix.lock" >&2
  exit 3
fi

events_file="$campaign_root/events.tsv"
plan_file="$campaign_root/plan.tsv"
if [[ ! -f "$events_file" ]]; then
  printf 'timestamp\tstage\tmethod\tmotion\tstate\toutput\tdetail\n' > "$events_file"
fi

append_event() {
  local stage="$1"
  local method="$2"
  local motion="$3"
  local state="$4"
  local out_dir="$5"
  local detail="${6:-}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$stage" "$method" "$motion" \
    "$state" "$out_dir" "$detail" >> "$events_file"
}

write_plan() {
  printf 'order\tmethod\tmotion\tenv_config\tformal_output\n' > "$plan_file"
  local order=0
  local method motion env_config out_dir
  for method in "${methods[@]}"; do
    for motion in "${motions[@]}"; do
      order=$((order + 1))
      env_config="data/envs/paper_benchmark/${method}_${motion}_env.yaml"
      out_dir="$campaign_root/${method}_${motion}_2k_8192_seed${seed}"
      printf '%d\t%s\t%s\t%s\t%s\n' \
        "$order" "$method" "$motion" "$env_config" "$out_dir" >> "$plan_file"
    done
  done
}

preflight() {
  local method motion env_config
  for method in "${methods[@]}"; do
    [[ -f "${arg_files[$method]}" ]] || {
      printf 'ERROR: missing arg file: %s\n' "${arg_files[$method]}" >&2
      return 1
    }
    [[ -f "${agent_files[$method]}" ]] || {
      printf 'ERROR: missing agent file: %s\n' "${agent_files[$method]}" >&2
      return 1
    }
    for motion in "${motions[@]}"; do
      env_config="data/envs/paper_benchmark/${method}_${motion}_env.yaml"
      [[ -f "$env_config" ]] || {
        printf 'ERROR: missing env config: %s\n' "$env_config" >&2
        return 1
      }
    done
  done
  [[ -f data/assets/objects/climbing_box.usd ]] || {
    printf 'ERROR: missing Isaac Lab Climb asset: data/assets/objects/climbing_box.usd\n' >&2
    return 1
  }
  [[ -f data/assets/objects/climbing_box.xml ]] || {
    printf 'ERROR: missing Climb source asset: data/assets/objects/climbing_box.xml\n' >&2
    return 1
  }
  [[ -f tools/paper_eval/evaluate_checkpoint.py ]] || {
    printf 'ERROR: missing final evaluator: tools/paper_eval/evaluate_checkpoint.py\n' >&2
    return 1
  }

  # Isaac Lab training is CUDA-only.  Fail before writing a misleading job
  # STARTED/FAILED pair when the launcher itself is running in a CPU-only
  # shell or container.
  if ! "$python_bin" - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print(
        "ERROR: CUDA is unavailable to the benchmark launcher; "
        "run this script from the GPU-enabled host/session.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
  then
    return 1
  fi
}

wait_for_existing_training() {
  local poll_seconds="${WAIT_POLL_SECONDS:-30}"
  if [[ "$wait_for_gpu" != true ]]; then
    return
  fi

  local pids
  while pids="$(pgrep -f '[m]imickit/run.py' || true)" && [[ -n "$pids" ]]; do
    printf '[%s] Waiting for existing MimicKit training PID(s): %s\n' \
      "$(date --iso-8601=seconds)" "$(tr '\n' ' ' <<< "$pids")"
    sleep "$poll_seconds"
  done
}

run_job() {
  local stage="$1"
  local method="$2"
  local motion="$3"
  local num_envs="$4"
  local max_samples="$5"
  local save_int_models="$6"
  local root="$7"

  local env_config="data/envs/paper_benchmark/${method}_${motion}_env.yaml"
  local suffix
  if [[ "$stage" == "smoke" ]]; then
    suffix="smoke"
  elif [[ "$stage" == "scale_smoke" ]]; then
    suffix="scale_smoke"
  else
    suffix="2k_8192"
  fi
  local out_dir="$root/${method}_${motion}_${suffix}_seed${seed}"
  local done_file="$out_dir/DONE"
  local checkpoint_file="$out_dir/checkpoint.pt"
  local console_file="$out_dir/console.log"
  mkdir -p "$out_dir"

  local eval_name="final"
  local eval_num_envs=256
  local eval_steps=300
  if [[ "$stage" != "formal" ]]; then
    eval_name="$stage"
    eval_num_envs=2
    eval_steps=2
  fi
  local eval_dir="$out_dir/eval/$eval_name"
  local eval_summary="$eval_dir/summary.json"
  if [[ -f "$done_file" && -s "$eval_summary" ]]; then
    printf '[%s] SKIP %s/%s/%s (DONE)\n' \
      "$(date --iso-8601=seconds)" "$stage" "$method" "$motion"
    append_event "$stage" "$method" "$motion" "SKIPPED_DONE" "$out_dir"
    return 0
  fi

  local -a resume_args=()
  local redirect_mode="fresh"
  if [[ -f "$checkpoint_file" ]]; then
    resume_args=(--resume_file "$checkpoint_file")
    redirect_mode="resume"
  fi

  append_event "$stage" "$method" "$motion" "STARTED" "$out_dir" \
    "num_envs=$num_envs,max_samples=$max_samples,$redirect_mode"
  printf '[%s] START %s/%s/%s (%s envs, target %s samples, %s)\n' \
    "$(date --iso-8601=seconds)" "$stage" "$method" "$motion" \
    "$num_envs" "$max_samples" "$redirect_mode"

  local -a cmd=(
    "$python_bin" mimickit/run.py
    --arg_file "${arg_files[$method]}"
    --env_config "$env_config"
    --out_dir "$out_dir"
    --num_envs "$num_envs"
    --max_samples "$max_samples"
    --rand_seed "$seed"
    --save_int_models "$save_int_models"
    "${resume_args[@]}"
  )

  printf '[%s] COMMAND' "$(date --iso-8601=seconds)" >> "$console_file"
  printf ' %q' "${cmd[@]}" >> "$console_file"
  printf '\n' >> "$console_file"

  if PYTHONUNBUFFERED=1 "${cmd[@]}" >> "$console_file" 2>&1; then
    :
  else
    local rc=$?
    append_event "$stage" "$method" "$motion" "FAILED" "$out_dir" "exit=$rc"
    printf '[%s] FAILED %s/%s/%s (exit %s); see %s\n' \
      "$(date --iso-8601=seconds)" "$stage" "$method" "$motion" \
      "$rc" "$console_file" >&2
    return "$rc"
  fi

  if [[ ! -s "$out_dir/model.pt" || ! -s "$checkpoint_file" || ! -s "$out_dir/log.txt" ]]; then
    append_event "$stage" "$method" "$motion" "FAILED_ARTIFACT_CHECK" \
      "$out_dir" "missing model.pt/checkpoint.pt/log.txt"
    printf 'ERROR: %s/%s/%s exited 0 but required artifacts are missing\n' \
      "$stage" "$method" "$motion" >&2
    return 4
  fi

  # Every smoke also loads its checkpoint through the shared evaluator for two
  # steps; this catches method-specific observation, metric, and Climb-object
  # failures before any 524M-sample job starts. Formal runs use the fixed
  # 256-episode, 300-step paper protocol.
  if [[ "$stage" == "smoke" || "$stage" == "scale_smoke"
        || "$stage" == "formal" ]]; then
    local eval_console="$eval_dir/console.log"
    mkdir -p "$eval_dir"

    if [[ -s "$eval_summary" ]]; then
      append_event "$stage" "$method" "$motion" "EVAL_SKIPPED_DONE" \
        "$eval_dir" "summary.json exists"
    else
      local -a eval_cmd=(
        "$python_bin" tools/paper_eval/evaluate_checkpoint.py
        --model-file "$out_dir/model.pt"
        --env-config "$env_config"
        --agent-config "${agent_files[$method]}"
        --engine-config data/engines/isaac_lab_engine.yaml
        --method "$method"
        --motion "$motion"
        --num-envs "$eval_num_envs"
        --steps "$eval_steps"
        --start-mode phase0
        --condition nominal
        --seed "$seed"
        --out-dir "$eval_dir"
      )
      append_event "$stage" "$method" "$motion" "EVAL_STARTED" "$eval_dir"
      printf '[%s] EVAL %s/%s %s checkpoint (%s envs x %s steps)\n' \
        "$(date --iso-8601=seconds)" "$method" "$motion" "$eval_name" \
        "$eval_num_envs" "$eval_steps"
      printf '[%s] COMMAND' "$(date --iso-8601=seconds)" >> "$eval_console"
      printf ' %q' "${eval_cmd[@]}" >> "$eval_console"
      printf '\n' >> "$eval_console"

      if PYTHONUNBUFFERED=1 "${eval_cmd[@]}" >> "$eval_console" 2>&1; then
        :
      else
        local eval_rc=$?
        append_event "$stage" "$method" "$motion" "EVAL_FAILED" \
          "$eval_dir" "exit=$eval_rc"
        printf '[%s] EVAL FAILED %s/%s (exit %s); see %s\n' \
          "$(date --iso-8601=seconds)" "$method" "$motion" \
          "$eval_rc" "$eval_console" >&2
        return "$eval_rc"
      fi

      if [[ ! -s "$eval_summary"
            || ! -s "$eval_dir/episodes.npz"
            || ! -s "$eval_dir/timeseries.npz" ]]; then
        append_event "$stage" "$method" "$motion" \
          "EVAL_FAILED_ARTIFACT_CHECK" "$eval_dir" \
          "missing summary.json/episodes.npz/timeseries.npz"
        printf 'ERROR: final evaluator omitted required artifacts for %s/%s\n' \
          "$method" "$motion" >&2
        return 5
      fi
      append_event "$stage" "$method" "$motion" "EVAL_DONE" "$eval_dir"
    fi
  fi

  printf 'stage=%s\nmethod=%s\nmotion=%s\nnum_envs=%s\ntarget_samples=%s\nfinished=%s\n' \
    "$stage" "$method" "$motion" "$num_envs" "$max_samples" \
    "$(date --iso-8601=seconds)" > "$done_file"
  append_event "$stage" "$method" "$motion" "DONE" "$out_dir"
  printf '[%s] DONE %s/%s/%s\n' \
    "$(date --iso-8601=seconds)" "$stage" "$method" "$motion"
}

run_stage() {
  local stage="$1"
  local num_envs="$2"
  local max_samples="$3"
  local save_int_models="$4"
  local root="$5"
  local method motion

  for method in "${methods[@]}"; do
    for motion in "${motions[@]}"; do
      run_job "$stage" "$method" "$motion" "$num_envs" "$max_samples" \
        "$save_int_models" "$root"
    done
  done
}

write_plan
preflight
wait_for_existing_training

printf 'Campaign plan: %s\n' "$plan_file"
printf 'Event log:    %s\n' "$events_file"
printf 'Formal budget: %d methods x %d motions x %d samples = %d samples\n' \
  "${#methods[@]}" "${#motions[@]}" "$formal_samples" \
  $(( ${#methods[@]} * ${#motions[@]} * formal_samples ))
printf '%s\n' \
  'ETA warning: current 8192-env runs take about 2.2 h each; 24 formal jobs' \
  'need at least ~52 h and AMP/Climb can extend the campaign to 2--3 days.'

if [[ "$run_smoke" == true ]]; then
  run_stage "smoke" "$smoke_envs" "$smoke_samples" false "$smoke_root"
fi

if [[ "$run_scale_smoke" == true ]]; then
  # Run is sufficient here: this stage validates 8192-env simulator memory,
  # the 262144-sample on-policy rollout, and AMP/ADD's 200000-slot replay cap.
  for method in "${methods[@]}"; do
    run_job "scale_smoke" "$method" "run" "$scale_smoke_envs" \
      "$scale_smoke_samples" false "$scale_smoke_root"
  done
fi

if [[ "$run_formal" == true ]]; then
  run_stage "formal" "$formal_envs" "$formal_samples" true "$campaign_root"
fi

printf '[%s] Entire requested matrix is complete.\n' "$(date --iso-8601=seconds)"
