#!/usr/bin/env bash
set -Eeuo pipefail

# Serial 2x2 ablation campaign for DARE on Climb.
#
# The existing Full DARE controller is the (+ group, + calibration) cell:
#   output/paper_benchmark/dare_climb_2k_8192_seed0
# This launcher executes only the three missing cells, in order:
#   wogroup        (- group embedding, + anchor calibration)
#   wocalibration  (+ group embedding, fixed kappa=1)
#   base           (- group embedding, fixed kappa=1)
#
# Each formal run is protected by a 64-env smoke and a 8192-env scale smoke.
# Interrupted jobs resume from checkpoint.pt; a valid completed job is skipped.

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
    "Runs the three missing Climb DARE ablations serially using env_isaaclab."
}

while (($# > 0)); do
  case "$1" in
    --smoke-only)
      run_smoke=true
      run_scale_smoke=false
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

readonly motion="climb"
readonly seed=0
readonly env_config="data/envs/paper_benchmark/dare_climb_env.yaml"
readonly engine_config="data/engines/isaac_lab_engine.yaml"
readonly args_file="args/paper_benchmark/dare_2k_8192_args.txt"
readonly full_dare_output="output/paper_benchmark/dare_climb_2k_8192_seed0"

variants=(wogroup wocalibration base)
declare -A agent_configs=(
  [wogroup]="data/agents/ablations/dare_climb_wogroup_agent.yaml"
  [wocalibration]="data/agents/ablations/dare_climb_wocalibration_agent.yaml"
  [base]="data/agents/ablations/dare_climb_base_agent.yaml"
)
declare -A group_embedding=(
  [wogroup]="false"
  [wocalibration]="true"
  [base]="false"
)
declare -A anchor_calibration=(
  [wogroup]="true"
  [wocalibration]="false"
  [base]="false"
)

readonly campaign_root="output/paper_benchmark/ablations"
readonly smoke_root="output/paper_benchmark_smoke/ablations"
readonly scale_root="output/paper_benchmark_scale_smoke/ablations"
readonly plan_file="$campaign_root/climb_dare_ablation_plan.tsv"
readonly events_file="$campaign_root/climb_dare_ablation_events.tsv"
readonly launcher_log="$campaign_root/climb_dare_ablation_launcher.log"
mkdir -p "$campaign_root" "$smoke_root" "$scale_root"

readonly smoke_envs=64
readonly smoke_iters=2
readonly scale_envs=8192
readonly scale_iters=3
readonly formal_envs=8192
readonly formal_iters=2000
readonly steps_per_iter=32
readonly smoke_samples=$((smoke_envs * steps_per_iter * smoke_iters))
readonly scale_samples=$((scale_envs * steps_per_iter * scale_iters))
readonly formal_samples=$((formal_envs * steps_per_iter * formal_iters))

exec 9>"$campaign_root/.climb_dare_ablation.lock"
if ! flock -n 9; then
  printf 'ERROR: another Climb DARE ablation launcher holds the campaign lock\n' >&2
  exit 3
fi

if [[ ! -f "$events_file" ]]; then
  printf 'timestamp\tstage\tvariant\tstate\toutput\tdetail\n' > "$events_file"
fi

append_event() {
  local stage="$1"
  local variant="$2"
  local state="$3"
  local out_dir="$4"
  local detail="${5:-}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$stage" "$variant" "$state" \
    "$out_dir" "$detail" >> "$events_file"
}

write_plan() {
  printf 'order\tvariant\tgroup_embedding\tanchor_calibration\tagent_config\tformal_output\tstatus\n' > "$plan_file"
  printf '0\tfull_dare\ttrue\ttrue\tdata/agents/dare_humanoid_agent.yaml\t%s\texisting\n' \
    "$full_dare_output" >> "$plan_file"
  local order=0
  local variant
  for variant in "${variants[@]}"; do
    order=$((order + 1))
    printf '%d\t%s\t%s\t%s\t%s\t%s\tpending\n' \
      "$order" "$variant" "${group_embedding[$variant]}" \
      "${anchor_calibration[$variant]}" "${agent_configs[$variant]}" \
      "$campaign_root/climb_${variant}_2k_8192_seed${seed}" \
      >> "$plan_file"
  done
}

preflight() {
  local file
  for file in "$env_config" "$engine_config" "$args_file"; do
    [[ -f "$file" ]] || {
      printf 'ERROR: missing required file: %s\n' "$file" >&2
      return 1
    }
  done
  for variant in "${variants[@]}"; do
    [[ -f "${agent_configs[$variant]}" ]] || {
      printf 'ERROR: missing agent config: %s\n' \
        "${agent_configs[$variant]}" >&2
      return 1
    }
  done
  [[ -s "$full_dare_output/model.pt" ]] || {
    printf 'ERROR: missing existing Full DARE model: %s/model.pt\n' \
      "$full_dare_output" >&2
    return 1
  }
  [[ -f tools/paper_eval/evaluate_checkpoint.py ]] || {
    printf 'ERROR: missing shared evaluator\n' >&2
    return 1
  }
  "$python_bin" - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("ERROR: CUDA is unavailable to the ablation launcher.", file=sys.stderr)
    raise SystemExit(1)
PY
}

wait_for_existing_training() {
  [[ "$wait_for_gpu" == true ]] || return
  local poll_seconds="${WAIT_POLL_SECONDS:-30}"
  local pids
  while pids="$(pgrep -f '[m]imickit/run.py' || true)" && [[ -n "$pids" ]]; do
    printf '[%s] Waiting for existing MimicKit training PID(s): %s\n' \
      "$(date --iso-8601=seconds)" "$(tr '\n' ' ' <<< "$pids")"
    sleep "$poll_seconds"
  done
}

checkpoint_reached_budget() {
  local checkpoint_file="$1"
  local target_samples="$2"
  local agent_config="$3"
  "$python_bin" - "$checkpoint_file" "$target_samples" "$env_config" \
    "$agent_config" "$engine_config" <<'PY'
import hashlib
import sys
import torch

checkpoint_file, target = sys.argv[1], int(sys.argv[2])
files = dict(zip(
    ("env_config_sha256", "agent_config_sha256", "engine_config_sha256"),
    sys.argv[3:6],
))

def digest(filename):
    result = hashlib.sha256()
    with open(filename, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

try:
    checkpoint = torch.load(
        checkpoint_file, map_location="cpu", weights_only=False, mmap=True)
    actual = int(checkpoint["trainer_state"]["sample_count"])
    saved = checkpoint["metadata"]["checkpoint_context"]
    expected = {key: digest(path) for key, path in files.items()}
except Exception as exc:
    print(f"ERROR: cannot validate checkpoint {checkpoint_file}: {exc}", file=sys.stderr)
    raise SystemExit(1)

if saved != expected:
    print(f"ERROR: checkpoint configuration mismatch: saved={saved}, expected={expected}", file=sys.stderr)
    raise SystemExit(1)
if actual < target:
    print(f"ERROR: checkpoint has {actual} samples, below {target}", file=sys.stderr)
    raise SystemExit(1)
PY
}

run_job() {
  local stage="$1"
  local variant="$2"
  local num_envs="$3"
  local max_samples="$4"
  local save_int_models="$5"
  local root="$6"
  local agent_config="${agent_configs[$variant]}"
  local suffix
  case "$stage" in
    smoke) suffix="smoke" ;;
    scale_smoke) suffix="scale_smoke" ;;
    formal) suffix="2k_8192" ;;
    *) printf 'ERROR: unsupported stage: %s\n' "$stage" >&2; return 2 ;;
  esac

  local out_dir="$root/climb_${variant}_${suffix}_seed${seed}"
  local done_file="$out_dir/DONE"
  local checkpoint_file="$out_dir/checkpoint.pt"
  local console_file="$out_dir/console.log"
  local eval_name="final"
  local eval_envs=256
  local eval_steps=300
  if [[ "$stage" != formal ]]; then
    eval_name="$stage"
    eval_envs=2
    eval_steps=2
  fi
  local eval_dir="$out_dir/eval/$eval_name"
  local eval_summary="$eval_dir/summary.json"
  mkdir -p "$out_dir"

  if [[ -f "$done_file" && -s "$eval_summary" ]] \
      && checkpoint_reached_budget "$checkpoint_file" "$max_samples" "$agent_config"; then
    printf '[%s] SKIP %s/%s (DONE)\n' "$(date --iso-8601=seconds)" \
      "$stage" "$variant"
    append_event "$stage" "$variant" "SKIPPED_DONE" "$out_dir"
    return 0
  fi

  local -a resume_args=()
  local resume_mode="fresh"
  if [[ -f "$checkpoint_file" ]]; then
    resume_args=(--resume_file "$checkpoint_file")
    resume_mode="resume"
  fi
  append_event "$stage" "$variant" "STARTED" "$out_dir" \
    "num_envs=$num_envs,max_samples=$max_samples,$resume_mode"
  printf '[%s] START %s/%s (%s envs, target %s samples, %s)\n' \
    "$(date --iso-8601=seconds)" "$stage" "$variant" "$num_envs" \
    "$max_samples" "$resume_mode"

  local -a command=(
    "$python_bin" mimickit/run.py
    --arg_file "$args_file"
    --agent_config "$agent_config"
    --env_config "$env_config"
    --out_dir "$out_dir"
    --num_envs "$num_envs"
    --max_samples "$max_samples"
    --rand_seed "$seed"
    --save_int_models "$save_int_models"
    "${resume_args[@]}"
  )
  printf '[%s] COMMAND' "$(date --iso-8601=seconds)" >> "$console_file"
  printf ' %q' "${command[@]}" >> "$console_file"
  printf '\n' >> "$console_file"

  if PYTHONUNBUFFERED=1 "${command[@]}" >> "$console_file" 2>&1; then
    :
  else
    local rc=$?
    append_event "$stage" "$variant" "FAILED" "$out_dir" "exit=$rc"
    printf '[%s] FAILED %s/%s (exit %s); see %s\n' \
      "$(date --iso-8601=seconds)" "$stage" "$variant" "$rc" \
      "$console_file" >&2
    return "$rc"
  fi

  if [[ ! -s "$out_dir/model.pt" || ! -s "$checkpoint_file" || ! -s "$out_dir/log.txt" ]]; then
    append_event "$stage" "$variant" "FAILED_ARTIFACT_CHECK" "$out_dir" \
      "missing model.pt/checkpoint.pt/log.txt"
    return 4
  fi
  if ! checkpoint_reached_budget "$checkpoint_file" "$max_samples" "$agent_config"; then
    append_event "$stage" "$variant" "FAILED_BUDGET_CHECK" "$out_dir"
    return 5
  fi

  mkdir -p "$eval_dir"
  if [[ ! -s "$eval_summary" ]]; then
    local eval_console="$eval_dir/console.log"
    local -a eval_command=(
      "$python_bin" tools/paper_eval/evaluate_checkpoint.py
      --model-file "$out_dir/model.pt"
      --env-config "$env_config"
      --agent-config "$agent_config"
      --engine-config "$engine_config"
      --method dare
      --motion "$motion"
      --num-envs "$eval_envs"
      --steps "$eval_steps"
      --start-mode phase0
      --condition nominal
      --seed "$seed"
      --out-dir "$eval_dir"
    )
    append_event "$stage" "$variant" "EVAL_STARTED" "$eval_dir"
    printf '[%s] EVAL %s/%s (%s envs x %s steps)\n' \
      "$(date --iso-8601=seconds)" "$stage" "$variant" "$eval_envs" "$eval_steps"
    printf '[%s] COMMAND' "$(date --iso-8601=seconds)" >> "$eval_console"
    printf ' %q' "${eval_command[@]}" >> "$eval_console"
    printf '\n' >> "$eval_console"
    if PYTHONUNBUFFERED=1 "${eval_command[@]}" >> "$eval_console" 2>&1; then
      :
    else
      local rc=$?
      append_event "$stage" "$variant" "EVAL_FAILED" "$eval_dir" "exit=$rc"
      return "$rc"
    fi
    [[ -s "$eval_summary" && -s "$eval_dir/episodes.npz" && -s "$eval_dir/timeseries.npz" ]] || {
      append_event "$stage" "$variant" "EVAL_FAILED_ARTIFACT_CHECK" "$eval_dir"
      return 6
    }
    append_event "$stage" "$variant" "EVAL_DONE" "$eval_dir"
  fi

  printf 'stage=%s\nvariant=%s\nmotion=%s\nnum_envs=%s\ntarget_samples=%s\nfinished=%s\n' \
    "$stage" "$variant" "$motion" "$num_envs" "$max_samples" \
    "$(date --iso-8601=seconds)" > "$done_file"
  append_event "$stage" "$variant" "DONE" "$out_dir"
  printf '[%s] DONE %s/%s\n' "$(date --iso-8601=seconds)" "$stage" "$variant"
}

run_job_with_retries() {
  local stage="$1"
  local variant="$2"
  local max_attempts=2
  [[ "$stage" == formal ]] && max_attempts=4
  local attempt=1
  local rc=0
  while ((attempt <= max_attempts)); do
    if run_job "$@"; then
      return 0
    else
      rc=$?
    fi
    if ((attempt == max_attempts)); then
      append_event "$stage" "$variant" "RETRIES_EXHAUSTED" "output" \
        "attempts=$max_attempts,exit=$rc"
      return "$rc"
    fi
    append_event "$stage" "$variant" "RETRYING" "output" \
      "attempt=$attempt,exit=$rc"
    sleep 10
    attempt=$((attempt + 1))
  done
}

run_stage() {
  local stage="$1"
  local num_envs="$2"
  local max_samples="$3"
  local save_int_models="$4"
  local root="$5"
  local variant
  for variant in "${variants[@]}"; do
    run_job_with_retries "$stage" "$variant" "$num_envs" "$max_samples" \
      "$save_int_models" "$root"
  done
}

write_plan
preflight
wait_for_existing_training

printf 'Climb DARE 2x2 plan: %s\n' "$plan_file"
printf 'Event log: %s\n' "$events_file"
printf 'Formal queue: %d variants x %d samples = %d samples\n' \
  "${#variants[@]}" "$formal_samples" \
  "$((${#variants[@]} * formal_samples))"

if [[ "$run_smoke" == true ]]; then
  run_stage smoke "$smoke_envs" "$smoke_samples" false "$smoke_root"
fi
if [[ "$run_scale_smoke" == true ]]; then
  run_stage scale_smoke "$scale_envs" "$scale_samples" false "$scale_root"
fi
if [[ "$run_formal" == true ]]; then
  run_stage formal "$formal_envs" "$formal_samples" true "$campaign_root"
fi

printf '[%s] Climb DARE ablation campaign complete.\n' "$(date --iso-8601=seconds)"
