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
method_filter=""
motion_filters=()

usage() {
  printf '%s\n' \
    "Usage: $0 [--smoke-only|--scale-smoke-only|--formal-only] [--method NAME] [--motion NAME ...] [--no-wait] [--python PATH]" \
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
    --method)
      shift
      if (($# == 0)); then
        printf 'ERROR: --method requires a method name\n' >&2
        exit 2
      fi
      method_filter="$1"
      ;;
    --motion)
      shift
      if (($# == 0)); then
        printf 'ERROR: --motion requires a motion name\n' >&2
        exit 2
      fi
      motion_filters+=("$1")
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

methods=(dare add deepmimic amp)
motions=(climb backflip crawl roll getup_facedown spinkick)

run_methods=("${methods[@]}")
if [[ -n "$method_filter" ]]; then
  method_known=false
  for method in "${run_methods[@]}"; do
    if [[ "$method" == "$method_filter" ]]; then
      method_known=true
      break
    fi
  done
  if [[ "$method_known" != true ]]; then
    printf 'ERROR: unsupported method filter: %s\n' "$method_filter" >&2
    exit 2
  fi
  run_methods=("$method_filter")
fi

run_motions=("${motions[@]}")
if ((${#motion_filters[@]} > 0)); then
  run_motions=()
  for requested_motion in "${motion_filters[@]}"; do
    motion_known=false
    for motion in "${motions[@]}"; do
      if [[ "$motion" == "$requested_motion" ]]; then
        motion_known=true
        break
      fi
    done
    if [[ "$motion_known" != true ]]; then
      printf 'ERROR: unsupported motion filter: %s\n' "$requested_motion" >&2
      exit 2
    fi

    motion_duplicate=false
    for motion in "${run_motions[@]}"; do
      if [[ "$motion" == "$requested_motion" ]]; then
        motion_duplicate=true
        break
      fi
    done
    if [[ "$motion_duplicate" != true ]]; then
      run_motions+=("$requested_motion")
    fi
  done
fi

declare -A arg_files=(
  [deepmimic]="args/paper_benchmark/deepmimic_2k_8192_args.txt"
  [amp]="args/paper_benchmark/amp_2k_8192_args.txt"
  [add]="args/paper_benchmark/add_2k_8192_args.txt"
  [dare]="args/paper_benchmark/dare_2k_8192_args.txt"
)

declare -A agent_files=(
  [deepmimic]="data/agents/deepmimic_humanoid_ppo_agent.yaml"
  [amp]="data/agents/amp_humanoid_agent.yaml"
  [add]="data/agents/add_humanoid_agent.yaml"
  [dare]="data/agents/dare_humanoid_agent.yaml"
)

smoke_envs=64
steps_per_iter=32
smoke_iters=2
scale_smoke_envs=8192
scale_smoke_iters=3
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
  for motion in "${run_motions[@]}"; do
    for method in "${run_methods[@]}"; do
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
  for method in "${run_methods[@]}"; do
    [[ -f "${arg_files[$method]}" ]] || {
      printf 'ERROR: missing arg file: %s\n' "${arg_files[$method]}" >&2
      return 1
    }
    [[ -f "${agent_files[$method]}" ]] || {
      printf 'ERROR: missing agent file: %s\n' "${agent_files[$method]}" >&2
      return 1
    }
    for motion in "${run_motions[@]}"; do
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

checkpoint_reached_budget() {
  local checkpoint_file="$1"
  local target_samples="$2"
  local env_config="$3"
  local agent_config="$4"
  local engine_config="$5"
  "$python_bin" - "$checkpoint_file" "$target_samples" \
    "$env_config" "$agent_config" "$engine_config" <<'PY'
import hashlib
import sys
import torch

checkpoint_file = sys.argv[1]
target_samples = int(sys.argv[2])
config_files = {
    "env_config_sha256": sys.argv[3],
    "agent_config_sha256": sys.argv[4],
    "engine_config_sha256": sys.argv[5],
}


def file_sha256(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


try:
    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    actual_samples = int(checkpoint["trainer_state"]["sample_count"])
    saved_context = checkpoint["metadata"]["checkpoint_context"]
    expected_context = {
        key: file_sha256(filename)
        for key, filename in config_files.items()
    }
except Exception as exc:
    print(
        f"ERROR: unable to validate checkpoint budget for "
        f"{checkpoint_file}: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1)

if saved_context != expected_context:
    print(
        f"ERROR: checkpoint {checkpoint_file} was produced by different "
        f"configuration files; saved={saved_context}, "
        f"expected={expected_context}.",
        file=sys.stderr,
    )
    raise SystemExit(1)

if actual_samples < target_samples:
    print(
        f"ERROR: checkpoint {checkpoint_file} contains {actual_samples} "
        f"samples, below target {target_samples}.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
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
    if checkpoint_reached_budget "$checkpoint_file" "$max_samples" \
      "$env_config" "${agent_files[$method]}" \
      data/engines/isaac_lab_engine.yaml; then
      printf '[%s] SKIP %s/%s/%s (DONE)\n' \
        "$(date --iso-8601=seconds)" "$stage" "$method" "$motion"
      append_event "$stage" "$method" "$motion" "SKIPPED_DONE" "$out_dir"
      return 0
    fi
    append_event "$stage" "$method" "$motion" "INVALID_DONE" "$out_dir" \
      "checkpoint below target; resuming"
    printf '[%s] INVALID DONE %s/%s/%s; checkpoint is below budget, resuming\n' \
      "$(date --iso-8601=seconds)" "$stage" "$method" "$motion" >&2
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

  # Isaac Sim can occasionally terminate after printing a Python traceback
  # while still returning a successful process status during plugin teardown.
  # Never evaluate or mark a job DONE unless the full training checkpoint
  # itself proves that the requested physics-sample budget was reached.
  if ! checkpoint_reached_budget "$checkpoint_file" "$max_samples" \
    "$env_config" "${agent_files[$method]}" \
    data/engines/isaac_lab_engine.yaml; then
    append_event "$stage" "$method" "$motion" "FAILED_BUDGET_CHECK" \
      "$out_dir" "checkpoint below target after trainer exit"
    printf 'ERROR: %s/%s/%s exited before reaching its sample budget\n' \
      "$stage" "$method" "$motion" >&2
    return 6
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

run_job_with_retries() {
  local stage="$1"
  local method="$2"
  local motion="$3"
  local max_attempts=2
  if [[ "$stage" == "formal" ]]; then
    max_attempts=4
  fi

  local attempt=1
  local rc=0
  while ((attempt <= max_attempts)); do
    if run_job "$@"; then
      return 0
    else
      rc=$?
    fi

    if ((attempt == max_attempts)); then
      append_event "$stage" "$method" "$motion" "RETRIES_EXHAUSTED" \
        "output" "attempts=$max_attempts,exit=$rc"
      return "$rc"
    fi

    append_event "$stage" "$method" "$motion" "RETRYING" "output" \
      "attempt=$attempt,exit=$rc"
    printf '[%s] RETRY %s/%s/%s after exit %s (attempt %s/%s)\n' \
      "$(date --iso-8601=seconds)" "$stage" "$method" "$motion" \
      "$rc" "$((attempt + 1))" "$max_attempts" >&2
    attempt=$((attempt + 1))
    sleep 10
  done
}

run_stage() {
  local stage="$1"
  local num_envs="$2"
  local max_samples="$3"
  local save_int_models="$4"
  local root="$5"
  local method motion

  # Motion-major order: every method finishes the current clip before the next
  # clip.  Methods themselves start at DARE.
  for motion in "${run_motions[@]}"; do
    for method in "${run_methods[@]}"; do
      run_job_with_retries "$stage" "$method" "$motion" "$num_envs" "$max_samples" \
        "$save_int_models" "$root"
    done
  done
}

write_plan
preflight
wait_for_existing_training

printf 'Campaign plan: %s\n' "$plan_file"
printf 'Event log:    %s\n' "$events_file"
printf 'Selected formal budget: %d methods x %d motions x %d samples = %d samples\n' \
  "${#run_methods[@]}" "${#run_motions[@]}" "$formal_samples" \
  $(( ${#run_methods[@]} * ${#run_motions[@]} * formal_samples ))
printf '%s\n' \
  "ETA warning: the selected campaign contains $((${#run_methods[@]} * ${#run_motions[@]})) formal jobs;" \
  'current 8192-env runs take about 2.2 h each, while Climb can take longer.'

if [[ "$run_smoke" == true ]]; then
  run_stage "smoke" "$smoke_envs" "$smoke_samples" false "$smoke_root"
fi

if [[ "$run_scale_smoke" == true ]]; then
  for method in "${run_methods[@]}"; do
    scale_motions=(climb)
    for motion in "${scale_motions[@]}"; do
      motion_selected=false
      for requested_motion in "${run_motions[@]}"; do
        if [[ "$motion" == "$requested_motion" ]]; then
          motion_selected=true
          break
        fi
      done
      if [[ "$motion_selected" != true ]]; then
        continue
      fi
      run_job_with_retries "scale_smoke" "$method" "$motion" \
        "$scale_smoke_envs" "$scale_smoke_samples" false "$scale_smoke_root"
    done
  done
fi

if [[ "$run_formal" == true ]]; then
  run_stage "formal" "$formal_envs" "$formal_samples" true "$campaign_root"
fi

printf '[%s] Entire requested matrix is complete.\n' "$(date --iso-8601=seconds)"
