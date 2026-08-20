#!/usr/bin/env bash
set -Eeuo pipefail

# Four-condition Roll campaign for the Hinge-SN Aligned ADD ablation.
#
# Default order:
#   E1--E4: 64 env x 32 steps x 2 iterations (all smokes first), then
#   E1--E4: 8192 env x 32 steps x 2000 iterations (formal runs).
#
# The script never launches jobs concurrently. Re-running skips a job only
# after its checkpoint reaches the target budget and its shared paper
# evaluator artifacts exist. An interrupted job resumes from checkpoint.pt.
# It is safe to detach, for example:
#   nohup tools/hinge_sn_add/run_serial_roll_2k.sh \
#     > output/hinge_sn_add/launcher.log 2>&1 &

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
python_bin="${MIMICKIT_PYTHON:-/home/y/miniconda3/envs/env_isaaclab/bin/python}"
wait_for_gpu=true
run_smoke=true
run_formal=true
check_only=false
signal_test_dir=""

usage() {
  printf '%s\n' \
    "Usage: $0 [--smoke-only|--formal-only|--check-only] [--no-wait] [--python PATH]" \
    "" \
    "Environment:" \
    "  MIMICKIT_PYTHON   Python executable (default: env_isaaclab Python)" \
    "  WAIT_POLL_SECONDS Poll interval while another run.py owns the GPU"
}

while (($# > 0)); do
  case "$1" in
    --smoke-only)
      run_smoke=true
      run_formal=false
      ;;
    --formal-only)
      run_smoke=false
      run_formal=true
      ;;
    --check-only)
      check_only=true
      ;;
    --signal-test-dir)
      shift
      if (($# == 0)); then
        printf 'ERROR: --signal-test-dir requires a directory\n' >&2
        exit 2
      fi
      signal_test_dir="$1"
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

# Every long-running Python command gets its own process group.  The launcher
# can therefore stop both the direct child and any workers it spawned without
# accidentally signalling the shell/nohup process that owns this script.
active_child_pid=""
active_child_label=""
events_file=""

run_managed_child() {
  local label="$1"
  local log_file="$2"
  shift 2
  if [[ -n "$active_child_pid" ]]; then
    printf 'ERROR: managed child already active: %s (PID %s)\n' \
      "$active_child_label" "$active_child_pid" >&2
    return 70
  fi

  setsid env PYTHONUNBUFFERED=1 "$@" >> "$log_file" 2>&1 &
  active_child_pid=$!
  active_child_label="$label"
  local child_pid="$active_child_pid"
  local rc
  set +e
  wait "$child_pid"
  rc=$?
  set -e
  if [[ "$active_child_pid" == "$child_pid" ]]; then
    active_child_pid=""
    active_child_label=""
  fi
  return "$rc"
}

forward_signal() {
  local signal_name="$1"
  local exit_code="$2"
  # A second signal must not recursively enter this handler while it waits for
  # the process group it just stopped.
  trap - TERM INT HUP
  if [[ -n "$active_child_pid" ]]; then
    local child_pid="$active_child_pid"
    local child_label="$active_child_label"
    printf '[%s] %s received; forwarding to %s process group %s\n' \
      "$(date --iso-8601=seconds)" "$signal_name" \
      "$child_label" "$child_pid" >&2
    if ! kill -s "$signal_name" -- "-$child_pid" 2>/dev/null; then
      kill -s "$signal_name" "$child_pid" 2>/dev/null || true
    fi
    set +e
    wait "$child_pid"
    set -e
    active_child_pid=""
    active_child_label=""
  fi
  if [[ -n "$events_file" && -f "$events_file" ]]; then
    printf '%s\tsignal\tlauncher\tINTERRUPTED\t-\t%s\n' \
      "$(date --iso-8601=seconds)" "$signal_name" >> "$events_file"
  fi
  exit "$exit_code"
}

trap 'forward_signal TERM 143' TERM
trap 'forward_signal INT 130' INT
trap 'forward_signal HUP 129' HUP

# Private hook used by test_serial_launcher_signals.sh.  It exercises the real
# process-group and trap implementation without importing Isaac Lab or
# starting a training job.
if [[ -n "$signal_test_dir" ]]; then
  mkdir -p "$signal_test_dir"
  run_managed_child signal-test "$signal_test_dir/child.log" \
    bash -c '
      printf "%s\n" "$$" > "$1/leader.pid"
      sleep 300 &
      worker=$!
      printf "%s\n" "$worker" > "$1/worker.pid"
      wait "$worker"
    ' bash "$signal_test_dir"
  exit $?
fi

cd "$repo_dir"

variants=(
  e1_hinge_sn_gp
  e2_hinge_sn_nogp
  e3_hinge_sn_margin_reward
  e4_hinge_sn_cr
)

declare -A agent_files=(
  [e1_hinge_sn_gp]="data/agents/hinge_sn_gp_aligned_add_humanoid_agent.yaml"
  [e2_hinge_sn_nogp]="data/agents/hinge_sn_nogp_aligned_add_humanoid_agent.yaml"
  [e3_hinge_sn_margin_reward]="data/agents/hinge_sn_margin_reward_aligned_add_humanoid_agent.yaml"
  [e4_hinge_sn_cr]="data/agents/hinge_sn_cr_aligned_add_humanoid_agent.yaml"
)

declare -A smoke_args=(
  [e1_hinge_sn_gp]="args/hinge_sn_add/e1_hinge_sn_gp_roll_smoke_args.txt"
  [e2_hinge_sn_nogp]="args/hinge_sn_add/e2_hinge_sn_nogp_roll_smoke_args.txt"
  [e3_hinge_sn_margin_reward]="args/hinge_sn_add/e3_hinge_sn_margin_reward_roll_smoke_args.txt"
  [e4_hinge_sn_cr]="args/hinge_sn_add/e4_hinge_sn_cr_roll_smoke_args.txt"
)

declare -A formal_args=(
  [e1_hinge_sn_gp]="args/hinge_sn_add/e1_hinge_sn_gp_roll_2k_8192_args.txt"
  [e2_hinge_sn_nogp]="args/hinge_sn_add/e2_hinge_sn_nogp_roll_2k_8192_args.txt"
  [e3_hinge_sn_margin_reward]="args/hinge_sn_add/e3_hinge_sn_margin_reward_roll_2k_8192_args.txt"
  [e4_hinge_sn_cr]="args/hinge_sn_add/e4_hinge_sn_cr_roll_2k_8192_args.txt"
)

declare -A smoke_outputs=(
  [e1_hinge_sn_gp]="output/hinge_sn_add_smoke/e1_hinge_sn_gp_roll_smoke_seed0"
  [e2_hinge_sn_nogp]="output/hinge_sn_add_smoke/e2_hinge_sn_nogp_roll_smoke_seed0"
  [e3_hinge_sn_margin_reward]="output/hinge_sn_add_smoke/e3_hinge_sn_margin_reward_roll_smoke_seed0"
  [e4_hinge_sn_cr]="output/hinge_sn_add_smoke/e4_hinge_sn_cr_roll_smoke_seed0"
)

declare -A formal_outputs=(
  [e1_hinge_sn_gp]="output/hinge_sn_add/e1_hinge_sn_gp_roll_2k_8192_seed0"
  [e2_hinge_sn_nogp]="output/hinge_sn_add/e2_hinge_sn_nogp_roll_2k_8192_seed0"
  [e3_hinge_sn_margin_reward]="output/hinge_sn_add/e3_hinge_sn_margin_reward_roll_2k_8192_seed0"
  [e4_hinge_sn_cr]="output/hinge_sn_add/e4_hinge_sn_cr_roll_2k_8192_seed0"
)

smoke_samples=$((64 * 32 * 2))
formal_samples=$((8192 * 32 * 2000))
campaign_root="output/hinge_sn_add"
smoke_root="output/hinge_sn_add_smoke"
eval_env="data/envs/aligned_add_humanoid_roll_eval_env.yaml"
engine_config="data/engines/isaac_lab_engine.yaml"
mkdir -p "$campaign_root" "$smoke_root"

# One process owns the complete serial queue, including its evaluations.
exec 9>"$campaign_root/.serial_roll.lock"
if ! flock -n 9; then
  printf 'ERROR: another Hinge-SN Roll launcher holds %s\n' \
    "$campaign_root/.serial_roll.lock" >&2
  exit 3
fi

events_file="$campaign_root/events.tsv"
plan_file="$campaign_root/plan.tsv"
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
  printf 'order\tstage\tvariant\tnum_envs\titerations\tmax_samples\tagent_config\toutput\n' \
    > "$plan_file"
  local order=0
  local variant
  if [[ "$run_smoke" == true ]]; then
    for variant in "${variants[@]}"; do
      order=$((order + 1))
      printf '%d\tsmoke\t%s\t64\t2\t%d\t%s\t%s\n' \
        "$order" "$variant" "$smoke_samples" "${agent_files[$variant]}" \
        "${smoke_outputs[$variant]}" >> "$plan_file"
    done
  fi
  if [[ "$run_formal" == true ]]; then
    for variant in "${variants[@]}"; do
      order=$((order + 1))
      printf '%d\tformal\t%s\t8192\t2000\t%d\t%s\t%s\n' \
        "$order" "$variant" "$formal_samples" "${agent_files[$variant]}" \
        "${formal_outputs[$variant]}" >> "$plan_file"
    done
  fi
}

preflight() {
  if [[ ! -x "$python_bin" ]]; then
    printf 'ERROR: Python executable not found: %s\n' "$python_bin" >&2
    return 1
  fi
  command -v flock >/dev/null || {
    printf 'ERROR: flock is required by the serial launcher\n' >&2
    return 1
  }
  command -v setsid >/dev/null || {
    printf 'ERROR: setsid is required for child process-group management\n' >&2
    return 1
  }

  local variant filename
  for variant in "${variants[@]}"; do
    for filename in "${agent_files[$variant]}" \
      "${smoke_args[$variant]}" "${formal_args[$variant]}"; do
      [[ -f "$filename" ]] || {
        printf 'ERROR: missing campaign file: %s\n' "$filename" >&2
        return 1
      }
    done
  done
  for filename in "$eval_env" "$engine_config" \
    tools/paper_eval/evaluate_checkpoint.py \
    tools/paper_eval/aggregate_results.py; do
    [[ -f "$filename" ]] || {
      printf 'ERROR: missing shared evaluation file: %s\n' "$filename" >&2
      return 1
    }
  done

  # Validate both the one-factor ablation and exact rollout budgets before CUDA
  # is initialized. This prevents a typo from consuming a multi-day campaign.
  "$python_bin" - <<'PY'
from pathlib import Path
import yaml

agents = {
    "e1": Path("data/agents/hinge_sn_gp_aligned_add_humanoid_agent.yaml"),
    "e2": Path("data/agents/hinge_sn_nogp_aligned_add_humanoid_agent.yaml"),
    "e3": Path("data/agents/hinge_sn_margin_reward_aligned_add_humanoid_agent.yaml"),
    "e4": Path("data/agents/hinge_sn_cr_aligned_add_humanoid_agent.yaml"),
}
cfg = {key: yaml.safe_load(path.read_text()) for key, path in agents.items()}
for key, value in cfg.items():
    assert value["agent_name"] == "HINGE_SN_ALIGNED_ADD", key
    assert value["disc_hinge_margin"] == 1.0, key
    assert value["disc_spectral_norm"] is True, key
    assert value["disc_sn_power_iterations"] == 1, key
    assert value["steps_per_iter"] == 32, key

expected = {
    "e1": (2, "add_softplus", 0.0),
    "e2": (0, "add_softplus", 0.0),
    "e3": (0, "smooth_margin", 0.0),
    "e4": (0, "add_softplus", 1.0),
}
for key, values in expected.items():
    actual = (
        cfg[key]["disc_grad_penalty"],
        cfg[key]["disc_reward_type"],
        cfg[key]["disc_consistency_weight"],
    )
    assert actual == values, (key, actual, values)
    assert cfg[key]["disc_consistency_noise_std"] == 0.01, key

# Remove only the intended experimental factors; everything else must match.
factor_keys = {
    "disc_grad_penalty",
    "disc_reward_type",
    "disc_consistency_weight",
}
reference = {k: v for k, v in cfg["e2"].items() if k not in factor_keys}
for key in ("e1", "e3", "e4"):
    candidate = {k: v for k, v in cfg[key].items() if k not in factor_keys}
    assert candidate == reference, f"uncontrolled config difference: {key}"

def parse_args(path):
    tokens = Path(path).read_text().split()
    return dict(zip(tokens[0::2], tokens[1::2]))

for path in Path("args/hinge_sn_add").glob("*_smoke_args.txt"):
    args = parse_args(path)
    assert int(args["--num_envs"]) == 64, path
    assert int(args["--max_samples"]) == 64 * 32 * 2, path
for path in Path("args/hinge_sn_add").glob("*_2k_8192_args.txt"):
    args = parse_args(path)
    assert int(args["--num_envs"]) == 8192, path
    assert int(args["--max_samples"]) == 8192 * 32 * 2000, path
PY

  if ! "$python_bin" - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print(
        "ERROR: CUDA is unavailable; run the launcher in the GPU-enabled "
        "host/session.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
  then
    return 1
  fi
}

wait_for_existing_training() {
  if [[ "$wait_for_gpu" != true ]]; then
    return
  fi
  local poll_seconds="${WAIT_POLL_SECONDS:-30}"
  local pids
  while pids="$(pgrep -f '[m]imickit/run.py' || true)" && [[ -n "$pids" ]]; do
    printf '[%s] Waiting for existing MimicKit training PID(s): %s\n' \
      "$(date --iso-8601=seconds)" "$(tr '\n' ' ' <<< "$pids")"
    sleep "$poll_seconds"
  done
}

checkpoint_samples() {
  local checkpoint_file="$1"
  "$python_bin" - "$checkpoint_file" <<'PY'
import sys
import torch

checkpoint = torch.load(
    sys.argv[1], map_location="cpu", weights_only=False, mmap=True
)
print(int(checkpoint["trainer_state"]["sample_count"]))
PY
}

run_job() {
  local stage="$1"
  local variant="$2"
  local arg_file out_dir target_samples save_int_models eval_name eval_envs eval_steps
  if [[ "$stage" == "smoke" ]]; then
    arg_file="${smoke_args[$variant]}"
    out_dir="${smoke_outputs[$variant]}"
    target_samples="$smoke_samples"
    save_int_models=false
    eval_name=smoke
    eval_envs=2
    eval_steps=2
  else
    arg_file="${formal_args[$variant]}"
    out_dir="${formal_outputs[$variant]}"
    target_samples="$formal_samples"
    save_int_models=true
    eval_name=final
    eval_envs=256
    eval_steps=300
  fi

  local checkpoint_file="$out_dir/checkpoint.pt"
  local model_file="$out_dir/model.pt"
  local console_file="$out_dir/console.log"
  local done_file="$out_dir/DONE"
  local eval_dir="$out_dir/eval/$eval_name"
  local eval_summary="$eval_dir/summary.json"
  mkdir -p "$out_dir"

  local actual_samples=0
  if [[ -f "$checkpoint_file" ]]; then
    if ! actual_samples="$(checkpoint_samples "$checkpoint_file")"; then
      append_event "$stage" "$variant" "INVALID_CHECKPOINT" "$out_dir"
      printf 'ERROR: unreadable checkpoint: %s\n' "$checkpoint_file" >&2
      return 7
    fi
  fi

  if [[ -f "$done_file" && "$actual_samples" -ge "$target_samples" \
        && -s "$model_file" && -s "$eval_summary" \
        && -s "$eval_dir/episodes.npz" && -s "$eval_dir/timeseries.npz" ]]; then
    printf '[%s] SKIP %s/%s (DONE)\n' \
      "$(date --iso-8601=seconds)" "$stage" "$variant"
    append_event "$stage" "$variant" "SKIPPED_DONE" "$out_dir"
    return 0
  fi

  if [[ "$actual_samples" -lt "$target_samples" ]]; then
    local -a resume_args=()
    local resume_state=fresh
    if [[ -f "$checkpoint_file" ]]; then
      resume_args=(--resume_file "$checkpoint_file")
      resume_state=resume
    fi
    local -a train_cmd=(
      "$python_bin" mimickit/run.py
      --arg_file "$arg_file"
      --save_int_models "$save_int_models"
      "${resume_args[@]}"
    )
    append_event "$stage" "$variant" "STARTED" "$out_dir" \
      "target_samples=$target_samples,$resume_state"
    printf '[%s] START %s/%s (target %s samples, %s)\n' \
      "$(date --iso-8601=seconds)" "$stage" "$variant" \
      "$target_samples" "$resume_state"
    printf '[%s] COMMAND' "$(date --iso-8601=seconds)" >> "$console_file"
    printf ' %q' "${train_cmd[@]}" >> "$console_file"
    printf '\n' >> "$console_file"
    if run_managed_child "$stage/$variant training" "$console_file" \
        "${train_cmd[@]}"; then
      :
    else
      local rc=$?
      append_event "$stage" "$variant" "FAILED" "$out_dir" "exit=$rc"
      printf '[%s] FAILED %s/%s (exit %s); see %s\n' \
        "$(date --iso-8601=seconds)" "$stage" "$variant" \
        "$rc" "$console_file" >&2
      return "$rc"
    fi
  else
    printf '[%s] TRAINING COMPLETE %s/%s; checking evaluation\n' \
      "$(date --iso-8601=seconds)" "$stage" "$variant"
    append_event "$stage" "$variant" "TRAIN_SKIPPED_BUDGET" "$out_dir" \
      "samples=$actual_samples"
  fi

  if [[ ! -s "$model_file" || ! -s "$checkpoint_file" \
        || ! -s "$out_dir/log.txt" ]]; then
    append_event "$stage" "$variant" "FAILED_ARTIFACT_CHECK" "$out_dir"
    printf 'ERROR: %s/%s is missing model.pt, checkpoint.pt, or log.txt\n' \
      "$stage" "$variant" >&2
    return 4
  fi
  if ! actual_samples="$(checkpoint_samples "$checkpoint_file")" \
      || [[ "$actual_samples" -lt "$target_samples" ]]; then
    append_event "$stage" "$variant" "FAILED_BUDGET_CHECK" "$out_dir" \
      "samples=$actual_samples,target=$target_samples"
    printf 'ERROR: %s/%s exited below target sample budget\n' \
      "$stage" "$variant" >&2
    return 6
  fi

  mkdir -p "$eval_dir"
  if [[ -s "$eval_summary" && -s "$eval_dir/episodes.npz" \
        && -s "$eval_dir/timeseries.npz" ]]; then
    append_event "$stage" "$variant" "EVAL_SKIPPED_DONE" "$eval_dir"
  else
    local eval_console="$eval_dir/console.log"
    local -a eval_cmd=(
      "$python_bin" tools/paper_eval/evaluate_checkpoint.py
      --model-file "$model_file"
      --env-config "$eval_env"
      --agent-config "${agent_files[$variant]}"
      --engine-config "$engine_config"
      --method "$variant"
      --motion roll
      --num-envs "$eval_envs"
      --steps "$eval_steps"
      --start-mode phase0
      --condition nominal
      --seed 0
      --out-dir "$eval_dir"
    )
    append_event "$stage" "$variant" "EVAL_STARTED" "$eval_dir"
    printf '[%s] EVAL %s/%s (%s envs x %s steps)\n' \
      "$(date --iso-8601=seconds)" "$stage" "$variant" \
      "$eval_envs" "$eval_steps"
    printf '[%s] COMMAND' "$(date --iso-8601=seconds)" >> "$eval_console"
    printf ' %q' "${eval_cmd[@]}" >> "$eval_console"
    printf '\n' >> "$eval_console"
    if run_managed_child "$stage/$variant evaluation" "$eval_console" \
        "${eval_cmd[@]}"; then
      :
    else
      local eval_rc=$?
      append_event "$stage" "$variant" "EVAL_FAILED" "$eval_dir" \
        "exit=$eval_rc"
      printf '[%s] EVAL FAILED %s/%s (exit %s); see %s\n' \
        "$(date --iso-8601=seconds)" "$stage" "$variant" \
        "$eval_rc" "$eval_console" >&2
      return "$eval_rc"
    fi
    if [[ ! -s "$eval_summary" || ! -s "$eval_dir/episodes.npz" \
          || ! -s "$eval_dir/timeseries.npz" ]]; then
      append_event "$stage" "$variant" "EVAL_FAILED_ARTIFACT_CHECK" \
        "$eval_dir"
      printf 'ERROR: evaluator omitted required artifacts for %s/%s\n' \
        "$stage" "$variant" >&2
      return 5
    fi
    append_event "$stage" "$variant" "EVAL_DONE" "$eval_dir"
  fi

  printf 'stage=%s\nvariant=%s\ntarget_samples=%s\nactual_samples=%s\nfinished=%s\n' \
    "$stage" "$variant" "$target_samples" "$actual_samples" \
    "$(date --iso-8601=seconds)" > "$done_file"
  append_event "$stage" "$variant" "DONE" "$out_dir"
  printf '[%s] DONE %s/%s\n' \
    "$(date --iso-8601=seconds)" "$stage" "$variant"
}

run_stage() {
  local stage="$1"
  local variant
  for variant in "${variants[@]}"; do
    run_job "$stage" "$variant"
  done
}

write_plan
preflight
if [[ "$check_only" == true ]]; then
  printf 'Preflight passed; no training or evaluation was started.\n'
  exit 0
fi
wait_for_existing_training

printf 'Campaign plan:  %s\n' "$plan_file"
printf 'Event log:     %s\n' "$events_file"
printf 'Formal budget: 4 x 8192 env x 32 steps x 2000 iterations = %d samples\n' \
  $((4 * formal_samples))
printf '%s\n' \
  'All four 64-env smokes finish before any formal job starts.' \
  'Any failed training, budget validation, or evaluation stops the queue.'

if [[ "$run_smoke" == true ]]; then
  run_stage smoke
fi

if [[ "$run_formal" == true ]]; then
  run_stage formal
  aggregate_dir="$campaign_root/aggregate"
  aggregate_console="$aggregate_dir/console.log"
  mkdir -p "$aggregate_dir"
  aggregate_cmd=(
    "$python_bin" tools/paper_eval/aggregate_results.py "$campaign_root"
    --out-dir "$aggregate_dir"
  )
  if run_managed_child "result aggregation" "$aggregate_console" \
      "${aggregate_cmd[@]}"; then
    :
  else
    rc=$?
    append_event aggregate all FAILED "$aggregate_dir" "exit=$rc"
    printf 'ERROR: final result aggregation failed; see %s\n' \
      "$aggregate_console" >&2
    exit "$rc"
  fi
  append_event aggregate all DONE "$aggregate_dir"
fi

printf '[%s] Entire requested Hinge-SN Roll campaign is complete.\n' \
  "$(date --iso-8601=seconds)"
