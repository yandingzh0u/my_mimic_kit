#!/usr/bin/env bash
set -Eeuo pipefail

# Restartable launcher for the formal 8192-env, 2000-iteration Roll run.
# Recommended detached invocation:
#   nohup setsid bash tools/phase_transition_critic/run_roll_2k.sh \
#     > output/phase_transition_critic_roll_2k_8192_seed0/launcher.log 2>&1 \
#     < /dev/null &

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_dir"

python_bin="${MIMICKIT_PYTHON:-/home/y/miniconda3/envs/env_isaaclab/bin/python}"
arg_file="args/phase_transition_critic_humanoid_roll_2k_8192_args.txt"
agent_file="data/agents/phase_transition_critic_humanoid_agent.yaml"
env_file="data/envs/phase_transition_critic_humanoid_roll_env.yaml"
engine_file="data/engines/isaac_lab_engine.yaml"
evaluator="tools/paper_eval/evaluate_checkpoint.py"
out_dir="output/phase_transition_critic_roll_2k_8192_seed0"
checkpoint_file="$out_dir/checkpoint.pt"
manifest_file="$out_dir/manifest.env"
num_envs=8192
steps_per_iter=32
target_iters=2000
target_samples=$((num_envs * steps_per_iter * target_iters))
seed=0

mkdir -p "$out_dir" output
exec 9>"$out_dir/launcher.lock"
if ! flock -n 9; then
  printf 'ERROR: another Roll launcher holds %s\n' \
    "$out_dir/launcher.lock" >&2
  exit 2
fi
exec 8>"output/.mimickit_cuda0_training.lock"
if ! flock -n 8; then
  printf '%s\n' \
    'ERROR: another launcher holds output/.mimickit_cuda0_training.lock' >&2
  exit 2
fi

write_atomic() {
  local target="$1"
  shift
  local tmp_file
  tmp_file="$(mktemp "$out_dir/.$(basename -- "$target").XXXXXX")"
  printf '%s\n' "$@" > "$tmp_file"
  mv -f -- "$tmp_file" "$target"
}

success=0
child_pid=""
on_exit() {
  local rc=$?
  if [[ "$success" -eq 0 && "$rc" -eq 0 ]]; then
    rc=1
  fi
  write_atomic "$out_dir/EXIT_STATUS" "$rc"
  write_atomic "$out_dir/finished_at.txt" "$(date --iso-8601=seconds)"
  if [[ "$success" -eq 0 ]]; then
    write_atomic "$out_dir/FAILED" \
      "exit_status=$rc" "failed_at=$(date --iso-8601=seconds)"
    write_atomic "$out_dir/STATUS" "FAILED (exit $rc)"
  fi
}
trap on_exit EXIT

forward_signal() {
  local signal="$1"
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill "-$signal" "$child_pid" 2>/dev/null || true
  fi
}
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap 'forward_signal HUP' HUP

for required in "$python_bin" "$arg_file" "$agent_file" "$env_file" \
                "$engine_file" "$evaluator"; do
  [[ -e "$required" ]] || {
    printf 'ERROR: required path is missing: %s\n' "$required" >&2
    exit 3
  }
done
[[ -x "$python_bin" ]] || {
  printf 'ERROR: Python executable is not executable: %s\n' \
    "$python_bin" >&2
  exit 3
}

# Formal runs execute only committed source. This is checked again on resume,
# so a commit hash never conceals modified Python code.
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  printf '%s\n' \
    'ERROR: refusing to run formal training from a dirty worktree.' >&2
  git status --short >&2
  exit 4
fi
git_branch="$(git symbolic-ref --quiet --short HEAD || true)"
[[ -n "$git_branch" ]] || {
  printf '%s\n' 'ERROR: formal training requires a named Git branch.' >&2
  exit 4
}
git_commit="$(git rev-parse --verify HEAD)"
python_path="$(readlink -f -- "$python_bin")"
arg_sha="$(sha256sum "$arg_file" | cut -d' ' -f1)"
agent_sha="$(sha256sum "$agent_file" | cut -d' ' -f1)"
env_sha="$(sha256sum "$env_file" | cut -d' ' -f1)"
engine_sha="$(sha256sum "$engine_file" | cut -d' ' -f1)"

# This doubles as a CUDA preflight. The signature deliberately excludes
# volatile utilization, temperature, and clock values.
runtime_signature="$("$python_bin" - <<'PY'
import json
import platform
import subprocess
import sys

import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("formal training requires a visible CUDA GPU")
props = torch.cuda.get_device_properties(0)
try:
    driver_uuid = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,uuid",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
except (FileNotFoundError, subprocess.CalledProcessError):
    driver_uuid = "unavailable"
print(json.dumps({
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "torch": str(torch.__version__),
    "torch_cuda": str(torch.version.cuda),
    "visible_gpu_count": torch.cuda.device_count(),
    "gpu0_name": props.name,
    "gpu0_memory": props.total_memory,
    "gpu0_capability": [props.major, props.minor],
    "driver_uuid": driver_uuid,
}, sort_keys=True, separators=(",", ":")))
PY
)"

# Reject accidental protocol edits before creating or trusting a manifest.
"$python_bin" - "$arg_file" "$agent_file" "$env_file" \
  "$num_envs" "$target_samples" "$out_dir" "$seed" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

arg_file, agent_file, env_file = map(Path, sys.argv[1:4])
tokens = shlex.split(arg_file.read_text(encoding="utf-8"), comments=True)
if len(tokens) % 2:
    raise SystemExit("formal argument file must contain option/value pairs")
args = dict(zip(tokens[0::2], tokens[1::2]))
expected = {
    "--num_envs": sys.argv[4],
    "--max_samples": sys.argv[5],
    "--out_dir": sys.argv[6],
    "--rand_seed": sys.argv[7],
    "--visualize": "false",
    "--save_int_models": "true",
    "--agent_config": str(agent_file),
    "--env_config": str(env_file),
    "--engine_config": "data/engines/isaac_lab_engine.yaml",
}
for key, value in expected.items():
    if args.get(key) != value:
        raise SystemExit(
            f"formal argument mismatch for {key}: "
            f"expected {value!r}, got {args.get(key)!r}"
        )
agent = yaml.safe_load(agent_file.read_text(encoding="utf-8"))
env = yaml.safe_load(env_file.read_text(encoding="utf-8"))
if agent.get("agent_name") != "PHASE_TRANSITION_CRITIC":
    raise SystemExit("formal agent is not PHASE_TRANSITION_CRITIC")
if int(agent.get("steps_per_iter", -1)) != 32:
    raise SystemExit("formal steps_per_iter must be 32")
if env.get("env_name") != "phase_transition_critic":
    raise SystemExit("formal environment is not phase_transition_critic")
if not str(env.get("motion_file", "")).endswith("humanoid_roll.pkl"):
    raise SystemExit("formal environment is not Roll")
PY

if [[ -s "$manifest_file" ]]; then
  # This file is generated below from quoted local values only.
  # shellcheck disable=SC1090
  source "$manifest_file"
  [[ "${MANIFEST_BRANCH:-}" == "$git_branch" \
     && "${MANIFEST_COMMIT:-}" == "$git_commit" ]] || {
    printf '%s\n' 'ERROR: Git identity differs from the run manifest.' >&2
    exit 5
  }
  [[ "${MANIFEST_ARG_SHA:-}" == "$arg_sha" \
     && "${MANIFEST_AGENT_SHA:-}" == "$agent_sha" \
     && "${MANIFEST_ENV_SHA:-}" == "$env_sha" \
     && "${MANIFEST_ENGINE_SHA:-}" == "$engine_sha" ]] || {
    printf '%s\n' 'ERROR: a configuration differs from the run manifest.' >&2
    exit 5
  }
  [[ "${MANIFEST_PYTHON_PATH:-}" == "$python_path" \
     && "${MANIFEST_RUNTIME:-}" == "$runtime_signature" ]] || {
    printf '%s\n' 'ERROR: Python/CUDA/GPU runtime differs from the manifest.' >&2
    exit 5
  }
else
  if [[ -e "$checkpoint_file" || -e "$out_dir/model.pt" \
        || -e "$out_dir/log.txt" || -e "$out_dir/train_metrics.jsonl" ]]; then
    printf '%s\n' \
      'ERROR: training artifacts exist but manifest.env is missing.' >&2
    exit 5
  fi
  manifest_tmp="$(mktemp "$out_dir/.manifest.env.XXXXXX")"
  {
    printf 'MANIFEST_VERSION=%q\n' 1
    printf 'MANIFEST_CREATED=%q\n' "$(date --iso-8601=seconds)"
    printf 'MANIFEST_HOST=%q\n' "$(hostname)"
    printf 'MANIFEST_BRANCH=%q\n' "$git_branch"
    printf 'MANIFEST_COMMIT=%q\n' "$git_commit"
    printf 'MANIFEST_ARG_SHA=%q\n' "$arg_sha"
    printf 'MANIFEST_AGENT_SHA=%q\n' "$agent_sha"
    printf 'MANIFEST_ENV_SHA=%q\n' "$env_sha"
    printf 'MANIFEST_ENGINE_SHA=%q\n' "$engine_sha"
    printf 'MANIFEST_PYTHON_PATH=%q\n' "$python_path"
    printf 'MANIFEST_RUNTIME=%q\n' "$runtime_signature"
    printf 'MANIFEST_NUM_ENVS=%q\n' "$num_envs"
    printf 'MANIFEST_STEPS_PER_ITER=%q\n' "$steps_per_iter"
    printf 'MANIFEST_TARGET_ITERS=%q\n' "$target_iters"
    printf 'MANIFEST_TARGET_SAMPLES=%q\n' "$target_samples"
    printf 'MANIFEST_SEED=%q\n' "$seed"
  } > "$manifest_tmp"
  mv -f -- "$manifest_tmp" "$manifest_file"
fi

existing_training="$(pgrep -af '[m]imickit/run.py' || true)"
if [[ -n "$existing_training" ]]; then
  printf 'ERROR: refusing to share the GPU with an existing MimicKit run:\n%s\n' \
    "$existing_training" >&2
  exit 6
fi

# Validate the actual PhaseTransitionCriticAgent continuation state: metadata,
# all three optimizers, fixed transition statistics/private counter, and the
# seven-field transition replay buffer.
checkpoint_samples() {
  "$python_bin" - "$1" "$agent_sha" "$env_sha" "$engine_sha" \
    "$num_envs" "$steps_per_iter" <<'PY'
import sys

import torch

checkpoint_file, agent_sha, env_sha, engine_sha, num_envs, steps = sys.argv[1:]
try:
    checkpoint = torch.load(
        checkpoint_file, map_location="cpu", weights_only=False, mmap=True
    )
except TypeError:
    checkpoint = torch.load(
        checkpoint_file, map_location="cpu", weights_only=False
    )
required = {
    "checkpoint_version", "metadata", "model_state_dict",
    "optimizer_state_dicts", "normalizer_training_states",
    "trainer_state", "exp_buffer_sampling_state", "rng_state",
    "replay_buffer_states",
}
missing = required.difference(checkpoint)
if missing:
    raise SystemExit(f"checkpoint sections missing: {sorted(missing)}")
if int(checkpoint["checkpoint_version"]) != 2:
    raise SystemExit("checkpoint version mismatch")
metadata = checkpoint["metadata"]
if metadata.get("agent_class") != "PhaseTransitionCriticAgent":
    raise SystemExit("checkpoint agent class mismatch")
if int(metadata.get("num_envs", -1)) != int(num_envs):
    raise SystemExit("checkpoint num_envs mismatch")
if int(metadata.get("world_size", -1)) != 1:
    raise SystemExit("checkpoint world_size mismatch")
expected_context = {
    "agent_config_sha256": agent_sha,
    "env_config_sha256": env_sha,
    "engine_config_sha256": engine_sha,
}
if metadata.get("checkpoint_context") != expected_context:
    raise SystemExit("checkpoint configuration metadata mismatch")

expected_optimizers = {
    "_actor_optimizer", "_critic_optimizer", "_disc_optimizer"
}
optimizers = checkpoint["optimizer_state_dicts"]
if set(optimizers) != expected_optimizers:
    raise SystemExit("checkpoint optimizer set mismatch")
for name, state in optimizers.items():
    if set(state) != {"optimizer", "steps"} or int(state["steps"]) < 0:
        raise SystemExit(f"checkpoint optimizer state is invalid: {name}")

replays = checkpoint["replay_buffer_states"]
if set(replays) != {"_disc_buffer"}:
    raise SystemExit("checkpoint replay-buffer set mismatch")
expected_fields = {
    "sim_state", "sim_motion", "ref_state", "ref_motion",
    "motion_id", "motion_phase", "motion_is_wrap",
}
if set(replays["_disc_buffer"].get("buffers", {})) != expected_fields:
    raise SystemExit("checkpoint transition replay fields mismatch")
required_model = {
    "_transition_state_mean", "_transition_state_scale",
    "_transition_motion_mean", "_transition_motion_scale",
    "_transition_private_counter", "_model._disc_layers.0.weight",
    "_model._disc_logits.weight",
}
if not required_model.issubset(checkpoint["model_state_dict"]):
    raise SystemExit("checkpoint transition critic state is incomplete")

trainer = checkpoint["trainer_state"]
samples = int(trainer["sample_count"])
next_iter = int(trainer["next_iter"])
samples_per_iter = int(num_envs) * int(steps)
if samples < 0 or samples % samples_per_iter:
    raise SystemExit("checkpoint sample_count is not iteration-aligned")
if next_iter != samples // samples_per_iter:
    raise SystemExit("checkpoint iteration/sample_count mismatch")
if int(trainer["exp_total_samples"]) != samples:
    raise SystemExit("checkpoint experience count mismatch")
print(samples)
PY
}

write_atomic "$out_dir/launcher.pid" "$$"
write_atomic "$out_dir/started_at.txt" "$(date --iso-8601=seconds)"
rm -f -- "$out_dir/FAILED"
write_atomic "$out_dir/STATUS" "PREFLIGHT"

eval_dir="$out_dir/eval/final"
eval_complete=false
if [[ -s "$eval_dir/summary.json" && -s "$eval_dir/episodes.npz" \
      && -s "$eval_dir/timeseries.npz" ]]; then
  eval_complete=true
fi

if [[ -e "$out_dir/DONE" ]]; then
  if [[ "$eval_complete" == true && -s "$checkpoint_file" ]]; then
    samples="$(checkpoint_samples "$checkpoint_file")"
    if ((samples == target_samples)); then
      write_atomic "$out_dir/STATUS" "DONE"
      success=1
      printf 'Roll run is already complete at %s samples.\n' "$samples"
      exit 0
    fi
  fi
  mv -- "$out_dir/DONE" \
    "$out_dir/DONE.invalid.$(date +%Y%m%dT%H%M%S)"
fi

resume_args=()
training_needed=true
if [[ -s "$checkpoint_file" ]]; then
  samples="$(checkpoint_samples "$checkpoint_file")"
  if ((samples < target_samples)); then
    resume_args=(--resume_file "$checkpoint_file")
    write_atomic "$out_dir/STATUS" \
      "RESUMING from $samples/$target_samples"
    printf 'Resuming from %s/%s samples.\n' "$samples" "$target_samples"
  elif ((samples == target_samples)); then
    training_needed=false
  else
    printf 'ERROR: checkpoint exceeds budget: %s > %s\n' \
      "$samples" "$target_samples" >&2
    exit 7
  fi
elif [[ -e "$out_dir/model.pt" || -e "$out_dir/log.txt" \
        || -e "$out_dir/train_metrics.jsonl" ]]; then
  printf '%s\n' \
    'ERROR: orphaned training artifacts exist without checkpoint.pt.' >&2
  exit 7
fi

if [[ "$training_needed" == true ]]; then
  train_cmd=(
    "$python_bin" mimickit/run.py
    --arg_file "$arg_file"
    "${resume_args[@]}"
  )
  {
    printf '[%s] COMMAND' "$(date --iso-8601=seconds)"
    printf ' %q' "${train_cmd[@]}"
    printf '\n'
  } >> "$out_dir/console.log"
  write_atomic "$out_dir/STATUS" "TRAINING"
  PYTHONUNBUFFERED=1 "${train_cmd[@]}" >> "$out_dir/console.log" 2>&1 &
  child_pid=$!
  write_atomic "$out_dir/train.pid" "$child_pid"
  if wait "$child_pid"; then
    child_pid=""
  else
    rc=$?
    child_pid=""
    exit "$rc"
  fi
fi

for artifact in model.pt checkpoint.pt log.txt train_metrics.jsonl; do
  [[ -s "$out_dir/$artifact" ]] || {
    printf 'ERROR: training exited without %s\n' "$artifact" >&2
    exit 8
  }
done
samples="$(checkpoint_samples "$checkpoint_file")"
if ((samples != target_samples)); then
  printf 'ERROR: checkpoint has %s/%s samples after a clean exit.\n' \
    "$samples" "$target_samples" >&2
  exit 8
fi

mkdir -p "$eval_dir"
if [[ "$eval_complete" != true ]]; then
  eval_cmd=(
    "$python_bin" "$evaluator"
    --model-file "$out_dir/model.pt"
    --env-config "$env_file"
    --agent-config "$agent_file"
    --engine-config "$engine_file"
    --method PhaseTransitionCritic
    --motion roll
    --num-envs 256
    --steps 300
    --start-mode phase0
    --condition nominal
    --seed "$seed"
    --out-dir "$eval_dir"
  )
  {
    printf '[%s] COMMAND' "$(date --iso-8601=seconds)"
    printf ' %q' "${eval_cmd[@]}"
    printf '\n'
  } >> "$eval_dir/console.log"
  write_atomic "$out_dir/STATUS" "EVALUATING"
  PYTHONUNBUFFERED=1 "${eval_cmd[@]}" >> "$eval_dir/console.log" 2>&1 &
  child_pid=$!
  write_atomic "$out_dir/eval.pid" "$child_pid"
  if wait "$child_pid"; then
    child_pid=""
  else
    rc=$?
    child_pid=""
    exit "$rc"
  fi
fi

for artifact in summary.json episodes.npz timeseries.npz; do
  [[ -s "$eval_dir/$artifact" ]] || {
    printf 'ERROR: evaluation exited without %s\n' "$artifact" >&2
    exit 9
  }
done
"$python_bin" - "$eval_dir/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    summary = json.load(stream)
metadata = summary.get("metadata", {})
protocol = summary.get("protocol", {})
if metadata.get("method") != "PhaseTransitionCritic":
    raise SystemExit("evaluation method label mismatch")
if metadata.get("motion") != "roll":
    raise SystemExit("evaluation motion mismatch")
if metadata.get("condition") != "nominal":
    raise SystemExit("evaluation condition mismatch")
if int(metadata.get("seed", -1)) != 0:
    raise SystemExit("evaluation seed mismatch")
if protocol.get("start_mode") != "phase0":
    raise SystemExit("evaluation start mode mismatch")
if int(protocol.get("num_episodes", -1)) != 256:
    raise SystemExit("evaluation episode count mismatch")
if int(protocol.get("requested_steps", -1)) != 300:
    raise SystemExit("evaluation horizon mismatch")
PY

write_atomic "$out_dir/DONE" \
  "samples=$samples" \
  "completed_at=$(date --iso-8601=seconds)" \
  "evaluation=$eval_dir/summary.json"
rm -f -- "$out_dir/FAILED"
write_atomic "$out_dir/STATUS" "DONE"
success=1
printf 'Roll training and fixed-protocol evaluation completed successfully.\n'
