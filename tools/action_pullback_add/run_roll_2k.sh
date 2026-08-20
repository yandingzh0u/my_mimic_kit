#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible, restartable launcher for the 8192-env, 2000-iteration Roll run.
# Launch in the background with:
#   nohup setsid bash tools/action_pullback_add/run_roll_2k.sh \
#     > output/action_pullback_add_roll_2k_8192_seed0/launcher.log 2>&1 \
#     < /dev/null &

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${MIMICKIT_PYTHON:-/home/y/miniconda3/envs/env_isaaclab/bin/python}"
arg_file="args/action_pullback_add_humanoid_roll_2k_8192_args.txt"
agent_file="data/agents/action_pullback_add_humanoid_agent.yaml"
env_file="data/envs/action_pullback_add_humanoid_roll_env.yaml"
engine_file="data/engines/isaac_lab_engine.yaml"
out_dir="output/action_pullback_add_roll_2k_8192_seed0"
checkpoint_file="$out_dir/checkpoint.pt"
target_samples=524288000

mkdir -p "$out_dir"
exec 9>"$out_dir/launcher.lock"
if ! flock -n 9; then
  printf 'ERROR: another Roll launcher holds %s\n' "$out_dir/launcher.lock" >&2
  exit 2
fi

existing_training="$(pgrep -af '[m]imickit/run.py' || true)"
if [[ -n "$existing_training" ]]; then
  printf 'ERROR: refusing to share the GPU with an existing MimicKit run:\n%s\n' \
    "$existing_training" >&2
  exit 2
fi

printf '%s\n' "$$" > "$out_dir/launcher.pid"
printf 'RUNNING\n' > "$out_dir/STATUS"
printf '%s\n' "$(date --iso-8601=seconds)" > "$out_dir/started_at.txt"

success=0
train_pid=""
on_exit() {
  local rc=$?
  printf '%s\n' "$rc" > "$out_dir/EXIT_STATUS"
  printf '%s\n' "$(date --iso-8601=seconds)" > "$out_dir/finished_at.txt"
  if [[ "$success" -eq 0 ]]; then
    printf 'FAILED (exit %s)\n' "$rc" > "$out_dir/STATUS"
  fi
}
trap on_exit EXIT

forward_signal() {
  local signal="$1"
  if [[ -n "$train_pid" ]] && kill -0 "$train_pid" 2>/dev/null; then
    kill "-$signal" "$train_pid" 2>/dev/null || true
  fi
}
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap 'forward_signal HUP' HUP

for required in "$python_bin" "$arg_file" "$agent_file" "$env_file" \
                "$engine_file" tools/paper_eval/evaluate_checkpoint.py; do
  if [[ ! -e "$required" ]]; then
    printf 'ERROR: required path is missing: %s\n' "$required" >&2
    exit 3
  fi
done

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  printf 'ERROR: refusing to start a formal run from a dirty worktree.\n' >&2
  git status --short >&2
  exit 4
fi

git_sha="$(git rev-parse HEAD)"
agent_sha="$(sha256sum "$agent_file" | cut -d' ' -f1)"
env_sha="$(sha256sum "$env_file" | cut -d' ' -f1)"
engine_sha="$(sha256sum "$engine_file" | cut -d' ' -f1)"
manifest="$out_dir/manifest.env"

if [[ -s "$manifest" ]]; then
  # The manifest contains only values produced below by git/sha256sum.
  # shellcheck disable=SC1090
  source "$manifest"
  [[ "${MANIFEST_GIT_SHA:-}" == "$git_sha" ]] || {
    printf 'ERROR: Git commit differs from the original run manifest.\n' >&2
    exit 5
  }
  [[ "${MANIFEST_AGENT_SHA:-}" == "$agent_sha" \
     && "${MANIFEST_ENV_SHA:-}" == "$env_sha" \
     && "${MANIFEST_ENGINE_SHA:-}" == "$engine_sha" ]] || {
    printf 'ERROR: a training configuration differs from the manifest.\n' >&2
    exit 6
  }
else
  {
    printf 'MANIFEST_GIT_SHA=%q\n' "$git_sha"
    printf 'MANIFEST_AGENT_SHA=%q\n' "$agent_sha"
    printf 'MANIFEST_ENV_SHA=%q\n' "$env_sha"
    printf 'MANIFEST_ENGINE_SHA=%q\n' "$engine_sha"
    printf 'MANIFEST_HOST=%q\n' "$(hostname)"
    printf 'MANIFEST_PYTHON=%q\n' "$python_bin"
    printf 'MANIFEST_CREATED=%q\n' "$(date --iso-8601=seconds)"
  } > "$manifest"
fi

checkpoint_samples() {
  "$python_bin" - "$1" <<'PY'
import sys
import torch

checkpoint = torch.load(
    sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
metadata = checkpoint["metadata"]
trainer = checkpoint["trainer_state"]
if metadata["agent_class"] != "ActionPullbackADDAgent":
    raise SystemExit("checkpoint agent class mismatch")
if int(metadata["num_envs"]) != 8192 or int(metadata["world_size"]) != 1:
    raise SystemExit("checkpoint execution topology mismatch")
expected_optimizers = {
    "_actor_optimizer", "_critic_optimizer", "_disc_optimizer",
    "_response_optimizer",
}
if set(checkpoint["optimizer_state_dicts"]) != expected_optimizers:
    raise SystemExit("checkpoint optimizer set mismatch")
if "_response_self_norm" not in checkpoint["normalizer_training_states"]:
    raise SystemExit("checkpoint response normalizer is missing")
required_model_keys = {
    "_response_delta_norm._count",
    "_response_delta_norm._mean_abs",
    "_model._response_net.0.weight",
}
if not required_model_keys.issubset(checkpoint["model_state_dict"]):
    raise SystemExit("checkpoint response model state is incomplete")
print(int(trainer["sample_count"]))
PY
}

if [[ -s "$out_dir/DONE" && -s "$out_dir/eval/final/summary.json" ]]; then
  samples="$(checkpoint_samples "$checkpoint_file")"
  if (( samples >= target_samples )); then
    printf 'DONE\n' > "$out_dir/STATUS"
    success=1
    printf 'Roll run already complete at %s samples.\n' "$samples"
    exit 0
  fi
fi

resume_args=()
training_needed=true
if [[ -s "$checkpoint_file" ]]; then
  samples="$(checkpoint_samples "$checkpoint_file")"
  if (( samples < target_samples )); then
    resume_args=(--resume_file "$checkpoint_file")
    printf 'Resuming from %s samples.\n' "$samples"
  else
    training_needed=false
    printf 'Training budget already reached at %s samples; running evaluation.\n' \
      "$samples"
  fi
elif [[ -e "$out_dir/model.pt" || -e "$out_dir/log.txt" ]]; then
  printf 'ERROR: refusing to overwrite orphaned training artifacts without a checkpoint.\n' >&2
  exit 7
fi

if [[ "$training_needed" == true ]]; then
  cmd=(
    "$python_bin" mimickit/run.py
    --arg_file "$arg_file"
    "${resume_args[@]}"
  )
  {
    printf '[%s] COMMAND' "$(date --iso-8601=seconds)"
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } >> "$out_dir/console.log"

  PYTHONUNBUFFERED=1 "${cmd[@]}" >> "$out_dir/console.log" 2>&1 &
  train_pid=$!
  printf '%s\n' "$train_pid" > "$out_dir/train.pid"
  if wait "$train_pid"; then
    train_pid=""
  else
    rc=$?
    train_pid=""
    exit "$rc"
  fi
fi

for artifact in model.pt checkpoint.pt log.txt train_metrics.jsonl; do
  [[ -s "$out_dir/$artifact" ]] || {
    printf 'ERROR: training exited without %s\n' "$artifact" >&2
    exit 7
  }
done

samples="$(checkpoint_samples "$checkpoint_file")"
if (( samples < target_samples )); then
  printf 'ERROR: checkpoint has %s/%s samples after a clean exit.\n' \
    "$samples" "$target_samples" >&2
  exit 8
fi

eval_dir="$out_dir/eval/final"
mkdir -p "$eval_dir"
eval_cmd=(
  "$python_bin" tools/paper_eval/evaluate_checkpoint.py
  --model-file "$out_dir/model.pt"
  --env-config "$env_file"
  --agent-config "$agent_file"
  --engine-config "$engine_file"
  --method ActionPullbackADD
  --motion roll
  --num-envs 256
  --steps 300
  --start-mode phase0
  --condition nominal
  --seed 0
  --out-dir "$eval_dir"
)
{
  printf '[%s] COMMAND' "$(date --iso-8601=seconds)"
  printf ' %q' "${eval_cmd[@]}"
  printf '\n'
} >> "$eval_dir/console.log"
PYTHONUNBUFFERED=1 "${eval_cmd[@]}" >> "$eval_dir/console.log" 2>&1

for artifact in summary.json episodes.npz timeseries.npz; do
  [[ -s "$eval_dir/$artifact" ]] || {
    printf 'ERROR: evaluation exited without %s\n' "$artifact" >&2
    exit 9
  }
done

touch "$out_dir/DONE"
printf 'DONE\n' > "$out_dir/STATUS"
success=1
printf 'Roll training and final evaluation completed successfully.\n'
