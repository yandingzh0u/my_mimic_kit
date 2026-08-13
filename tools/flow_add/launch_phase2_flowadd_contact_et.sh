#!/usr/bin/env bash
# Wait for both ADD contact-ET 1k runs to finish, then launch FlowADD roll+spinkick.
set -euo pipefail
ROOT="/home/y/my_mimic_mixgrpo/MimicKit"
PY="/home/y/miniconda3/envs/env_isaaclab/bin/python"
NUM_ENVS=2048
MAX_SAMPLES=$((1000 * 32 * NUM_ENVS))
ROLL_ADD_DIR="$ROOT/output/add_roll_contact_et_1k_seed0"
SPIN_ADD_DIR="$ROOT/output/add_spinkick_contact_et_1k_seed0"
ROLL_FLOW_DIR="$ROOT/output/flowadd_roll_contact_et_1k_seed0"
SPIN_FLOW_DIR="$ROOT/output/flowadd_spinkick_contact_et_1k_seed0"
LOG="$ROOT/output/phase2_flowadd_launcher.log"

still_running() {
  pgrep -f "out_dir output/add_roll_contact_et_1k_seed0" >/dev/null \
    || pgrep -f "out_dir output/add_spinkick_contact_et_1k_seed0" >/dev/null \
    || pgrep -f "add_humanoid_roll_contact_et_env.yaml" >/dev/null \
    || pgrep -f "add_humanoid_spinkick_contact_et_env.yaml" >/dev/null
}

echo "[$(date -Is)] Waiting for ADD roll+spinkick to finish..." | tee -a "$LOG"
while still_running; do
  sleep 60
done
echo "[$(date -Is)] ADD phase done. Launching FlowADD roll+spinkick..." | tee -a "$LOG"

mkdir -p "$ROLL_FLOW_DIR" "$SPIN_FLOW_DIR"

nohup "$PY" "$ROOT/mimickit/run.py" \
  --mode train \
  --engine_config "$ROOT/data/engines/isaac_lab_engine.yaml" \
  --env_config "$ROOT/data/envs/flow_add_humanoid_roll_contact_et_env.yaml" \
  --agent_config "$ROOT/data/agents/flow_add_humanoid_agent.yaml" \
  --num_envs "$NUM_ENVS" \
  --max_samples "$MAX_SAMPLES" \
  --visualize false \
  --out_dir "$ROLL_FLOW_DIR" \
  --rand_seed 0 \
  > "$ROLL_FLOW_DIR/console.log" 2>&1 &
echo "[$(date -Is)] FlowADD roll PID=$!" | tee -a "$LOG"

sleep 20

nohup "$PY" "$ROOT/mimickit/run.py" \
  --mode train \
  --engine_config "$ROOT/data/engines/isaac_lab_engine.yaml" \
  --env_config "$ROOT/data/envs/flow_add_humanoid_spinkick_contact_et_env.yaml" \
  --agent_config "$ROOT/data/agents/flow_add_humanoid_agent.yaml" \
  --num_envs "$NUM_ENVS" \
  --max_samples "$MAX_SAMPLES" \
  --visualize false \
  --out_dir "$SPIN_FLOW_DIR" \
  --rand_seed 0 \
  > "$SPIN_FLOW_DIR/console.log" 2>&1 &
echo "[$(date -Is)] FlowADD spinkick PID=$!" | tee -a "$LOG"
echo "[$(date -Is)] Phase-2 launched." | tee -a "$LOG"
