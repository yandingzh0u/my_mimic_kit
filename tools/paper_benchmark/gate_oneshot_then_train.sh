#!/usr/bin/env bash
set -u

DARE_ROOT="/home/y/my_mimic_mixgrpo/MimicKit"
ONE_SHOT_DIR="${DARE_ROOT}/output/paper_benchmark/dare_10layer_cpl_oneshot_climb_500_8192_seed0_headless"
FORMAL_DIR="${DARE_ROOT}/output/paper_benchmark/dare_10layer_cpl_oneshot_climb_3000_8192_seed0_headless"
GATE_LOG="${DARE_ROOT}/output/paper_benchmark/dare_10layer_cpl_oneshot_gate.log"
ONE_SHOT_PID="1358750"

cd "${DARE_ROOT}" || exit 1
{
  echo "[$(date '+%F %T')] gate scheduled; initial wait 1200s"
  sleep 1200
  echo "[$(date '+%F %T')] 20-minute check reached"
} >> "${GATE_LOG}" 2>&1

# Do not start a second Isaac job while the 500-iteration gate is still using
# the GPU.  Once it exits, evaluate its final JSONL row.
while kill -0 "${ONE_SHOT_PID}" 2>/dev/null; do
  {
    echo "[$(date '+%F %T')] one-shot still running; waiting for completion"
    tail -n 1 "${ONE_SHOT_DIR}/train_metrics.jsonl" 2>/dev/null || true
  } >> "${GATE_LOG}" 2>&1
  sleep 60
done

GATE_RESULT=$(python - "${ONE_SHOT_DIR}/train_metrics.jsonl" \
  "output/paper_benchmark/dare_10layer_cpl_climb_3000_8192_seed0_headless/train_metrics.jsonl" \
  "output/paper_benchmark/dare_10layer_cpl_nocalib_climb_3000_8192_seed0_headless/train_metrics.jsonl" <<'PY'
import json
import math
import pathlib
import sys

test_path, rollout_path, no_calib_path = map(pathlib.Path, sys.argv[1:])

def rows(path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

test = rows(test_path)
rollout = rows(rollout_path)
no_calib = rows(no_calib_path)
if not test:
    print("FAIL no one-shot metrics", end="")
    raise SystemExit

last = test[-1]
iteration = int(last.get("Iteration", -1))
frozen = bool(last.get("Norm_Frozen", 0.0))
updates = int(last.get("Disc_Calibration_Updates", 0))
kappa = float(last.get("Disc_Kappa", float("nan")))
start0 = float(last.get("Test_Episode_Length_Start0", float("nan")))
core = ["Root_Pos_Err", "Body_Pos_Err", "Root_Rot_Err", "Body_Rot_Err", "Dof_Vel_Err"]
finite_core = all(math.isfinite(float(last.get(k, float("nan")))) for k in core)
finite_calib = math.isfinite(kappa) and math.isfinite(float(last.get("Disc_Logit_Center", float("nan")))) and math.isfinite(float(last.get("Disc_Logit_Std", float("nan"))))

passed = iteration >= 400 and frozen and updates >= 1 and finite_core and finite_calib and 0.2 <= kappa <= 2.0 and math.isfinite(start0)
print(("PASS" if passed else "FAIL") +
      f" iter={iteration} frozen={int(frozen)} updates={updates} kappa={kappa:.4f} start0={start0:.2f}", end="")
raise SystemExit(0 if passed else 1)
PY
)
echo "[$(date '+%F %T')] gate result: ${GATE_RESULT}" >> "${GATE_LOG}"

if [[ "${GATE_RESULT}" != PASS* ]]; then
  echo "[$(date '+%F %T')] formal 3000-iteration training NOT started" >> "${GATE_LOG}"
  exit 0
fi

if [[ -e "${FORMAL_DIR}" ]]; then
  echo "[$(date '+%F %T')] formal output already exists; refusing to overwrite" >> "${GATE_LOG}"
  exit 0
fi

mkdir -p "${FORMAL_DIR}"
echo "[$(date '+%F %T')] starting 3000-iteration one-shot training" >> "${GATE_LOG}"
PYTHONPATH=mimickit /home/y/miniconda3/envs/env_isaaclab/bin/python \
  mimickit/run.py --mode train --num_envs 8192 --devices cuda:0 \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/paper_benchmark/dare_climb_env.yaml \
  --agent_config data/agents/dare_10layer_cpl_oneshot_climb_agent.yaml \
  --visualize false --out_dir "${FORMAL_DIR}" \
  --max_samples 786432000 --rand_seed 0 --save_int_models true --logger txt \
  >> "${FORMAL_DIR}/console.log" 2>&1
echo "[$(date '+%F %T')] formal training exited with code $?" >> "${GATE_LOG}"
