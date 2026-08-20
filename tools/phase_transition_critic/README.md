# Phase-matched transition audit

This directory keeps the offline decision path separate from RL training.

## Data that exists today

- `output/aligned_add_roll_2k_8192_seed0/model.pt` is the successful historical
  Aligned ADD checkpoint. It has no transition-evaluation NPZ.
- `output/add_roll_seed0/model.pt` is the historical ADD checkpoint. It also has
  no transition-evaluation NPZ, and its training protocol is not matched to the
  Aligned ADD run (10 s with pose termination versus 2 s without it).
- `output/action_pullback_add_roll_2k_8192_seed0/eval/final/` is a measured Roll
  shortcut: completion is zero and episode winding is about 0.014--0.045. Its
  NPZ files contain physical aggregate errors but no complete discriminator
  feature transition.
- Old replay buffers cannot replace rollout collection. They contain randomly
  sampled post-step observations; temporal adjacency, and therefore the actual
  policy transition, is unavailable.

Consequently, none of the existing NPZ artifacts is sufficient for a valid
transition-critic ranking claim.

## Collect held-out transitions

Use the exact same evaluation protocol for the successful and shortcut policy.
For example (run with the Isaac Lab environment, not plain system Python):

```bash
/home/y/miniconda3/envs/env_isaaclab/bin/python \
  tools/phase_transition_critic/collect_rollout.py \
  --model-file output/aligned_add_roll_2k_8192_seed0/model.pt \
  --env-config data/envs/aligned_add_humanoid_roll_eval_env.yaml \
  --agent-config output/aligned_add_roll_2k_8192_seed0/agent_config.yaml \
  --engine-config output/aligned_add_roll_2k_8192_seed0/engine_config.yaml \
  --num-envs 256 --steps 300 --start-mode grid \
  --out output/phase_transition_audit/aligned_roll/transitions.npz
```

Collect a known shortcut checkpoint separately. Do not concatenate a
success/shortcut label into either file, and do not use either file to fit the
critic. The raw schema is:

```text
x_t, x_t1, r_t, r_t1       [num_rows, phi_dim]
episode_id, step_index      [num_rows]
phase, alive                [num_rows]
```

Here `x` is the policy state in raw ADD `phi` coordinates and `r` is the
phase-matched reference state. The one-step motions are derived without loss:
`x_t1 - x_t` and `r_t1 - r_t`.

Validate each dump before scoring:

```bash
python tools/phase_transition_critic/rollout_contract.py TRANSITIONS.npz
```

## Gate a fitted critic

Score the two held-out files through the critic's centered advantage
`A(candidate, reference)` with `score_rollouts.py`. It constructs the same
wrong-phase reference hard negatives as training, measures pose/velocity
gradient RMS, and audits held-out interpolation gradient norms:

```bash
/home/y/miniconda3/envs/env_isaaclab/bin/python \
  tools/phase_transition_critic/score_rollouts.py \
  --critic-model-file output/phase_transition_critic_roll_smoke_seed0/model.pt \
  --critic-env-config data/envs/phase_transition_critic_humanoid_roll_env.yaml \
  --critic-agent-config data/agents/phase_transition_critic_humanoid_agent.yaml \
  --success-transitions output/phase_transition_audit/aligned_roll/transitions.npz \
  --shortcut-transitions output/phase_transition_audit/shortcut_roll/transitions.npz \
  --out output/phase_transition_audit/scores.npz
```

Then run the strict gate:

```bash
python tools/phase_transition_critic/offline_gate.py \
  --scores output/phase_transition_audit/scores.npz \
  --out-json output/phase_transition_audit/gate.json
```

The default gate requires all of the following:

- reference ranks above successful Roll with probability at least 0.75;
- successful Roll ranks above shortcut with probability at least 0.75;
- the correct reference transition ranks above a wrong-phase reference hard
  negative with paired probability at least 0.75;
- both pose and velocity blocks have nonzero gradient RMS and neither accounts
  for more than 90% of their combined sensitivity;
- both next-state error and motion error affect the score and neither accounts
  for more than 90% of their combined sensitivity;
- centered reference score is numerically zero, scores are noncollapsed, and
  gradient norms remain approximately one.

All probabilities are computed after reducing transitions to episode means.
Passing this gate is only permission to spend a small RL budget; it is not a
claim that Roll has been learned.
