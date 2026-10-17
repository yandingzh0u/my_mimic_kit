# RL objectives and reward agents

DARE no longer inherits `PPOAgent`. The main dependency is now:

```text
DAREAgent → ADDAgent → AMPAgent → OnPolicyAgent → BaseAgent
                                      │
                                      └─ algorithm object (default: PPO)
```

Responsibilities:

- `dare_agent.py`, `add_agent.py`, `amp_agent.py`: discriminator, replay and
  reward construction. These run before algorithm target construction; the
  discriminator update still runs after the policy/value updates.
- `on_policy_agent.py`: rollout collection, normalization, phase weights,
  optimizer scheduling, minibatches and policy diagnostics.
- `algorithms/ppo.py`: TD-lambda targets, normalized advantages, value loss
  and clipped policy loss. It consumes the existing reward; it neither owns
  nor imports the discriminator, environment implementation or replay logic.

`PPOAgent` is a six-line compatibility entry for existing plain-PPO configs
and checkpoint class names. It contains no second copy of the algorithm.
The Actor/value network module names are unchanged to preserve model keys.

Old configs continue to select PPO without edits. New configs may explicitly
set `algorithm: "PPO"`. An unknown name fails; it never silently falls back.
An objective provides `build_train_data(agent)`,
`compute_actor_loss(agent, batch)` and `compute_critic_loss(agent, batch)`.
The `agent` argument supplies the existing model, normalizers and buffer;
there is no attribute-forwarding proxy or dynamic method replacement in the
production implementation.

This is the first extraction, not a claim that every historical algorithm has
already been converted. ASE keeps its specialized latent-conditioned losses;
the existing REPPO adapter overrides its own targets/losses/update and does not
instantiate an unused PPO object. REPPO changes and training are out of scope
for the PPO preservation audit.

The refactor deliberately preserves operation order, random draws, SGD state,
model keys, positive-first SN, logging semantics and evaluation frequency.
It does not fix historical diagnostics or change a numerical reduction while
claiming baseline equivalence.

Verification (a new output directory is required):

```bash
PYTHONPATH=mimickit OMP_NUM_THREADS=4 \
/home/y/miniconda3/envs/env_isaaclab/bin/python -B -m pytest -q \
  tests/test_ppo_isolation.py
/home/y/miniconda3/envs/env_isaaclab/bin/python -B -u \
  tools/paper_benchmark/check_ppo_isolation.py \
  --out output/diagnostics/ppo_isolation_new
```

The physical audit freezes two source trees, extracts the old implementation
directly from commit `58373240451d6ab093c7e3a154164a94f9dabab3`, and runs each
for three actual iterations with 8192 environments and archived configs.
`all_iterations.jsonl` records every full-precision row without changing the
100-iteration evaluation/output schedule. `comparison.json` separates exact
log equality, post-update tensor equality, and historical iteration-0 equality.
Wall time, throughput and peak allocation are reported but cannot reasonably
be required to equal across runs. The old archive has no iteration-1/2 rows;
a current run of old source is explicitly labeled as a new reference run.
