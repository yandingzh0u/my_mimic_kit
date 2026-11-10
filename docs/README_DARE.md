# DARE with a constrained monotonic discriminator

The maintained DARE implementation uses one Constrained Monotonic Neural
Network (CMNN), following [Runje & Shankaranarayana, ICML 2023](https://proceedings.mlr.press/v202/runje23a.html).
It replaces ADD's discriminator; the actor, critic, BCE labels and ADD reward
mapping remain the same.

Paired policy/reference observations are converted to seven physical tracking
errors: root position, root rotation, body position, body rotation, joint
velocity, root velocity and root angular velocity. The errors are scaled by
their running mean absolute magnitude, without centering, preserving zero.
One `7 → 341 → 512 → 1` network learns their joint energy `E` using
nonnegative weights, spectral normalization and mixed monotonic ELU
activations. Its discriminator score is `s(e) = b - (E(e) - E(0))`.
Reducing an error while holding the others fixed cannot reduce the score.
The reward uses ADD's existing numerically clipped `-log(1 - sigmoid(s))`
mapping, with no extra multiplier. GP is disabled.

The supported environment uses global, single-frame, 172-dimensional ADD
observations for the 15-body/28-DOF humanoid. Replay retains paired raw
observations so that physical rotation errors can be reconstructed correctly.
PPO minibatches use uniform transition sampling and reference reset times
use the motion library's normal sampling. There are no phase-bin balancing,
reference-centered action, structured-negative, CPL, multi-discriminator or
seven-encoder paths in the maintained DARE code.

Train backflip for 2000 updates with 8192 environments and seed 0:

```bash
python mimickit/run.py \
  --arg_file args/paper_benchmark/dare_2k_8192_args.txt \
  --env_config data/envs/paper_benchmark/dare_backflip_env.yaml \
  --out_dir output/dare_cmnn_backflip
```

The agent configuration is
[`dare_monotone_humanoid_agent.yaml`](../data/agents/dare_monotone_humanoid_agent.yaml).
Model and full training checkpoints from the completed CMNN run remain
compatible. Strict resume still checks configuration hashes, so resume an
existing run with its original frozen agent/environment/engine configurations.
Older DARE architectures must use their frozen experiment source.
Historical results and source snapshots under `output/` are retained for
reproduction; they are not runtime dependencies.
