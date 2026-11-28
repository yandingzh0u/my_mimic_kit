# MIRA with a constrained monotonic discriminator

MIRA replaces ADD's discriminator with a single monotone ReLU network.
It retains ADD's actor, critic, BCE labels and reward mapping.

Paired policy/reference observations are converted to 64 physical errors:
root position distance, root rotation angle, 14 non-root body position
distances, 14 non-root world-body rotation angles, 28 absolute joint velocity
errors, three root velocity errors and three root angular velocity errors.
Body positions are root-relative in world axes. Rotation angles use a stable
6D-to-quaternion conversion, including rotations near pi.

The original seven evaluator metrics supply running mean-absolute scales,
broadcast to their corresponding components without centering. Inputs are
not averaged before entering the network. The zero Root-relative body
position and duplicate Root body rotation are omitted from the input; the
original seven metrics are reconstructed for evaluation and scale statistics.
One `64 → 341 → 512 → 1` network learns their joint energy `E` using
nonnegative linear contractions and ReLU in both hidden layers. The final
linear layer has an ordinary learnable scalar bias. The
discriminator score is `s(e) = -E(e)`. Reducing an error while holding the
others fixed cannot reduce the score.

The classification loss is the sum of the positive BCE at zero error and
the mean negative BCE on the policy/replay batch; MIRA has no discriminator
logit regularizer. The reward uses ADD's numerically clipped
`-log(1 - sigmoid(s))` mapping, with no extra multiplier. GP is disabled.

The optimizer learns the same matrices used in the forward pass. After each
discriminator SGD step, negative entries in its candidate `U` are clipped to
zero. This gives `C`, the nearest nonnegative matrix in Frobenius norm. If
`C` has spectral norm above one, `C` is divided by that norm. This is the
smallest uniform reduction that enforces the bound and leaves the ratios of
positive entries intact. The two steps are a feasibility repair, not the
Frobenius projection onto the joint constraint set. They are applied outside
backpropagation, so a zero connection can become positive in a later update.
No auxiliary weight coordinates or new loss coefficients are needed.
The unit sensitivity bound applies to the normalized input coordinates.

The supported environment uses global, single-frame, 172-dimensional ADD
observations for the 15-body/28-DOF humanoid. Replay retains paired raw
observations so that physical rotation errors can be reconstructed correctly.
PPO minibatches use uniform transition sampling and reference reset times
use the motion library's normal sampling. There are no phase-bin balancing,
reference-centered action, structured-negative, CPL, multi-discriminator or
seven-encoder paths in the maintained MIRA code.

Train backflip for 2000 updates with 8192 environments and seed 0:

```bash
python mimickit/run.py \
  --arg_file PAPER/configs/args/mira_2k_8192_args.txt \
  --env_config PAPER/configs/envs/mira_backflip_env.yaml \
  --out_dir output/mira_relu_backflip
```

The agent configuration is
[`mira_monotone_humanoid_agent.yaml`](../data/agents/mira_monotone_humanoid_agent.yaml).
Each physical component is divided by its running group scale and by the
square root of the group size.

The direct-weight ReLU implementation uses checkpoint schema 29. Earlier MIRA checkpoints
must be evaluated or resumed with their original source and configuration,
using the run's frozen snapshot where available. Historical results are kept
under `output/`. They record the versions that were actually trained.

To play a soccer checkpoint trained with the current implementation:

```bash
python mimickit/run.py \
  --arg_file args/mira_monotone_humanoid_soccer_cmu_10_02_args.txt \
  --mode test --num_envs 1 --visualize true \
  --model_file /path/to/new/model.pt
```
