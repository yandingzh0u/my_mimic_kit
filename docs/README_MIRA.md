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
nonnegative weights, learnable diagonal normalization and ReLU in both hidden
layers. The final linear layer has an ordinary learnable scalar bias. The
discriminator score is `s(e) = -E(e)`. Reducing an error while holding the
others fixed cannot reduce the score.

The classification loss is the sum of the positive BCE at zero error and
the mean negative BCE on the policy/replay batch. The existing head weight
penalty is added separately. The reward uses ADD's numerically clipped
`-log(1 - sigmoid(s))` mapping, with no extra multiplier. GP is disabled.

All three linear layers use the non-residual construction in
[Araujo et al., ICLR 2023](https://arxiv.org/abs/2303.03169), Theorem 1 and
Appendix C.3: `W = P diag(sqrt(q / (P.T @ (P @ q))))`, with `P >= 0` and
positive learned `q`. This adds 917 log-q parameters, retains the L2 operator
norm bound, and introduces no free output gain or new loss coefficient.
The ordinary discriminator SGD and its existing weight decay also update
these parameters. A deterministic Perron initialization recovers SN for a
shared strictly positive raw weight matrix; this does not imply identical
random initialization of separately constructed old and new networks.
Training uses stable log-domain reductions, including explicit zero-weight
handling, without an SN module. The unit sensitivity bound applies to the
normalized 64-dimensional input, not unnormalized physical units. The scalar Head L2 penalty remains nearly
constant under this construction; restoring that penalty is not its purpose.

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

The ReLU implementation uses checkpoint schema 26. Earlier MIRA checkpoints
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
