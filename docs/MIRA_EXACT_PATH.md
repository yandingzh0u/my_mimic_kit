# Exact sensitivity of MIRA's scalar nonnegative ReLU reward

This note describes a controlled Backflip experiment, not a change to the
maintained agent or a claim that the new update improves tracking. The
discriminator, physical error input, biases, BCE, reward and PPO are the same in
both arms. Only the post-SGD constraint is different.

## Derivation

Let `E(z)` be a finite scalar ReLU network, with unrestricted finite biases and
nonnegative linear weights `W_1,...,W_L` in input-to-output order. The error
domain is the complete orthant `z >= 0` in normalized coordinates. Write the row
vector `a = W_L ... W_1`.

For each linear stage, `|W x - W y| <= W |x-y|` entrywise since `W >= 0`.
ReLU satisfies the same entrywise comparison with unit gain. Thus

```text
|E(z') - E(z)| <= a |z' - z| <= ||a||_2 ||z' - z||_2.
```

For a nonnegative unit vector `v`, divide the forward pass at `t v` by `t`.
Positive homogeneity of ReLU absorbs the division and all biases vanish as
`t -> infinity`. Every intermediate product in the limit is nonnegative, so
ReLU is the identity there, and `E(t v)/t -> a v`. If `a != 0`, take
`v = a.T/||a||_2`; the energy difference quotient relative to the origin tends
to `||a||_2`. If `a = 0` the entrywise bound already shows that `E` is constant.

For the mathematical ADD reward, `r = min(R, softplus(-E))` with the existing
constant upper cap `R`. Its log, as a function of `E`, is nonincreasing and
1-Lipschitz: in the uncapped region the magnitude of its derivative is
`sigmoid(-E)/softplus(-E) <= 1`, since `log(1+x) >= x/(1+x)` for `x >= 0`.
In the capped region the derivative is zero. Along the same ray with `a != 0`,
`E(t v) -> infinity` and `log r(t v) = -E(t v) + o(1)`. Consequently,

```text
Lip_2(E) = Lip_2(log r) = ||W_L ... W_1||_2  on the full nonnegative orthant.
```

The log appears only in the certificate; the policy still receives the
original reward. Real-valued reward positivity is assumed for the logarithm;
the current finite-precision implementation can round small rewards to zero.
The certificate is a valid upper bound on any subset of the orthant. Equality
does not require or imply that the proving ray occurs in physical rollouts.

If only error group `g` changes, the same proof gives exact groupwise worst-case
log-reward sensitivity `||a_g||_2`, by using a ray supported in that group. With
the existing group RMS scaling and fixed physical scales during the comparison,
one unit is a normalized RMS error difference. The individual budgets obey
`sum_g ||a_g||_2^2 = ||a||_2^2`. These are worst-case sensitivities, not average
group importance or a claim about policy behavior. For example, worsening only
group `g` by a nonnegative perturbation of normalized norm `delta` can reduce
the mathematical reward by at most a fraction
`1 - exp(-||a_g||_2 * delta)` of its current value. This follows by
exponentiating the log-reward bound and using monotonicity.

For a bounded box `0 <= z <= u`, there is also an exact certificate. Let
`D_l(u)` be the diagonal ReLU mask at the upper corner, with a gate set to
one only if its preactivation is strictly positive. Each preactivation is
coordinatewise nondecreasing, so gates absent at that corner are inactive
throughout the box. Applying the same entrywise bound gives

```text
a_u = W_L D_(L-1)(u) W_(L-1) ... D_1(u) W_1
Lip_2(E on [0,u]) = ||a_u||_2.
```

For equality, take an interior point close enough to the corner that all
strictly positive gates remain open. There the network is affine with
gradient `a_u`. On zero-width axes, omit their entries from that gradient.
The magnitude of the uncapped derivative of log reward with respect to
energy, `rho(E) = sigmoid(-E)/softplus(-E)`, is increasing in `E`. Thus the
exact box constant for the mathematical log reward is
`rho(E(u)) ||a_u||_2` if the upper corner is uncapped, and zero otherwise
because the whole box is then capped. This describes the existing reward;
the policy does not receive its logarithm.

## What changes, and what does not

The exact constraint admits more **parameter values** than three separate
spectral caps. It need not admit more **functions**. For this scalar positive
ReLU chain, there is a stronger statement: every function obeying the exact
cap has a representation of the same width obeying all the layerwise caps.

Here is a constructive proof. If `a = 0`, the function is constant. Otherwise
put `v = a.T`, propagate `f_0 = v`, `f_l = W_l f_(l-1)` forward, and
propagate `b_L = 1`, `b_(l-1) = W_l.T b_l` backward. In particular,
`b_0 = f_0` and `f_L = ||a||_2^2`. The matrix of nonnegative path flows

```text
F_l[i,j] = b_l[i] W_l[i,j] f_(l-1)[j]
```

has incoming and outgoing marginals `p_l = f_l * b_l` and
`p_(l-1) = f_(l-1) * b_(l-1)`. The same total `||a||_2^2` is carried through
every layer. On live neurons, let `D_l = diag(sqrt(b_l / f_l))`. Then the
balanced matrix `M_l = D_l W_l D_(l-1)^(-1)` is exactly
`F_l[i,j] / sqrt(p_l[i] p_(l-1)[j])`. Rowwise Cauchy–Schwarz gives
`(M_l x)[i]^2 <= sum_j F_l[i,j] x[j]^2 / p_(l-1)[j]`.
Sum over rows and use the column marginals to obtain `||M_l x||_2 <= ||x||_2`.
Equality holds at `x = sqrt(p_(l-1))`, which the matrix maps to
`sqrt(p_l)` with the same norm. Therefore `||M_l||_2 = 1`.

Rescale hidden weights and biases by these positive diagonal changes of
coordinates; the rescalings commute with ReLU. At the input `D_0 = I`. At the
output undo `D_L = 1 / ||a||_2` and keep the original output bias. The same
function now has hidden norms 1 and final norm `||a||_2`. Zero paths are handled
by deleting input-independent constants (folding them into successor biases)
or units with no output path; the removed coordinates can be zero-padded to
keep widths. The input marginal `p_0 = a.T * a.T` shows that the squared
group sensitivities are exactly the input-group shares of this conserved flow.

For a supported chain the construction also gives the sharp equal-layer
bound. Write `G = ||a||_2 > 0`. A further scalar change of units on hidden
layer `l` by `G^(l/L)` gives every layer norm `G^(1/L)` without changing the
function. Among positive diagonal changes of the hidden units, the minimum
possible product of layer norms is therefore `G`, and the minimum possible
maximum layer norm is `G^(1/L)`. Both lower bounds follow from
submultiplicativity, and the formulas above attain them on the active chain.

Thus the training comparison tests conditioning, regularization and compute,
not added function capacity. The diagonal transformation above is used for
the proof only. The experiment does not apply it during training.

## Experimental update

After each discriminator step, clip all negative weights to zero. Starting from
the last row, compute `a` with matrix-vector products. Then apply

```text
W_L <- W_L / max(1, ||a||_2).
```

For fixed earlier weights and a fixed readout direction, this is the smallest
common decrease needed to restore a unit global gain. It is not the Euclidean
projection onto the joint feasible set. No extra coefficient is introduced.
All existing biases stay ordinary trainable biases. Internal layer norms may
exceed one; the update does not fix hidden-unit scaling. The existing readout
penalty can therefore act differently even with the same coefficient: a
function-preserving change of hidden scales can redistribute its effect. Track
individual layer norms and head norms as well as global gain and reward stats.

## Attribution

- Gouk et al., [Regularisation of Neural Networks by Enforcing Lipschitz
  Continuity](https://arxiv.org/abs/1804.04368): the per-layer post-update
  baseline, specialized to nonnegative weights in MIRA.
- Virmaux and Scaman, [Lipschitz Regularity of Deep Neural Networks](https://arxiv.org/abs/1805.10965):
  the general gated Jacobian product and the difficulty of exact computation
  for unrestricted networks. The equality here is restricted to the stated
  nonnegative scalar ReLU class.
- Gonon et al., [A Path-Norm Toolkit for Modern Networks](https://arxiv.org/abs/2310.01225),
  and Neyshabur et al., [Path-SGD](https://arxiv.org/abs/1506.02617): related
  path-based bounds and invariances, not a citation for this exact ADD reward
  identity or its measured imitation behavior.
- Biswas, [Hidden Synergy: L1 Weight Normalization and 1-Path-Norm
  Regularization](https://arxiv.org/abs/2404.19112) discusses an
  absolute matrix product for a mixed-norm path bound. The comparison here
  concerns the Euclidean norm of a scalar nonnegative chain and its relative
  ADD reward sensitivity.

Prior work on path norms is relevant. This derivation does not establish
priority for the positive-network result. Replacing a conservative constraint
with a tighter one also does not imply better classification or better tracking.

## Backflip pressure test

Both arms trained for 100 updates from initialization (seed 0, 8,192
environments, 26,214,400 transitions per arm) with the same features, network,
output bias, BCE and policy settings. They used matching actor and critic
initialization hashes and identical discriminator probe outputs. The earlier
baseline does not provide full RNG-state hashes. The maintained training code
was not switched to the candidate.

In the phase-zero deterministic evaluation, both checkpoints failed at step
25 in all 256 environments. The post-step errors over those same first 25
steps are reported below. This window covers about 0.83 seconds and does not
test a completed Backflip.

| Error | Per-layer cap | Whole-network path cap | Relative change |
| --- | ---: | ---: | ---: |
| Root position (m) | 0.16338 | 0.15544 | −4.86% |
| Root rotation (rad) | 0.18875 | 0.16935 | −10.28% |
| Body position (m) | 0.14465 | 0.13692 | −5.35% |
| Body rotation (rad) | 0.62881 | 0.60679 | −3.50% |
| Joint velocity (rad/s) | 2.20648 | 2.17504 | −1.42% |
| Root linear velocity (m/s) | 0.52144 | 0.49864 | −4.37% |
| Root angular velocity (rad/s) | 0.61068 | 0.60317 | −1.23% |

Median training-update time over updates 2–99 was 4.299 s versus 4.016 s
(6.6% less in this run). Independent float64 checks of the final weight
files gave a path gain of 0.998780 versus 1.000000. Candidate layer norms
were 2.71724, 2.54384 and 0.14499. Both final rollout audits had no numerically
zero rewards or upper clipping in their samples. Training reward standard
deviation at update 100 was 0.17514 versus 0.17710; BCE was 0.50757 versus
0.52763. Better early tracking on one seed does not establish a better final
policy or any advantage on other motions. See
[`comparison.json`](../output/diagnostics/mira_exact_path_pair_backflip/comparison.json)
for the fixed-window errors and the separate online diagnostics.

## Architecture-only compatibility check

The frozen experiment arms match the maintained sources for the discriminator
forward, the input normalizer, the optimizer hook, positive and negative BCE,
the readout regularizer, and the ADD reward and its aggregation. The candidate
changes only the permissible weight set and how the existing optimizer hook
repairs out-of-set weights. The hook remains outside backpropagation.

The 100-update baseline checkpoint was feasible under the candidate rule
without any changes to its tensors (path gain 0.99877993). Applying that rule
once and comparing on 16,384 real paired replay samples and the zero-error
positive gave identical FP32 logits, ADD rewards and discriminator loss
including its readout penalty. This verifies an unchanged starting function,
not equality between the two training trajectories.

On those same paired replay observations, each independently trained
100-update model had a similar distribution of analytical log-reward
gradient norms. Using each model's saved normalizer, the medians were
0.92927 (per layer) and 0.92715 (path cap); the corresponding 95th percentiles
were 0.98089 and 0.98104. On the coordinatewise envelope of the sampled
inputs, the exact constants above were 0.99849 and 1.00000, matched by
autograd near the upper corners. The box contains the replay sample; its
upper corner need not correspond to a reachable humanoid state.

The [300-update continuation driver](../PAPER/tools/benchmark/continue_mira_exact_path_backflip.py)
extends each arm from its own 100-update checkpoint in a separate directory.
It restores model, optimizers and replay and restarts the physics simulation
at the same update in both arms. It compares any completed checkpoints with
the same phase-zero deterministic evaluation and retains the common frame
prefix if either policy terminates early. This is a longer controlled check,
not a replacement of the maintained MIRA network.
