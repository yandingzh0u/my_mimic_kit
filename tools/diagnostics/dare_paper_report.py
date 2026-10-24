"""Export reproducible numerical gates, a figure, and a LaTeX research note."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dare_paper_stress import ROOT, Disc, load_disc, radial_probe
from dare_ray_candidate import RayDisc, optimal_intercept


def numerical_gates(source):
    result = dict(constant_distance_roots=[], input_scale_checks=[])
    for value in [0., 1e-10, 1., 1e3, 1e6]:
        distance = torch.full((10,), value, device="cuda:0")
        actual = optimal_intercept(distance)
        torch.testing.assert_close(actual, distance[0] / 2)
        result["constant_distance_roots"].append(dict(distance=value, intercept=float(actual)))
    model = RayDisc(source).eval().requires_grad_(False)
    for value in [0., 1e-6, 1e-3, 1., 1e3, 1e6]:
        x = (value * torch.randn(32, 172, device="cuda:0")).requires_grad_(True)
        y = model(x)
        grad, = torch.autograd.grad(y.sum(), x)
        finite = bool(torch.isfinite(y).all() and torch.isfinite(grad).all())
        assert finite
        result["input_scale_checks"].append(dict(scale=value, finite_outputs_and_gradients=finite))
    axes = torch.cat([torch.eye(172, device="cuda:0"), -torch.eye(172, device="cuda:0")])
    distances = model.distance(axes)
    assert (distances > 0).all()
    result["signed_axes"] = dict(count=344, min_distance=float(distances.min()), max_distance=float(distances.max()))
    # A 1-Lipschitz biased CPL can fold a ray back onto the origin. An
    # anchored norm head alone does not imply radial monotonicity.
    points = torch.tensor([0., .5, 1., 1.5, 2.])
    folded = points - 2 * F.relu(points - 1)
    result["biased_cpl_counterexample"] = dict(inputs=points.tolist(), distance=folded.abs().tolist())
    return result


def initialization_gates(state):
    result = []
    for seed in range(8):
        torch.manual_seed(seed)
        original = Disc(state, load_weights=False).cuda().eval().requires_grad_(False)
        ray = RayDisc(original).eval().requires_grad_(False)
        x = torch.randn(512, 172, device="cuda:0")
        entry = dict(seed=seed, baseline=radial_probe(original, x), ray=radial_probe(ray, x))
        assert entry["ray"]["nonmonotone_ray_fraction"] == 0
        result.append(entry)
    return result


def export_figure(out, live, rows):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.25), constrained_layout=True)
    iterations = [r["Iteration"] for r in rows]
    ax = axes[0]
    ax.plot(iterations, [r["Test_Episode_Length_Start0"] for r in rows], color="#2878a0", lw=2)
    ax.set(xlabel="Training iteration", ylabel="Start-zero episode length (steps)",
           title="Archived policy training")
    ax.axvline(1700, ls=":", color="gray", lw=1)
    ax = axes[1]
    checkpoints = list(live["checkpoints"])
    locations = list(range(len(checkpoints)))
    width = .35
    original = [100*live["checkpoints"][k]["baseline"]["heldout_radial"]["nonmonotone_ray_fraction"] for k in checkpoints]
    ray = [100*live["checkpoints"][k]["ray"]["heldout_radial"]["nonmonotone_ray_fraction"] for k in checkpoints]
    ax.bar([x-width/2 for x in locations], original, width, color="#2878a0", label="Original head")
    ax.bar([x+width/2 for x in locations], ray, width, color="#d88426", label="Proposed head")
    ax.set(xticks=locations, xticklabels=checkpoints, xlabel="Source checkpoint iteration",
           ylabel="Nonmonotone rays (%)", ylim=(0, 100),
           title="Fresh simulator residuals")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    original = [live["checkpoints"][k]["baseline"]["heldout_balanced_bce"] for k in checkpoints]
    ray = [live["checkpoints"][k]["ray"]["heldout_balanced_bce"] for k in checkpoints]
    ax.bar([x-width/2 for x in locations], original, width, color="#2878a0")
    ax.bar([x+width/2 for x in locations], ray, width, color="#d88426")
    ax.set(xticks=locations, xticklabels=checkpoints, xlabel="Source checkpoint iteration",
           ylabel="Balanced BCE", title="Frozen-feature diagnostic")
    fig.savefig(out / "dare_ray_stress.pdf")
    fig.savefig(out / "dare_ray_stress.png", dpi=190)
    plt.close(fig)


def write_tex(out, live, stress):
    table = "\n".join(
        f"{k} & {100*v['baseline']['heldout_radial']['nonmonotone_ray_fraction']:.2f} & "
        f"{100*v['ray']['heldout_radial']['nonmonotone_ray_fraction']:.2f} & "
        f"{v['baseline']['heldout_balanced_bce']:.4f} & {v['ray']['heldout_balanced_bce']:.4f} \\\\"
        for k, v in live["checkpoints"].items())
    template = r"""\documentclass[10pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,hyperref}
\title{A Ray Consistent Extension of Single Anchor Adversarial Imitation}
\date{Research note, 24 October 2026}
\begin{document}
\maketitle
\noindent This note specifies an isolated candidate extension to the archived
DARE baseline. The extension introduces no loss term or manually selected loss
weight. Its geometric guarantees and diagnostic results do not establish improved
closed-loop learning performance; the candidate has not undergone a complete PPO
training comparison.

\section{Existing baseline and motivation}
DARE classifies the zero tracking residual against current-policy and replay
residuals using balanced binary cross entropy (BCE). The discriminator has seven
semantic encoders, a shared fusion layer, seven convex potential blocks, and a
scalar linear readout. The reference run uses a 172-dimensional residual. At
iteration 2999 the recorded start-zero evaluation length is 299 steps, the
training-window completion fraction is 0.99149, and the recorded body-position
error is 0.01999 m. These statistics have different evaluation protocols and
must not be equated with an independently measured deployment success rate.

Lipschitz control bounds score variation but does not imply that decreasing all
residuals proportionally improves the score. In particular, a biased scalar
CPL $h(x)=x-2\operatorname{ReLU}(x-1)$ is 1-Lipschitz but satisfies
$|h(1)|=1$ and $|h(2)|=0$. Merely replacing a scalar readout with an
anchor-distance head therefore does not ensure monotonicity along rays.

\section{Homogeneous distance parameterization}
Let $z_i=e_i/\max(s_i,10^{-4})$ denote the existing coordinatewise mean-absolute
normalization. Retain the existing semantic encoders and CPL trunk, remove all
hidden biases, and replace the scalar linear head with a Euclidean feature norm.
ReLU, linear maps without biases, and bias-free CPL blocks are positively
homogeneous. Their composition $H_\theta$ therefore satisfies
$H_\theta(az)=aH_\theta(z)$ for $a\geq0$.
Define
\begin{equation}
 d_\theta(z)=\|H_\theta(z)\|_2,\qquad
 f_\theta(z)=b-d_\theta(z),\qquad
 r_\theta(z)=2\min\{\operatorname{softplus}(f_\theta(z)),\log 10^4\}.
 \label{eq:head}
\end{equation}
The reward scale and numerical cap are inherited from the archived configuration.
This parameterization changes the discriminator function class, while retaining
the original balanced BCE objective. It removes hidden biases and the scalar
linear head; it adds no learned temperature or regularization coefficient.

\paragraph{Ray consistency.}
For fixed weights, intercept, and normalizer, the anchor attains a global
maximum of the score, and $f_\theta(az)=b-a d_\theta(z)$. Consequently
$r_\theta(a_1z)\geq r_\theta(a_2z)$ whenever $0\leq a_1\leq a_2$.
Where the reward is uncapped, its derivative along a ray is
$-2d_\theta(z)\sigma(b-a d_\theta(z))\leq0$.
This holds at initialization and is independent of the accuracy of spectral
normalization. Strict decrease additionally requires $d_\theta(z)>0$ and an
uncapped reward. The result does not imply coordinatewise Pareto monotonicity,
physical reachability of scaled residuals, an injective representation, or
monotonic scores across changing network snapshots.

\paragraph{Conditional Lipschitz bound.}
If each actual linear operator has spectral norm at most one, the group-separable
frontend, fusion layer, and CPL blocks are nonexpansive. The reverse triangle
inequality then makes $d_\theta$ and $f_\theta$ 1-Lipschitz and $r_\theta$
2-Lipschitz in normalized residual coordinates. This conditional statement
must be distinguished from the current one-power-iteration spectral-normalization
implementation: measured individual operator norms reach approximately 1.010
over the inspected checkpoints. Ray consistency itself does not need this bound.

\section{Automatic intercept without a new learning rate}
For a negative minibatch with distances $d_1,\ldots,d_n$, keep the existing loss
$L(b)=\tfrac12\operatorname{softplus}(-b)+
\tfrac1{2n}\sum_i\operatorname{softplus}(b-d_i)$.
Eliminate its scalar intercept subproblem by solving
\begin{equation}
 \sigma(-b^*)=\frac1n\sum_{i=1}^n\sigma(b^*-d_i),\qquad
 \tfrac12\min_i d_i\leq b^*\leq\tfrac12\max_i d_i.
 \label{eq:calibration}
\end{equation}
The objective is strictly convex for finite distances, so this solution is
unique. If every distance equals $d$, then $b^*=d/2$; if all distances vanish,
$b^*=0$. The prototype solves this one-dimensional problem using float64
log-domain bisection. The iteration limit is a numerical precision setting,
not a task-specific trade-off coefficient. The intercept is stored as model
state, not optimized with a separate learning rate.

At each discriminator step, concatenate current and replay residuals as in the
baseline, compute the distances, solve for $b^*$ without autograd, and apply one
ordinary update to $H_\theta$ using the same BCE. The envelope theorem justifies
holding the optimal intercept fixed in the partial gradient with respect to
$\theta$. During policy rollouts keep both the intercept and normalizer fixed.
Actor/critic losses, phase balancing, replay capacity, optimizer settings, and
the original reward transform remain part of the existing training protocol.

\section{Executed diagnostic gates}
Four archived policies generated fresh stochastic rollouts with random reference
starts: 512 environments and 64 steps per checkpoint, with explicit resets after
termination. This yields 32,768 physical residual observations per checkpoint.
Ray tests use 2,048 selected residuals per checkpoint and 12 scales from zero to
16. Scaling an observed residual is an input-space diagnostic; the scaled
vectors need not be physically reachable. A separate slice calibrates the
candidate intercept. The candidate reuses source features after removing their
biases and receives no policy updates in this test. The data split is by samples,
not independent episodes, so BCE values are diagnostic rather than a statistical
generalization estimate.

\begin{center}
\begin{tabular}{rrrrr}
\toprule
Checkpoint & Original bad rays (\%) & Proposed bad rays (\%) & Original BCE & Proposed BCE\\
\midrule
@@TABLE@@
\bottomrule
\end{tabular}
\end{center}
The mature baseline already passes the sampled radial tests. The observed
benefit is an architectural guarantee from the start of training. Frozen-feature
BCE is slightly worse at some mature checkpoints; it must not be reported as a
uniform performance improvement.

Additional gates cover six checkpoints on shared replay, random directions,
fixed-radius adversarial searches, eight fresh network initializations,
344 signed coordinate axes, all-zero inputs, and scales from $10^{-6}$ to
$10^6$. Input and feature gradients remain finite at the tested scales.
Constant-distance intercepts agree with $d/2$ up to distance $10^6$.
Three replay fitting runs vary minibatch order while sharing the same archived
starting weights; these are not independent RL training seeds.

\begin{figure}[h]
 \centering\includegraphics[width=\linewidth]{dare_ray_stress.pdf}
 \caption{Archived policy learning and frozen-feature geometry diagnostics.
 The proposed head has not undergone a full PPO comparison.}
\end{figure}

\section{Required paper experiments and reporting scope}
Compare the unchanged baseline with a norm head retaining biases, a homogeneous
norm head with an SGD intercept, and the full automatically calibrated variant.
Reuse the archived optimizer and PPO settings without tuning. An initial complete
comparison should cover climb, get-up, and backflip with at least three independent
training seeds, followed by all target motions for the final transfer claim.
Report fixed-start completion, physical tracking errors, learning curves,
environment samples, and wall time. Keep evaluation and reset protocols identical.
Log the radial failure rate through training, especially before iteration 100.
Only a complete RL comparison can establish improved sample efficiency or final
control performance.

There are two baseline reporting issues. First, the shared-storage logging
aggregation aliases \texttt{disc\_loss} and its detached classification loss.
A constant true loss of one is logged as 0.0615234375 across 32 steps. Absolute
loss values should be rechecked before they appear in a paper. Second, the
existing $10^{-4}$ normalization floor prevents unconditional invariance to
arbitrary changes of units. Scaling both raw inputs and fitted statistics by
$10^{-3}$ changes the final discriminator logit by 0.1826 on average in the
shared-replay audit. The proposed head inherits this normalizer and does not
resolve that limitation.

The supported claim is \emph{no additional reward weights or regularization
coefficients, with a task-independent training configuration}. This is distinct
from eliminating PPO, architecture, or optimizer hyperparameters. Existing CPL
architecture and bias-free one-class representations require appropriate prior
work attribution; no claim of first use is made here.

\begin{thebibliography}{9}
\bibitem{add} Z. Zhang et al. Physics-Based Motion Imitation with Adversarial
Differential Discriminators. SIGGRAPH Asia, 2025.
\url{https://add-moo.github.io/}
\bibitem{cpl} L. Meunier et al. A Dynamical System Perspective for Lipschitz
Neural Networks. ICML, 2022.
\url{https://proceedings.mlr.press/v162/meunier22a.html}
\bibitem{svdd} L. Ruff et al. Deep One-Class Classification. ICML, 2018.
\url{https://proceedings.mlr.press/v80/ruff18a.html}
\bibitem{sn} PyTorch. Spectral normalization documentation.
\url{https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.spectral_norm.html}
\end{thebibliography}
\end{document}
"""
    (out / "method_note.tex").write_text(template.replace("@@TABLE@@", table))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    archive = ROOT / "output/paper_benchmark/dare_10layer_cpl_phasebalanced_climb_3000_8192_seed0_headless_20261012"
    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    source, state, _ = load_disc(archive / "model.pt", "cuda:0")
    disc_state = {k.removeprefix("_model."): v for k, v in state.items() if k.startswith("_model._disc_")}
    report = dict(source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  torch_version=torch.__version__, cuda_device=torch.cuda.get_device_name(),
                  numerical=numerical_gates(source),
                  fresh_initializations=initialization_gates(disc_state),
                  source_hashes={})
    for relative in ["mimickit/learning/dare_agent.py", "mimickit/learning/dare_model.py",
                     "mimickit/learning/diff_normalizer.py", "mimickit/learning/ppo_agent.py"]:
        report["source_hashes"][relative] = hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    (out / "numerical").mkdir(exist_ok=True)
    (out / "numerical/summary.json").write_text(json.dumps(report, indent=2)+"\n")
    live = json.loads((out / "live/summary.json").read_text())
    stress = json.loads((out / "baseline_stress/summary.json").read_text())
    rows = [json.loads(line) for line in (archive / "train_metrics.jsonl").read_text().splitlines()]
    export_figure(out, live, rows)
    write_tex(out, live, stress)
    print("Exported numerical gates, LaTeX note, and PDF/PNG figure to", out)


if __name__ == "__main__":
    main()
