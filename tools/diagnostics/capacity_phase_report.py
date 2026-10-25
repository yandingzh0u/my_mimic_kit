"""Summarize the same-target capacity audit and write its finite-data lemma."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    stress = json.loads((args.out/"capacity_stress.json").read_text())
    live = json.loads((args.out/"live.json").read_text())
    lines = [
        "容量约束配额重分配：同目标审计（2026-10-25）",
        "",
        "前版的明确不足：控制时间格可能只有 1~2 条数据，等配额仍分配 6~7 条。",
        "均匀复用处理了重复次数的不均衡，但没有把冗余配额转移到尚未使用的数据。",
        "K=min(F,B_min,N) 仅是时长与预算自适应，不是统计最优分箱定理。",
        "",
        "新机制",
        "先按旧规则得到等配额 m_k，再把超出 N_k 的部分退回。",
        "用容量封顶的水位分配，把退回的配额分到其他层，同时保留所有原先有用的配额。",
        "当 K_present<=B<=sum N_k 时：sum m'_k=B，1<=m'_k<=N_k，m'_k>=min(m_k,N_k)。",
        "层内每个 minibatch 独立无放回，不对整个 epoch 施加一次性覆盖约束。",
        "loss 权重仍为 rho_k/m'_k；phase target、full weights、advantage normalization 完全相同。",
        "没有引入梯度方差估计、阈值、密度筛选或新的损失系数。",
        "这不是 Neyman 全局最优配额；它只利用精确的样本容量消除确定存在的浪费。",
        "",
        "固定 rollout、参数、目标时的协方差比较",
        "令 Sigma_k=(1/N_k) sum_i (g_i-gbar_k)(g_i-gbar_k)^T，r_k=m_k mod N_k。",
        "旧版均匀复用的层均值协方差为 r_k(N_k-r_k)/(m_k^2(N_k-1))*Sigma_k。",
        "若 m_k<=N_k，新配额不减少有用样本数，无放回均值的协方差随配额增加而下降。",
        "若 m_k>N_k，新配额正好取满该层 N_k 条，层内采样协方差归零。",
        "故新估计器的整体协方差在半正定序下不高于旧版，且估计相同的有限 rollout 目标。",
        "N_k=1 时旧新版该层均值都确定；收益来自将多余计算转移给其他层。",
        "B>sum N_k 时全局重复不可避免：包含全部数据，并按每条数据的实际重复次数修正权重，",
        "使每条数据的总权重仍为 rho_k/N_k，此时等于完整 rollout 的目标梯度。",
        "以上不证明最终 tracking error、更高成功率或 PPO 全局收敛。",
        "",
        "压力测试",
        json.dumps(stress["allocations"], ensure_ascii=False),
        json.dumps(stress["sampling"], ensure_ascii=False),
        "57 项针对旧原型与新扩展的单元测试通过，不是全仓库测试声明。",
        "合成梯度的 Monte Carlo 验证：",
        json.dumps(stress["monte_carlo"], ensure_ascii=False),
        "GPU 的极端容量不均衡测试：",
        json.dumps(stress["gpu"], ensure_ascii=False),
        "",
        "真实 Climb PPO 梯度",
        f"训练权重来自 seed 0 的成功 baseline；审计随机种子={live['seed']}。",
        f"环境数={live['num_envs']}；先连续运行 {live['warmup_steps']} 步，再取 {live['steps']} 步。",
        f"每个检查点每种方法估计 {live['resamples']} 个 minibatch 梯度；无模型或 optimizer 更新。",
        "均匀复用与容量重分配使用配对随机排列；其目标权重和 advantage tensor 逐元素相同。",
        "CI 仅反映该固定 rollout 上的 minibatch 抽样误差，不代表跨训练种子的不确定性。",
    ]
    for checkpoint, row in live["checkpoints"].items():
        old = row["methods"]["adaptive_balanced"]
        new = row["methods"]["adaptive_capacity"]
        reduction = 100*(1-new["gradient_variance"]/old["gradient_variance"])
        paired = row["capacity_same_target"]
        lines.append(f"checkpoint {checkpoint}: old variance={old['gradient_variance']:.6f}, "
                     f"new variance={new['gradient_variance']:.6f}, reduction={reduction:.3f}%")
        lines.append(f"  actor batch={row['actor_batch']}, min unique indices: "
                     f"{old['minimum_unique_in_batch']} -> {new['minimum_unique_in_batch']}")
        lines.append(f"  paired MSE ratio 95% interval={paired['paired_bootstrap_95ci_ratio']}")
        lines.append(f"  full target difference={paired['full_target_difference']}")
        lines.append(f"  natural rollout coverage={json.dumps(new['coverage'], ensure_ascii=False)}")
        lines.append(f"  separate coverage-initialized rollout={json.dumps(row['coverage_reset'], ensure_ascii=False)}")
    lines += [
        "",
        "明确限制",
        "末期自然 rollout 缺失两个控制格；采样器没有填补或隐瞒它们。",
        "独立的覆盖初始化 rollout 包含全部 299 个控制格；它不是同一个 buffer 的事后补齐。",
        "覆盖重置保证按 reset requests 计，不保证所有短 rollout 都齐全。",
        "该原型没有接入生产训练默认路径，也未开始新的 3000 轮训练。",
        "当前只审计了 frozen-policy PPO actor 梯度；critic 梯度、后续 PPO epoch 和多训练 seed 还需验证。",
        "F=299 是控制分辨率，容量重分配不是统计最优分箱，也没有证明语义事件分箱会更好。",
        "按梯度方差分配预算仍有潜在收益，但其估计成本、误差与实际收益尚未验证，不能并入本结果。",
    ]
    (args.out/"report.txt").write_text("\n".join(lines)+"\n")
    tex = r"""% Same-target finite-rollout result; not a tracking-error or convergence theorem.
\paragraph{Capacity-constrained phase allocation.}
Let $\rho_k$ denote the fixed temporal mass of each occupied phase stratum,
$N_k$ its eligible population size and $m_k$ its original equal quota.
If $|\mathcal A|\leq B\leq\sum_kN_k$, we redistribute redundant quotas to obtain
\begin{equation}
 \sum_km'_k=B,\qquad 1\leq m'_k\leq N_k,\qquad
 m'_k\geq\min(m_k,N_k).
\end{equation}
All strata with $m_k>N_k$ receive exactly $N_k$ draws. Remaining slots
are assigned by capacity-capped water filling, retaining all useful original
remainder slots. We draw independently without replacement inside each
stratum at every minibatch, and use
\begin{equation}
 \widehat L_{\mathrm{cap}}=\sum_{k\in\mathcal A}
 \frac{\rho_k}{m'_k}\sum_{i\in S_k}\ell_{\mathrm{PPO}}(z_i).
\end{equation}
The full-rollout target, including advantage normalization, is unchanged.
No additional loss term or tunable coefficient is introduced.

\paragraph{Finite-rollout covariance dominance.}
Fix the rollout, network parameters, target masses and original quotas.
Let $\Sigma_k$ be the population covariance of per-transition gradients
with denominator $N_k$. For $N_k>1$, the original balanced-reuse mean has
covariance $r_k(N_k-r_k)\Sigma_k/[m_k^2(N_k-1)]$, where
$r_k=m_k\bmod N_k$. For a stratum with $m_k\leq N_k$, increasing its quota
without replacement cannot increase its covariance. For $m_k>N_k$,
the capacity-constrained estimator includes the entire population and has
zero within-stratum sampling covariance. Independent within-stratum
permutations therefore give
\[
 \operatorname{Cov}(\widehat g_{\mathrm{cap}})
 \preceq\operatorname{Cov}(\widehat g_{\mathrm{balanced}}).
\]
Both estimators target the same finite-rollout surrogate gradient.
The $N_k=1$ case is deterministic in both estimators.
This is a sampling result, not a claim about final tracking error,
global optimal allocation or end-to-end PPO convergence.
"""
    # Preserve the manuscript once it has been edited after its initial draft.
    draft_path = args.out/"capacity_method.tex"
    if not draft_path.exists():
        draft_path.write_text(tex)


if __name__ == "__main__":
    main()
