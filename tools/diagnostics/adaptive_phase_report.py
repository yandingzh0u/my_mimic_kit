"""Build a concise, reproducible research report and standalone figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    stress = json.loads((args.out/"stress.json").read_text())
    live = json.loads((args.out/"live.json").read_text())
    methods = ["historical_8_wr", "balanced_8", "adaptive_balanced"]
    checkpoints = list(live["checkpoints"])
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.subplots_adjust(bottom=.28, top=.90, left=.11, right=.98)
    x = np.arange(len(checkpoints))
    labels = ["8 strata, replacement", "8 strata, balanced reuse", "Adaptive, balanced reuse"]
    for j, (method, label) in enumerate(zip(methods, labels)):
        ratios, lo, hi = [], [], []
        for ckpt in checkpoints:
            row = live["checkpoints"][ckpt]["methods"][method]
            ratio = row["gradient_mse_mean"] / live["checkpoints"][ckpt]["methods"][methods[0]]["gradient_mse_mean"]
            ratios.append(ratio)
            ci = row["mse_ratio_bootstrap_95ci_vs_8_wr"]
            lo.append(max(0, ratio-ci[0]))
            hi.append(max(0, ci[1]-ratio))
        ax.bar(x+(j-1)*.24, ratios, width=.23, label=label, yerr=[lo, hi], capsize=3)
    ax.set(xticks=x, xticklabels=checkpoints, xlabel="Archived training iteration",
           ylabel="Gradient sampling MSE / original 8-stratum MSE", ylim=(0, 1.35),
           title="Frozen-policy PPO audit on physical Climb rollouts")
    ax.axhline(1, color="black", linewidth=.7, linestyle="--")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.text(.02, .02, "512 environments; one-motion warm-up; 128 draws per method.\n"
             "Each method is measured against its own full-rollout gradient; adaptive strata change the target.\n"
             "Intervals: bootstrap uncertainty from minibatch draws, not across-seed training uncertainty.",
             fontsize=8)
    fig.savefig(args.out/"gradient_audit.png", dpi=180, bbox_inches="tight")
    fig.savefig(args.out/"gradient_audit.pdf", bbox_inches="tight")
    lines = [
        "控制分辨率自适应相位采样：研究结果（2026-10-24）",
        "",
        "建议：采用动作控制时间步确定阶段分辨率，batch 预算限制最大层数；",
        "每个非空层都有配额，层内无放回或均匀复用。另用参考初始化轮转处理采集覆盖。",
        "这是独立原型，未接入 PPO 默认训练路径，未改判别器、奖励或损失项。",
        "",
        "定义与权重",
        "F=max(1,ceil(T/dt)); K=min(F,B_min,N).",
        "native_i=min(floor(F*phase_i),F-1); bin_i=floor(K*native_i/F).",
        "合并边界必须落在 native control cell 边界，不能拆开一个控制格。",
        "c_k=ceil((k+1)F/K)-ceil(kF/K); A={k:N_k>0}; rho_k=c_k/sum_{a in A} c_a.",
        "m_k=floor(B/|A|) 或 ceil(B/|A|)，余数配额按轮转分配。",
        "L_hat=sum_{k in A} rho_k/m_k * sum_j loss_PPO(z_kj).",
        "每条抽样权重 rho_k/m_k 替换旧 phase_weight；不允许两套权重相乘。",
        "advantage 的加权均值和标准差也必须由同一个 rollout 目标 rho_k/N_k 计算。",
        "F=K 且所有层存在时 rho_k=1/F。合并后不能一律用 1/K：层宽可能不同。",
        "8 层与细控制格的经验目标不同，不能宣称完全等价；这也不是最优分箱定理。",
        "",
        "重复与方差",
        "令 m_k=q_k*N_k+r_k。每次独立生成层内排列，完整使用 q_k 次，再使用前 r_k 条。",
        "每条数据出现 q_k 或 q_k+1 次；不是跨整个 epoch 每条只用一次的失败方案。",
        "在固定 rollout、固定参数和配额下，以 population covariance Sigma_k 定义：",
        "Cov(gbar_bal)=r_k(N_k-r_k)/(m_k^2(N_k-1))*Sigma_k（N_k>1）。",
        "有放回抽样 Cov(gbar_wr)=Sigma_k/m_k；因此相同比较目标与配额下前者不更大。",
        "N_k=1 时层内抽样方差为零，但没有增加独立训练数据。",
        "这不证明总体 RL 梯度无偏、PPO 收敛、更低最终 tracking error 或更高成功率。",
        "",
        "覆盖保证的精确范围",
        "1. K<=B_min，所有非空 eligible bin 每个 minibatch 至少一条。",
        "2. 空层和缺失 native control cell 分别报告，不通过合并或 nan_to_num 隐藏。",
        "3. 参考重置用一个互质步长的置换循环：每连续 F 次该 motion 的重置请求覆盖每个控制格一次。",
        "4. 初始状态需要被记录进 rollout；只有首次集合中所有格被采集时，才能宣称 buffer 完整覆盖。",
        "5. F 比有限采样预算更大时，不能承诺每次 rollout 含所有控制格；保证按重置请求计。",
        "6. 轮转初始化具有依赖性，不能宣称与 IID RSI 在条件分布上等价。",
        "7. 实机原型针对单条 motion；多 motion 需分别维护轮转状态并保留 motion selection weights。",
        "",
        "压力测试",
        json.dumps(stress["fuzz"], ensure_ascii=False),
        json.dumps(stress["reset_rotor"], ensure_ascii=False),
        "另外：6 个配额设置各 20000 次 Monte Carlo，经验均值/方差符合解析式；",
        "GPU 使用 262144 个样本，测试 16384/32768 batch 和最多十亿控制格的元数据。",
        "长动作不会分配十亿大小的数组。极端测试中出现空层时，如实报告，不记为覆盖成功。",
        "38 项新增单元测试通过；不是对全仓库测试通过的声明。",
        "真实时间戳还检查乘除法的浮点边界；若 dtype 无法表示所要求的控制格，明确报错。",
        "",
        "物理 rollout 的真实 PPO 梯度审计",
        "512 环境，三个检查点；每个检查点连续运行 299 步后采集 32 步。",
        "使用仓库真实奖励、带终止处理的 TD-lambda、随机动作掩码、advantage clip 及完整 actor loss。",
        "冻结所有 normalizer 和模型；每种采样器抽样 128 次，合计 1152 次 minibatch 梯度估计。",
        "actor batch=2048，critic budget=1024；它们按环境数缩放，非 8192 环境的完整训练。",
        "var_ratio_vs_8_wr：数值越低越好；adaptive 的分母是旧目标，分子是自适应自身目标的方差。",
    ]
    for ckpt, row in live["checkpoints"].items():
        lines.append(f"checkpoint {ckpt}:")
        for method in methods:
            d = row["methods"][method]
            lines.append(f"  {method}: variance={d['gradient_variance']:.6f}, ratio={d['variance_ratio_vs_8_wr']:.6f}, min_pool={d['smallest_pool']}, "
                         f"relative_noise={d['gradient_variance']/d['full_gradient_norm']**2:.6f}")
        lines.append(f"  adaptive target: cosine_vs_8={row['adaptive_full_target_cosine_vs_8']:.6f}, "
                     f"relative_gradient_change={row['adaptive_full_target_relative_change_vs_8']:.6f}")
        lines.append(f"  reset coverage: {json.dumps(row['coverage_reset'], ensure_ascii=False)}")
    lines += ["", "判断与论文使用范围",
        "均匀复用是当前最可靠的改进：同目标的解析降方差保证和实测都成立。",
        "自适应控制格解决固定 8 的任意性，但额外降方差不在所有检查点成立，且目标发生实质变化。",
        "可以作为方法小节与机制证据；完整训练结果出来前，不得写成已降低七项 tracking error。",
        "最有信息量的训练消融：旧 8 层 WR；8 层均匀复用；自适应均匀复用；后者再加覆盖初始化。",
        "除这些变化外保持网络、损失、优化器、训练步数、初始化、评估协议一致，至少比较多个 seed。",
        "优先记录所有七项 err、phase0/random 起点成功与长度、空格占比、最小池大小、重复率及训练耗时。",
        "通用分层 SGD 的成熟基础：Zhao & Zhang (2014), https://arxiv.org/abs/1405.3080 .",
        "不把通用分层采样、有限总体校正或参考状态随机初始化宣称为首次提出。",
    ]
    (args.out/"report.txt").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
