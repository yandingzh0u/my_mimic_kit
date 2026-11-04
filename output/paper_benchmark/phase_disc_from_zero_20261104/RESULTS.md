# 从零训练：3000 轮附近与 3100 轮结果

3100 轮训练和全部独立评估已经完成。未加载预训练模型或恢复 checkpoint，seed 0，8192 环境，每轮 32 步。

3000 轮附近的模型有局部收益：随机起点 RootPos 降低 16.97%，从动作开头测试降低 1.37%；其他指标有升有降，两种评估都满足各项 ERR 最多恶化 10% 的验收线。3100 轮对比中 RootRot 增加 14.02%，未满足验收线，不能把本次结果表述为稳定整体优势。

下表均为新方案相对各自轮次历史 baseline 的变化，负值表示误差下降。每种起点各使用 8192 个严格配对 episode，最多 299 步、确定性均值动作。

| ERR | 3000 附近：随机起点 | 3000 附近：从头起点 | 3100：随机起点 |
|---|---:|---:|---:|
| RootPos | -16.97% | -1.37% | -7.91% |
| RootRot | +2.68% | +4.56% | +14.02% |
| BodyPos | -0.87% | -1.84% | +4.46% |
| BodyRot | +2.44% | +2.94% | +7.50% |
| DofVel | +0.92% | -2.14% | +1.38% |
| RootVel | -0.25% | +1.58% | -2.11% |
| RootAngVel | +1.66% | -1.87% | +1.92% |

3000 附近随机起点自然失败数：baseline 24/8192，新方案 25/8192；从头起点两者都没有失败，完整动作 completion 都为 100%。3100 随机起点自然失败数：24 → 23。

轮次口径：新方案保存的 model_0000003000.pt 是零起始编号，实际完成 3001 次更新（786694144 transitions）；历史 baseline 完成 3000 次更新（786432000 transitions），预算相差 0.0333%。不是精确同预算的新两组训练。3100 对比的两者预算一致。

历史 baseline 的配置文件与本次配置逐字节一致，已重新评估 3000 baseline；本次没有重跑其训练。两种起点评估都核对 episode ID、起点 phase、motion ID 和预期步数。一条 seed 和一个动作不支持跨 seed、跨任务优势。

最终 checkpoint：next_iter=3100，sample_count=812646400，resume_count=0，历史 iteration=3099，难度定义 bce_segment_phase_history_v1。训练源码和模拟输入 hash 核验通过。

- [3000 详细结果](/home/y/my_project/MimicKit_phase_disc_causal/output/research/phase_disc_from_zero_20261104/comparison_3000/REPORT.md)
- [3000 比较数据](/home/y/my_project/MimicKit_phase_disc_causal/output/research/phase_disc_from_zero_20261104/comparison_3000/comparisons.json)
- [3100 详细结果](/home/y/my_project/MimicKit_phase_disc_causal/output/research/phase_disc_from_zero_20261104/FINAL_REPORT.md)
- [3100 比较数据](/home/y/my_project/MimicKit_phase_disc_causal/output/research/phase_disc_from_zero_20261104/comparisons.json)
- [完整训练状态](/home/y/my_project/MimicKit_phase_disc_causal/output/research/phase_disc_from_zero_20261104/status.json)
- [3000 评估状态](/home/y/my_project/MimicKit_phase_disc_causal/output/research/phase_disc_from_zero_20261104/comparison_3000/status.json)
