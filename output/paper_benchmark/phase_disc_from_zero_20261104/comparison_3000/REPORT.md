# 3000 轮附近：从零训练采样方案与历史 baseline

Candidate file label 3000 is zero-based: 3001 completed updates versus 3000 for the historical baseline. Candidate has 0.0333% more transitions; this is not an exactly budget-matched fresh two-arm training experiment.

两者配置文件逐字节一致，分别使用 8192 个配对起点、确定性均值动作、299 步评估。

## random

| ERR | baseline | BCE 片段阶段采样 | 相对变化 |
|---|---:|---:|---:|
| RootPos | 0.052702 | 0.043756 | -16.97% |
| RootRot | 0.029906 | 0.030708 | +2.68% |
| BodyPos | 0.017124 | 0.016975 | -0.87% |
| BodyRot | 0.064177 | 0.065742 | +2.44% |
| DofVel | 0.353943 | 0.357197 | +0.92% |
| RootVel | 0.081170 | 0.080965 | -0.25% |
| RootAngVel | 0.186268 | 0.189370 | +1.66% |

自然终止计数：{"baseline": {"fail": 24, "null": 0, "succ": 8168, "time": 0}, "candidate": {"fail": 25, "null": 0, "succ": 8167, "time": 0}}

七项 ERR 中降低项数：3/7。
## phase0

| ERR | baseline | BCE 片段阶段采样 | 相对变化 |
|---|---:|---:|---:|
| RootPos | 0.051423 | 0.050719 | -1.37% |
| RootRot | 0.033939 | 0.035485 | +4.56% |
| BodyPos | 0.019352 | 0.018997 | -1.84% |
| BodyRot | 0.071305 | 0.073400 | +2.94% |
| DofVel | 0.471155 | 0.461079 | -2.14% |
| RootVel | 0.100458 | 0.102047 | +1.58% |
| RootAngVel | 0.260277 | 0.255412 | -1.87% |

自然终止计数：{"baseline": {"fail": 0, "null": 0, "succ": 8192, "time": 0}, "candidate": {"fail": 0, "null": 0, "succ": 8192, "time": 0}}

七项 ERR 中降低项数：4/7。

完整动作 completion：100.0000% → 100.0000%。

单 seed、单动作和历史对照不能证明稳定优势。随机起点不用于正式完整动作完成率。全部差异如实报告，未根据结果搜索参数。
