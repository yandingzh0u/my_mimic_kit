# MM-ADD Roll experiment

## Problem

ADD learns how to combine position, rotation, and velocity errors, but its PPO
objective still maximizes the time-average return.  In Roll, the difficult
inverted transition occupies only a short interval.  Consequently, good
rewards in the easy standing, lying, and recovery phases can compensate for a
failure to execute the actual roll.  This is a phase-wise objective imbalance,
not necessarily a failure of the ADD discriminator.

## Change

MM-ADD keeps the Aligned ADD policy input, discriminator, BCE objective,
gradient penalty, learned reward, network architecture, and PPO settings.  It
partitions the reference trajectory at the control rate and optimizes a max-min
objective over phase-wise ADD returns.  Exponentiated dual updates place more
optimization mass on the currently worst-tracked phases.  Phase labels are
training metadata only; neither the policy nor the discriminator observes
them.

## Roll result (seed 0)

All values below come from the same deterministic evaluator: phase-zero start,
60 control steps (2 seconds), 256 environments, no automatic reset, and the
same Roll reference.  Lower tracking error is better.

| Metric | Aligned ADD, 8192 envs x 2000 iterations | MM-ADD, 4096 envs x 2000 iterations |
|---|---:|---:|
| Completion rate | 1.000 | 1.000 |
| Root position error | 0.1293 | **0.1031** |
| Root rotation error | 0.1237 | **0.1048** |
| Body position error | 0.0586 | **0.0510** |
| Body rotation error | 0.1916 | **0.1820** |
| DoF velocity error | **0.6565** | 0.8093 |
| Root velocity error | 0.3996 | **0.3522** |
| Root angular-velocity error | 0.6462 | **0.6364** |
| Paper position error | 0.0672 | **0.0579** |

At approximately the same training sample count (262 million), MM-ADD completes
Roll while the 8192-environment Aligned ADD checkpoint at 1000 iterations does
not (completion rates 1.000 and 0.000, respectively).  Thus this run suggests
improved phase coverage and sample efficiency.  The remaining weakness is the
23.3% higher DoF velocity error, indicating that full-body velocity fidelity
lags behind the spatial trajectory.

These results are evidence from one training seed and one phase-zero evaluation
condition.  A paper-level reliability claim still requires multiple training
seeds and randomized or grid-based starting phases.

## Artifacts

- Training log: `output/mm_aligned_add/roll_300_4096_seed0/log.txt`
- Structured training metrics: `output/mm_aligned_add/roll_300_4096_seed0/train_metrics.jsonl`
- MM-ADD final evaluation: `output/paper_eval/mm_vs_aligned/mm_add_final_phase0/summary.json`
- Aligned ADD final evaluation: `output/paper_eval/mm_vs_aligned/aligned_add_final_phase0/summary.json`
- Aligned ADD equal-sample evaluation: `output/paper_eval/mm_vs_aligned/aligned_add_262m_phase0/summary.json`
