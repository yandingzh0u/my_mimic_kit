# R2 NO-GO: generic-run collapse

This snapshot preserves the implementation and milestone artifacts of the
absolute conditional-mismatch reward ablation before the clean three-stage CPG
rebuild.

## Provenance

- Base commit: `a8ea5eb4b6ecabb52da8c03e3bc87bb889c3a667`
- Source branch: `archive/r2-no-go-generic-run`
- PPO checkpoint: iteration 1100 / 144,310,272 transitions
- Deployment data: 54 locomotion clips
- Encoder/Flow fit data in this invalid ablation: 41 clips
- Failure: walk commands converged to a stable generic-run basin while episode
  length approached the environment timeout ceiling.

This result is retained as an engineering failure and policy-negative stress
corpus. It is not a clean R2 scientific ablation because the prior fit support
did not cover the full PPO deployment support and the old validation gate used
micro-averaged results that masked the walk failure.

## External artifacts

Milestone artifacts are stored outside Git at:

`/home/y/experiment_archives/MimicKit/R2_NO_GO_generic_run/`

The archive contains:

- the formal Stage 1 encoder artifact and logs;
- the formal conditional/NULL Flow artifact and logs;
- PPO checkpoints 0, 100, 500, 900, and 1100;
- the 13M condition-response evaluation;
- `MANIFEST.sha256` for byte-level verification.

No checkpoint is committed to ordinary Git.

## Reuse boundary

The new method may selectively reuse motion features, encoder/Flow building
blocks, paired scoring, runtime support checks, and tests. The old
absolute-`S_c` reward, micro gate, format-v2 publication contract, and PPO
configuration remain archive-only.
