# MIRA measured experiment data

## Current manuscript

Run `python PAPER/tools/build_mira_tracking_table.py` to regenerate
`mira_tracking_table.tex` and `mira_tracking_results.json`. The table
covers the six motions in Fig. 2: Sofa Sit, Box Carry, Soccer, Backflip,
Climb, and Roll. The JSON records per-motion errors, source files and hashes,
episode counts, and sample counts. The figure generator uses its
`curve_logs` paths for the same six motions at equal sample counts.

The three motions with objects use independent reference-start evaluations
of the final checkpoints in `PAPER/build/interaction_eval/`, with the same
time window for every method (171, 197, and 129 steps, respectively). The
other three use the latest reference-start training log with a mean test
length equal to the motion horizon. Climb uses 299 steps for DeepMimic, ADD,
and MIRA. The ADD record is from `add_climb_2k_8192_seed0_resume500_v3`.
AMP has no logged full-horizon Backflip or Climb record and has no
six-motion aggregate; its three-motion aggregate is reported alongside
the other methods. Curve endpoints are training-time tests through
524,288,000 samples and need not match the table's selected records.

## Current supplement and motion-quality figures

Run `python PAPER/tools/build_mira_supplement.py` from the repository root
after regenerating the tracking manifest. It reads the raw contact-motion
and Backflip traces, exports `mira_supplement_results.json`, and writes
the native LaTeX tables and two vector time-series figures. The manifest
records each trace and checkpoint hash. Only measured seed 0 is used.

- `mira_motion_quality_table.tex` and `mira_per_motion_quality_table.tex`:
  planar root-speed correlation, foot--ground support timing IoU, stance
  slip, and body jerk. The macro gives equal weight to Sofa Sit, Box Carry,
  and Soccer. Replicas come from the same trained checkpoint, not 256 seeds.
- `mira_per_motion_table.tex`: the full six-motion tracking breakdown.
  Missing full-window AMP entries stay blank in the aggregate and appear
  as dashes in the tables.
- `mira_matched_ablation_table.tex` and `mira_full_ablation_table.tex`:
  independent Backflip rollouts from the final six checkpoints, with
  errors compared within 60-step or 26-step common prefixes.

To recreate the quality traces, run the following with Isaac Lab Python
from the repository root (serial execution):

```bash
for motion in sofa003 carry_box soccer_cmu_10_02; do
  for method in AMP DeepMimic ADD MIRA; do
    python PAPER/tools/evaluation/record_motion_quality.py "$motion" "$method" \
      --destination PAPER/build/mira_motion_metric_audit
  done
done
```

For each key in `mira_supplement_results.json` under `ablations`, use
the recorded `model_file` and its run's `env_config.yaml`,
`agent_config.yaml`, and `engine_config.yaml` with
`PAPER/tools/evaluation/evaluate_checkpoint.py --motion backflip --steps 60
--num-envs 256 --start-mode phase0 --seed 0`. Write the output to
`PAPER/build/appendix_backflip_eval/<key>`.

## Legacy tracking benchmark and training-time ablation

Regenerate the earlier six-motion tracking assets and legacy Backflip
training-time ablation diagnostics from logs:

    python PAPER/tools/build_mira_experiments.py

Active results contain **measured seed 0 only**. They contain no random seed
deviations, error bands, estimated seed uncertainty, or unmeasured motion data.
Historical placeholders and superseded figure assets have been removed.

**MIRA Climb uses 3000 PPO updates (786,432,000 environment samples);
all other terminal results use 2000 updates (524,288,000 samples).**
Every run uses 8192 environments. The six motions are Roll, Climb, Backflip,
Crawl, Get-Up, and Spin Kick. Terminal macro-averages are therefore not
equal-budget comparisons.
Run lookup discovers completed MIRA logs across date suffixes instead of
assuming that every motion was trained on the same date.

## Files

- mira_measured_seed0.json: canonical measured values, nominal motion horizons,
  logged test lengths, full-horizon checks, source paths and SHA-256 hashes.
- mira_results_audit.json and .md: equal-weight six-motion averages,
  raw per-motion values and completion for all four methods, ablation
  diagnostics and limitations. No method is used as a privileged reference.
- mira_per_motion_seed0.csv: all 24 method-by-motion rows, with all seven ERRs.
- mira_learning_curve_data.json: measured per-motion curves and their
  equal-weight six-motion mean aligned by each run's actual terminal-budget
  fraction. The primary axis is grid_training_progress; per_run_grid_samples
  records the different actual sample locations. Legacy grid_samples is a
  2000-update reference schedule only. Interpolation is linear between
  logged evaluation points; no unmeasured suffix is created for other runs.
- mira_main_table.tex and mira_ablation_table.tex: the earlier tracking
  and ablation tables, respectively. Paper tables are native LaTeX.

## Protocol and interpretation

These tables and learning curves use the **reference-start tests recorded
during training**, with one measured seed per configuration. They do not use
the independently collected baseline eval/final/summary.json files, whose
protocol differs. Existing independent reports are recorded as available
metadata, never mixed into the current comparisons.

The MIRA Climb terminal record has logged iteration 2999, samples 786432000,
and resume segment 2. Its measured test length is 299 steps. The source
directory retains its original 2000-update name after continuation; its
actual budget is recorded explicitly in every per-run manifest/CSV row.
The exact 3000-update weights are int_models/model_0000002999.pt inside
that run. Baseline Climb records remain at their measured 2000-update
budgets. Tracking errors cover only executed timesteps, so lower error
does not by itself imply full-motion success.

Learning curves use a training-progress axis: the same percentage maps to
3000 times that fraction for MIRA Climb and 2000 times that fraction for
each other run. This describes relative training progress rather than
equal-sample efficiency. There is no zero-update measurement; the shared
progress grid begins at 0.0005 and ends at 1. No initial or trailing
measurements are invented.

A completion check in the ablation table or audit means the mean logged test length
reaches the configured motion horizon, with a tolerance of 0.001 simulation
step. It is a full-horizon survival check rather than an independent
motion-specific success-rate estimate. The nominal horizons are Roll 60,
Climb 299, Backflip 60, Crawl 77, Get-Up 91, and Spin Kick 60 steps.

AMP, DeepMimic, ADD, and MIRA are reported using the same raw-value layout
with their actual budgets. Per-motion and aggregate results need not rank methods
identically; the results do not support universal superiority. Additional
independent training seeds and a unified independent checkpoint evaluation
for the six-motion tracking benchmark remain **TODO**.

Backflip ablations use the same total sample budget. The monotone-modeling
ablation jointly changes input representation, normalization, and weight
signs, so that comparison cannot isolate any one of those changes. The
diagonal-normalization ablation preserves monotone modeling. Sampled
gradient diagnostics are local measurements; norms from different input
representations cannot be interpreted as identical physical-space
sensitivity. The original head regularizer is retained in all variants.

The legacy tracking table contains the macro-average and seven-error MIRA
per-motion panels. The per-motion panel has no completion column, and
the former main-table completion panel is removed. The Backflip ablation
table retains its completion column. A brief dagger note identifies the
MIRA Climb 3000-update result in the main table.

## Archived physical outcome evaluation

`mira_interaction_results.json` contains separate checkpoint evaluations
for Box Carry, Sofa Sit, and CMU 10_02 Soccer: exact success and horizon
completion counts, physical thresholds, tracking errors averaged over
executed steps per trial and then over trials, and checkpoint hashes.
These are fixed-reference-start simulator replicas of one training seed
per method, with $524{,}288{,}000$ samples per policy. They are not random
initial conditions or independent training runs. These task-specific success
counts are not used in the current manuscript.

Run `PAPER/tools/evaluation/evaluate_interactions.py TASK METHOD
--destination PAPER/build/interaction_eval/TASK/METHOD` in the Isaac Lab
environment for each task and method, capturing output in the destination's
`eval.log`. For Box Carry and Soccer, also run the MIRA configuration with
`--controller guide` and the destination `TASK/guide`. The guide control
uses zero residual action. Then run
`PAPER/tools/evaluation/build_interaction_table.py` to check the saved
trajectories and regenerate the result file and LaTeX table.

The data generator does not write figure files. Current vector plots and
editable chart slides are generated by `PAPER/tools/build_mira_figures.py`.
Actual simulator captures and the G1 method diagram have separate generators,
listed in `PAPER/FIGURE_GUIDE.md`.
