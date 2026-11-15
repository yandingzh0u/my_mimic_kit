# MIRA experiment draft data

Regenerate the tables and plots from the repository's measured logs:

```bash
python PAPER/tools/build_mira_experiments.py --synthetic-seed-placeholders
```

The flag explicitly enables random **layout-only** seed deviations. Centers
remain measured seed-0 values. Synthetic slots 1 and 2 are symmetric around
that center and are not independent runs, empirical uncertainty, or evidence
of statistical significance. Tables, figures, and the experiment text label
them accordingly. Omit the flag to generate seed-0-only values and curves.

`mira_measured_seed0.json` contains the measured final values, source paths,
and SHA-256 hashes. `mira_synthetic_seed_placeholders.json` stores synthetic
layout values separately. No generated values are written to training logs.

The target is three independent seeds per configuration. Currently the
generator reads verified seed-0 runs only. When seed 1 and 2 are available,
extend its run manifest to read those logs and replace the layout deviations
with the sample standard deviation of **measured per-seed macro-averages**.
Remove the synthetic annotations only after doing so.

All main results use exactly 2000 updates (524,288,000 samples). The Climb
run's later 2500/3000-update records are deliberately excluded. Macro-averages
in the main table require all six motions. Preliminary learning curves use
the same currently measured three-motion cohort (Backflip, Climb, Crawl)
for every compared method, as explicitly authorized for this draft. Their
captions and image labels identify the three-motion average. Missing motions
are never filled with random values. Missing current MIRA runs:
Roll, Get-Up, Spin Kick. Their table cells and the full MIRA aggregate remain
TODO in the main table. Available per-motion values are retained there.

Backflip ablation diagnostics use measured seed-0 data, without synthetic
observations. Completion checks/crosses refer to this seed's reference-start
evaluation reaching 60 steps; additional-seed completion is pending. The
original head regularizer is retained in all variants, with different
effective behavior after normalization is removed. The full run resumed
after an interruption, with the same cumulative sample budget.

The six keyframe panels are explicitly TODO because current MIRA renders
are unavailable. Older DARE images are preserved on disk but are not
relabelled as MIRA rollouts.
