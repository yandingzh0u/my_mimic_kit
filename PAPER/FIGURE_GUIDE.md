# MIRA manuscript and editable figures

The compiled manuscript is `icra2027_paper.pdf`. Its title, abstract,
introduction, method, experiments, and conclusion use the current MIRA
formulation.

## Tables

All manuscript tables are native LaTeX. They are not PowerPoint assets or
embedded table screenshots:

- Configuration: `experiments_mira.tex`.
- Main comparison: `generated/mira_main_table.tex`.
- Backflip ablations: `generated/mira_ablation_table.tex`.

AMP, DeepMimic, ADD, and MIRA are reported together. The quantitative comparison is presented in the LaTeX table; there is no
redundant seven-error bar chart or baseline-ratio heatmap.

## Editable PowerPoint sources

| File in `figures/` | Contents |
| --- | --- |
| `mira_pipeline.pptx` | Reference-layout method illustration with rendered Unitree G1 icons, aligned sample cards, pastel fills, the constrained discriminator, softplus reward, and PPO control loop; layer sizes and training settings remain in the text. |
| `mira_results_editable.pptx` | Native charts for learning curves and discriminator diagnostics only. No tables, heatmaps, or repeated numerical-comparison figure. |
| `mira_rollout_screenshots.pptx` | Actual Isaac Lab camera captures: Get-Up and Crawl first with six frames each, then Backflip, Climb, Roll, and Spin Kick with twelve frames each (60 total). Image placement and labels remain editable. |

Plots and the method illustration have vector PDF exports and PNG previews.
The screenshot PDF embeds the actual rendered raster frames. PowerPoint
charts retain editable data; the overview uses native shapes and the
rendered Unitree G1 model images. The G1 illustrations belong only to the method diagram; the experimental policies and humanoid remain unchanged.

## Data and interpretation

All active results use measured seed 0. MIRA Climb uses the actual 3000-update
checkpoint (786,432,000 samples); other results use 2000 updates
(524,288,000 samples). The manuscript briefly records this budget distinction.
Main learning curves average the same six motions equally, aligned by the
fraction of each run's actual training budget. They do not represent equal
absolute-sample efficiency. There are no simulated extra seeds or uncertainty
bands. Source log paths and SHA-256 hashes are recorded in
`generated/mira_measured_seed0.json`.

The comparison uses deterministic reference-start tests recorded during
training. Independent evaluation under one unified protocol and further
training seeds remain TODO. Errors cover executed trajectory portions.
Completion marks remain in the Backflip ablation table; the main table has
no completion column or completion panel. Bold values mark the lowest error
for each metric in panel (a) only; panel (b) uses regular-weight values.

The action figure contains real RGB captures from Isaac Lab executions of
the frozen policies, with the actual climbing box retained. Climb uses the
3000-update model and 12 keyframes. Backflip, Roll, and Spin Kick use their
2000-update models and 12 keyframes each; Get-Up and Crawl use six each
and appear first in the figure. Changing camera, lighting, and visual colors
does not change policy weights or physical behavior. Raw PNG captures,
frame timestamps, checkpoint hashes, and camera provenance are under
`PAPER/artifacts/screenshots/mira_20261115/`. These are
illustrative rollouts, not additional training seeds or success-rate samples.

## Regeneration

Run from the repository root with Python, Matplotlib, python-pptx, PyMuPDF,
and LibreOffice installed:

```bash
python PAPER/tools/build_mira_experiments.py
python PAPER/tools/build_mira_figures.py
python PAPER/tools/build_mira_pipeline.py
python PAPER/tools/build_mira_screenshot_figure.py
cd PAPER
mkdir -p build/latex
/home/y/.local/bin/tectonic --only-cached --outdir build/latex icra2027_paper.tex
cp build/latex/icra2027_paper.pdf icra2027_paper.pdf
```

The screenshot-figure generator reads saved RGB captures. Actual simulator
capture uses `PAPER/tools/capture_mira_screenshots.py` with the Isaac Lab
Python environment and a selected frozen run; it does not train the policy. Superseded diagrams, placeholder panels, and redundant plots have been
removed. The data generator produces tables and measured data only; current
plots are produced exclusively by `build_mira_figures.py`.
