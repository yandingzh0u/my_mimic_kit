# MIRA paper

The manuscript, figures and paper tools live in this directory. Experiment
logs, checkpoints and frozen source snapshots live directly under `output/`
at the repository root, as requested. Commands below run from the repository
root; shared simulator code, motion data, robot assets, and common agent/engine
configurations remain in their usual repository directories.

| Location | Contents |
| --- | --- |
| `icra2027_paper.tex`, `mira.pdf` | Main manuscript and compiled paper |
| `experiments_mira.tex`, `appendix_mira.tex` | Experiment section and supplementary material |
| `figures/` | Current figures, editable PowerPoint sources, and G1 illustration assets |
| `generated/` | Measured data exports, provenance, and native LaTeX tables |
| `configs/envs/`, `configs/args/` | Paper-specific environment and training configurations |
| `tools/` | Data/figure generation, screenshot capture, benchmark and evaluation tools |
| `tests/` | Tests for paper evaluation and benchmark configurations |
| `artifacts/screenshots/` | Original Isaac Lab RGB screenshots and capture metadata |
| `artifacts/rollouts/` | Saved rollout poses and metadata |
| `build/` | Compilation logs and verification records |
| `backups/` | Archived obsolete files and the cleanup manifest |

See [FIGURE_GUIDE.md](FIGURE_GUIDE.md) for figure details and
[generated/README.md](generated/README.md) for data interpretation.

## Regenerate and compile

```bash
python PAPER/tools/build_mira_experiments.py
python PAPER/tools/build_mira_tracking_table.py
python PAPER/tools/build_mira_figures.py
python PAPER/tools/build_mira_supplement.py
python PAPER/tools/build_mira_pipeline_reference.py
python PAPER/tools/build_mira_screenshot_figure.py
cd PAPER
mkdir -p build/latex
/home/y/.local/bin/tectonic --only-cached --outdir build/latex icra2027_paper.tex
cp build/latex/icra2027_paper.pdf mira.pdf
```

Use the Isaac Lab Python environment for simulator capture, benchmark tools,
and evaluation tests. For example:

```bash
python PAPER/tools/benchmark/run_mira_diagonal.py --help
python PAPER/tools/evaluation/evaluate_checkpoint.py --help
python -m pytest PAPER/tests/test_paper_eval_metrics.py PAPER/tests/test_paper_eval_artifacts.py
```

## Relocation and provenance

[relocation_manifest.json](relocation_manifest.json) records the previous and
current directories, including experiment runs under `output/`. Historical
experiment records, frozen source snapshots,
model files, and archive members keep their original bytes. Paths recorded
inside historical records are preserved; current run directories use `mira_`.
`tools/paths.py` translates those recorded paths to their current locations;
the current benchmark runner can resume moved runs without editing their
frozen training code. Current generated data uses the new paths.

Large artifacts and build outputs remain excluded from Git, as before.
