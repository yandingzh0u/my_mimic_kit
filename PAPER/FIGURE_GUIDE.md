# MIRA manuscript figures

The compiled manuscript is [mira.pdf](mira.pdf). Tables use native LaTeX.
The six-motion comparison and the three-motion quality comparison put AMP,
DeepMimic, ADD, and MIRA on the same footing; their numerical values are
not repeated as bar charts.

| Figure | Source | Contents |
| --- | --- | --- |
| 1 | `figures/mira_pipeline_reference.pptx` | Reward-learning overview with Unitree G1 illustrations and the normalized error space. |
| 2 | `figures/mira_rollout_screenshots.pptx` | Actual Isaac Lab RGB captures: six frames each for Sofa Sit, Box Carry, and Soccer, and twelve each for Backflip, Climb, and Roll. |
| 3 | `tools/build_mira_supplement.py` | Soccer root-speed and ground-support timing from the final checkpoints, exported as a vector PDF. |
| Appendix | `figures/mira_results_editable.pptx` | Editable charts for the six-motion learning curves and Backflip discriminator diagnostics. |
| Appendix | `tools/build_mira_supplement.py` | Roll and Climb joint-velocity learning curves at their actual sample counts. |

Fig. 2 is illustrative. Soccer uses the final checkpoint from the tables,
Climb uses the 3000-update policy, and the Sofa Sit and Box Carry rows
retain their earlier illustrative captures. The experimental humanoid
is the 28-DoF model; G1 appears only in the method illustration. The raw
frames, timestamps, hashes, and cameras are documented under
`artifacts/screenshots/`. The screenshot builder verifies the RGB hashes
and the Soccer checkpoint hash before regenerating the editable slide.

Fig. 3 uses the actual reference and simulator traces from the same
seed-0 policies used in the three-motion quality table. Contact spans in
the plot use the majority across simulator replicas; the table's IoU
uses all foot/time/replica labels. The training curves use the actual
environment sample counts and do not extrapolate beyond measured tests.

See [README.md](README.md) for compilation and
[generated/README.md](generated/README.md) for trace collection and metric
regeneration. The Soccer row can be recaptured with Isaac Lab Python:

```bash
python PAPER/tools/capture_mira_screenshots.py \
  --run-dir output/mira_cmnn64_group_rms_soccer_cmu_10_02_2000_8192_seed0_20261122 \
  --out-dir PAPER/artifacts/screenshots/mira_20261124_soccer \
  --current-run --steps 1 31 57 69 80 130
python PAPER/tools/build_mira_screenshot_figure.py
```
