# PARC Jump Gap 005

`parc_jump_gap_005.npz` contains the heightfield and reference contact labels from `dec_release/initial_aug/jumping/run_jump_gap_005_dm.pkl` in the [official PARC dataset](https://huggingface.co/datasets/mxucg/PARC) (MIT). The source member SHA-256 is `496d774478d8b204b0af22340147ca5b78b981c11030e15b6e3575ceb6621f11`.

For Isaac Lab, the coplanar cells are merged into three surfaces and the two vertical trench walls. This preserves the official cell centers, step heights, and edges in the reference coordinate system. Each grid cell covers a 0.4 m square centered at `min_point + 0.4 * [i, j]`. The terrain covers approximately `x = [-0.840, 3.160]`, `y = [-0.991, 9.009]`, with a 1.2 m trench at `y = [3.409, 4.609]`, a floor at `z = -3`, and banks at `z = 0`.

Use it with `data/motions/humanoid/parc/humanoid_run_jump_gap04.pkl`, the native copy of `dec_release/initial/jumping/run_jump_gap_005_dm.pkl`. Its 196 frames at 30 Hz give the same root positions and joint orientations as the paired source. The simulator places the motion and terrain at the same per-environment origin; neither has been shifted relative to the other.
