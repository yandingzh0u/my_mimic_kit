# External interaction assets

This directory contains the local inputs used to prepare the G1 interaction
experiments.  Files under `raw/` are intentionally ignored by Git.

- `raw/samp/sofa.obj` and `raw/samp/sofa_stageII.pkl` come from SAMP
  (Hassan et al., ICCV 2021).  The SAMP research-material license permits
  non-commercial research use but prohibits redistribution.
- `raw/force/260923_10_2_4.npz` and `raw/force/box_huge.ply` come from FORCE
  (Zhang et al., 3DV 2025).
- `raw/unitree/g1_29dof_rev_1_0.urdf` comes from Unitree's public
  `unitree_ros` G1 description.

Generated MimicKit motions contain retargeted robot states rather than copies
of the original human body model.  Re-run the preparation scripts only after
obtaining each dataset from its official source and accepting its license.

For `sofa003`, the local retargeted motion is `data/motions/humanoid/humanoid_sofa003.pkl`
and the matching asset is `data/assets/objects/samp_sofa003.usd`. These two files
are excluded from Git. The November 2026 interaction runs preserve their original
environment YAML as `frozen_env.yaml` for strict checkpoint resume on the original
machine, where legacy asset paths resolve to the relocated files.
