"""Intrinsic reference context for the CPMD discriminator metric.

The context is the current reference configuration expressed with the same
single-frame features as ADD, but always in a heading-normalized, root-relative
frame. In particular, global x/y translation is zeroed. This tensor is
training-side information for the discriminator metric; it is never part of
the actor observation.
"""

import envs.add_env as add_env


def compute_intrinsic_context(root_pos, root_rot, root_vel, root_ang_vel,
                              joint_rot, dof_vel, body_pos):
    """Build a single-frame, intrinsic ADD observation.

    ``add_env.compute_disc_obs`` operates on a short time axis. CPMD context
    is instantaneous, so each synchronized reference tensor is given a
    singleton time dimension. ``global_obs=False`` removes global heading and
    x/y translation while retaining root height, local velocities, joint
    configuration, and root-relative body configuration.
    """
    return add_env.compute_disc_obs(
        root_pos=root_pos.unsqueeze(1),
        root_rot=root_rot.unsqueeze(1),
        root_vel=root_vel.unsqueeze(1),
        root_ang_vel=root_ang_vel.unsqueeze(1),
        joint_rot=joint_rot.unsqueeze(1),
        dof_vel=dof_vel.unsqueeze(1),
        body_pos=body_pos.unsqueeze(1),
        global_obs=False,
    )
