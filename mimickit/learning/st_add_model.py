import gymnasium.spaces as spaces
import math
import numpy as np
import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util

class STADDModel(add_model.ADDModel):
    """Structured discriminator for ST-ADD / ZA-STADD.

    The discriminator observation differential is split into three fixed
    segments and each is scored by its own encoder:

        state differential      -> state encoder    -> z_s
        motion trajectory       -> motion encoder   -> z_m
        winding residual        -> rotation encoder -> z_r

    The rotation branch only ever sees the accumulated directed-rotation
    residuals, so it cannot classify from root position, joints, or body
    positions. The three branch logits are fused into ONE logit; policy
    reward comes only from the fused logit. Two fusion modes:

    disc_fusion: "mean" (ST-ADD)
        z = (z_s + z_m + z_r) / 3

    disc_fusion: "za" (ZA-STADD, zero-anchored smooth-Tchebycheff)
        b   = z(0)                      exact per-branch ideal anchor at the
                                        universal ADD zero differential
        d   = b - z(Delta)              anchored deficit (additive per-branch
                                        logit bias cancels exactly)
        z   = mean(b) - tau * log[ (1/3) sum_i exp(d_i / tau) ]

        with gradient dz/dz_i = softmax(d/tau)_i: the branch currently
        furthest from its own perfect-tracking anchor automatically gets the
        largest weight (smooth bottleneck, no learned fusion parameters).
        tau -> inf recovers the mean fusion exactly; tau -> 0 approaches
        mean(b) - max_i d_i. The anchor is detached in the fusion so the
        bottleneck cannot cheat by moving the reference point; z_i(0) itself
        keeps training through the positive BCE paths.

    Each branch is calibrated by its own auxiliary zero-vs-policy
    discrimination loss (see STADDAgent), not by learned fusion weights that
    could re-suppress a branch.
    """

    def __init__(self, config, env):
        super().__init__(config, env)
        return

    def _build_disc(self, config, env):
        total_dim = env.get_disc_obs_space().shape[0]
        state_dim = env.get_disc_state_obs_dim()
        motion_dim = env.get_disc_traj_motion_obs_dim()
        rot_dim = env.get_disc_traj_rot_obs_dim()
        assert state_dim + motion_dim + rot_dim == total_dim

        self._disc_state_dim = state_dim
        self._disc_motion_dim = motion_dim
        self._disc_rot_dim = rot_dim

        state_net = config["disc_state_net"]
        motion_net = config["disc_motion_net"]
        rot_net = config["disc_rot_net"]

        self._fusion_mode = config.get("disc_fusion", "mean")
        assert self._fusion_mode in ["mean", "za"]
        self._fusion_tau = float(config.get("disc_fusion_tau", 2.0))
        assert self._fusion_tau > 0.0

        self._disc_state_layers, self._disc_state_logits = self._build_disc_branch(state_net, state_dim)
        self._disc_motion_layers, self._disc_motion_logits = self._build_disc_branch(motion_net, motion_dim)
        self._disc_rot_layers, self._disc_rot_logits = self._build_disc_branch(rot_net, rot_dim)
        return

    def get_fusion_mode(self):
        return self._fusion_mode

    def get_fusion_tau(self):
        return self._fusion_tau

    def _build_disc_branch(self, net_name, in_dim):
        init_output_scale = 1.0
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=[in_dim], dtype=np.float32)
        input_dict = {"disc_obs": obs_space}
        layers, _ = net_builder.build_net(net_name, input_dict, activation=self._activation)

        layers_out_size = torch_util.calc_layers_out_size(layers)
        logits = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.uniform_(logits.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(logits.bias)
        return layers, logits

    def _split_disc_obs(self, disc_obs):
        s = self._disc_state_dim
        m = self._disc_motion_dim
        state_obs = disc_obs[..., :s]
        motion_obs = disc_obs[..., s:s + m]
        rot_obs = disc_obs[..., s + m:]
        return state_obs, motion_obs, rot_obs

    def _eval_branch_logits(self, disc_obs):
        state_obs, motion_obs, rot_obs = self._split_disc_obs(disc_obs)
        z_state = self._disc_state_logits(self._disc_state_layers(state_obs))
        z_motion = self._disc_motion_logits(self._disc_motion_layers(motion_obs))
        z_rot = self._disc_rot_logits(self._disc_rot_layers(rot_obs))
        return z_state, z_motion, z_rot

    def eval_zero_anchor(self):
        """Branch logits at the universal ADD ideal differential [1, 3].

        The diff normalizer is scale-only (0 maps to 0), so the zero input
        here is exactly the perfect-tracking differential for every skill.
        """
        ref = self._disc_state_logits.weight
        total_dim = self._disc_state_dim + self._disc_motion_dim + self._disc_rot_dim
        zeros = torch.zeros([1, total_dim], device=ref.device, dtype=ref.dtype)
        z_state, z_motion, z_rot = self._eval_branch_logits(zeros)
        return torch.cat([z_state, z_motion, z_rot], dim=-1)

    def eval_disc_branches(self, disc_obs):
        z_state, z_motion, z_rot = self._eval_branch_logits(disc_obs)

        if (self._fusion_mode == "mean"):
            z_fused = (z_state + z_motion + z_rot) / 3.0
        else:
            z = torch.cat([z_state, z_motion, z_rot], dim=-1)
            anchor = self.eval_zero_anchor().detach()
            deficit = anchor - z
            lse = torch.logsumexp(deficit / self._fusion_tau, dim=-1, keepdim=True) - math.log(z.shape[-1])
            z_fused = torch.mean(anchor, dim=-1, keepdim=True) - self._fusion_tau * lse

        return z_state, z_motion, z_rot, z_fused

    def eval_disc(self, disc_obs):
        _, _, _, z_fused = self.eval_disc_branches(disc_obs)
        return z_fused

    def get_disc_logit_weights(self):
        weights = [torch.flatten(self._disc_state_logits.weight),
                   torch.flatten(self._disc_motion_logits.weight),
                   torch.flatten(self._disc_rot_logits.weight)]
        return torch.cat(weights)

    def get_disc_params(self):
        params = (list(self._disc_state_layers.parameters()) + list(self._disc_state_logits.parameters())
                  + list(self._disc_motion_layers.parameters()) + list(self._disc_motion_logits.parameters())
                  + list(self._disc_rot_layers.parameters()) + list(self._disc_rot_logits.parameters()))
        return params
