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
    positions. Two head modes:

    disc_head: "independent"
        Each branch has its own scalar head. The branch logits then live on
        arbitrary, unrelated scales (head fan-ins 1024/512/128 with identical
        init give std(z) ratios ~2.4/1.7 at birth, and training preserves a
        ~2x margin gap), which makes any cross-branch comparison of logits or
        anchored deficits scale-unidentifiable.

    disc_head: "shared"
        Root fix for the scale problem: each encoder output is projected to a
        common k-dim space, normalized by a single parameter-free LayerNorm,
        and scored by ONE shared scalar head

            u_i = LayerNorm(P_i h_i),   z_i = w^T u_i + b

        so all three logits are measured by the same ruler w on unit-scale
        features. The per-branch scale degree of freedom is removed by
        parameterization instead of post-hoc statistics.

    The branch logits are fused into ONE logit; policy reward comes only from
    the fused logit. Two fusion modes:

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

        self._head_mode = config.get("disc_head", "independent")
        assert self._head_mode in ["independent", "shared"]

        if (self._head_mode == "independent"):
            self._disc_state_layers, self._disc_state_logits = self._build_disc_branch(state_net, state_dim)
            self._disc_motion_layers, self._disc_motion_logits = self._build_disc_branch(motion_net, motion_dim)
            self._disc_rot_layers, self._disc_rot_logits = self._build_disc_branch(rot_net, rot_dim)
        else:
            head_dim = int(config.get("disc_head_dim", 128))
            self._disc_state_layers = self._build_disc_encoder(state_net, state_dim)
            self._disc_motion_layers = self._build_disc_encoder(motion_net, motion_dim)
            self._disc_rot_layers = self._build_disc_encoder(rot_net, rot_dim)

            self._disc_state_proj = torch.nn.Linear(torch_util.calc_layers_out_size(self._disc_state_layers), head_dim)
            self._disc_motion_proj = torch.nn.Linear(torch_util.calc_layers_out_size(self._disc_motion_layers), head_dim)
            self._disc_rot_proj = torch.nn.Linear(torch_util.calc_layers_out_size(self._disc_rot_layers), head_dim)

            # parameter-free normalization: the only scale left is the shared w
            self._disc_head_norm = torch.nn.LayerNorm(head_dim, elementwise_affine=False)
            self._disc_shared_logits = torch.nn.Linear(head_dim, 1)
            head_bound = math.sqrt(3.0 / head_dim)
            torch.nn.init.uniform_(self._disc_shared_logits.weight, -head_bound, head_bound)
            torch.nn.init.zeros_(self._disc_shared_logits.bias)
        return

    def get_fusion_mode(self):
        return self._fusion_mode

    def get_fusion_tau(self):
        return self._fusion_tau

    def _build_disc_encoder(self, net_name, in_dim):
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=[in_dim], dtype=np.float32)
        input_dict = {"disc_obs": obs_space}
        layers, _ = net_builder.build_net(net_name, input_dict, activation=self._activation)
        return layers

    def _build_disc_branch(self, net_name, in_dim):
        init_output_scale = 1.0
        layers = self._build_disc_encoder(net_name, in_dim)

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
        h_state = self._disc_state_layers(state_obs)
        h_motion = self._disc_motion_layers(motion_obs)
        h_rot = self._disc_rot_layers(rot_obs)

        if (self._head_mode == "independent"):
            z_state = self._disc_state_logits(h_state)
            z_motion = self._disc_motion_logits(h_motion)
            z_rot = self._disc_rot_logits(h_rot)
        else:
            u_state = self._disc_head_norm(self._disc_state_proj(h_state))
            u_motion = self._disc_head_norm(self._disc_motion_proj(h_motion))
            u_rot = self._disc_head_norm(self._disc_rot_proj(h_rot))
            z_state = self._disc_shared_logits(u_state)
            z_motion = self._disc_shared_logits(u_motion)
            z_rot = self._disc_shared_logits(u_rot)
        return z_state, z_motion, z_rot

    def eval_zero_anchor(self):
        """Branch logits at the universal ADD ideal differential [1, 3].

        The diff normalizer is scale-only (0 maps to 0), so the zero input
        here is exactly the perfect-tracking differential for every skill.
        """
        ref = self.get_disc_logit_weights()
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
        if (self._head_mode == "independent"):
            weights = [torch.flatten(self._disc_state_logits.weight),
                       torch.flatten(self._disc_motion_logits.weight),
                       torch.flatten(self._disc_rot_logits.weight)]
            return torch.cat(weights)
        else:
            return torch.flatten(self._disc_shared_logits.weight)

    def get_disc_params(self):
        params = (list(self._disc_state_layers.parameters())
                  + list(self._disc_motion_layers.parameters())
                  + list(self._disc_rot_layers.parameters()))
        if (self._head_mode == "independent"):
            params += (list(self._disc_state_logits.parameters())
                       + list(self._disc_motion_logits.parameters())
                       + list(self._disc_rot_logits.parameters()))
        else:
            params += (list(self._disc_state_proj.parameters())
                       + list(self._disc_motion_proj.parameters())
                       + list(self._disc_rot_proj.parameters())
                       + list(self._disc_shared_logits.parameters()))
        return params
