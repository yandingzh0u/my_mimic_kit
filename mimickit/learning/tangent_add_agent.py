import numpy as np
import torch

import learning.add_agent as add_agent
import envs.temporal_add_obs as temporal_add_obs

class TangentADDAgent(add_agent.ADDAgent):
    """ADD + gated fixed tangent reward (TangentADD, stage 1).

    The ADD branch is inherited untouched: same ADDModel, same disc loss,
    same softplus disc reward (r_ADD = disc_reward_scale * softplus(logit)).
    A separate, always non-negative tangent branch is added on top:

        r_tan = w_t * exp(-E_tan / (2 sigma^2)),   w_t = exp(-E_cfg / (2 rho^2))
        r     = r_ADD + lambda_tan * r_tan

    E_cfg is computed from the pre-step configuration error and E_tan from
    the post-step generalized velocity residual (exposed by TangentADDEnv),
    each dimension-balanced over three groups with fixed physical scales
    loaded from a frozen calibration file (never running statistics).

    With tangent_reward.scale = 0 this agent is numerically identical to ADD:
    no extra data is recorded, no reward is added, no networks or random
    numbers are created beyond ADD's.
    """
    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        self._build_tangent_reward_params(config, env)
        return

    def _build_tangent_reward_params(self, config, env):
        tangent_config = config.get("tangent_reward", dict())

        self._tangent_reward_scale = float(tangent_config.get("scale", 0.0))
        self._tangent_gate_radius = float(tangent_config.get("gate_radius", 1.0))
        self._tangent_error_sigma = float(tangent_config.get("error_sigma", 1.0))
        self._tangent_group_weights = [float(w) for w in tangent_config.get("group_weights", [1.0 / 3.0] * 3)]

        assert self._tangent_reward_scale >= 0.0
        assert self._tangent_gate_radius > 0.0
        assert self._tangent_error_sigma > 0.0
        assert len(self._tangent_group_weights) == 3

        self._tangent_cfg_group_dims = env.get_tangent_cfg_group_dims()
        self._tangent_vel_group_dims = env.get_tangent_vel_group_dims()

        scale_file = tangent_config.get("scale_file", None)
        if (self._enable_tangent_reward()):
            assert scale_file is not None, "tangent_reward.scale > 0 requires tangent_reward.scale_file"

        if (scale_file is not None):
            scales = np.load(scale_file)
            self._tangent_cfg_scales = [float(s) for s in scales["cfg_scales"]]
            self._tangent_tan_scales = [float(s) for s in scales["tan_scales"]]

            assert all(s > 0.0 for s in self._tangent_cfg_scales)
            assert all(s > 0.0 for s in self._tangent_tan_scales)
            assert list(scales["cfg_group_dims"]) == self._tangent_cfg_group_dims, \
                "calibration file group dims do not match the character"
            assert list(scales["vel_group_dims"]) == self._tangent_vel_group_dims, \
                "calibration file group dims do not match the character"
        else:
            self._tangent_cfg_scales = None
            self._tangent_tan_scales = None
        return

    def _enable_tangent_reward(self):
        return self._tangent_reward_scale > 0.0

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)

        if (self._enable_tangent_reward()):
            self._exp_buffer.record("tangent_cfg_err", next_info["tangent_cfg_err"])
            self._exp_buffer.record("tangent_vel_resid", next_info["tangent_vel_resid"])
        return

    def _compute_rewards(self):
        info = super()._compute_rewards()

        if (self._enable_tangent_reward()):
            cfg_err = self._exp_buffer.get_data_flat("tangent_cfg_err")
            vel_resid = self._exp_buffer.get_data_flat("tangent_vel_resid")

            with torch.no_grad():
                gate_w, tangent_r, e_cfg, e_tan = temporal_add_obs.calc_tangent_rewards(
                    cfg_err=cfg_err,
                    vel_resid=vel_resid,
                    cfg_group_dims=self._tangent_cfg_group_dims,
                    vel_group_dims=self._tangent_vel_group_dims,
                    group_weights=self._tangent_group_weights,
                    cfg_scales=self._tangent_cfg_scales,
                    tan_scales=self._tangent_tan_scales,
                    gate_radius=self._tangent_gate_radius,
                    error_sigma=self._tangent_error_sigma)

                r = self._exp_buffer.get_data_flat("reward")
                r = r + self._tangent_reward_scale * tangent_r
                self._exp_buffer.set_data_flat("reward", r)

                tangent_r_std, tangent_r_mean = torch.std_mean(tangent_r)
                info["tangent_reward_mean"] = tangent_r_mean
                info["tangent_reward_std"] = tangent_r_std
                info["tangent_gate_mean"] = torch.mean(gate_w)
                info["tangent_gate_frac"] = torch.mean((gate_w > 0.1).float())
                info["tangent_e_cfg_mean"] = torch.mean(e_cfg)
                info["tangent_e_tan_mean"] = torch.mean(e_tan)
        return info
