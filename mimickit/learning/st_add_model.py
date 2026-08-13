import gymnasium.spaces as spaces
import numpy as np
import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util

class STADDModel(add_model.ADDModel):
    """Structured discriminator for ST-ADD.

    The discriminator observation differential is split into three fixed
    segments and each is scored by its own encoder:

        state differential      -> state encoder    -> z_s
        motion trajectory       -> motion encoder   -> z_m
        winding residual        -> rotation encoder -> z_r

    The rotation branch only ever sees the accumulated directed-rotation
    residuals, so it cannot classify from root position, joints, or body
    positions. The three branch logits are additively fused with fixed equal
    weights into ONE logit:

        z_ST = (z_s + z_m + z_r) / 3

    Each branch is calibrated to the same scale by its own auxiliary
    zero-vs-policy discrimination loss (see STADDAgent), not by learned
    fusion weights that could re-suppress a branch. Policy reward comes only
    from the fused logit.
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

        self._disc_state_layers, self._disc_state_logits = self._build_disc_branch(state_net, state_dim)
        self._disc_motion_layers, self._disc_motion_logits = self._build_disc_branch(motion_net, motion_dim)
        self._disc_rot_layers, self._disc_rot_logits = self._build_disc_branch(rot_net, rot_dim)
        return

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

    def eval_disc_branches(self, disc_obs):
        state_obs, motion_obs, rot_obs = self._split_disc_obs(disc_obs)
        z_state = self._disc_state_logits(self._disc_state_layers(state_obs))
        z_motion = self._disc_motion_logits(self._disc_motion_layers(motion_obs))
        z_rot = self._disc_rot_logits(self._disc_rot_layers(rot_obs))
        z_fused = (z_state + z_motion + z_rot) / 3.0
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
