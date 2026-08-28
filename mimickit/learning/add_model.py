import gymnasium.spaces as spaces
import numpy as np
import torch

import learning.amp_model as amp_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util

class ADDModel(amp_model.AMPModel):
    def __init__(self, config, env):
        self._disc_geometry = config.get("disc_geometry", "add")
        self._disc_spectral_norm = bool(
            config.get("disc_spectral_norm", False))
        self._disc_group_balanced_metric = bool(
            config.get("disc_group_balanced_metric", False))
        if self._disc_geometry not in {"add", "ref_concat"}:
            raise ValueError(
                "Unsupported ADD discriminator geometry: {}".format(
                    self._disc_geometry))
        super().__init__(config, env)
        return

    def eval_disc(self, diff, context=None):
        disc_input = self.build_disc_input(diff, context)
        return self.eval_disc_input(disc_input)

    def eval_disc_input(self, disc_input):
        h = self._disc_layers(disc_input)
        return self._disc_logits(h)

    def build_disc_input(self, diff, context=None):
        if self._disc_group_balanced_metric:
            diff = diff * self._disc_metric_scale
        if self._disc_geometry == "add":
            return diff
        if self._disc_geometry == "ref_concat":
            if context is None:
                raise ValueError("RefConcat requires reference context")
            return torch.cat([diff, context], dim=-1)
        raise ValueError("Unsupported ADD discriminator geometry")

    def _build_disc(self, config, env):
        disc_obs_space = env.get_disc_obs_space()
        self._disc_obs_dim = int(np.prod(disc_obs_space.shape))
        if self._disc_group_balanced_metric:
            if self._disc_geometry != "add":
                raise ValueError(
                    "Group-balanced metric requires direct ADD geometry")
            groups = env.get_disc_error_groups()
            dims = torch.tensor(
                [len(indices) for _, indices in groups], dtype=torch.float32)
            calibration_dim = torch.mean(dims)
            scale = torch.ones(self._disc_obs_dim, dtype=torch.float32)
            for group_id, (_, indices) in enumerate(groups):
                scale[list(indices)] = torch.sqrt(
                    calibration_dim / dims[group_id])
            self.register_buffer("_disc_metric_scale", scale)

        input_dim = self._disc_obs_dim
        if self._disc_geometry == "ref_concat":
            input_dim *= 2
        disc_input_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(input_dim,),
            dtype=disc_obs_space.dtype)
        input_dict = {"disc_obs": disc_input_space}
        self._disc_layers, _ = net_builder.build_net(
            config["disc_net"], input_dict, activation=self._activation)
        if self._disc_spectral_norm:
            for layer in self._disc_layers.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.utils.parametrizations.spectral_norm(layer)

        layers_out_size = torch_util.calc_layers_out_size(self._disc_layers)
        self._disc_logits = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.uniform_(self._disc_logits.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self._disc_logits.bias)
        if self._disc_spectral_norm:
            torch.nn.utils.parametrizations.spectral_norm(self._disc_logits)
        return
