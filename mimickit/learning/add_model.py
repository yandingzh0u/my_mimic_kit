import gymnasium.spaces as spaces
import numpy as np
import torch

import learning.amp_model as amp_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util

class ADDModel(amp_model.AMPModel):
    def __init__(self, config, env):
        self._disc_geometry = config.get("disc_geometry", "add")
        self._metric_max = float(config.get("metric_max", 5.0))
        if self._disc_geometry not in {
                "add", "ref_concat", "global_metric",
                "conditioned_metric"}:
            raise ValueError(
                "Unsupported ADD discriminator geometry: {}".format(
                    self._disc_geometry))
        if self._metric_max < 1.0:
            raise ValueError("metric_max must be at least 1")
        super().__init__(config, env)
        return

    def eval_disc(self, diff, context=None):
        disc_input = self.build_disc_input(diff, context)
        return self.eval_disc_input(disc_input)

    def eval_disc_input(self, disc_input):
        h = self._disc_layers(disc_input)
        return self._disc_logits(h)

    def build_disc_input(self, diff, context=None):
        if self._disc_geometry == "add":
            return diff
        if self._disc_geometry == "ref_concat":
            if context is None:
                raise ValueError("RefConcat requires reference context")
            return torch.cat([diff, context], dim=-1)
        return self.transform_diff(diff, context)

    def transform_diff(self, diff, context=None):
        weights = self.calc_metric_weights(context)
        return torch.sqrt(weights) * diff

    def calc_metric_weights(self, context=None):
        if self._disc_geometry == "global_metric":
            logits = self._metric_logits
        elif self._disc_geometry == "conditioned_metric":
            if context is None:
                raise ValueError("Conditioned metric requires reference context")
            h = self._metric_layers(context)
            logits = self._metric_out(h)
        else:
            raise ValueError("Current geometry does not define a metric")

        # The mean weight is exactly one.  With a = log(metric_max)/2,
        # exp(a*tanh(.))/mean(exp(a*tanh(.))) lies in
        # [1/metric_max, metric_max], bounding the condition number while
        # preventing a trivial global rescaling of the discriminator input.
        log_radius = 0.5 * np.log(self._metric_max)
        weights = torch.exp(log_radius * torch.tanh(logits))
        return weights / torch.mean(weights, dim=-1, keepdim=True)

    def get_metric_stats(self, context=None):
        with torch.no_grad():
            weights = self.calc_metric_weights(context)
            return {
                "metric_weight_min": torch.min(weights),
                "metric_weight_max": torch.max(weights),
                "metric_weight_std": torch.std(weights),
            }

    def get_metric_grad_norm(self):
        if self._disc_geometry == "global_metric":
            params = [self._metric_logits]
        elif self._disc_geometry == "conditioned_metric":
            params = (list(self._metric_layers.parameters())
                      + list(self._metric_out.parameters()))
        else:
            return torch.zeros((), device=self._disc_logits.weight.device)

        grad_sq_sum = torch.zeros((), device=self._disc_logits.weight.device)
        for param in params:
            if param.grad is not None:
                grad_sq_sum += torch.sum(torch.square(param.grad.detach()))
        return torch.sqrt(grad_sq_sum)

    def get_disc_params(self):
        params = super().get_disc_params()
        if self._disc_geometry == "global_metric":
            params += [self._metric_logits]
        elif self._disc_geometry == "conditioned_metric":
            params += list(self._metric_layers.parameters())
            params += list(self._metric_out.parameters())
        return params

    def _build_disc(self, config, env):
        disc_obs_space = env.get_disc_obs_space()
        self._disc_obs_dim = int(np.prod(disc_obs_space.shape))
        self._build_metric(config, disc_obs_space)

        input_dim = self._disc_obs_dim
        if self._disc_geometry == "ref_concat":
            input_dim *= 2
        disc_input_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(input_dim,),
            dtype=disc_obs_space.dtype)
        input_dict = {"disc_obs": disc_input_space}
        self._disc_layers, _ = net_builder.build_net(
            config["disc_net"], input_dict, activation=self._activation)

        layers_out_size = torch_util.calc_layers_out_size(self._disc_layers)
        self._disc_logits = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.uniform_(self._disc_logits.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self._disc_logits.bias)
        return

    def _build_metric(self, config, context_space):
        if self._disc_geometry == "global_metric":
            self._metric_logits = torch.nn.Parameter(
                torch.zeros(self._disc_obs_dim))
        elif self._disc_geometry == "conditioned_metric":
            metric_net = config.get("metric_net", "fc_2layers_256units")
            self._metric_layers, _ = net_builder.build_net(
                metric_net, {"context": context_space},
                activation=self._activation)
            layers_out_size = torch_util.calc_layers_out_size(
                self._metric_layers)
            self._metric_out = torch.nn.Linear(
                layers_out_size, self._disc_obs_dim)
            torch.nn.init.zeros_(self._metric_out.weight)
            torch.nn.init.zeros_(self._metric_out.bias)
        return
