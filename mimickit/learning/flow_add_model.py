import numpy as np
import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util

# structured differential-flow discriminator (FlowADD)
DISC_MODE_FLOW = "flow"
# radial-only ablation, tangential component disabled
DISC_MODE_RADIAL = "radial"
# plain MLP on the concatenated input [delta, v] (Concat-ADD baseline)
DISC_MODE_CONCAT = "concat"

DISC_MODES = [DISC_MODE_FLOW, DISC_MODE_RADIAL, DISC_MODE_CONCAT]

class FlowADDModel(add_model.ADDModel):
    """ADD discriminator extended from pointwise differential scalarization
    D(delta) to differential-flow scalarization D(delta, v), where
    v_t = norm(delta_t) - norm(delta_t-1) is the flow of the tracking-error
    differential.

    For disc_mode "flow"/"radial" the logit is:
        z(delta, v) = f(delta) + q(delta, v)
        q(delta, v) = A(delta)^T v - 0.5 * v^T G(delta) v,  A = G v*
        v*(delta)   = -alpha(delta) * delta + |delta| * P_perp(delta) u(delta)
    with alpha > 0 (softplus) and G diagonal positive definite, so that q is
    strictly concave in v, maximized at the preferred flow v*, and q(delta, 0) = 0.
    """
    def __init__(self, config, env):
        super().__init__(config, env)
        return

    def get_disc_mode(self):
        return self._disc_mode

    def eval_disc(self, disc_obs, disc_flow):
        if (self._disc_mode == DISC_MODE_CONCAT):
            disc_in = torch.cat([disc_obs, disc_flow], dim=-1)
            h = self._disc_layers(disc_in)
            logit = self._disc_logits(h)
        else:
            h = self._disc_layers(disc_obs)
            f = self._disc_logits(h)

            v_star, _, G = self._eval_flow(disc_obs, h)
            # q(delta, v) = (G v*)^T v - 0.5 v^T G v, G diagonal
            # q(delta, 0) = 0 and argmax_v q(delta, v) = v*
            q = torch.sum(G * v_star * disc_flow, dim=-1, keepdim=True) \
                - 0.5 * torch.sum(G * torch.square(disc_flow), dim=-1, keepdim=True)
            logit = f + q
        return logit

    def eval_disc_flow(self, disc_obs):
        assert(self._disc_mode != DISC_MODE_CONCAT)
        h = self._disc_layers(disc_obs)
        v_star, v_star_tan, G = self._eval_flow(disc_obs, h)
        return v_star, v_star_tan, G

    def get_disc_logit_weights(self):
        weights = [torch.flatten(self._disc_logits.weight)]
        if (self._disc_mode != DISC_MODE_CONCAT):
            weights.append(torch.flatten(self._disc_flow_alpha.weight))
            weights.append(torch.flatten(self._disc_flow_metric.weight))
            if (self._disc_mode == DISC_MODE_FLOW):
                weights.append(torch.flatten(self._disc_flow_tangent.weight))
        return torch.cat(weights)

    def get_disc_params(self):
        params = super().get_disc_params()
        if (self._disc_mode != DISC_MODE_CONCAT):
            params += list(self._disc_flow_alpha.parameters())
            params += list(self._disc_flow_metric.parameters())
            if (self._disc_mode == DISC_MODE_FLOW):
                params += list(self._disc_flow_tangent.parameters())
        return params

    def _build_disc(self, config, env):
        self._disc_mode = config.get("disc_mode", DISC_MODE_FLOW)
        assert(self._disc_mode in DISC_MODES), "Unsupported disc_mode: {}".format(self._disc_mode)

        self._disc_flow_g_min = config.get("disc_flow_g_min", 1e-3)
        self._disc_flow_proj_eps = config.get("disc_flow_proj_eps", 1e-6)

        init_output_scale = 1.0
        net_name = config["disc_net"]

        input_dict = self._build_disc_input_dict(env)
        self._disc_layers, layers_info = net_builder.build_net(net_name, input_dict,
                                                               activation=self._activation)

        layers_out_size = torch_util.calc_layers_out_size(self._disc_layers)
        self._disc_logits = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.uniform_(self._disc_logits.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(self._disc_logits.bias)

        if (self._disc_mode != DISC_MODE_CONCAT):
            disc_obs_space = env.get_disc_obs_space()
            diff_dim = int(np.prod(disc_obs_space.shape))

            # zero init: alpha = softplus(0) > 0 gives an inward radial flow prior,
            # G = softplus(0) + g_min is well conditioned, u = 0 disables the
            # tangential flow at init, so training starts close to plain ADD
            self._disc_flow_alpha = torch.nn.Linear(layers_out_size, 1)
            torch.nn.init.zeros_(self._disc_flow_alpha.weight)
            torch.nn.init.zeros_(self._disc_flow_alpha.bias)

            self._disc_flow_metric = torch.nn.Linear(layers_out_size, diff_dim)
            torch.nn.init.zeros_(self._disc_flow_metric.weight)
            torch.nn.init.zeros_(self._disc_flow_metric.bias)

            if (self._disc_mode == DISC_MODE_FLOW):
                self._disc_flow_tangent = torch.nn.Linear(layers_out_size, diff_dim)
                torch.nn.init.zeros_(self._disc_flow_tangent.weight)
                torch.nn.init.zeros_(self._disc_flow_tangent.bias)
        return

    def _build_disc_input_dict(self, env):
        obs_space = env.get_disc_obs_space()
        input_dict = {"disc_obs": obs_space}
        if (self._disc_mode == DISC_MODE_CONCAT):
            input_dict["disc_flow"] = obs_space
        return input_dict

    def _eval_flow(self, disc_obs, h):
        # v*(delta) = -alpha(delta) * delta + |delta| * P_perp(delta) u(delta)
        alpha = torch.nn.functional.softplus(self._disc_flow_alpha(h))
        G = torch.nn.functional.softplus(self._disc_flow_metric(h)) + self._disc_flow_g_min

        v_star_tan = self._eval_tangential_flow(disc_obs, h)
        v_star = -alpha * disc_obs + v_star_tan
        return v_star, v_star_tan, G

    def _eval_tangential_flow(self, disc_obs, h):
        if (self._disc_mode != DISC_MODE_FLOW):
            return torch.zeros_like(disc_obs)

        u = self._disc_flow_tangent(h)
        delta_sq = torch.sum(torch.square(disc_obs), dim=-1, keepdim=True)
        # P_perp(delta) u = u - delta (delta^T u) / (|delta|^2 + eps)
        proj = torch.sum(disc_obs * u, dim=-1, keepdim=True) / (delta_sq + self._disc_flow_proj_eps)
        u_perp = u - proj * disc_obs
        # eps inside the sqrt keeps the gradient finite at delta = 0, which is
        # required for the gradient penalty on the ideal point (delta, v) = (0, 0)
        v_star_tan = torch.sqrt(delta_sq + self._disc_flow_proj_eps) * u_perp
        return v_star_tan
