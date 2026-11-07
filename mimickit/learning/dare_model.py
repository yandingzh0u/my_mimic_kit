import torch
import torch.nn.functional as F

import learning.add_model as add_model
from learning.coordinate_disc import CoordinateInteractionEncoder
import learning.nets.net_builder as net_builder


class ConvexPotentialBlock(torch.nn.Module):
    """A square 1-Lipschitz convex-potential residual block.

    For a spectrally-normalized W, the map
        x -> x - 2 W^T ReLU(Wx + b)
    is the CPL layer used for the deep-trunk stress test.  The transpose is
    tied to the same W; no extra projection or regularizer is introduced.
    """

    def __init__(self, width, activation):
        super().__init__()
        linear = torch.nn.Linear(width, width)
        torch.nn.init.zeros_(linear.bias)
        self.linear = torch.nn.utils.parametrizations.spectral_norm(linear)
        self.activation = activation()

    def forward(self, inputs):
        # Read the parametrized weight once so one forward does not advance
        # spectral-normalization power iteration twice.
        weight = self.linear.weight
        pre = F.linear(inputs, weight, self.linear.bias)
        return inputs - 2.0 * F.linear(self.activation(pre), weight.transpose(-1, -2))


class CoordinateDiscLayers(torch.nn.Module):
    """Coordinate tokens, one similarity interaction, and the original CPL trunk."""

    def __init__(self, input_dim, trunk_widths, activation):
        super().__init__()
        self.frontend = CoordinateInteractionEncoder(input_dim, token_dim=6)
        self.token_dim = self.frontend.token_dim
        self.total_width = self.frontend.total_width

        trunk = []
        in_size = self.total_width
        if not trunk_widths:
            raise ValueError("CPL trunk requires at least one fusion layer")
        fusion_width = trunk_widths[0]
        trunk.extend((self._build_linear(in_size, fusion_width),
                      activation()))
        in_size = fusion_width
        for out_size in trunk_widths[1:]:
            if out_size != in_size:
                raise ValueError(
                    "CPL requires square hidden trunk layers, got "
                    "{} -> {}".format(in_size, out_size))
            trunk.append(ConvexPotentialBlock(in_size, activation))
        self.trunk = torch.nn.Sequential(*trunk)
        self.out_features = in_size

    def _build_linear(self, in_features, out_features):
        layer = torch.nn.Linear(in_features, out_features)
        torch.nn.init.zeros_(layer.bias)
        return torch.nn.utils.parametrizations.spectral_norm(layer)

    def forward(self, inputs):
        return self.trunk(self.frontend(inputs))


class DAREModel(add_model.ADDModel):
    """Group-free coordinate interaction followed by DARE's CPL discriminator."""

    def _build_disc(self, config, env):
        input_dict = {"disc_obs": env.get_disc_obs_space()}
        # Preserve the historical RNG draws before constructing the actual
        # discriminator; skipping this build changes every seeded weight.
        base_layers, _ = net_builder.build_net(
            config["disc_net"], input_dict, activation=self._activation)
        linears = [layer for layer in base_layers.modules()
                   if isinstance(layer, torch.nn.Linear)]
        if len(linears) < 2:
            raise ValueError(
                "DARE requires a shared discriminator trunk")

        self._disc_layers = CoordinateDiscLayers(
            input_dim=env.get_disc_obs_space().shape[0],
            trunk_widths=[layer.out_features for layer in linears[1:]],
            activation=self._activation)
        disc_out = self._disc_layers.out_features

        self._disc_logits = torch.nn.Linear(disc_out, 1, bias=True)
        torch.nn.init.uniform_(self._disc_logits.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self._disc_logits.bias)
        torch.nn.utils.parametrizations.spectral_norm(self._disc_logits)
