import torch
import torch.nn.functional as F

import learning.add_model as add_model


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


class CPLDiscLayers(torch.nn.Module):
    """Project the full differential directly into the CPL trunk."""

    def __init__(self, input_dim, hidden_sizes, activation):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("CPL trunk requires at least one fusion layer")
        in_size = hidden_sizes[0]
        trunk = [self._build_linear(input_dim, in_size), activation()]
        for out_size in hidden_sizes[1:]:
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
        return self.trunk(inputs)


class DAREModel(add_model.ADDModel):
    """A global differential projection followed by convex-potential blocks."""

    def _build_disc(self, config, env):
        obs_shape = env.get_disc_obs_space().shape
        if len(obs_shape) != 1:
            raise ValueError("DARE requires a vector differential input")
        self._disc_layers = CPLDiscLayers(
            input_dim=obs_shape[0],
            hidden_sizes=config["disc_hidden_sizes"],
            activation=self._activation)
        disc_out = self._disc_layers.out_features

        self._disc_logits = torch.nn.Linear(disc_out, 1, bias=True)
        torch.nn.init.uniform_(self._disc_logits.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self._disc_logits.bias)
        torch.nn.utils.parametrizations.spectral_norm(self._disc_logits)

    def eval_disc_raw(self, disc_obs):
        return self._disc_logits(self._disc_layers(disc_obs))
