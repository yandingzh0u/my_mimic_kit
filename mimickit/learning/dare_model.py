import torch
import torch.nn.functional as F

import learning.add_model as add_model
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


class LearnableCReLUStem(torch.nn.Module):
    """Learnable column-orthogonal lift with paired signed responses.

    A local seed initializes from the completed fixed-CReLU experiment's
    lift without consuming the policy/discriminator random stream.
    Householder parametrization preserves column orthogonality as the
    projection directions learn.  CReLU retains both signs and preserves
    distance to zero; it does not preserve every pairwise distance.
    """

    def __init__(self, input_dim, output_width=1022):
        super().__init__()
        if output_width < 2 or output_width % 2:
            raise ValueError("CReLU output width must be positive and even")
        lift_width = output_width // 2
        if input_dim < 1 or input_dim > lift_width:
            raise ValueError(
                "The isometric lift requires input_dim <= output_width / 2")
        self.output_width = output_width
        self.lift_width = lift_width
        generator = torch.Generator(device="cpu").manual_seed(0)
        raw = torch.randn(lift_width, input_dim, generator=generator,
                          device="cpu", dtype=torch.float64)
        lift, _ = torch.linalg.qr(raw, mode="reduced")
        self.lift = torch.nn.Parameter(lift.to(dtype=torch.get_default_dtype()))
        torch.nn.utils.parametrizations.orthogonal(
            self, "lift", orthogonal_map="householder",
            use_trivialization=False)

    def forward(self, inputs):
        projected = F.linear(inputs, self.lift)
        return torch.cat((F.relu(projected), F.relu(-projected)), dim=-1)


class GroupFreeDiscLayers(torch.nn.Module):
    """Learnable semiorthogonal CReLU and the original fusion/CPL trunk."""

    def __init__(self, input_dim, trunk_widths, activation):
        super().__init__()
        self.stem = LearnableCReLUStem(input_dim, output_width=1022)

        trunk = []
        in_size = self.stem.output_width
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
        return self.trunk(self.stem(inputs))


class DAREModel(add_model.ADDModel):
    """Group-free learnable CReLU followed by DARE's CPL discriminator."""

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

        self._disc_layers = GroupFreeDiscLayers(
            input_dim=linears[0].in_features,
            trunk_widths=[layer.out_features for layer in linears[1:]],
            activation=self._activation)
        disc_out = self._disc_layers.out_features

        self._disc_logits = torch.nn.Linear(disc_out, 1, bias=True)
        torch.nn.init.uniform_(self._disc_logits.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self._disc_logits.bias)
        torch.nn.utils.parametrizations.spectral_norm(self._disc_logits)
