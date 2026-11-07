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


class GroupConvexPotentialEncoder(torch.nn.Module):
    """Zero-anchored group embedding with a bounded residual step.

    Zero padding is an isometry.  For ||W||_2 <= 1 and 0 < step < 1,
    E(x) = Px - step W^T [ReLU(WPx + b) - ReLU(b)]
    has distance bounds [1 - step, 1].  Spectral normalization is a
    numerical approximation; verify effective singular values separately.
    """

    def __init__(self, input_width, output_width, step):
        super().__init__()
        if not 0 < input_width <= output_width:
            raise ValueError("Group input must fit within the encoder width")
        if not 0.0 < step < 1.0:
            raise ValueError("Group encoder step must be strictly between 0 and 1")
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        self.step = float(step)
        linear = torch.nn.Linear(output_width, output_width)
        torch.nn.init.zeros_(linear.bias)
        self.linear = torch.nn.utils.parametrizations.spectral_norm(linear)

    def forward(self, inputs):
        lifted = F.pad(inputs, (0, self.output_width - self.input_width))
        # Reuse the same effective weight for both tied operations.  The
        # zero-anchor term stays in the graph, including its bias gradient.
        weight = self.linear.weight
        responses = F.relu(F.linear(lifted, weight, self.linear.bias))
        responses = responses - F.relu(self.linear.bias)
        return lifted - self.step * F.linear(responses, weight.transpose(-1, -2))


class GroupSeparableDiscLayers(torch.nn.Module):
    """Independent group potential encoders and the original shared CPL trunk."""

    def __init__(self, groups, first_width, trunk_widths, activation, encoder_step):
        super().__init__()
        num_groups = len(groups)
        self.group_width = first_width // num_groups
        if self.group_width < 1:
            raise ValueError("Discriminator width must cover every error group")
        self.total_width = self.group_width * num_groups

        self.encoders = torch.nn.ModuleList()
        for group_id, (_, indices) in enumerate(groups):
            self.register_buffer(
                "group_indices_{}".format(group_id),
                torch.tensor(indices, dtype=torch.long))
            self.encoders.append(GroupConvexPotentialEncoder(
                len(indices), self.group_width, encoder_step))

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
        encoded = []
        for group_id, encoder in enumerate(self.encoders):
            indices = getattr(
                self, "group_indices_{}".format(group_id))
            group_input = torch.index_select(inputs, -1, indices)
            encoded.append(encoder(group_input))
        return self.trunk(torch.cat(encoded, dim=-1))


class DAREModel(add_model.ADDModel):
    """Group potential encoding followed by the original CPL discriminator."""

    def _build_disc(self, config, env):
        input_dict = {"disc_obs": env.get_disc_obs_space()}
        base_layers, _ = net_builder.build_net(
            config["disc_net"], input_dict, activation=self._activation)
        linears = [layer for layer in base_layers.modules()
                   if isinstance(layer, torch.nn.Linear)]
        if len(linears) < 2:
            raise ValueError(
                "DARE requires a shared discriminator trunk")

        self._disc_layers = GroupSeparableDiscLayers(
            groups=env.get_disc_error_groups(),
            first_width=linears[0].out_features,
            trunk_widths=[layer.out_features for layer in linears[1:]],
            activation=self._activation,
            encoder_step=config.get("disc_encoder_step", 0.5))
        disc_out = self._disc_layers.out_features

        self._disc_logits = torch.nn.Linear(disc_out, 1, bias=True)
        torch.nn.init.uniform_(self._disc_logits.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self._disc_logits.bias)
        torch.nn.utils.parametrizations.spectral_norm(self._disc_logits)

    def eval_disc_raw(self, disc_obs):
        return self._disc_logits(self._disc_layers(disc_obs))

    def get_disc_group_width(self):
        return self._disc_logits.weight.new_tensor(
            float(self._disc_layers.group_width))

    def get_disc_group_total_width(self):
        return self._disc_logits.weight.new_tensor(
            float(self._disc_layers.total_width))
