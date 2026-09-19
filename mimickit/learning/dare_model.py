import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder


def _orthonormalize(weight):
    """Project ``weight`` onto the (semi-)orthogonal group via the polar factor.

    Tall matrices (rows >= cols) become column-orthonormal, W^T W = I, i.e. an
    isometric embedding of R^cols.  Square/wide matrices become row-orthonormal,
    W W^T = I, a partial isometry that discards only its null space.

    This replaces spectral normalization as the constraint on the hidden
    discriminator: SN only pins the largest singular value and leaves the rest
    of the spectrum free to shrink (measured sigma_max = 1.0 with a 0.6 plateau
    on the trained Climb trunk, i.e. RMS gain 0.37).

    The polar factor is taken from an SVD rather than a Newton-Schulz iteration.
    Newton-Schulz was tried first and is not usable here: after rescaling by
    sigma_max, small singular values converge only linearly (sigma <- sigma *
    (3 - sigma^2) / 2 grows a 0.005 direction by ~1.5x per step), so five steps
    left the 1022x1022 trunk with sigma_min = 0.005 and ||W^T W - I||_F = 9.9.
    The SVD is exact and costs ~1 ms on the 1022x1022 trunk, against ~4.3 s for
    a whole training iteration.
    """
    if weight.shape[0] < weight.shape[1]:
        return _orthonormalize(weight.transpose(0, 1)).transpose(0, 1)
    left, _, right = torch.linalg.svd(weight, full_matrices=False)
    return left @ right


class SemiOrthogonal(torch.nn.Module):
    """Parametrization keeping a linear weight (semi-)orthogonal.

    Registered through ``torch.nn.utils.parametrize`` so the raw parameter stays
    unconstrained while every forward pass sees an exactly (semi-)orthogonal
    operator.  Together with GroupSort this makes the hidden discriminator
    norm-preserving; neither spectral normalization nor ReLU provides that.
    """

    def forward(self, weight):
        return _orthonormalize(weight)


class GroupSort2(torch.nn.Module):
    """GroupSort(2): sorts each adjacent pair ascending.

    Its Jacobian is a permutation matrix (identity or swap, no sign flips), so
    the activation is an exact isometry, unlike ReLU which zeroes roughly half
    the units and is where most of the hidden contraction comes from.
    """

    def __init__(self, width=None):
        super().__init__()
        if width is not None and width % 2 != 0:
            raise ValueError(
                "GroupSort(2) requires an even width, got {}".format(width))

    def forward(self, inputs):
        low = torch.minimum(inputs[..., 0::2], inputs[..., 1::2])
        high = torch.maximum(inputs[..., 0::2], inputs[..., 1::2])
        return torch.stack((low, high), dim=-1).flatten(-2)


class GroupSeparableDiscLayers(torch.nn.Module):
    """Group-separable discriminator backbone.

    ``geometry="semi_orthogonal"`` (default) uses the isometry-preserving
    construction: semi-orthogonal linears, GroupSort(2) activations and a
    *square* trunk of width ``group_width * num_groups``.

    The square trunk is essential, not cosmetic.  A rectangular trunk
    R^1022 -> R^w is a partial isometry whose row space meets the (172
    dimensional) encoder image in about w / 1022 of its dimensions, which caps
    the hidden isometry at w / 1022 regardless of how orthogonal it is.
    Measured with random semi-orthogonal weights: w=512 -> 0.500, w=640 ->
    0.625, w=1022 -> 1.000.

    ``geometry="full_sn"`` reproduces the historical spectral-norm + ReLU
    backbone with the configured trunk widths, so the existing ablations stay
    reproducible.
    """

    def __init__(self, groups, first_width, trunk_widths, activation,
                 geometry="semi_orthogonal"):
        super().__init__()
        num_groups = len(groups)
        self.geometry = geometry
        self.group_width = first_width // num_groups
        if self.group_width < 1:
            raise ValueError(
                "Discriminator width must cover every error group")
        if geometry == "semi_orthogonal" and self.group_width % 2 != 0:
            raise ValueError(
                "GroupSort(2) needs an even per-group width, got {}"
                .format(self.group_width))

        self.encoders = torch.nn.ModuleList()
        for group_id, (_, indices) in enumerate(groups):
            self.register_buffer(
                "group_indices_{}".format(group_id),
                torch.tensor(indices, dtype=torch.long))
            layer = self._build_linear(len(indices), self.group_width)
            self.encoders.append(torch.nn.Sequential(
                layer, self._build_activation(activation)))

        self.total_width = self.group_width * num_groups
        if geometry == "semi_orthogonal":
            # Square trunk: keeps every direction of the encoder image.
            trunk_widths = [self.total_width]
        trunk = []
        in_size = self.total_width
        for out_size in trunk_widths:
            trunk.append(self._build_linear(in_size, out_size))
            trunk.append(self._build_activation(activation))
            in_size = out_size
        self.trunk = torch.nn.Sequential(*trunk)
        self.out_features = in_size

    def _build_linear(self, in_features, out_features):
        layer = torch.nn.Linear(in_features, out_features)
        torch.nn.init.zeros_(layer.bias)
        if self.geometry == "semi_orthogonal":
            torch.nn.utils.parametrize.register_parametrization(
                layer, "weight", SemiOrthogonal())
        else:
            torch.nn.utils.parametrizations.spectral_norm(layer)
        return layer

    def _build_activation(self, activation):
        if self.geometry == "semi_orthogonal":
            return GroupSort2()
        return activation()

    def forward(self, inputs):
        encoded = []
        for group_id, encoder in enumerate(self.encoders):
            indices = getattr(
                self, "group_indices_{}".format(group_id))
            group_input = torch.index_select(inputs, -1, indices)
            encoded.append(encoder(group_input))
        return self.trunk(torch.cat(encoded, dim=-1))


class DAREModel(add_model.ADDModel):
    """DARE critic with the original calibrated classifier output."""

    def __init__(self, config, env):
        super().__init__(config, env)
        self.register_buffer("_disc_logit_scale", torch.ones(()))
        self.register_buffer("_disc_logit_center", torch.zeros(()))
        self.register_buffer(
            "_disc_logit_calibrated", torch.zeros((), dtype=torch.bool))

    def _build_disc(self, config, env):
        input_dict = {"disc_obs": env.get_disc_obs_space()}
        base_layers, _ = net_builder.build_net(
            config["disc_net"], input_dict, activation=self._activation)
        linears = [layer for layer in base_layers.modules()
                   if isinstance(layer, torch.nn.Linear)]
        if len(linears) < 2:
            raise ValueError(
                "DARE requires a shared discriminator trunk")

        self._disc_group_embedding = bool(
            config.get("disc_group_embedding", True))
        self._disc_geometry = config.get(
            "disc_hidden_geometry", "semi_orthogonal")
        if self._disc_group_embedding:
            self._disc_layers = GroupSeparableDiscLayers(
                groups=env.get_disc_error_groups(),
                first_width=linears[0].out_features,
                trunk_widths=[layer.out_features for layer in linears[1:]],
                activation=self._activation,
                geometry=self._disc_geometry)
            disc_out = self._disc_layers.out_features
        else:
            # The ablation retains DARE's Full-SN critic and all subsequent
            # training machinery, but replaces the semantic direct sum by the
            # ordinary dense first layer used by a flat discriminator.
            self._disc_layers = base_layers
            for layer in linears:
                torch.nn.utils.parametrizations.spectral_norm(layer)
            disc_out = linears[-1].out_features

        # The output map stays unit-L2-norm in both geometries.  Spectral
        # normalization of a (1, n) layer pins ||w||_2 = 1 exactly, which is the
        # same constraint the new geometry applies to its hidden layers.
        self._disc_logits = torch.nn.Linear(disc_out, 1, bias=True)
        torch.nn.init.uniform_(self._disc_logits.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self._disc_logits.bias)
        torch.nn.utils.parametrizations.spectral_norm(self._disc_logits)

    def eval_disc_hidden(self, disc_obs):
        """Hidden discriminator representation h(x), used for diagnostics."""
        return self._disc_layers(disc_obs)

    def eval_disc_raw(self, disc_obs):
        return self._disc_logits(self._disc_layers(disc_obs))

    def get_disc_hidden_geometry(self):
        return self._disc_geometry

    def eval_disc(self, disc_obs):
        # One-shot affine logit standardization: the calibrated logit is
        # centered on the softplus transition region and unit-scaled by the
        # balanced calibration spread, i.e. z = (f - c_f) / s_f.
        return self._disc_logit_scale * (
            self.eval_disc_raw(disc_obs) - self._disc_logit_center)

    @torch.no_grad()
    def set_disc_logit_calibration(self, center, scale):
        """Set the affine calibration (center c_f, kappa = 1 / s_f)."""
        self._disc_logit_center.fill_(float(center))
        self._disc_logit_scale.fill_(float(scale))
        self._disc_logit_calibrated.fill_(True)

    @torch.no_grad()
    def set_disc_logit_scale(self, scale):
        # Legacy scale-only entry point (zero center), kept for compatibility.
        self._disc_logit_center.fill_(0.0)
        self._disc_logit_scale.fill_(float(scale))
        self._disc_logit_calibrated.fill_(True)

    def is_disc_logit_calibrated(self):
        return bool(self._disc_logit_calibrated.item())

    def get_disc_logit_scale(self):
        return self._disc_logit_scale.clone()

    def get_disc_logit_center(self):
        return self._disc_logit_center.clone()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Legacy v6 model/checkpoint state dicts have no calibration buffers.
        state_dict.setdefault(prefix + "_disc_logit_scale", torch.ones(()))
        state_dict.setdefault(prefix + "_disc_logit_center", torch.zeros(()))
        state_dict.setdefault(
            prefix + "_disc_logit_calibrated",
            torch.zeros((), dtype=torch.bool))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def get_disc_group_width(self):
        if not self._disc_group_embedding:
            return self._disc_logits.weight.new_zeros(())
        return self._disc_logits.weight.new_tensor(
            float(self._disc_layers.group_width))

    def get_disc_group_total_width(self):
        if not self._disc_group_embedding:
            return self._disc_logits.weight.new_zeros(())
        return self._disc_logits.weight.new_tensor(
            float(self._disc_layers.total_width))

    def uses_disc_group_embedding(self):
        return self._disc_group_embedding
