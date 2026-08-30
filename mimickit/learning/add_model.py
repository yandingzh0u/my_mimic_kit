import torch

import learning.amp_model as amp_model
import learning.nets.net_builder as net_builder


class SemanticAnchoredEmbedding(torch.nn.Module):
    """Group-factorized, centered, contractive ADD embedding."""

    def __init__(self, groups, first_width, trunk_widths, activation):
        super().__init__()
        num_groups = len(groups)
        self.group_width = first_width // num_groups
        if self.group_width < 1:
            raise ValueError("Discriminator width must cover every error group")

        self.encoders = torch.nn.ModuleList()
        for group_id, (_, indices) in enumerate(groups):
            self.register_buffer(
                "group_indices_{}".format(group_id),
                torch.tensor(indices, dtype=torch.long))
            self.encoders.append(torch.nn.Sequential(
                self._build_linear(len(indices), self.group_width),
                activation()))

        self.total_width = self.group_width * num_groups
        trunk = []
        in_size = self.total_width
        for out_size in trunk_widths:
            trunk.extend((self._build_linear(in_size, out_size), activation()))
            in_size = out_size
        self.trunk = torch.nn.Sequential(*trunk)
        self.out_features = in_size

    @staticmethod
    def _build_linear(in_features, out_features):
        layer = torch.nn.Linear(in_features, out_features)
        torch.nn.init.zeros_(layer.bias)
        torch.nn.utils.parametrizations.spectral_norm(layer)
        return layer

    @staticmethod
    def _centered_forward(module, x):
        """Evaluate x and its zero anchor in one parametrized forward."""
        anchor = torch.zeros(
            (1, x.shape[-1]), device=x.device, dtype=x.dtype)
        values = module(torch.cat((x, anchor), dim=0))
        return values[:-1] - values[-1:]

    def forward(self, diff):
        shape = diff.shape[:-1]
        flat_diff = diff.reshape(-1, diff.shape[-1])

        groups = []
        for group_id, encoder in enumerate(self.encoders):
            indices = getattr(self, "group_indices_{}".format(group_id))
            group_diff = torch.index_select(flat_diff, -1, indices)
            groups.append(self._centered_forward(encoder, group_diff))

        semantic_features = torch.cat(groups, dim=-1)
        embedding = self._centered_forward(self.trunk, semantic_features)
        return embedding.reshape(*shape, self.out_features)


class ADDModel(amp_model.AMPModel):
    """Semantic Anchored Distance ADD discriminator."""

    def _build_disc(self, config, env):
        # Preserve the established network-builder RNG ordering.
        disc_obs_space = env.get_disc_obs_space()
        base_layers, _ = net_builder.build_net(
            config["disc_net"], {"disc_obs": disc_obs_space},
            activation=self._activation)
        linears = [
            layer for layer in base_layers
            if isinstance(layer, torch.nn.Linear)
        ]
        if len(linears) < 2:
            raise ValueError(
                "Semantic Anchored Distance ADD requires a shared trunk")

        self._disc_layers = SemanticAnchoredEmbedding(
            groups=env.get_disc_error_groups(),
            first_width=linears[0].out_features,
            trunk_widths=[layer.out_features for layer in linears[1:]],
            activation=self._activation)
        self._disc_bias = torch.nn.Parameter(torch.zeros(1))

    def eval_disc_embedding(self, diff):
        return self._disc_layers(diff)

    def eval_disc_distance(self, diff):
        embedding = self.eval_disc_embedding(diff)
        return torch.linalg.vector_norm(embedding, ord=2, dim=-1)

    def eval_disc_with_distance(self, diff):
        distance = self.eval_disc_distance(diff)
        logit = (self._disc_bias - distance).unsqueeze(-1)
        return logit, distance

    def eval_disc(self, diff):
        logit, _ = self.eval_disc_with_distance(diff)
        return logit

    def get_disc_params(self):
        return list(self._disc_layers.parameters()) + [self._disc_bias]

    def get_disc_bias(self):
        return self._disc_bias

    def get_disc_group_width(self):
        return self._disc_bias.new_tensor(float(self._disc_layers.group_width))

    def get_disc_group_total_width(self):
        return self._disc_bias.new_tensor(float(self._disc_layers.total_width))
