import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder
from learning.semantic_block_sn import SemanticBlockEqualizedLinear


class SBEFSNADDModel(add_model.ADDModel):
    """Dense ADD discriminator with a block-equalized Full-SN first layer."""

    def _build_disc(self, config, env):
        input_dict = {"disc_obs": env.get_disc_obs_space()}
        base_layers, _ = net_builder.build_net(
            config["disc_net"], input_dict, activation=self._activation)
        linears = [layer for layer in base_layers.modules()
                   if isinstance(layer, torch.nn.Linear)]
        if len(linears) < 1:
            raise ValueError("SBE-FSN ADD requires a dense discriminator")

        layers = [
            SemanticBlockEqualizedLinear(
                in_features=linears[0].in_features,
                out_features=linears[0].out_features,
                groups=env.get_disc_error_groups(),
                power_iterations=1,
                bias=True),
            self._activation(),
        ]
        for layer in linears[1:]:
            dense = torch.nn.Linear(
                layer.in_features, layer.out_features, bias=True)
            torch.nn.init.zeros_(dense.bias)
            layers.extend((
                torch.nn.utils.parametrizations.spectral_norm(dense),
                self._activation()))
        self._disc_layers = torch.nn.Sequential(*layers)

        output = torch.nn.Linear(linears[-1].out_features, 1, bias=True)
        torch.nn.init.uniform_(output.weight, -1.0, 1.0)
        torch.nn.init.zeros_(output.bias)
        self._disc_logits = torch.nn.utils.parametrizations.spectral_norm(
            output)

    def get_semantic_layer(self):
        return self._disc_layers[0]

    def get_disc_lipschitz_bound(self):
        return self._disc_logits.bias.new_tensor(1.0)

    def get_disc_semantic_gain_mean(self):
        layer = self.get_semantic_layer()
        return torch.reciprocal(layer.composite_spectral_value())

    def get_disc_semantic_gain_spread(self):
        # Under the parameterization every estimated block norm is exactly
        # the same common composite scale.  Exact-SVD drift is tested offline;
        # do not put seven SVDs in every discriminator minibatch.
        return self._disc_logits.bias.new_zeros(())

    def get_disc_semantic_energy_ratio(self):
        layer = self.get_semantic_layer()
        weight = layer.normalized_weight()
        energies = []
        for group_id in range(layer.num_groups):
            indices = getattr(layer, "group_indices_{}".format(group_id))
            block = torch.index_select(weight, 1, indices)
            energies.append(torch.linalg.matrix_norm(block, ord="fro"))
        energies = torch.stack(energies)
        return energies.max() / energies.min().clamp_min(1e-12)
