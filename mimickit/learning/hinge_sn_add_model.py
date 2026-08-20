import torch

import learning.add_model as add_model


class HingeSNADDModel(add_model.ADDModel):
    """ADD model with optional spectral normalization on the discriminator.

    The actor and critic are built by the unmodified ADD/PPO model path.  SN is
    registered only after the discriminator has been initialized, and only on
    its Linear modules.
    """

    def __init__(self, config, env, spectral_norm=True,
                 sn_power_iterations=1):
        super().__init__(config, env)

        self._disc_sn_layers = []
        if spectral_norm:
            self._apply_disc_spectral_norm(sn_power_iterations)
        return

    def _apply_disc_spectral_norm(self, power_iterations):
        if power_iterations < 1:
            raise ValueError("disc_sn_power_iterations must be at least 1")

        disc_linear_layers = [
            module for module in self._disc_layers.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        disc_linear_layers.append(self._disc_logits)

        # Registering torch's SN parametrization initializes its private power
        # iteration vectors randomly.  Restore the global streams afterwards
        # so enabling SN does not silently change subsequent policy sampling.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = None
        if torch.cuda.is_available():
            cuda_rng_states = torch.cuda.get_rng_state_all()

        try:
            for layer in disc_linear_layers:
                torch.nn.utils.parametrizations.spectral_norm(
                    layer, name="weight",
                    n_power_iterations=power_iterations)
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

        self._disc_sn_layers = disc_linear_layers
        return

    def get_disc_logit_weights(self):
        """Return the effective output weight without another SN update."""
        if len(self._disc_sn_layers) == 0:
            return super().get_disc_logit_weights()

        parametrization = self._disc_logits.parametrizations.weight[0]
        was_training = parametrization.training
        parametrization.eval()
        try:
            weight = self._disc_logits.weight
        finally:
            parametrization.train(was_training)
        return torch.flatten(weight)

    def get_disc_sn_diagnostics(self):
        """Return cheap raw-weight singular-value estimates from SN's u/v."""
        diagnostics = {}
        with torch.no_grad():
            for layer_idx, layer in enumerate(self._disc_sn_layers):
                parametrizations = layer.parametrizations.weight
                sn = parametrizations[0]
                raw_weight = parametrizations.original
                sigma = torch.dot(sn._u, torch.mv(raw_weight, sn._v)).abs()
                diagnostics["disc_sn_raw_sigma_{}".format(layer_idx)] = sigma
        return diagnostics
