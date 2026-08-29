import torch

import learning.amp_model as amp_model


class ScaleSeparatedGraNHead(torch.nn.Module):
    """GraN head whose gain is the regularized output-vector magnitude."""

    EPSILON = 1.0e-8

    def __init__(self, input_size, initial_direction):
        super().__init__()
        direction = initial_direction.detach().reshape(1, input_size)
        direction = direction / torch.linalg.vector_norm(direction)
        self.weight = torch.nn.Parameter(direction)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        return

    def forward(self, features, zero_features, disc_input, create_graph):
        alpha = torch.linalg.vector_norm(self.weight)
        direction = self.weight / torch.clamp_min(alpha, self.EPSILON)

        # ADD's expert differential is exactly zero.  Anchor the learned shape
        # at that point so gradient normalization cannot amplify an arbitrary
        # hidden-layer offset when ||grad h(0)|| is small.
        zero_shape = torch.nn.functional.linear(zero_features, direction)
        raw_shape = (torch.nn.functional.linear(features, direction)
                     - zero_shape)

        input_grad = torch.autograd.grad(
            raw_shape.sum(), disc_input,
            create_graph=create_graph, retain_graph=True,
            only_inputs=True)[0]
        input_grad_norm = torch.linalg.vector_norm(
            input_grad, dim=-1, keepdim=True)
        normalizer = input_grad_norm / (
            torch.square(input_grad_norm) + self.EPSILON)
        normalized_shape = raw_shape * normalizer
        logits = self.bias + alpha * normalized_shape

        local_lipschitz = alpha * torch.square(input_grad_norm) / (
            torch.square(input_grad_norm) + self.EPSILON)
        stats = {
            "alpha": alpha,
            "zero_shape_mean": zero_shape.mean(),
            "raw_shape_mean": raw_shape.mean(),
            "normalized_shape_mean": normalized_shape.mean(),
            "input_grad_norm_mean": input_grad_norm.mean(),
            "input_grad_norm_std": input_grad_norm.std(unbiased=False),
            "normalizer_mean": normalizer.mean(),
            "local_lipschitz_mean": local_lipschitz.mean()
        }
        return logits, stats


class GraNADDModel(amp_model.AMPModel):
    """Direct-differential ADD with a scale-separated GraN logit head."""

    def __init__(self, config, env):
        self._capture_gran_diagnostics = False
        self._gran_diagnostic_index = 0
        self._gran_diagnostics = {}
        super().__init__(config, env)
        return

    def _build_disc(self, config, env):
        super()._build_disc(config, env)
        old_head = self._disc_logits
        input_size = old_head.in_features
        initial_direction = old_head.weight.detach().clone()
        del self._disc_logits
        self._gran_head = ScaleSeparatedGraNHead(
            input_size=input_size,
            initial_direction=initial_direction)
        return

    def eval_disc(self, disc_obs):
        outer_grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            disc_input = disc_obs.detach().requires_grad_(True)
            features = self._disc_layers(disc_input)
            zero_features = self._disc_layers(
                torch.zeros_like(disc_input[:1]))
            logits, stats = self._gran_head(
                features, zero_features, disc_input,
                create_graph=outer_grad_enabled)

        if self._capture_gran_diagnostics:
            label = "pos" if self._gran_diagnostic_index == 0 else "neg"
            for name, value in stats.items():
                if name == "alpha":
                    self._gran_diagnostics["gran_alpha"] = value.detach()
                else:
                    self._gran_diagnostics[
                        "gran_{}_{}".format(label, name)] = value.detach()
            self._gran_diagnostic_index += 1

        if not outer_grad_enabled:
            logits = logits.detach()
        return logits

    def begin_gran_diagnostics(self):
        self._capture_gran_diagnostics = True
        self._gran_diagnostic_index = 0
        self._gran_diagnostics = {}
        return

    def end_gran_diagnostics(self):
        self._capture_gran_diagnostics = False
        return dict(self._gran_diagnostics)

    def get_disc_logit_weights(self):
        return torch.flatten(self._gran_head.weight)

    def get_disc_params(self):
        return (list(self._disc_layers.parameters())
                + list(self._gran_head.parameters()))
