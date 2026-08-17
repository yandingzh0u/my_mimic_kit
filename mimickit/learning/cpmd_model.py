"""Fixed-budget, reference-conditioned differential metric for CPMD."""

import torch

import learning.add_model as add_model


class CPMDModel(add_model.ADDModel):
    """ADD actor/critic with a structured contextual discriminator.

    The discriminator receives a normalized ADD differential ``delta`` and an
    intrinsic reference context ``context``.  Context can only redistribute a
    fixed metric budget; it cannot produce a logit on its own::

        z(delta, c) = b - m0 ||delta||^2
            - kappa ||V(c) delta||^2 / (||V(c)||_F^2 + eps).

    Consequently ``z(0, c) == b`` for every context, while ``m0 > 0`` keeps a
    nonzero cost on every differential coordinate.
    """

    SCHEMA_VERSION = 2

    def _build_disc(self, config, env):
        self._disc_dim = int(env.get_disc_obs_space().shape[0])
        self._context_dim = int(env.get_cpmd_context_dim())

        self._metric_rank = int(config.get("metric_rank", 8))
        self._metric_base_weight = float(
            config.get("metric_base_weight", 0.01))
        self._metric_context_budget = float(
            config.get("metric_context_budget", 1.0))
        self._metric_norm_eps = float(config.get("metric_norm_eps", 1e-8))
        context_hidden = config.get("metric_context_hidden", [256, 128])
        self._cpmd_schema_version = int(
            config.get("cpmd_schema_version", self.SCHEMA_VERSION))

        if self._disc_dim <= 0:
            raise ValueError("CPMD differential dimension must be positive")
        if self._context_dim <= 0:
            raise ValueError("CPMD context dimension must be positive")
        if not 0 < self._metric_rank <= self._disc_dim:
            raise ValueError(
                "metric_rank must be in [1, differential dimension]")
        if self._metric_base_weight <= 0.0:
            raise ValueError("metric_base_weight must be strictly positive")
        if self._metric_context_budget < 0.0:
            raise ValueError("metric_context_budget must be nonnegative")
        if self._metric_norm_eps <= 0.0:
            raise ValueError("metric_norm_eps must be strictly positive")
        if self._cpmd_schema_version != self.SCHEMA_VERSION:
            raise ValueError(
                "Unsupported CPMD schema version: expected {}, got {}".format(
                    self.SCHEMA_VERSION, self._cpmd_schema_version))

        if hasattr(env, "get_cpmd_schema_version"):
            env_schema = int(env.get_cpmd_schema_version())
            if env_schema != self.SCHEMA_VERSION:
                raise ValueError(
                    "CPMD model/environment schema mismatch: model {}, env {}"
                    .format(self.SCHEMA_VERSION, env_schema))

        if not isinstance(context_hidden, (list, tuple)):
            raise TypeError("metric_context_hidden must be a list or tuple")
        hidden_dims = [int(width) for width in context_hidden]
        if any(width <= 0 for width in hidden_dims):
            raise ValueError("metric_context_hidden widths must be positive")
        self._metric_context_hidden = tuple(hidden_dims)

        layer_dims = [self._context_dim] + hidden_dims
        layers = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            linear = torch.nn.Linear(in_dim, out_dim)
            torch.nn.init.xavier_uniform_(linear.weight)
            torch.nn.init.zeros_(linear.bias)
            layers.extend([linear, self._activation()])

        output_in_dim = layer_dims[-1]
        output_dim = self._metric_rank * self._disc_dim
        output = torch.nn.Linear(output_in_dim, output_dim)
        torch.nn.init.xavier_uniform_(output.weight)
        # A nonzero output bias ensures V(c) is nonzero even for a zero
        # context, avoiding the zero-gradient point of the normalized energy.
        torch.nn.init.normal_(output.bias, mean=0.0, std=1e-2)
        layers.append(output)
        self._metric_context_net = torch.nn.Sequential(*layers)

        self._disc_bias = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "_cpmd_schema",
            torch.tensor(self.SCHEMA_VERSION, dtype=torch.int64),
        )
        return

    def eval_disc(self, disc_obs, context):
        """Evaluate the contextual differential logit."""
        return self.eval_metric_terms(disc_obs, context)["logit"]

    def eval_metric_terms(self, delta, context):
        """Evaluate the logit and expose the metric's diagnostic terms.

        All scalar quantities retain a final singleton dimension.
        ``metric_diag`` contains the diagonal of the normalized low-rank
        matrix A(c), which makes its allocation across differential
        coordinates inspectable without ever materializing a full ``D x D``
        matrix.
        """
        if delta.shape[:-1] != context.shape[:-1]:
            raise ValueError(
                "delta and context must have identical leading dimensions")
        if delta.shape[-1] != self._disc_dim:
            raise ValueError(
                "Expected differential dimension {}, got {}".format(
                    self._disc_dim, delta.shape[-1]))
        if context.shape[-1] != self._context_dim:
            raise ValueError(
                "Expected context dimension {}, got {}".format(
                    self._context_dim, context.shape[-1]))

        v = self._metric_context_net(context)
        v = v.reshape(*context.shape[:-1], self._metric_rank, self._disc_dim)

        projected = torch.matmul(v, delta.unsqueeze(-1)).squeeze(-1)
        v_norm_sq = torch.sum(torch.square(v), dim=(-2, -1)).unsqueeze(-1)
        denom = v_norm_sq + self._metric_norm_eps

        base = self._metric_base_weight * torch.sum(
            torch.square(delta), dim=-1, keepdim=True)
        contextual = self._metric_context_budget * torch.sum(
            torch.square(projected), dim=-1, keepdim=True) / denom
        logit = self._disc_bias - base - contextual

        diag = torch.sum(torch.square(v), dim=-2) / denom
        trace = torch.sum(diag, dim=-1, keepdim=True)
        v_norm = torch.sqrt(v_norm_sq)

        return {
            "logit": logit,
            "base_energy": base,
            "context_energy": contextual,
            "v_norm_sq": v_norm_sq,
            "v_norm": v_norm,
            "trace": trace,
            "metric_diag": diag,
        }

    def eval_zero_logit(self, batch_size, device, dtype):
        """Return the context-independent positive logit ``z(0, c) = b``."""
        if batch_size < 0:
            raise ValueError("batch_size must be nonnegative")
        bias = self._disc_bias.to(device=device, dtype=dtype)
        return bias.reshape(1, 1).expand(int(batch_size), 1)

    def get_disc_params(self):
        return list(self._metric_context_net.parameters()) + [self._disc_bias]

    def get_disc_logit_weights(self):
        return torch.flatten(self._disc_bias)

    def get_disc_dim(self):
        return self._disc_dim

    def get_delta_dim(self):
        return self._disc_dim

    def get_context_dim(self):
        return self._context_dim

    def get_metric_rank(self):
        return self._metric_rank

    def get_metric_base_weight(self):
        return self._metric_base_weight

    def get_metric_context_budget(self):
        return self._metric_context_budget

    def get_metric_norm_eps(self):
        return self._metric_norm_eps

    def get_metric_context_hidden(self):
        return self._metric_context_hidden

    def get_metric_context_net(self):
        return self._metric_context_net

    def get_cpmd_schema_version(self):
        return self._cpmd_schema_version
