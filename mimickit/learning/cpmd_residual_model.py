"""ADD discriminator with a zero-initialized bilinear motion correction."""

import torch

import learning.add_model as add_model


class CPMDResidualModel(add_model.ADDModel):
    """Stock ADD logit plus linear and symmetric bilinear history terms."""

    def _build_nets(self, config, env):
        super()._build_nets(config, env)
        self._build_context_residual(env)
        return

    def _build_context_residual(self, env):
        motion_dim = env.get_cpmd_motion_dim()
        assert motion_dim > 1
        self._cpmd_motion_dim = motion_dim

        pair_idx = torch.triu_indices(motion_dim, motion_dim, offset=1)
        self.register_buffer("_context_pair_i", pair_idx[0])
        self.register_buffer("_context_pair_j", pair_idx[1])

        # Direct upper-triangle parameters contain no redundant antisymmetric
        # or diagonal directions. Both terms start at exact ADD.
        self._context_linear = torch.nn.Parameter(torch.zeros(motion_dim))
        self._context_bilinear = torch.nn.Parameter(
            torch.zeros(pair_idx.shape[1]))

        base_param_ids = {id(p) for p in super().get_disc_params()}
        context_param_ids = {id(p) for p in self.get_context_params()}
        assert base_param_ids.isdisjoint(context_param_ids)
        return

    def eval_context(self, delta_motion, sum_motion):
        """Evaluate ``u^T h + 1/4 h^T A s`` with symmetric zero-diagonal A."""
        assert delta_motion.shape[-1] == self._cpmd_motion_dim
        assert sum_motion.shape[-1] == self._cpmd_motion_dim

        linear = torch.sum(delta_motion * self._context_linear, dim=-1)
        pair_terms = (
            delta_motion[..., self._context_pair_i]
            * sum_motion[..., self._context_pair_j]
            + delta_motion[..., self._context_pair_j]
            * sum_motion[..., self._context_pair_i]
        )
        bilinear = 0.25 * torch.sum(
            pair_terms * self._context_bilinear, dim=-1)
        total = linear + bilinear
        return total, linear, bilinear

    def eval_context_residual(self, delta_motion, sum_motion):
        total, _, _ = self.eval_context(delta_motion, sum_motion)
        return total

    def eval_combined(self, disc_obs, delta_motion, sum_motion):
        base_logit = self.eval_disc(disc_obs).squeeze(-1)
        return base_logit + self.eval_context_residual(
            delta_motion, sum_motion)

    def get_context_params(self):
        return [self._context_linear, self._context_bilinear]

    def get_context_logit_weights(self):
        return torch.cat([
            torch.flatten(self._context_linear),
            torch.flatten(self._context_bilinear),
        ])

    def get_context_num_pairs(self):
        return self._context_bilinear.numel()

    def get_context_linear_norm(self):
        return torch.linalg.vector_norm(self._context_linear)

    def get_context_bilinear_norm(self):
        return torch.linalg.vector_norm(self._context_bilinear)

    def get_disc_params(self):
        """Return only stock ADD parameters for the base optimizer."""
        return super().get_disc_params()
