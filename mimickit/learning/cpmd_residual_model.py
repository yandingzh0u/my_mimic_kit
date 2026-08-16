import torch

import learning.add_model as add_model


class CPMDResidualModel(add_model.ADDModel):
    """ADD discriminator with a bounded low-rank contextual residual.

    The inherited discriminator consumes only the original ADD differential.
    Context is kept in a parameter-disjoint branch so the stock ADD optimizer
    cannot update it accidentally.
    """

    def _build_nets(self, config, env):
        super()._build_nets(config, env)
        self._build_context_residual(config, env)
        return

    def _build_context_residual(self, config, env):
        history_dim = env.get_cpmd_history_dim()
        ref_motion_dim = env.get_cpmd_ref_motion_dim()
        rank = int(config.get("cpmd_rank", 16))
        residual_bound = float(config.get("cpmd_residual_bound", 1.0))

        assert history_dim > 0
        assert ref_motion_dim > 0
        assert rank > 0
        assert residual_bound > 0.0

        self._cpmd_history_dim = history_dim
        self._cpmd_ref_motion_dim = ref_motion_dim
        self._cpmd_rank = rank
        self._cpmd_residual_bound = residual_bound

        # U and V start as usable random features.  Only w is zero initialized:
        # this makes the initial combined logit exactly ADD while preserving a
        # nonzero first-step gradient for w on nonzero contextual inputs.
        self._context_hist_proj = torch.nn.Linear(history_dim, rank, bias=False)
        self._context_ref_proj = torch.nn.Linear(ref_motion_dim, rank, bias=False)
        self._context_logits = torch.nn.Linear(rank, 1, bias=False)

        torch.nn.init.xavier_uniform_(self._context_hist_proj.weight)
        torch.nn.init.xavier_uniform_(self._context_ref_proj.weight)
        torch.nn.init.zeros_(self._context_logits.weight)

        base_param_ids = {id(p) for p in super().get_disc_params()}
        context_param_ids = {id(p) for p in self.get_context_params()}
        assert base_param_ids.isdisjoint(context_param_ids)
        return

    def eval_context(self, hist_err, ref_motion):
        """Return the bounded residual and its unbounded raw logit."""
        assert hist_err.shape[-1] == self._cpmd_history_dim
        assert ref_motion.shape[-1] == self._cpmd_ref_motion_dim

        hist_features = self._context_hist_proj(hist_err)
        ref_features = torch.tanh(self._context_ref_proj(ref_motion))
        interaction = hist_features * ref_features
        raw_residual = self._context_logits(interaction)

        bound = self._cpmd_residual_bound
        bounded_residual = bound * torch.tanh(raw_residual / bound)
        return bounded_residual, raw_residual

    def eval_context_residual(self, hist_err, ref_motion):
        bounded_residual, _ = self.eval_context(hist_err, ref_motion)
        return bounded_residual

    def eval_combined(self, disc_obs, hist_err, ref_motion):
        """Evaluate ADD plus its detached-confidence contextual correction."""
        base_logit = self.eval_disc(disc_obs)
        residual = self.eval_context_residual(hist_err, ref_motion)
        gate = torch.sigmoid(base_logit).detach()
        return base_logit + gate * residual

    def get_context_params(self):
        params = list(self._context_hist_proj.parameters())
        params += list(self._context_ref_proj.parameters())
        params += list(self._context_logits.parameters())
        return params

    def get_context_logit_weights(self):
        return torch.flatten(self._context_logits.weight)

    def get_context_rank(self):
        return self._cpmd_rank

    def get_context_residual_bound(self):
        return self._cpmd_residual_bound

    def get_context_weight_norm(self):
        params = [torch.sum(torch.square(p)) for p in self.get_context_params()]
        return torch.sqrt(torch.sum(torch.stack(params)))

    def get_disc_params(self):
        """Return only stock ADD parameters for the base optimizer."""
        return super().get_disc_params()
