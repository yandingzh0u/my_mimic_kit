"""Conditional discriminator initialized exactly from the ADD discriminator."""

import torch

import learning.add_model as add_model


class CPMDConditionalModel(add_model.ADDModel):
    """Score a paired error and reference context with one discriminator.

    The stock ADD discriminator is built first.  Its first layer is then
    widened by the error-memory and context dimensions.  Existing columns and
    every downstream parameter are copied unchanged; new columns are exactly
    zero.  Consequently the initial conditional discriminator is numerically
    identical to ADD for every input.
    """

    SCHEMA_VERSION = 1

    def _build_nets(self, config, env):
        super()._build_nets(config, env)
        self._expand_disc_input(env)
        return

    def _expand_disc_input(self, env):
        self._disc_state_dim = int(env.get_disc_state_obs_dim())
        self._error_memory_dim = int(env.get_cpmd_error_dim())
        self._context_dim = int(env.get_cpmd_context_dim())
        self._error_dim = self._disc_state_dim + self._error_memory_dim
        self._conditional_dim = self._error_dim + self._context_dim

        first = self._disc_layers[0]
        assert isinstance(first, torch.nn.Linear)
        assert first.in_features == self._disc_state_dim

        expanded = torch.nn.Linear(
            self._conditional_dim,
            first.out_features,
            bias=first.bias is not None,
            device=first.weight.device,
            dtype=first.weight.dtype,
        )
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :self._disc_state_dim].copy_(first.weight)
            if first.bias is not None:
                expanded.bias.copy_(first.bias)
        self._disc_layers[0] = expanded
        self.register_buffer(
            "_cpmd_cond_schema",
            torch.tensor(self.SCHEMA_VERSION, dtype=torch.int64),
        )
        return

    def eval_cond(self, error_obs, ref_context):
        assert error_obs.shape[-1] == self._error_dim
        assert ref_context.shape[-1] == self._context_dim
        disc_input = torch.cat([error_obs, ref_context], dim=-1)
        hidden = self._disc_layers(disc_input)
        return self._disc_logits(hidden)

    def eval_disc(self, disc_obs):
        """Evaluate the ADD specialization with zero added coordinates."""
        assert disc_obs.shape[-1] == self._disc_state_dim
        shape = list(disc_obs.shape[:-1])
        error_memory = torch.zeros(
            shape + [self._error_memory_dim],
            dtype=disc_obs.dtype,
            device=disc_obs.device,
        )
        context = torch.zeros(
            shape + [self._context_dim],
            dtype=disc_obs.dtype,
            device=disc_obs.device,
        )
        return self.eval_cond(
            torch.cat([disc_obs, error_memory], dim=-1), context)

    def get_disc_state_dim(self):
        return self._disc_state_dim

    def get_error_memory_dim(self):
        return self._error_memory_dim

    def get_context_dim(self):
        return self._context_dim

    def get_error_dim(self):
        return self._error_dim

    def get_conditional_dim(self):
        return self._conditional_dim

    def get_added_input_weights(self):
        return self._disc_layers[0].weight[:, self._disc_state_dim:]

    def get_error_memory_input_weights(self):
        start = self._disc_state_dim
        end = start + self._error_memory_dim
        return self._disc_layers[0].weight[:, start:end]

    def get_context_input_weights(self):
        return self._disc_layers[0].weight[:, self._error_dim:]
