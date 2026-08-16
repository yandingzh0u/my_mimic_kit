"""Isolated ADD tracker with a paired temporal-context critic."""

import gymnasium.spaces as spaces
import numpy as np
import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util


class CPMDConditionalModel(add_model.ADDModel):
    """Keep the stock ADD discriminator and add a disjoint context critic.

    The inherited ``_disc_layers`` and ``_disc_logits`` are the complete
    172-D ADD branch.  The context critic only sees the motion-error memory
    and phase-matched reference context.  Its output head starts at zero, so
    the initial context veto is exactly inactive without changing ADD.
    """

    SCHEMA_VERSION = 2

    def _build_nets(self, config, env):
        super()._build_nets(config, env)
        self._build_context_critic(config, env)
        return

    def _build_context_critic(self, config, env):
        self._error_memory_dim = int(env.get_cpmd_error_dim())
        self._context_dim = int(env.get_cpmd_context_dim())
        self._context_input_dim = self._error_memory_dim + self._context_dim

        input_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=[self._context_input_dim],
            dtype=np.float32,
        )
        net_name = config.get("context_net", config["disc_net"])
        self._context_layers, _ = net_builder.build_net(
            net_name,
            {"context_obs": input_space},
            activation=self._activation,
        )
        output_dim = torch_util.calc_layers_out_size(self._context_layers)
        self._context_logits = torch.nn.Linear(output_dim, 1)
        torch.nn.init.zeros_(self._context_logits.weight)
        torch.nn.init.zeros_(self._context_logits.bias)

        self.register_buffer(
            "_cpmd_cond_schema",
            torch.tensor(self.SCHEMA_VERSION, dtype=torch.int64),
        )

        base_ids = {id(p) for p in self.get_disc_params()}
        context_ids = {id(p) for p in self.get_context_params()}
        assert base_ids.isdisjoint(context_ids)
        return

    def eval_context(self, error_memory, ref_context):
        assert error_memory.shape[-1] == self._error_memory_dim
        assert ref_context.shape[-1] == self._context_dim
        context_input = torch.cat([error_memory, ref_context], dim=-1)
        hidden = self._context_layers(context_input)
        return self._context_logits(hidden)

    def get_context_params(self):
        return (list(self._context_layers.parameters())
                + list(self._context_logits.parameters()))

    def get_context_logit_weights(self):
        return torch.flatten(self._context_logits.weight)

    def get_error_memory_dim(self):
        return self._error_memory_dim

    def get_context_dim(self):
        return self._context_dim

    def get_context_input_dim(self):
        return self._context_input_dim

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        schema_key = prefix + "_cpmd_cond_schema"
        if schema_key in state_dict:
            schema = int(state_dict[schema_key].item())
            if schema != self.SCHEMA_VERSION:
                error_msgs.append(
                    "CPMD conditional checkpoint schema {} is incompatible "
                    "with schema {}".format(schema, self.SCHEMA_VERSION))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
