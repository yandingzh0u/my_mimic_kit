import gymnasium.spaces as spaces
import numpy as np
import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util


class PhaseTransitionCriticModel(add_model.ADDModel):
    """ADD actor/critic with a conditional one-step transition critic.

    The critic receives two normalized blocks:

    ``u = [(ref_next_state - sim_next_state) / state_scale,
           (ref_motion - sim_motion) / motion_scale]``

    and a phase-matched reference context

    ``c = [(ref_state - state_mean) / state_scale,
           (ref_motion - motion_mean) / motion_scale]``.

    The public score is unconstrained.  Anchoring, the Wasserstein objective,
    and the actor reward are defined by the agent from score differences
    ``F(u, c) - F(0, c)``.
    """

    def _build_disc(self, config, env):
        self._transition_dim = 2 * int(env.get_disc_obs_space().shape[0])
        critic_input_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * self._transition_dim,),
            dtype=np.float32,
        )
        input_dict = {"transition_input": critic_input_space}
        self._disc_layers, _ = net_builder.build_net(
            config["disc_net"], input_dict, activation=self._activation)

        output_size = torch_util.calc_layers_out_size(self._disc_layers)
        self._disc_logits = torch.nn.Linear(output_size, 1)
        init_scale = float(config.get("transition_init_output_scale", 0.01))
        if init_scale <= 0:
            raise ValueError("transition_init_output_scale must be positive")
        torch.nn.init.uniform_(
            self._disc_logits.weight, -init_scale, init_scale)
        torch.nn.init.zeros_(self._disc_logits.bias)

    @property
    def transition_dim(self):
        return self._transition_dim

    def eval_transition_score(self, transition_error, reference_context):
        """Evaluate ``F(u, c)`` for identically shaped ``2D`` blocks."""
        if transition_error.shape != reference_context.shape:
            raise ValueError(
                "transition error and context must have identical shapes")
        if transition_error.shape[-1] != self._transition_dim:
            raise ValueError(
                "expected transition blocks of size {}, got {}".format(
                    self._transition_dim, transition_error.shape[-1]))
        critic_input = torch.cat(
            [transition_error, reference_context], dim=-1)
        hidden = self._disc_layers(critic_input)
        return self._disc_logits(hidden)

    def eval_anchored_score(self, transition_error, reference_context):
        """Evaluate ``A(u,c)=F(u,c)-F(0,c)``."""
        score = self.eval_transition_score(
            transition_error, reference_context)
        zero_score = self.eval_transition_score(
            torch.zeros_like(transition_error), reference_context)
        return score - zero_score

    def eval_disc(self, transition_error, reference_context=None):
        """Compatibility alias; this critic always requires its context."""
        if reference_context is None:
            raise ValueError(
                "phase transition critic requires a paired reference context")
        return self.eval_transition_score(
            transition_error, reference_context)

    def get_disc_logit_weights(self):
        return torch.flatten(self._disc_logits.weight)

    def get_disc_params(self):
        return (
            list(self._disc_layers.parameters())
            + list(self._disc_logits.parameters())
        )
