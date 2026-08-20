import numpy as np
import torch

import learning.add_model as add_model


class ActionPullbackADDModel(add_model.ADDModel):
    """ADD actor/critic/discriminator plus an action-to-feature response model.

    The response model predicts the one-control-step feature increment
    produced by a normalized action.  It deliberately receives only the
    character's self observation; reference error and reference motion are
    excluded so the model cannot explain a desired transition by reading the
    command instead of learning the controlled dynamics.
    """

    def _build_nets(self, config, env):
        super()._build_nets(config, env)
        # Building the auxiliary response network must not change the random
        # stream used by the policy, rollout exploration, or PPO minibatches.
        # This makes pullback_weight=0 a genuine functional control rather
        # than a different random initialization in disguise.
        cpu_rng_state = torch.get_rng_state()
        try:
            self._build_response(config, env)
        finally:
            torch.set_rng_state(cpu_rng_state)

    def _build_response(self, config, env):
        if not hasattr(env, "get_aligned_self_obs_dim"):
            raise ValueError(
                "Action-pullback ADD requires the aligned differential "
                "command environment")

        self._response_self_dim = int(env.get_aligned_self_obs_dim())
        self._response_dim = int(env.get_aligned_command_dim())
        action_dim = int(np.prod(env.get_action_space().shape))
        hidden_units = config.get("response_hidden_units", [512, 512])
        if len(hidden_units) == 0 or any(int(width) <= 0 for width in hidden_units):
            raise ValueError("response_hidden_units must contain positive widths")

        layers = []
        input_dim = self._response_self_dim + action_dim
        for width in hidden_units:
            width = int(width)
            linear = torch.nn.Linear(input_dim, width)
            torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            torch.nn.init.zeros_(linear.bias)
            layers.extend([linear, self._activation()])
            input_dim = width

        output = torch.nn.Linear(input_dim, self._response_dim)
        # The response model is fitted before the actor update. A zero output
        # head makes an untrained model supply no arbitrary action direction;
        # supervised response updates establish its Jacobian naturally.
        torch.nn.init.zeros_(output.weight)
        torch.nn.init.zeros_(output.bias)
        layers.append(output)
        self._response_net = torch.nn.Sequential(*layers)

    def eval_response(self, norm_self_obs, norm_action):
        if norm_self_obs.shape[-1] != self._response_self_dim:
            raise ValueError("normalized self observation has wrong size")
        response_input = torch.cat([norm_self_obs, norm_action], dim=-1)
        return self._response_net(response_input)

    def get_response_params(self):
        return list(self._response_net.parameters())
