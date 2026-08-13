import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util

class FlowADDModel(add_model.ADDModel):
    """Tangent-error flow discriminator D(x_t-1, x_t) (FlowADD, route A).

    x_t is ADD's normalized tracking-error differential. The discriminator is
    a plain MLP on the joint input

        z_t = f([x_t, v_t]),    v_t = x_t - x_t-1

    v_t is the explicit tangent error: because x_t = phi(ref_t) - phi(sim_t),
    the flow v_t equals the reference feature tangent minus the agent feature
    tangent over one step. The correct motion direction therefore enters the
    discriminator through the observed reference tangent itself, instead of
    being inferred from negative-only labels (the failure mode of the learned
    antisymmetric circulation matrix this design replaces). Wrong-direction
    motion, standing still while the reference moves, and lie-down/get-up
    shortcuts all produce a nonzero v_t even where x_t is momentarily small.

    Perfect tracking gives (x_t, v_t) = (0, 0), so ADD's single universal
    positive sample is preserved for any motion.

    disc_tangent_input = False is a strict falsification switch: it zeroes the
    tangent channel while keeping identical parameters and the same
    joint-input computation graph (so the gradient penalty setup is
    unchanged), reducing the model to ADD's pointwise scalarizer f([x_t, 0]).

    Optionally builds a fixed, group-balanced diagonal potential
    E(x) = 0.5 * sum_i w_i x_i^2 over semantic feature groups. The agent uses
    it as an explicit progress / absolute-energy reward outside the
    discriminator BCE; it is constant, never trained, and never in the logit.
    """
    def __init__(self, config, env):
        super().__init__(config, env)
        return

    def eval_disc(self, disc_obs, disc_obs_prev):
        disc_flow = disc_obs - disc_obs_prev
        if (not self._tangent_input):
            # strict tangent-off ablation: keep x_t-1 in the graph so the
            # joint-input gradient penalty sees the same inputs, but make the
            # tangent channel exactly zero
            disc_flow = 0.0 * disc_flow
        disc_in = torch.cat([disc_obs, disc_flow], dim=-1)
        h = self._disc_layers(disc_in)
        logit = self._disc_logits(h)
        return logit

    def is_tangent_input_enabled(self):
        return self._tangent_input

    def has_fixed_potential(self):
        return self._potential_diag_weights is not None

    def eval_potential_energy(self, disc_obs, clip=None):
        """Evaluates E(x) = 0.5 * sum_i w_i x_i^2 for the fixed potential."""
        assert(self.has_fixed_potential())
        if (clip is not None):
            disc_obs = torch.clamp(disc_obs, -clip, clip)
        return 0.5 * torch.sum(self._potential_diag_weights * torch.square(disc_obs), dim=-1)

    def eval_potential_group_energies(self, disc_obs, clip=None):
        """Returns each semantic group's contribution to the fixed energy."""
        assert(self.has_fixed_potential())
        if (clip is not None):
            disc_obs = torch.clamp(disc_obs, -clip, clip)

        groups = torch.split(disc_obs, self._potential_group_dims, dim=-1)
        energies = []
        for group, weight in zip(groups, self._potential_group_weights):
            energies.append(0.5 * weight * torch.mean(torch.square(group), dim=-1))
        return torch.stack(energies, dim=-1)

    def get_num_potential_groups(self):
        assert(self.has_fixed_potential())
        return len(self._potential_group_dims)

    def _build_disc(self, config, env):
        self._tangent_input = config.get("disc_tangent_input", True)

        init_output_scale = 1.0
        net_name = config["disc_net"]

        input_dict = self._build_disc_input_dict(env)
        self._disc_layers, layers_info = net_builder.build_net(net_name, input_dict,
                                                               activation=self._activation)

        layers_out_size = torch_util.calc_layers_out_size(self._disc_layers)
        self._disc_logits = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.uniform_(self._disc_logits.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(self._disc_logits.bias)

        self._build_fixed_potential(config, env)
        return

    def _build_disc_input_dict(self, env):
        obs_space = env.get_disc_obs_space()
        input_dict = {"disc_obs": obs_space, "disc_flow": obs_space}
        return input_dict

    def _build_fixed_potential(self, config, env):
        group_dims = config.get("disc_flow_potential_group_dims", None)
        group_weights = config.get("disc_flow_potential_group_weights", None)

        if (group_dims is None and group_weights is None):
            self._potential_group_dims = None
            self._potential_group_weights = None
            self.register_buffer("_potential_diag_weights", None)
            return

        assert(group_dims is not None and group_weights is not None)
        assert(len(group_dims) == len(group_weights))
        disc_obs_space = env.get_disc_obs_space()
        diff_dim = int(disc_obs_space.shape[-1])
        assert(sum(group_dims) == diff_dim)
        assert(all(v > 0 for v in group_dims))
        assert(all(v >= 0 for v in group_weights))
        weight_sum = float(sum(group_weights))
        assert(weight_sum > 0)

        self._potential_group_dims = tuple(int(v) for v in group_dims)
        self._potential_group_weights = tuple(
            float(v) / weight_sum for v in group_weights)

        # Each semantic group contributes its weighted mean-square error, so
        # large replicated pose groups cannot drown out the three-dimensional
        # global root translation error.
        diag_weights = []
        for group_dim, group_weight in zip(
                self._potential_group_dims, self._potential_group_weights):
            per_coord_weight = group_weight / group_dim
            diag_weights += [per_coord_weight] * group_dim
        self.register_buffer("_potential_diag_weights",
                             torch.tensor(diag_weights, dtype=torch.float32))
        return
