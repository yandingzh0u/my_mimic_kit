import numpy as np
import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util

# full potential-circulation flow discriminator (FlowADD-P+C)
DISC_MODE_FLOW = "flow"
# potential (progress) term only (FlowADD-P ablation)
DISC_MODE_POTENTIAL = "potential"
# circulation term only (FlowADD-C ablation)
DISC_MODE_CIRCULATION = "circulation"
# plain MLP on the concatenated input [x_t, v_t] (Concat-ADD baseline)
DISC_MODE_CONCAT = "concat"

DISC_MODES = [DISC_MODE_FLOW, DISC_MODE_POTENTIAL, DISC_MODE_CIRCULATION, DISC_MODE_CONCAT]

class FlowADDModel(add_model.ADDModel):
    """ADD discriminator extended from pointwise differential scalarization
    D(x_t) to a potential-circulation differential-flow scalarization
    D(x_t-1, x_t), where x_t is the normalized tracking-error differential.

    For the structured modes the logit is:
        z = f(x_t) + q(x_t-1, x_t)
        q = q_prog + q_circ
        q_prog = E_S(x_t-1) - E_S(x_t),  E_S(x) = 0.5 x^T S x,  S = L L^T >= 0
        q_circ = x_t-1^T A x_t,          A = B - B^T (antisymmetric)

    q_prog scores whether the error energy is making progress toward the ideal
    point, q_circ scores the oriented rotation of the error vector across
    objectives (it vanishes for pure scaling x_t = c * x_t-1 and flips sign
    under time reversal). No state-conditioned preferred flow is assumed, so
    the same x can evolve differently in different motions. With S = A = 0 the
    model reduces exactly to ADD, and the ideal point (0, 0) stays the single
    universal positive sample.
    """
    def __init__(self, config, env):
        super().__init__(config, env)
        return

    def get_disc_mode(self):
        return self._disc_mode

    def eval_disc(self, disc_obs, disc_obs_prev):
        if (self._disc_mode == DISC_MODE_CONCAT):
            disc_flow = disc_obs - disc_obs_prev
            disc_in = torch.cat([disc_obs, disc_flow], dim=-1)
            h = self._disc_layers(disc_in)
            logit = self._disc_logits(h)
        else:
            f = self.eval_static_score(disc_obs)

            q_prog, q_circ = self.eval_flow_scores(disc_obs, disc_obs_prev)
            if (self._potential_in_logit):
                flow_score = q_prog + q_circ
            else:
                # In reward-shaping mode the potential is deliberately kept
                # out of the policy-vs-ideal BCE.  Otherwise every real policy
                # transition -- including one that reduces error -- is a
                # negative example and the BCE systematically suppresses P.
                flow_score = q_circ
            logit = f + flow_score.unsqueeze(-1)
        return logit

    def eval_static_score(self, disc_obs):
        """ADD's pointwise scalarizer f(x_t), without the flow terms."""
        assert(self._disc_mode != DISC_MODE_CONCAT)
        h = self._disc_layers(disc_obs)
        f = self._disc_logits(h)
        return f

    def eval_flow_scores(self, disc_obs, disc_obs_prev):
        assert(self._disc_mode != DISC_MODE_CONCAT)

        if (self.has_potential()):
            # q_prog = E_S(x_prev) - E_S(x_t)
            energy = self.eval_potential_energy(disc_obs)
            energy_prev = self.eval_potential_energy(disc_obs_prev)
            q_prog = energy_prev - energy
        else:
            q_prog = torch.zeros(disc_obs.shape[:-1], device=disc_obs.device, dtype=disc_obs.dtype)

        if (self.has_circulation()):
            # q_circ = x_prev^T A x_t = x_prev^T B x_t - x_t^T B x_prev
            B = self._disc_flow_circulation
            q_circ = torch.sum(torch.matmul(disc_obs_prev, B) * disc_obs, dim=-1) \
                     - torch.sum(torch.matmul(disc_obs, B) * disc_obs_prev, dim=-1)
        else:
            q_circ = torch.zeros(disc_obs.shape[:-1], device=disc_obs.device, dtype=disc_obs.dtype)

        return q_prog, q_circ

    def eval_potential_energy(self, disc_obs, clip=None):
        """Evaluates E_S(x) = 0.5 x^T S x for the PSD potential."""
        assert(self.has_potential())
        if (clip is not None):
            disc_obs = torch.clamp(disc_obs, -clip, clip)
        Lx = torch.matmul(disc_obs, self._disc_flow_potential)
        return 0.5 * torch.sum(torch.square(Lx), dim=-1)

    def eval_potential_group_energies(self, disc_obs, clip=None):
        """Returns each semantic group's contribution to the fixed energy."""
        assert(self._potential_group_dims is not None), \
            "Group energies require a fixed group-balanced potential"
        if (clip is not None):
            disc_obs = torch.clamp(disc_obs, -clip, clip)

        groups = torch.split(disc_obs, self._potential_group_dims, dim=-1)
        energies = []
        for group, weight in zip(groups, self._potential_group_weights):
            energies.append(0.5 * weight * torch.mean(torch.square(group), dim=-1))
        return torch.stack(energies, dim=-1)

    def eval_potential_shaping(self, disc_obs, disc_obs_prev, discount, clip=None):
        """Linear energy-progress shaping.

        With discount=1 this is the raw progress E(x_prev)-E(x_curr), which
        intentionally changes the objective to prefer low intermediate error.
        With the policy discount it recovers standard potential shaping.
        """
        energy = self.eval_potential_energy(disc_obs, clip=clip)
        energy_prev = self.eval_potential_energy(disc_obs_prev, clip=clip)
        return energy_prev - discount * energy

    def is_potential_in_logit(self):
        return self._potential_in_logit

    def get_circulation_matrix(self):
        """Returns the antisymmetric circulation matrix A = B - B^T."""
        assert(self.has_circulation())
        B = self._disc_flow_circulation
        A = B - B.t()
        return A

    def get_flow_matrix_norms(self):
        assert(self._disc_mode != DISC_MODE_CONCAT)

        if (self.has_potential()):
            L = self._disc_flow_potential
            S = torch.matmul(L, L.t())
            s_norm = torch.norm(S)
        else:
            s_norm = torch.zeros([1], device=self._disc_logits.weight.device)

        if (self.has_circulation()):
            a_norm = torch.norm(self.get_circulation_matrix())
        else:
            a_norm = torch.zeros([1], device=self._disc_logits.weight.device)

        return s_norm, a_norm

    def get_disc_logit_weights(self):
        weights = [torch.flatten(self._disc_logits.weight)]
        if (self._regularize_flow_matrices
                and self.has_potential()
                and self._disc_flow_potential.requires_grad):
            weights.append(torch.flatten(self._disc_flow_potential))
        # C still participates in the discriminator and keeps its historical
        # logit regularization even when a fixed external P is exempt.
        if (self.has_circulation()):
            weights.append(torch.flatten(self._disc_flow_circulation))
        return torch.cat(weights)

    def get_disc_params(self):
        params = super().get_disc_params()
        if (self.has_potential()):
            params += [self._disc_flow_potential]
        if (self.has_circulation()):
            params += [self._disc_flow_circulation]
        return params

    def has_potential(self):
        return self._disc_mode in [DISC_MODE_FLOW, DISC_MODE_POTENTIAL]

    def has_circulation(self):
        return self._disc_mode in [DISC_MODE_FLOW, DISC_MODE_CIRCULATION]

    def _build_disc(self, config, env):
        self._disc_mode = config.get("disc_mode", DISC_MODE_FLOW)
        assert(self._disc_mode in DISC_MODES), "Unsupported disc_mode: {}".format(self._disc_mode)
        self._potential_in_logit = config.get("disc_flow_potential_in_logit", True)
        self._regularize_flow_matrices = config.get("disc_flow_regularize_matrices", True)
        self._potential_group_dims = None
        self._potential_group_weights = None

        init_output_scale = 1.0
        net_name = config["disc_net"]

        input_dict = self._build_disc_input_dict(env)
        self._disc_layers, layers_info = net_builder.build_net(net_name, input_dict,
                                                               activation=self._activation)

        layers_out_size = torch_util.calc_layers_out_size(self._disc_layers)
        self._disc_logits = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.uniform_(self._disc_logits.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(self._disc_logits.bias)

        if (self._disc_mode != DISC_MODE_CONCAT):
            disc_obs_space = env.get_disc_obs_space()
            diff_dim = int(np.prod(disc_obs_space.shape))
            flow_init_scale = config.get("disc_flow_init_scale", 1.0)

            if (self.has_potential()):
                fixed_group_dims = config.get("disc_flow_potential_group_dims", None)
                fixed_group_weights = config.get("disc_flow_potential_group_weights", None)
                if (fixed_group_dims is not None or fixed_group_weights is not None):
                    assert(fixed_group_dims is not None and fixed_group_weights is not None)
                    assert(len(fixed_group_dims) == len(fixed_group_weights))
                    assert(sum(fixed_group_dims) == diff_dim)
                    assert(all(v > 0 for v in fixed_group_dims))
                    assert(all(v >= 0 for v in fixed_group_weights))
                    weight_sum = float(sum(fixed_group_weights))
                    assert(weight_sum > 0)
                    self._potential_group_dims = tuple(int(v) for v in fixed_group_dims)
                    self._potential_group_weights = tuple(
                        float(v) / weight_sum for v in fixed_group_weights)

                    # Each semantic group contributes its weighted mean-square
                    # error, so large replicated pose groups cannot drown out
                    # the three-dimensional global root translation error.
                    diag_weights = []
                    for group_dim, group_weight in zip(
                            self._potential_group_dims, self._potential_group_weights):
                        per_coord_weight = group_weight / group_dim
                        diag_weights += [per_coord_weight] * group_dim
                    diag_weights = torch.tensor(diag_weights, dtype=torch.float32)
                    L0 = torch.diag(torch.sqrt(diag_weights))
                    self._disc_flow_potential = torch.nn.Parameter(L0, requires_grad=False)
                else:
                    # small random init: S = L L^T is quadratic in L, so L = 0 is a
                    # saddle point with zero gradient and a too small init starves
                    # the progress term of gradient signal; with std = scale / d the
                    # initial error energy is E_S(x) ~ scale^2 / 2 for normalized x,
                    # small relative to the f logits but with healthy gradients
                    L0 = torch.randn([diff_dim, diff_dim]) * (flow_init_scale / diff_dim)
                    self._disc_flow_potential = torch.nn.Parameter(L0)

            if (self.has_circulation()):
                # q_circ is linear in B, so B = 0 has non-zero gradient and the
                # model starts exactly as ADD (A = 0)
                self._disc_flow_circulation = torch.nn.Parameter(torch.zeros([diff_dim, diff_dim]))
        return

    def _build_disc_input_dict(self, env):
        obs_space = env.get_disc_obs_space()
        input_dict = {"disc_obs": obs_space}
        if (self._disc_mode == DISC_MODE_CONCAT):
            input_dict["disc_flow"] = obs_space
        return input_dict
