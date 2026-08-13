import math

import torch
import torch.nn.functional as F

import learning.add_agent as add_agent
import learning.flow_add_model as flow_add_model
import learning.diff_normalizer as diff_normalizer
import util.torch_util as torch_util

# number of samples used to estimate the flow diagnostics each iteration
FLOW_STAT_SAMPLES = 4096

DISC_REWARD_SOFTPLUS = "softplus"
DISC_REWARD_CENTERED_LOG_D = "centered_log_d"
DISC_REWARD_TYPES = [DISC_REWARD_SOFTPLUS, DISC_REWARD_CENTERED_LOG_D]


def calc_flow_disc_reward(logits, reward_type, reward_scale, reward_min=None):
    """Maps discriminator logits to policy rewards.

    ``softplus`` is the original ADD reward ``-log(1 - D)``.  Its slope
    vanishes for bad (large negative-logit) states and its convexity can give
    zero-sum temporal flow terms a cycle bonus.

    ``centered_log_d`` uses ``log(D) + log(2)``.  It is zero at an undecided
    discriminator, strictly increasing, and concave.  The centering constant
    does not change the policy objective for fixed-length episodes, while the
    non-saturating negative-logit slope provides a recovery signal when early
    termination is disabled.
    """
    assert(reward_type in DISC_REWARD_TYPES)

    if (reward_type == DISC_REWARD_SOFTPLUS):
        # Preserve ADD's historical numerical cap exactly.
        prob = torch.sigmoid(logits)
        reward = -torch.log(torch.clamp_min(1.0 - prob, 0.0001))
    else:
        reward = F.logsigmoid(logits) + math.log(2.0)

    reward = reward_scale * reward
    if (reward_min is not None):
        reward = torch.clamp_min(reward, reward_min)
    return reward


def calc_fixed_potential_rewards(energy, energy_prev, progress_scale,
                                 progress_discount, abs_energy_weight):
    """Returns linear progress and absolute-energy reward components.

    The progress term provides short-horizon credit for reducing error.  The
    non-positive absolute term prevents a policy from receiving zero shaping
    forever after settling at a large, constant tracking error.
    """
    progress_reward = progress_scale * (energy_prev - progress_discount * energy)
    abs_energy_reward = -abs_energy_weight * energy
    return progress_reward, abs_energy_reward


def calc_group_abs_rewards(group_energy, base_weight, extra_group_weights):
    """Returns per-group and total non-positive absolute-energy rewards."""
    weights = torch.as_tensor(extra_group_weights, device=group_energy.device,
                              dtype=group_energy.dtype)
    assert(weights.shape[-1] == group_energy.shape[-1])
    group_rewards = -(base_weight + weights) * group_energy
    return group_rewards, torch.sum(group_rewards, dim=-1)


class FlowADDAgent(add_agent.ADDAgent):
    """ADD agent with a tangent-error flow discriminator D(x_t-1, x_t).

    x_t is ADD's tracking-error differential in the normalized differential
    space and the discriminator consumes [x_t, v_t] with v_t = x_t - x_t-1,
    so the reference tangent supervises the correct motion direction
    explicitly. Positive samples are the ideal transition (x_t-1, x_t) =
    (0, 0), negatives are policy transitions, and the gradient penalty is
    applied on the joint input [x_t-1, x_t]. The policy reward mapping is
    configurable; the default remains ADD's ``-log(1-D)``.

    A fixed, group-balanced tracking-error potential can additionally shape
    the reward outside the discriminator: a linear progress term
    E(x_t-1) - discount * E(x_t) plus a non-positive absolute-energy term.
    Keeping it out of the BCE avoids the label conflict where genuinely
    improving policy transitions are still negatives and the potential gets
    suppressed.
    """
    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _load_params(self, config):
        super()._load_params(config)

        # Keep the original ADD reward as the default.  No-ET experiments can
        # opt into the non-saturating, concave centered log-D reward.
        self._disc_reward_type = config.get("disc_reward_type", DISC_REWARD_SOFTPLUS)
        assert(self._disc_reward_type in DISC_REWARD_TYPES), \
            "Unsupported discriminator reward type: {:s}".format(self._disc_reward_type)
        self._disc_reward_min = config.get("disc_reward_min", None)

        # Fixed P reward outside the discriminator nonlinearity.  Progress and
        # absolute energy are explicit, separately weighted components.
        self._disc_flow_potential_reward_scale = config.get(
            "disc_flow_potential_reward_scale", 0.0)
        self._disc_flow_potential_fixed_norm = config.get(
            "disc_flow_potential_fixed_norm", None)
        self._disc_flow_potential_obs_clip = config.get(
            "disc_flow_potential_obs_clip", None)
        self._disc_flow_potential_discount = config.get(
            "disc_flow_potential_discount", self._discount)
        self._disc_flow_potential_abs_energy_weight = config.get(
            "disc_flow_potential_abs_energy_weight", 0.0)
        self._disc_flow_potential_abs_group_weights = config.get(
            "disc_flow_potential_abs_group_weights", None)
        self._disc_flow_potential_group_names = config["model"].get(
            "disc_flow_potential_group_names", None)
        assert(self._disc_flow_potential_reward_scale >= 0)
        assert(0 <= self._disc_flow_potential_discount <= 1)
        assert(self._disc_flow_potential_abs_energy_weight >= 0)
        if (self._disc_flow_potential_abs_group_weights is not None):
            assert(all(v >= 0 for v in self._disc_flow_potential_abs_group_weights))
        self._use_external_potential_reward = (
            self._disc_flow_potential_reward_scale > 0
            or self._disc_flow_potential_abs_energy_weight > 0
            or (self._disc_flow_potential_abs_group_weights is not None
                and any(v > 0 for v in self._disc_flow_potential_abs_group_weights)))
        if (self._use_external_potential_reward):
            assert(self._disc_flow_potential_fixed_norm is not None), \
                "Potential shaping requires a fixed normalization scale"
        return

    def _build_normalizers(self):
        super()._build_normalizers()

        if (self._use_external_potential_reward):
            disc_obs_space = self._env.get_disc_obs_space()
            disc_obs_dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
            scale = torch.tensor(self._disc_flow_potential_fixed_norm,
                                 device=self._device, dtype=disc_obs_dtype)
            if (scale.numel() == 1):
                scale = torch.full(disc_obs_space.shape, scale.item(),
                                   device=self._device, dtype=disc_obs_dtype)
            assert(tuple(scale.shape) == tuple(disc_obs_space.shape))
            assert(torch.all(scale > 0))
            self._disc_potential_norm = diff_normalizer.DiffNormalizer(
                disc_obs_space.shape, device=self._device,
                init_mean=scale, dtype=disc_obs_dtype)
        else:
            self._disc_potential_norm = None
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = flow_add_model.FlowADDModel(model_config, self._env)
        if (self._use_external_potential_reward):
            assert(self._model.has_fixed_potential()), \
                "Potential shaping requires disc_flow_potential_group_dims/weights"
            num_groups = self._model.get_num_potential_groups()
            if (self._disc_flow_potential_group_names is not None):
                assert(len(self._disc_flow_potential_group_names) == num_groups)
            if (self._disc_flow_potential_abs_group_weights is not None):
                assert(len(self._disc_flow_potential_abs_group_weights) == num_groups)
            else:
                self._disc_flow_potential_abs_group_weights = (0.0,) * num_groups
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)

        self._exp_buffer.record("disc_obs_prev", next_info["disc_obs_prev"])
        self._exp_buffer.record("disc_obs_demo_prev", next_info["disc_obs_demo_prev"])
        return

    def _store_disc_replay_data(self):
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")
        disc_obs_prev = self._exp_buffer.get_data_flat("disc_obs_prev")
        disc_obs_demo_prev = self._exp_buffer.get_data_flat("disc_obs_demo_prev")

        n = disc_obs.shape[0]
        rand_idx = torch.randperm(n, device=self._device, dtype=torch.long)

        if (self._disc_buffer.is_full()):
            num_samples = min(n, self._disc_replay_samples)
        else:
            num_samples = n

        idx = rand_idx[:num_samples]
        disc_data = {
            "disc_obs": disc_obs[idx].unsqueeze(1),
            "disc_obs_demo": disc_obs_demo[idx].unsqueeze(1),
            "disc_obs_prev": disc_obs_prev[idx].unsqueeze(1),
            "disc_obs_demo_prev": disc_obs_demo_prev[idx].unsqueeze(1)
        }
        self._disc_buffer.push(disc_data)
        return

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")
        disc_obs_prev = self._exp_buffer.get_data_flat("disc_obs_prev")
        disc_obs_demo_prev = self._exp_buffer.get_data_flat("disc_obs_demo_prev")

        obs_diff = disc_obs_demo - disc_obs
        obs_diff_prev = disc_obs_demo_prev - disc_obs_prev

        norm_obs_diff = self._disc_obs_norm.normalize(obs_diff)
        norm_obs_diff_prev = self._disc_obs_norm.normalize(obs_diff_prev)

        reward_parts = self._calc_reward_components(norm_obs_diff, norm_obs_diff_prev)
        disc_base_r = reward_parts["base"]
        progress_r = reward_parts["progress"]
        abs_energy_r = reward_parts["absolute"]
        potential_r = progress_r + abs_energy_r
        disc_r = disc_base_r + potential_r
        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)

        r = self._task_reward_weight * task_r + self._disc_reward_weight * disc_r
        self._exp_buffer.set_data_flat("reward", r)

        if (self._need_normalizer_update()):
            self._disc_obs_norm.record(obs_diff)

        info = {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std,
            "disc_base_reward_mean": torch.mean(disc_base_r),
            "disc_base_reward_std": torch.std(disc_base_r),
            "disc_p_reward_mean": torch.mean(potential_r),
            "disc_p_reward_std": torch.std(potential_r),
            "disc_prog_reward_mean": torch.mean(progress_r),
            "disc_prog_reward_std": torch.std(progress_r),
            "disc_abs_reward_mean": torch.mean(abs_energy_r),
            "disc_abs_reward_std": torch.std(abs_energy_r)
        }
        if (self._use_external_potential_reward):
            energy = reward_parts["energy"]
            info.update({
                "disc_energy_mean": torch.mean(energy),
                "disc_energy_std": torch.std(energy),
                "disc_energy_p95": torch.quantile(energy, 0.95),
                "disc_energy_max": torch.max(energy)
            })
            group_energy = reward_parts["group_energy"]
            abs_group_reward = reward_parts["abs_group"]
            info["disc_abs_root_mean"] = torch.mean(abs_group_reward[..., 0])
            info["disc_abs_root_std"] = torch.std(abs_group_reward[..., 0])
            if (self._disc_flow_potential_group_names is not None):
                for group_idx, group_name in enumerate(
                        self._disc_flow_potential_group_names):
                    info["disc_e_{:s}".format(group_name)] = \
                        torch.mean(group_energy[..., group_idx])
        flow_info = self._compute_flow_stats(norm_obs_diff, norm_obs_diff_prev)
        info.update(flow_info)
        return info

    def _calc_disc_rewards(self, norm_obs_diff, norm_obs_diff_prev):
        reward_parts = self._calc_reward_components(norm_obs_diff, norm_obs_diff_prev)
        return reward_parts["base"] + reward_parts["progress"] + reward_parts["absolute"]

    def _calc_reward_components(self, norm_obs_diff, norm_obs_diff_prev):
        with torch.no_grad():
            disc_inputs = {"disc_obs": norm_obs_diff, "disc_obs_prev": norm_obs_diff_prev}
            disc_logits = torch_util.eval_minibatch(self._model.eval_disc, disc_inputs, self._disc_eval_batch_size)
            disc_logits = disc_logits.squeeze(-1)
            disc_base_r = calc_flow_disc_reward(logits=disc_logits,
                                                reward_type=self._disc_reward_type,
                                                reward_scale=self._disc_reward_scale,
                                                reward_min=self._disc_reward_min)

            if (self._use_external_potential_reward):
                potential_obs = self._disc_potential_norm.normalize(
                    self._disc_obs_norm.unnormalize(norm_obs_diff))
                potential_obs_prev = self._disc_potential_norm.normalize(
                    self._disc_obs_norm.unnormalize(norm_obs_diff_prev))
                energy = self._model.eval_potential_energy(
                    potential_obs, clip=self._disc_flow_potential_obs_clip)
                energy_prev = self._model.eval_potential_energy(
                    potential_obs_prev, clip=self._disc_flow_potential_obs_clip)
                progress_r, _ = calc_fixed_potential_rewards(
                    energy=energy,
                    energy_prev=energy_prev,
                    progress_scale=self._disc_flow_potential_reward_scale,
                    progress_discount=self._disc_flow_potential_discount,
                    abs_energy_weight=self._disc_flow_potential_abs_energy_weight)
                group_energy = self._model.eval_potential_group_energies(
                    potential_obs, clip=self._disc_flow_potential_obs_clip)
                abs_group_r, abs_energy_r = calc_group_abs_rewards(
                    group_energy=group_energy,
                    base_weight=self._disc_flow_potential_abs_energy_weight,
                    extra_group_weights=self._disc_flow_potential_abs_group_weights)
            else:
                progress_r = torch.zeros_like(disc_base_r)
                abs_energy_r = torch.zeros_like(disc_base_r)
                energy = torch.zeros_like(disc_base_r)
                group_energy = None
                abs_group_r = None
        return {
            "base": disc_base_r,
            "progress": progress_r,
            "absolute": abs_energy_r,
            "energy": energy,
            "group_energy": group_energy,
            "abs_group": abs_group_r
        }

    def _compute_disc_loss(self, batch):
        disc_obs = batch["disc_obs"]
        tar_disc_obs = batch["disc_obs_demo"]
        disc_obs_prev = batch["disc_obs_prev"]
        tar_disc_obs_prev = batch["disc_obs_demo_prev"]

        # positive sample is the ideal transition (x_t-1, x_t) = (0, 0)
        pos_diff = self._pos_diff.clone()
        pos_diff = pos_diff.unsqueeze(dim=0)
        pos_diff_prev = torch.zeros_like(pos_diff)
        pos_diff.requires_grad_(True)
        pos_diff_prev.requires_grad_(True)
        disc_pos_logit = self._model.eval_disc(pos_diff, pos_diff_prev)
        disc_pos_logit = disc_pos_logit.squeeze(-1)

        diff_obs = tar_disc_obs - disc_obs
        diff_obs_prev = tar_disc_obs_prev - disc_obs_prev

        replay_data = self._disc_buffer.sample(diff_obs.shape[0])
        replay_diff = replay_data["disc_obs_demo"] - replay_data["disc_obs"]
        replay_diff_prev = replay_data["disc_obs_demo_prev"] - replay_data["disc_obs_prev"]
        diff_obs = torch.cat([diff_obs, replay_diff], dim=0)
        diff_obs_prev = torch.cat([diff_obs_prev, replay_diff_prev], dim=0)

        norm_diff_obs = self._disc_obs_norm.normalize(diff_obs)
        norm_diff_obs_prev = self._disc_obs_norm.normalize(diff_obs_prev)
        norm_diff_obs.requires_grad_(True)
        norm_diff_obs_prev.requires_grad_(True)
        disc_neg_logit = self._model.eval_disc(norm_diff_obs, norm_diff_obs_prev)
        disc_neg_logit = disc_neg_logit.squeeze(-1)

        disc_loss_pos = self._disc_loss_pos(disc_pos_logit)
        disc_loss_neg = self._disc_loss_neg(disc_neg_logit)
        disc_loss = 0.5 * (disc_loss_pos + disc_loss_neg)

        # logit reg
        logit_weights = self._model.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty on the joint input [x_t-1, x_t]
        disc_neg_grads = torch.autograd.grad(disc_neg_logit, [norm_diff_obs, norm_diff_obs_prev],
                                             grad_outputs=torch.ones_like(disc_neg_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_neg_grad_squared = torch.sum(torch.square(disc_neg_grads[0]), dim=-1) \
                                + torch.sum(torch.square(disc_neg_grads[1]), dim=-1)

        disc_pos_grads = torch.autograd.grad(disc_pos_logit, [pos_diff, pos_diff_prev],
                                             grad_outputs=torch.ones_like(disc_pos_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_pos_grad_squared = torch.sum(torch.square(disc_pos_grads[0]), dim=-1) \
                                + torch.sum(torch.square(disc_pos_grads[1]), dim=-1)

        disc_grad_penalty = 0.5 * (torch.mean(disc_neg_grad_squared) + torch.mean(disc_pos_grad_squared))
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        disc_neg_acc, disc_pos_acc = self._compute_disc_acc(disc_neg_logit, disc_pos_logit)
        disc_pos_logit_mean = torch.mean(disc_pos_logit)
        disc_neg_logit_mean = torch.mean(disc_neg_logit)

        disc_info = {
            "disc_loss": disc_loss,
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_logit_loss": disc_logit_loss.detach(),
            "disc_pos_acc": disc_pos_acc.detach(),
            "disc_neg_acc": disc_neg_acc.detach(),
            "disc_pos_logit": disc_pos_logit_mean.detach(),
            "disc_neg_logit": disc_neg_logit_mean.detach()
        }
        return disc_info

    def _compute_flow_stats(self, norm_obs_diff, norm_obs_diff_prev):
        with torch.no_grad():
            n = norm_obs_diff.shape[0]
            num_samples = min(n, FLOW_STAT_SAMPLES)
            idx = torch.randperm(n, device=self._device, dtype=torch.long)[:num_samples]
            diff = norm_obs_diff[idx]
            diff_prev = norm_obs_diff_prev[idx]

            logit = self._model.eval_disc(diff, diff_prev).squeeze(-1)
            r_full = calc_flow_disc_reward(logits=logit,
                                           reward_type=self._disc_reward_type,
                                           reward_scale=self._disc_reward_scale,
                                           reward_min=self._disc_reward_min)

            if (self._disc_reward_min is None):
                reward_clip_frac = torch.zeros([], device=logit.device)
            else:
                r_unclipped = calc_flow_disc_reward(logits=logit,
                                                    reward_type=self._disc_reward_type,
                                                    reward_scale=self._disc_reward_scale,
                                                    reward_min=None)
                reward_clip_frac = torch.mean((r_unclipped < self._disc_reward_min).float())

            # tangent influence: feeding x_t-1 = x_t zeroes the tangent
            # channel, so this measures how much the discriminator actually
            # uses v_t, in logit units and in reward units
            logit_static = self._model.eval_disc(diff, diff).squeeze(-1)
            tangent_gap = torch.mean(torch.abs(logit - logit_static))
            r_static = calc_flow_disc_reward(logits=logit_static,
                                             reward_type=self._disc_reward_type,
                                             reward_scale=self._disc_reward_scale,
                                             reward_min=self._disc_reward_min)
            tangent_r_gap = torch.mean(torch.abs(r_full - r_static))

            # orientation probes: how much the logit changes when the
            # transition is reversed or when x_t-1 is shuffled across the batch
            logit_rev = self._model.eval_disc(diff_prev, diff).squeeze(-1)
            rev_gap = torch.mean(torch.abs(logit - logit_rev))

            perm = torch.randperm(num_samples, device=self._device, dtype=torch.long)
            logit_shuf = self._model.eval_disc(diff, diff_prev[perm]).squeeze(-1)
            shuffle_gap = torch.mean(torch.abs(logit - logit_shuf))

            flow_norm = torch.mean(torch.norm(diff - diff_prev, dim=-1))

        info = {
            "disc_flow_norm": flow_norm,
            "disc_flow_tangent_gap": tangent_gap,
            "disc_flow_tangent_r_gap": tangent_r_gap,
            "disc_flow_rev_gap": rev_gap,
            "disc_flow_shuffle_gap": shuffle_gap,
            "disc_reward_clip_frac": reward_clip_frac
        }
        return info
