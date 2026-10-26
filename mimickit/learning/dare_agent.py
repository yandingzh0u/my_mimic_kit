import math
import torch

import learning.add_agent as add_agent
import learning.dare_model as dare_model
from learning.phase_sampler import PhaseSampler, control_cells, supported_resolution
import util.mp_util as mp_util


class DAREAgent(add_agent.ADDAgent):
    """Zero-differential adversarial learning with a CPL discriminator."""

    def _load_params(self, config):
        super()._load_params(config)
        durations, timestep = self._env.get_motion_sampling_spec()
        if len(durations) != 1 or mp_util.get_num_procs() != 1:
            raise ValueError("DARE phase sampling currently supports one motion and one process")
        self._phase_control_cells = control_cells(durations[0], timestep)

    def _build_phase_weights(self):
        """Use the finest partition supported by both PPO populations."""
        phase = self._exp_buffer.get_data("diag_phase")
        actor_budget = math.ceil(self._actor_batch_size * self.get_num_envs())
        critic_budget = math.ceil(self._critic_batch_size * self.get_num_envs())
        eligible = self._exp_buffer.get_data_flat("rand_action_mask") == 1
        resolution = supported_resolution(phase, self._phase_control_cells,
                                          [actor_budget, critic_budget], [eligible, None])
        self._critic_phase_sampler = PhaseSampler(phase, resolution)
        self._actor_phase_sampler = PhaseSampler(phase, resolution, eligible)
        weights = (self._critic_phase_sampler.full_weights() * phase.numel()).float().reshape(phase.shape)
        info = {"phase_weight_mean": weights.mean(), "phase_weight_max": weights.max(),
                "phase_motion_control_cells": weights.new_tensor(float(self._phase_control_cells)),
                "phase_resolution_bins": weights.new_tensor(float(resolution))}
        for name, sampler in (("actor", self._actor_phase_sampler), ("critic", self._critic_phase_sampler)):
            for key, value in sampler.coverage.items():
                info["phase_" + name + "_" + key] = weights.new_tensor(float(value))
        return weights, info

    def _actor_population_weights(self):
        return self._actor_phase_sampler.full_weights()

    def _sample_ppo_batch(self, batch_size, actor):
        sampler = self._actor_phase_sampler if actor else self._critic_phase_sampler
        indices, weights, _ = sampler.sample(batch_size)
        batch = self._exp_buffer.sample_indices(indices)
        batch["phase_weight"] = weights.float()
        return batch

    def _build_model(self, config):
        self._model = dare_model.DAREModel(config["model"], self._env)

    def _compute_disc_loss(self, batch):
        """Classify the zero anchor against current and replay residuals."""
        # Preserve the positive-first spectral-normalization update order.
        pos_logits = self._model.eval_disc(
            self._pos_diff.unsqueeze(0)).squeeze(-1)
        current_diff = batch["disc_obs_demo"] - batch["disc_obs"]
        replay_data = self._disc_buffer.sample(current_diff.shape[0])
        replay_diff = (replay_data["disc_obs_demo"]
                       - replay_data["disc_obs"])

        raw_diff = torch.cat((current_diff, replay_diff), dim=0)
        norm_diff = self._disc_obs_norm.normalize(raw_diff)
        neg_logits = self._model.eval_disc(norm_diff).squeeze(-1)
        pos_loss = self._disc_loss_pos(pos_logits)
        neg_loss = self._disc_loss_neg(neg_logits)
        disc_loss = 0.5 * (pos_loss + neg_loss)
        neg_acc, pos_acc = self._compute_disc_acc(neg_logits, pos_logits)

        return {
            "disc_loss": disc_loss,
            "disc_cls_loss": disc_loss.detach(),
            "disc_pos_acc": pos_acc.detach(),
            "disc_neg_acc": neg_acc.detach(),
            "disc_pos_logit": pos_logits.mean().detach(),
            "disc_neg_logit": neg_logits.mean().detach(),
            "disc_group_width": norm_diff.new_tensor(float(self._model._disc_layers.group_width)),
            "disc_group_total_width": norm_diff.new_tensor(float(self._model._disc_layers.total_width)),
        }
