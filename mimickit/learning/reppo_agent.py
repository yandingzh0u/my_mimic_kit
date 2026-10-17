"""Compact DARE/REPPO robot adapter; original PPO and discriminator stay intact.

Robot migration keeps the actor's SGD state and uses centered entropy with
the original Gaussian entropy as its floor. This is a REPPO adaptation, not
a reproduction of the paper's from-scratch architecture/entropy convention.
"""
import copy
import math

import torch

from learning.dare_agent import DAREAgent
from learning.dare_model import DAREModel
from learning.distribution_gaussian_diag import DistributionGaussianDiag
from learning.mp_optimizer import MPOptimizer
from learning.reppo import (CategoricalQ, Duals, actor_objective,
                            frozen_parameters, soft_lambda_returns)


class REPPOModel(DAREModel):
    def _build_critic(self, config, env):
        self.q = CategoricalQ(env.get_obs_space().shape[0],
            env.get_action_space().shape[0], *config.get("q_support", [-100., 100.]))

    def get_critic_params(self):
        return list(self.q.parameters())

    def eval_critic(self, obs):
        return self.q.value(obs, self.eval_actor(obs).mean).unsqueeze(-1)


class REPPOAgent(DAREAgent):
    def _build_algorithm(self, config):
        # This existing adapter overrides targets and both losses itself.
        # Do not instantiate an unused PPO objective during the refactor.
        return None

    def _build_model(self, config):
        if config["model"]["actor_std_type"] != "CONSTANT":
            raise ValueError("REPPO requires a learnable constant Gaussian std")
        self._model = REPPOModel(config["model"], self._env)
        self._duals = Duals()
        entropy = self.get_action_size() * (
            math.log(config["model"]["action_std"]) + .5 * math.log(2 * math.pi * math.e))
        self.register_buffer("_entropy_ref", torch.tensor(entropy, dtype=torch.float32))
        self._kl_limit = float(config.get("reppo_kl_limit", .01))

    def _build_optimizer(self, config):
        self._actor_optimizer = MPOptimizer(config["actor_optimizer"],
            [p for p in self._model.get_actor_params() if p.requires_grad])
        self._critic_optimizer = MPOptimizer(config["critic_optimizer"], self._model.get_critic_params())
        self._disc_optimizer = MPOptimizer(config["disc_optimizer"], self._model.get_disc_params())
        self._dual_optimizer = MPOptimizer(config["dual_optimizer"], list(self._duals.parameters()))

    def _sync_optimizer(self):
        super()._sync_optimizer()
        self._dual_optimizer.sync()

    def load(self, in_file):
        checkpoint = self._torch_load(in_file)
        if self._is_training_checkpoint(checkpoint) and checkpoint["metadata"]["agent_class"] == "DAREAgent":
            warm_q = self._torch_load(self._config["migration_q"])
            self.migrate(checkpoint, warm_q["state_dict"])
        else:
            super().load(in_file)

    @torch.no_grad()
    def migrate(self, checkpoint, q_state):
        """Strict legacy import: retire V, keep every other original tensor.

        Q must be calibrated on this actor's real trajectories. Existing mean
        momentum is mapped by parameter identity, not shifted by the new std.
        """
        source = checkpoint["model_state_dict"]
        legacy = {k: v for k, v in source.items() if not k.startswith(
            ("_model._critic_layers.", "_model._critic_out."))}
        current = self.state_dict()
        expected = {k for k in current if not k.startswith(
            ("_model.q.", "_duals.", "_entropy_ref"))}
        if set(legacy) != expected:
            raise ValueError("Legacy DARE tensor keys do not match the robot adapter")
        self.load_state_dict({**current, **legacy})
        self._model.q.load_state_dict(q_state)
        self._entropy_ref.copy_(self._model._action_dist._logstd_net.sum()
            + .5 * self.get_action_size() * math.log(2 * math.pi * math.e))
        old = checkpoint["optimizer_state_dicts"]["_actor_optimizer"]
        new = self._actor_optimizer.state_dict()
        old_group, new_group = old["optimizer"]["param_groups"][0], new["optimizer"]["param_groups"][0]
        if "momentum" not in old_group or "momentum" not in new_group:
            raise ValueError("Robot migration must retain the original SGD optimizer")
        if len(old_group["params"]) != len(new_group["params"]) - 1:
            raise ValueError("Expected exactly one newly learnable std parameter")
        ids = iter(old_group["params"])
        std = self._model._action_dist._logstd_net
        for key, param in zip(new_group["params"], self._actor_optimizer._param_list):
            if param is not std:
                state = old["optimizer"]["state"].get(next(ids), {})
                new["optimizer"]["state"][key] = copy.deepcopy(state)
        if next(ids, None) is not None:
            raise ValueError("Legacy actor optimizer parameter count mismatch")
        new_group.update({k: v for k, v in old_group.items() if k != "params"})
        new["steps"] = old["steps"]
        self._actor_optimizer.load_state_dict(new)
        self._disc_optimizer.load_state_dict(copy.deepcopy(
            checkpoint["optimizer_state_dicts"]["_disc_optimizer"]))
        self._disc_buffer.load_state_dict(checkpoint["replay_buffer_states"]["_disc_buffer"])
        for name, module in self.named_modules():
            if name in checkpoint["normalizer_training_states"]:
                module.load_training_state_dict(checkpoint["normalizer_training_states"][name])
        # A frozen mature normalization is part of the migration contract.
        self._normalizer_samples = 0
        self._sync_optimizer()

    @torch.no_grad()
    def _build_train_data(self):
        self.eval()
        self._store_disc_replay_data()
        info = self._compute_rewards()
        buffer = self._exp_buffer
        means, logstds, values, entropies = [], [], [], []
        for obs, next_obs in zip(buffer.get_data_flat("obs").split(4096),
                                 buffer.get_data_flat("next_obs").split(4096)):
            dist = self._model.eval_actor(self._obs_norm.normalize(obs))
            means.append(dist.mean)
            logstds.append(dist.logstd)
            next_obs = self._obs_norm.normalize(next_obs)
            dist = self._model.eval_actor(next_obs)
            values.append(self._model.q.value(next_obs, dist.sample()))
            # Exact Gaussian entropy: the zero-centered bonus is zero at
            # migration. No sampled entropy noise enters Q targets.
            entropies.append(dist.entropy() - self._entropy_ref)
        buffer.set_data_flat("old_mean", torch.cat(means))
        buffer.set_data_flat("old_logstd", torch.cat(logstds))
        reward, done = buffer.get_data("reward"), buffer.get_data("done")
        target = soft_lambda_returns(reward, torch.cat(values).reshape_as(reward),
            -torch.cat(entropies).reshape_as(reward), done, self._discount,
            self._td_lambda, alpha=self._duals.alpha)
        outside = ((target < self._model.q.edges[0]) | (target > self._model.q.edges[-1])).float().mean()
        if not torch.isfinite(target).all() or outside > .01:
            raise RuntimeError(f"Invalid Q targets or HL support exceeded: {float(outside):.3%}")
        weights, phase_info = self._build_phase_weights()
        buffer.set_data("tar_val", target)
        buffer.set_data("phase_weight", weights)
        return {**info, **phase_info, "reppo_target_mean": target.mean(),
                "reppo_target_outside_support": outside}

    def _compute_critic_loss(self, batch):
        obs = self._obs_norm.normalize(batch["obs"])
        action = self._a_norm.normalize(batch["action"])
        return {"critic_loss": self._model.q.loss(obs, action,
            batch["tar_val"], batch["phase_weight"])}

    def _compute_actor_loss(self, batch):
        obs = self._obs_norm.normalize(batch["obs"])
        dist = self._model.eval_actor(obs)
        old = DistributionGaussianDiag(batch["old_mean"], batch["old_logstd"])
        loss, info = actor_objective(dist, old, self._model.q, obs,
            torch.randn_like(dist.mean), self._duals, batch["phase_weight"],
            self._kl_limit, self._entropy_ref)
        bound = self._compute_action_bound_loss(dist)
        bound = self._weighted_mean(bound, batch["phase_weight"])
        return {"actor_loss": loss + self._action_bound_weight * bound,
                "action_bound_loss": bound.detach(), **info}

    def _update_actor(self, batch_size, num_steps):
        buffer = self._exp_buffer
        ids = torch.linspace(0, buffer.get_sample_count() - 1,
            min(4096, buffer.get_sample_count()), device=self._device).long()
        audit_obs = self._obs_norm.normalize(buffer.get_data_flat("obs")[ids])
        old = {"mean": buffer.get_data_flat("old_mean")[ids],
               "logstd": buffer.get_data_flat("old_logstd")[ids]}
        phase = buffer.get_data_flat("diag_phase")[ids] if buffer.has_buffer("diag_phase") else None
        totals, max_post_kl = {}, 0.
        with frozen_parameters(self._model.q):
            for _ in range(num_steps):
                batch = buffer.sample(batch_size)
                info = self._compute_actor_loss(batch)
                self._actor_optimizer.step(info["actor_loss"])
                dual_loss = (self._duals.alpha * (info["reppo_entropy"] - self._entropy_ref)
                    + self._duals.beta * (self._kl_limit - info["reppo_kl"]))
                self._dual_optimizer.step(dual_loss)
                with torch.no_grad():
                    past = DistributionGaussianDiag(batch["old_mean"], batch["old_logstd"])
                    new = self._model.eval_actor(self._obs_norm.normalize(batch["obs"]))
                    max_post_kl = max(max_post_kl, float(past.kl(new).mean()))
                # Clone detached metrics: never mutate a live dual/Q/loss tensor.
                for key, value in info.items():
                    totals[key] = totals.get(key, 0.) + value.detach().clone()
        totals = {k: v / num_steps for k, v in totals.items()}
        totals.update(self._compute_policy_audit(audit_obs, old, phase))
        totals.update(reppo_max_post_step_mean_kl=max_post_kl,
            actor_grad_norm=self._actor_optimizer.get_last_grad_norm(),
            reppo_entropy_reference=self._entropy_ref.detach().clone(),
            reppo_std_mean=self._model._action_dist._logstd_net.detach().exp().mean())
        return totals
