import math

import torch
import torch.nn.functional as functional

import learning.aligned_add_agent as aligned_add_agent
import learning.hinge_sn_add_model as hinge_sn_add_model
import util.torch_util as torch_util


class HingeSNAlignedADDAgent(aligned_add_agent.AlignedADDAgent):
    """Aligned ADD with a configurable Hinge--SN discriminator.

    The differential construction, replay buffer, PPO updates, aligned policy
    observations, and task/reward mixing are inherited unchanged.  Four paper
    variants are selected only through the discriminator GP, reward type, and
    consistency coefficients in the agent configuration.
    """

    _REWARD_ADD_SOFTPLUS = "add_softplus"
    _REWARD_SMOOTH_MARGIN = "smooth_margin"

    def _load_params(self, config):
        super()._load_params(config)

        self._disc_hinge_margin = float(config["disc_hinge_margin"])
        self._disc_spectral_norm = bool(config["disc_spectral_norm"])
        self._disc_sn_power_iterations = int(
            config["disc_sn_power_iterations"])
        self._disc_reward_type = str(config["disc_reward_type"])
        self._disc_consistency_weight = float(
            config["disc_consistency_weight"])
        self._disc_consistency_noise_std = float(
            config["disc_consistency_noise_std"])

        if self._disc_hinge_margin <= 0:
            raise ValueError("disc_hinge_margin must be positive")
        if self._disc_sn_power_iterations < 1:
            raise ValueError("disc_sn_power_iterations must be at least 1")
        if self._disc_reward_type not in {
                self._REWARD_ADD_SOFTPLUS,
                self._REWARD_SMOOTH_MARGIN}:
            raise ValueError(
                "Unsupported disc_reward_type: {}".format(
                    self._disc_reward_type))
        if self._disc_consistency_weight < 0:
            raise ValueError("disc_consistency_weight must be nonnegative")
        if self._disc_consistency_noise_std < 0:
            raise ValueError("disc_consistency_noise_std must be nonnegative")

        # A private stream makes CR reproducible without consuming the global
        # PyTorch/CUDA stream used by rollouts and minibatch sampling.  The run
        # seed has already been installed as torch.initial_seed() at this point.
        self._disc_consistency_generator = torch.Generator(
            device=torch.device(self._device))
        self._disc_consistency_generator.manual_seed(torch.initial_seed())
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = hinge_sn_add_model.HingeSNADDModel(
            model_config, self._env,
            spectral_norm=self._disc_spectral_norm,
            sn_power_iterations=self._disc_sn_power_iterations)
        return

    def get_extra_state(self):
        """Include the private CR stream in weights and full checkpoints."""
        return {
            "disc_consistency_rng_state":
                self._disc_consistency_generator.get_state().clone()
        }

    def set_extra_state(self, state):
        if state is None:
            return
        rng_state = state.get("disc_consistency_rng_state", None)
        if rng_state is not None:
            self._disc_consistency_generator.set_state(rng_state.cpu())
        return

    def _sample_consistency_noise(self, reference):
        if self._disc_consistency_noise_std == 0:
            return torch.zeros_like(reference)
        noise = torch.randn(
            reference.shape, dtype=reference.dtype,
            device=reference.device,
            generator=self._disc_consistency_generator)
        return noise * self._disc_consistency_noise_std

    def _compute_disc_loss(self, batch):
        disc_obs = batch["disc_obs"]
        tar_disc_obs = batch["disc_obs_demo"]

        fresh_diff = tar_disc_obs - disc_obs
        fresh_count = fresh_diff.shape[0]

        replay_data = self._disc_buffer.sample(fresh_count)
        replay_disc_obs = replay_data["disc_obs"]
        replay_tar_disc_obs = replay_data["disc_obs_demo"]
        replay_diff = replay_tar_disc_obs - replay_disc_obs

        diff_obs = torch.cat([fresh_diff, replay_diff], dim=0)
        norm_diff_obs = self._disc_obs_norm.normalize(diff_obs)

        # GP=0 is a true fast path: no input requires_grad flag and no
        # autograd.grad/create_graph call are constructed.
        use_grad_penalty = self._disc_grad_penalty != 0
        if use_grad_penalty:
            norm_diff_obs.requires_grad_(True)

        pos_diff = self._pos_diff.clone().unsqueeze(dim=0)
        disc_inputs = [pos_diff, norm_diff_obs]

        use_consistency = (self._disc_consistency_weight != 0
                           and self._disc_consistency_noise_std != 0)
        if use_consistency:
            norm_fresh_diff = norm_diff_obs[:fresh_count]
            consistency_noise = self._sample_consistency_noise(
                norm_fresh_diff)
            perturbed_fresh_diff = norm_fresh_diff + consistency_noise
            disc_inputs.append(perturbed_fresh_diff)

        # One discriminator forward is intentional.  In training mode SN
        # updates its power-iteration buffers on every forward; concatenating
        # all views ensures every score in this loss uses the same SN weights.
        all_disc_inputs = torch.cat(disc_inputs, dim=0)
        all_disc_logits = self._model.eval_disc(all_disc_inputs).squeeze(-1)

        neg_end = 1 + norm_diff_obs.shape[0]
        disc_pos_logit = all_disc_logits[:1]
        disc_neg_logit = all_disc_logits[1:neg_end]
        disc_fresh_logit = disc_neg_logit[:fresh_count]

        margin = self._disc_hinge_margin
        disc_hinge_pos_loss = torch.mean(
            functional.relu(margin - disc_pos_logit))
        disc_hinge_neg_loss = torch.mean(
            functional.relu(margin + disc_neg_logit))
        disc_hinge_loss = disc_hinge_pos_loss + disc_hinge_neg_loss
        disc_loss = disc_hinge_loss

        zero = disc_loss.new_zeros(())
        disc_consistency_loss = zero
        if use_consistency:
            disc_perturbed_logit = all_disc_logits[neg_end:]
            disc_consistency_loss = torch.mean(torch.square(
                disc_fresh_logit - disc_perturbed_logit))
            disc_loss = (disc_loss
                         + self._disc_consistency_weight
                         * disc_consistency_loss)

        disc_grad_penalty = zero
        if use_grad_penalty:
            disc_neg_grad = torch.autograd.grad(
                disc_neg_logit, norm_diff_obs,
                grad_outputs=torch.ones_like(disc_neg_logit),
                create_graph=True, retain_graph=True, only_inputs=True)[0]
            disc_grad_penalty = torch.mean(torch.sum(
                torch.square(disc_neg_grad), dim=-1))
            disc_loss = (disc_loss
                         + self._disc_grad_penalty * disc_grad_penalty)

        disc_logit_loss = zero
        if self._disc_logit_reg != 0:
            logit_weights = self._model.get_disc_logit_weights()
            disc_logit_loss = torch.sum(torch.square(logit_weights))
            disc_loss = (disc_loss
                         + self._disc_logit_reg * disc_logit_loss)

        disc_neg_acc, disc_pos_acc = self._compute_disc_acc(
            disc_neg_logit, disc_pos_logit)
        disc_pos_logit_mean = torch.mean(disc_pos_logit)
        disc_neg_logit_mean = torch.mean(disc_neg_logit)

        disc_info = {
            "disc_loss": disc_loss,
            "disc_hinge_loss": disc_hinge_loss.detach(),
            "disc_hinge_pos_loss": disc_hinge_pos_loss.detach(),
            "disc_hinge_neg_loss": disc_hinge_neg_loss.detach(),
            "disc_hinge_pos_active_frac": torch.mean(
                (disc_pos_logit < margin).float()).detach(),
            "disc_hinge_neg_active_frac": torch.mean(
                (disc_neg_logit > -margin).float()).detach(),
            "disc_consistency_loss": disc_consistency_loss.detach(),
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_logit_loss": disc_logit_loss.detach(),
            "disc_pos_acc": disc_pos_acc.detach(),
            "disc_neg_acc": disc_neg_acc.detach(),
            "disc_pos_logit": disc_pos_logit_mean.detach(),
            "disc_neg_logit": disc_neg_logit_mean.detach(),
            "disc_score_gap": (
                disc_pos_logit_mean - disc_neg_logit_mean).detach(),
        }
        disc_info.update(self._model.get_disc_sn_diagnostics())
        return disc_info

    def _calc_disc_rewards(self, norm_disc_obs):
        with torch.no_grad():
            disc_inputs = {"disc_obs": norm_disc_obs}
            disc_logits = torch_util.eval_minibatch(
                self._model.eval_disc, disc_inputs,
                self._disc_eval_batch_size).squeeze(-1)

            if self._disc_reward_type == self._REWARD_ADD_SOFTPLUS:
                # Exactly ADD's -log(1-sigmoid(f)) mapping.
                disc_r = functional.softplus(disc_logits)
                disc_r *= self._disc_reward_scale
            elif self._disc_reward_type == self._REWARD_SMOOTH_MARGIN:
                margin = self._disc_hinge_margin
                denominator = math.log1p(math.exp(2.0 * margin))
                disc_r = functional.softplus(disc_logits + margin)
                disc_r *= self._disc_reward_scale / denominator
            else:
                raise RuntimeError(
                    "Unsupported disc_reward_type: {}".format(
                        self._disc_reward_type))
        return disc_r
