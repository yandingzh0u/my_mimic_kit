"""PPO targets and losses, independent of DARE rewards and rollout scheduling."""
import torch

import envs.base_env as base_env
import learning.rl_util as rl_util
import util.torch_util as torch_util


class PPO:
    def __init__(self, config):
        self._clip_ratio = config["ppo_clip_ratio"]
        self._norm_adv_clip = config["norm_adv_clip"]

    def build_train_data(self, agent):
        agent.eval()
        
        obs = agent._exp_buffer.get_data("obs")
        next_obs = agent._exp_buffer.get_data("next_obs")
        r = agent._exp_buffer.get_data("reward")
        done = agent._exp_buffer.get_data("done")
        rand_action_mask = agent._exp_buffer.get_data("rand_action_mask")
        
        norm_next_obs = agent._obs_norm.normalize(next_obs)
        next_critic_inputs = {"obs": norm_next_obs}
        next_vals = torch_util.eval_minibatch(agent._model.eval_critic, next_critic_inputs, agent._critic_eval_batch_size)
        next_vals = next_vals.squeeze(-1).detach()

        succ_val = agent._compute_succ_val()
        succ_mask = (done == base_env.DoneFlags.SUCC.value)
        next_vals[succ_mask] = succ_val

        fail_val = agent._compute_fail_val()
        fail_mask = (done == base_env.DoneFlags.FAIL.value)
        next_vals[fail_mask] = fail_val

        new_vals = rl_util.compute_td_lambda_return(r, next_vals, done, agent._discount, agent._td_lambda)

        norm_obs = agent._obs_norm.normalize(obs)
        critic_inputs = {"obs": norm_obs}
        vals = torch_util.eval_minibatch(agent._model.eval_critic, critic_inputs, agent._critic_eval_batch_size)
        vals = vals.squeeze(-1).detach()
        adv = new_vals - vals

        # PPO currently samples transitions uniformly from the rollout.  With
        # early termination this makes the actor/critic objective depend on
        # the policy's own phase occupancy: failed episodes contribute many
        # freshly reset states while late phases become underrepresented.
        # Reweight the existing motion-phase diagnostic so every occupied
        # phase bin contributes equally to both value and policy updates.
        phase_weight, phase_info = agent._build_phase_weights()

        # Termination-conditioned TD-target statistics are diagnostics only.
        # They make it possible to distinguish a critic-scale jump caused by
        # FAIL/TIME sampling from a genuine value-network update problem.
        td_flat = new_vals.detach().flatten()
        done_flat = done.flatten()

        def td_stats(mask):
            values = td_flat[mask]
            if values.numel() == 0:
                nan = torch.full((), float("nan"), device=td_flat.device)
                return (torch.zeros((), device=td_flat.device), nan, nan,
                        nan, nan, nan)
            quantiles = torch.quantile(values, torch.tensor(
                [0.1, 0.5, 0.9], device=values.device))
            return (torch.tensor(float(values.numel()), device=values.device),
                    values.mean(), values.std(unbiased=False), quantiles[0],
                    quantiles[1], quantiles[2])

        td_info = {}
        for name, flag in (("null", base_env.DoneFlags.NULL.value),
                           ("fail", base_env.DoneFlags.FAIL.value),
                           ("succ", base_env.DoneFlags.SUCC.value),
                           ("time", base_env.DoneFlags.TIME.value)):
            count, mean, std, p10, p50, p90 = td_stats(done_flat == flag)
            prefix = "td_target_{}".format(name)
            td_info.update({
                prefix + "_count": count,
                prefix + "_mean": mean,
                prefix + "_std": std,
                prefix + "_p10": p10,
                prefix + "_p50": p50,
                prefix + "_p90": p90,
            })
        
        rand_action_mask = (rand_action_mask == 1.0).flatten()
        adv_flat = adv.flatten()
        phase_weight_flat = phase_weight.flatten()
        rand_action_adv = adv_flat[rand_action_mask]
        rand_action_weight = phase_weight_flat[rand_action_mask]
        adv_mean, adv_std = agent._weighted_mean_std(
            rand_action_adv, rand_action_weight)
        norm_adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-5)
        norm_adv = torch.clamp(norm_adv, -self._norm_adv_clip, self._norm_adv_clip)
        
        agent._exp_buffer.set_data("tar_val", new_vals)
        agent._exp_buffer.set_data("adv", norm_adv)
        agent._exp_buffer.set_data("phase_weight", phase_weight)
        
        info = {
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            **phase_info,
            **td_info,
        }
        return info

    def compute_critic_loss(self, agent, batch):
        norm_obs = agent._obs_norm.normalize(batch["obs"])
        tar_val = batch["tar_val"]
        pred = agent._model.eval_critic(norm_obs)
        pred = pred.squeeze(-1)

        diff = tar_val - pred
        phase_weight = batch["phase_weight"]
        loss = agent._weighted_mean(torch.square(diff), phase_weight)

        info = {
            "critic_loss": loss
        }
        return info

    def compute_actor_loss(self, agent, batch):
        norm_obs = agent._obs_norm.normalize(batch["obs"])
        norm_a = agent._a_norm.normalize(batch["action"])
        old_a_logp = batch["a_logp"]
        adv = batch["adv"]
        rand_action_mask = batch["rand_action_mask"]
        phase_weight = batch["phase_weight"]

        # loss should only be computed using samples with random actions
        rand_action_mask = (rand_action_mask == 1.0)
        norm_obs = norm_obs[rand_action_mask]
        norm_a = norm_a[rand_action_mask]
        old_a_logp = old_a_logp[rand_action_mask]
        adv = adv[rand_action_mask]
        phase_weight = phase_weight[rand_action_mask]

        a_dist = agent._model.eval_actor(norm_obs)
        a_logp = a_dist.log_prob(norm_a)

        a_ratio = torch.exp(a_logp - old_a_logp)
        actor_loss0 = adv * a_ratio
        actor_loss1 = adv * torch.clamp(a_ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio)
        actor_objective = torch.minimum(actor_loss0, actor_loss1)
        actor_loss = -agent._weighted_mean(actor_objective, phase_weight)
        
        clip_frac = (torch.abs(a_ratio - 1.0) > self._clip_ratio).type(torch.float)
        clip_frac = agent._weighted_mean(clip_frac, phase_weight)
        imp_ratio = agent._weighted_mean(a_ratio, phase_weight)
        
        info = {
            "actor_loss": actor_loss,
            "clip_frac": clip_frac.detach(),
            "imp_ratio": imp_ratio.detach()
        }

        if (agent._action_bound_weight != 0):
            action_bound_loss = agent._compute_action_bound_loss(a_dist)
            if (action_bound_loss is not None):
                action_bound_loss = agent._weighted_mean(
                    action_bound_loss, phase_weight)
                actor_loss += agent._action_bound_weight * action_bound_loss
                info["action_bound_loss"] = action_bound_loss.detach()

        if (agent._action_entropy_weight != 0):
            action_entropy = a_dist.entropy()
            action_entropy = agent._weighted_mean(action_entropy, phase_weight)
            actor_loss += -agent._action_entropy_weight * action_entropy
            info["action_entropy"] = action_entropy.detach()
        
        if (agent._action_reg_weight != 0):
            action_reg_loss = a_dist.param_reg()
            action_reg_loss = agent._weighted_mean(action_reg_loss, phase_weight)
            actor_loss += agent._action_reg_weight * action_reg_loss
            info["action_reg_loss"] = action_reg_loss.detach()
        
        return info
