import torch

import learning.add_agent as add_agent
import learning.flow_add_model as flow_add_model
import util.torch_util as torch_util

# number of samples used to estimate the flow diagnostics each iteration
FLOW_STAT_SAMPLES = 4096

class FlowADDAgent(add_agent.ADDAgent):
    """ADD agent with a potential-circulation differential-flow discriminator
    D(x_t-1, x_t).

    x_t is ADD's tracking-error differential in the normalized differential
    space. Positive samples are the ideal transition (x_t-1, x_t) = (0, 0),
    negatives are policy transitions, and the gradient penalty is applied on
    the joint input [x_t-1, x_t]. The reward stays -log(1 - D).
    """
    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = flow_add_model.FlowADDModel(model_config, self._env)
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

        disc_r = self._calc_disc_rewards(norm_obs_diff, norm_obs_diff_prev)
        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)

        r = self._task_reward_weight * task_r + self._disc_reward_weight * disc_r
        self._exp_buffer.set_data_flat("reward", r)

        if (self._need_normalizer_update()):
            self._disc_obs_norm.record(obs_diff)

        info = {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std
        }
        flow_info = self._compute_flow_stats(norm_obs_diff, norm_obs_diff_prev)
        info.update(flow_info)
        return info

    def _calc_disc_rewards(self, norm_obs_diff, norm_obs_diff_prev):
        with torch.no_grad():
            disc_inputs = {"disc_obs": norm_obs_diff, "disc_obs_prev": norm_obs_diff_prev}
            disc_logits = torch_util.eval_minibatch(self._model.eval_disc, disc_inputs, self._disc_eval_batch_size)
            disc_logits = disc_logits.squeeze(-1)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self._device)))
            disc_r *= self._disc_reward_scale
        return disc_r

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
        if (self._model.get_disc_mode() == flow_add_model.DISC_MODE_CONCAT):
            return dict()

        with torch.no_grad():
            n = norm_obs_diff.shape[0]
            num_samples = min(n, FLOW_STAT_SAMPLES)
            idx = torch.randperm(n, device=self._device, dtype=torch.long)[:num_samples]
            diff = norm_obs_diff[idx]
            diff_prev = norm_obs_diff_prev[idx]

            q_prog, q_circ = self._model.eval_flow_scores(diff, diff_prev)

            eps = 1e-8
            # mean signed progress: > 0 means the error energy E_S is shrinking
            prog_mean = torch.mean(q_prog)
            # fraction of the flow score magnitude carried by the circulation term
            circ_ratio = torch.mean(torch.abs(q_circ)) \
                         / (torch.mean(torch.abs(q_prog)) + torch.mean(torch.abs(q_circ)) + eps)

            flow_norm = torch.mean(torch.norm(diff - diff_prev, dim=-1))
            s_norm, a_norm = self._model.get_flow_matrix_norms()

        info = {
            "disc_flow_prog_mean": prog_mean,
            "disc_flow_circ_ratio": circ_ratio,
            "disc_flow_norm": flow_norm,
            "disc_flow_s_norm": s_norm,
            "disc_flow_a_norm": a_norm
        }
        return info
