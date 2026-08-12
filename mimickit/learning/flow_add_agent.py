import torch

import learning.add_agent as add_agent
import learning.flow_add_model as flow_add_model
import util.torch_util as torch_util

# number of samples used to estimate the flow diagnostics each iteration
FLOW_STAT_SAMPLES = 4096

class FlowADDAgent(add_agent.ADDAgent):
    """ADD agent with a differential-flow discriminator D(delta, v).

    delta_t is ADD's tracking-error differential and
    v_t = norm(delta_t) - norm(delta_t-1) is its per-step flow, computed in the
    same normalized differential space. Positive samples are the ideal point
    (delta, v) = (0, 0), negatives are policy samples (delta_t, v_t), and the
    gradient penalty is applied on the joint input [delta, v].
    """
    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _load_params(self, config):
        super()._load_params(config)
        self._disc_flow_scale = config.get("disc_flow_scale", 1.0)
        self._disc_flow_shuffle = config.get("disc_flow_shuffle", False)
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
        disc_flow = self._compute_disc_flow(norm_obs_diff, norm_obs_diff_prev)

        if (self._disc_flow_shuffle):
            disc_flow = self._shuffle_flow(disc_flow)

        disc_r = self._calc_disc_rewards(norm_obs_diff, disc_flow)
        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)

        r = self._task_reward_weight * task_r + self._disc_reward_weight * disc_r
        self._exp_buffer.set_data_flat("reward", r)

        if (self._need_normalizer_update()):
            self._disc_obs_norm.record(obs_diff)

        info = {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std
        }
        flow_info = self._compute_flow_stats(norm_obs_diff, disc_flow)
        info.update(flow_info)
        return info

    def _calc_disc_rewards(self, norm_obs_diff, disc_flow):
        with torch.no_grad():
            disc_inputs = {"disc_obs": norm_obs_diff, "disc_flow": disc_flow}
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

        # positive sample is the ideal point (delta, v) = (0, 0)
        pos_diff = self._pos_diff.clone()
        pos_diff = pos_diff.unsqueeze(dim=0)
        pos_flow = torch.zeros_like(pos_diff)
        pos_diff.requires_grad_(True)
        pos_flow.requires_grad_(True)
        disc_pos_logit = self._model.eval_disc(pos_diff, pos_flow)
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
        flow_obs = self._compute_disc_flow(norm_diff_obs, norm_diff_obs_prev)

        if (self._disc_flow_shuffle):
            flow_obs = self._shuffle_flow(flow_obs)

        norm_diff_obs = norm_diff_obs.detach()
        flow_obs = flow_obs.detach()
        norm_diff_obs.requires_grad_(True)
        flow_obs.requires_grad_(True)
        disc_neg_logit = self._model.eval_disc(norm_diff_obs, flow_obs)
        disc_neg_logit = disc_neg_logit.squeeze(-1)

        disc_loss_pos = self._disc_loss_pos(disc_pos_logit)
        disc_loss_neg = self._disc_loss_neg(disc_neg_logit)
        disc_loss = 0.5 * (disc_loss_pos + disc_loss_neg)

        # logit reg
        logit_weights = self._model.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty on the joint input x = [delta, v]
        disc_neg_grads = torch.autograd.grad(disc_neg_logit, [norm_diff_obs, flow_obs],
                                             grad_outputs=torch.ones_like(disc_neg_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_neg_grad_squared = torch.sum(torch.square(disc_neg_grads[0]), dim=-1) \
                                + torch.sum(torch.square(disc_neg_grads[1]), dim=-1)

        disc_pos_grads = torch.autograd.grad(disc_pos_logit, [pos_diff, pos_flow],
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

    def _compute_disc_flow(self, norm_obs_diff, norm_obs_diff_prev):
        disc_flow = norm_obs_diff - norm_obs_diff_prev
        if (self._disc_flow_scale != 1.0):
            disc_flow = self._disc_flow_scale * disc_flow
        return disc_flow

    def _shuffle_flow(self, disc_flow):
        # FlowADD-Shuffle ablation: break the pairing between delta_t and v_t
        n = disc_flow.shape[0]
        rand_idx = torch.randperm(n, device=disc_flow.device, dtype=torch.long)
        disc_flow = disc_flow[rand_idx]
        return disc_flow

    def _compute_flow_stats(self, norm_obs_diff, disc_flow):
        if (self._model.get_disc_mode() == flow_add_model.DISC_MODE_CONCAT):
            return dict()

        with torch.no_grad():
            n = norm_obs_diff.shape[0]
            num_samples = min(n, FLOW_STAT_SAMPLES)
            idx = torch.randperm(n, device=self._device, dtype=torch.long)[:num_samples]
            diff = norm_obs_diff[idx]
            flow = disc_flow[idx]

            v_star, v_star_tan, G = self._model.eval_disc_flow(diff)

            eps = 1e-8
            # flow alignment: cos(v_t, v*(delta_t))
            flow_cos = torch.sum(flow * v_star, dim=-1) \
                       / (torch.norm(flow, dim=-1) * torch.norm(v_star, dim=-1) + eps)
            flow_alignment = torch.mean(flow_cos)

            # tangential contribution ratio R_perp = E|q_tan| / E|q|
            q_tan = torch.sum(G * v_star_tan * flow, dim=-1)
            q_total = torch.sum(G * v_star * flow, dim=-1) \
                      - 0.5 * torch.sum(G * torch.square(flow), dim=-1)
            tan_ratio = torch.mean(torch.abs(q_tan)) / (torch.mean(torch.abs(q_total)) + eps)

            flow_norm = torch.mean(torch.norm(flow, dim=-1))

        info = {
            "disc_flow_alignment": flow_alignment,
            "disc_flow_tan_ratio": tan_ratio,
            "disc_flow_norm": flow_norm
        }
        return info
