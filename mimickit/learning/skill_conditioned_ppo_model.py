import gymnasium.spaces as spaces
import numpy as np
import torch

import learning.ppo_model as ppo_model


class SkillConditionedPPOModel(ppo_model.PPOModel):
    """PPO actor/critic whose only skill command is a continuous 8-D z."""

    def __init__(self, config, env, latent_dim=8):
        self.latent_dim = int(latent_dim)
        if self.latent_dim != 8:
            raise ValueError("R2 requires latent_dim=8")
        super().__init__(config, env)

    def eval_actor(self, obs, latent):
        return super().eval_actor(self._condition(obs, latent))

    def eval_critic(self, obs, latent):
        return super().eval_critic(self._condition(obs, latent))

    def _build_actor_input_dict(self, env):
        return {"obs_skill": self._conditioned_obs_space(env)}

    def _build_critic_input_dict(self, env):
        return {"obs_skill": self._conditioned_obs_space(env)}

    def _conditioned_obs_space(self, env):
        obs_space = env.get_obs_space()
        if not isinstance(obs_space, spaces.Box) or len(obs_space.shape) != 1:
            raise TypeError("SkillConditionedPPOModel requires a flat Box observation")
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(np.prod(obs_space.shape)) + self.latent_dim,),
            dtype=obs_space.dtype,
        )

    def _condition(self, obs, latent):
        expected_shape = obs.shape[:-1] + (self.latent_dim,)
        if latent.shape != expected_shape:
            raise ValueError(
                "latent must have shape {}, got {}".format(
                    tuple(expected_shape), tuple(latent.shape)
                )
            )
        if not torch.isfinite(latent).all():
            raise ValueError("latent contains non-finite values")
        return torch.cat((obs, latent.to(device=obs.device, dtype=obs.dtype)), dim=-1)
