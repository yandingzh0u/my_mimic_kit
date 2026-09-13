import envs.deepmimic_env as deepmimic_env

class StaticObjectsEnv(deepmimic_env.DeepMimicEnv):
    """Backward-compatible alias for DeepMimic with ``objects`` configured.

    Static-object construction now lives in DeepMimicEnv so AMP, ADD, and
    residual-coordinate ADD can use exactly the same interaction geometry.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return
