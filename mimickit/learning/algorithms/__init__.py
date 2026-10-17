"""Objective selection. Reward agents do not inherit an RL algorithm."""


def build_algorithm(config):
    name = config.get("algorithm", "PPO")
    if name == "PPO":
        from learning.algorithms.ppo import PPO
        return PPO(config)
    raise ValueError(f"Unsupported on-policy algorithm: {name}")
