"""Compatibility entry point for plain PPO (training lives in OnPolicyAgent)."""
from learning.on_policy_agent import OnPolicyAgent


class PPOAgent(OnPolicyAgent):
    pass
