"""Retired reward mappings must not silently resume as original DARE."""
import pytest
import torch

from learning.dare_agent import DAREAgent


def bare_agent():
    agent = object.__new__(DAREAgent)
    torch.nn.Module.__init__(agent)
    agent._trajectory_allocation_enabled = False
    agent._allocation = None
    return agent


def test_original_checkpoint_and_current_checkpoint_are_accepted():
    agent = bare_agent()
    agent._load_checkpoint_extra_state({})
    agent._load_checkpoint_extra_state(agent._get_checkpoint_extra_state())


def test_retired_reward_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="reward mapping"):
        bare_agent()._load_checkpoint_extra_state({"disc_reward_type": "retired"})


def test_unsupported_reward_config_is_rejected_before_parent_initialization():
    with pytest.raises(ValueError, match="original ADD"):
        bare_agent()._load_params({"disc_reward_type": "retired"})


def test_allocation_mode_cannot_silently_change_on_resume():
    with pytest.raises(ValueError, match="allocation mode"):
        bare_agent()._load_checkpoint_extra_state({"trajectory_allocation_enabled": True})
