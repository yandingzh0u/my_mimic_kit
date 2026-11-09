"""Independent Body-split SN variant with raw logits and unscaled ADD rewards."""

import torch

from learning.add_agent import calc_unscaled_disc_reward
from learning.dare_agent import DAREAgent
from learning.dare_multi_agent import DAREMultiAgent
from learning.dare_body_split_sn_model import (
    BODY_SPLIT_SN_SCHEMA_VERSION, HIDDEN_WIDTHS, DAREBodySplitSNModel,
    hierarchical_parent_mean,
)
import util.torch_util as torch_util


REWARD_AGGREGATION = "mapped_root_plus_half_bodypos_half_bodyrot_plus_velocity_over_three"


class DAREBodySplitSNAgent(DAREMultiAgent):
    def _build_model(self, config):
        self._model = DAREBodySplitSNModel(config["model"], self._env)


    @torch.no_grad()
    def _calc_disc_rewards(self, norm_diff):
        logits = torch_util.eval_minibatch(
            self._model.eval_disc, {"disc_obs": norm_diff}, self._disc_eval_batch_size)
        # The baseline ADD clipping applies separately, before the hierarchy.
        return hierarchical_parent_mean(calc_unscaled_disc_reward(logits))

    def _get_checkpoint_extra_state(self):
        # Bypass DAREMulti's changing reward metadata; keep a separate schema.
        state = DAREAgent._get_checkpoint_extra_state(self)
        state["body_split_sn"] = {
            "version": BODY_SPLIT_SN_SCHEMA_VERSION,
            "groups": self._model.get_disc_error_groups(),
            "hidden_widths": HIDDEN_WIDTHS,
            "constraint": "ordinary_sn_hidden_and_head",
            "reward_aggregation": REWARD_AGGREGATION,
            "loss_aggregation": "sum_of_independent_losses",
        }
        return state

    def _load_checkpoint_extra_state(self, state):
        saved = state.get("body_split_sn", {})
        groups = tuple((name, tuple(indices)) for name, indices in saved.get("groups", ()))
        if (saved.get("version") != BODY_SPLIT_SN_SCHEMA_VERSION
                or groups != self._model.get_disc_error_groups()
                or tuple(saved.get("hidden_widths", ())) != HIDDEN_WIDTHS
                or saved.get("constraint") != "ordinary_sn_hidden_and_head"
                or saved.get("reward_aggregation") != REWARD_AGGREGATION
                or saved.get("loss_aggregation") != "sum_of_independent_losses"):
            raise ValueError("Cannot resume with different Body-split SN semantics")
        DAREAgent._load_checkpoint_extra_state(self, state)
