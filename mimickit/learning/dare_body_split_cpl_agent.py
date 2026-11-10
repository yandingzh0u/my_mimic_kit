"""CPL variant retaining the Body-split classifier loss and reward hierarchy."""

from learning.dare_agent import DAREAgent
from learning.dare_body_split_sn_agent import (
    DAREBodySplitSNAgent, REWARD_AGGREGATION,
)
from learning.dare_body_split_cpl_model import (
    BODY_SPLIT_CPL_SCHEMA_VERSION, CPL_CONSTRAINT, CPL_PLACEMENT,
    DAREBodySplitCPLModel, HIDDEN_WIDTHS,
)


class DAREBodySplitCPLAgent(DAREBodySplitSNAgent):
    def _build_model(self, config):
        self._model = DAREBodySplitCPLModel(config["model"], self._env)

    def _get_checkpoint_extra_state(self):
        state = DAREAgent._get_checkpoint_extra_state(self)
        state["body_split_cpl"] = {
            "version": BODY_SPLIT_CPL_SCHEMA_VERSION,
            "groups": self._model.get_disc_error_groups(),
            "hidden_widths": HIDDEN_WIDTHS,
            "cpl_depth": self._model.get_disc_cpl_depth(),
            "constraint": CPL_CONSTRAINT,
            "placement": CPL_PLACEMENT,
            "reward_aggregation": REWARD_AGGREGATION,
            "loss_aggregation": "sum_of_independent_losses",
        }
        return state

    def _load_checkpoint_extra_state(self, state):
        saved = state.get("body_split_cpl", {})
        groups = tuple((name, tuple(indices)) for name, indices in saved.get("groups", ()))
        if (saved.get("version") != BODY_SPLIT_CPL_SCHEMA_VERSION
                or groups != self._model.get_disc_error_groups()
                or tuple(saved.get("hidden_widths", ())) != HIDDEN_WIDTHS
                or saved.get("cpl_depth") != self._model.get_disc_cpl_depth()
                or saved.get("constraint") != CPL_CONSTRAINT
                or saved.get("placement") != CPL_PLACEMENT
                or saved.get("reward_aggregation") != REWARD_AGGREGATION
                or saved.get("loss_aggregation") != "sum_of_independent_losses"):
            raise ValueError("Cannot resume with different Body-split CPL semantics")
        DAREAgent._load_checkpoint_extra_state(self, state)
