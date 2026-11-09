"""Four independent ADD objectives, with fixed hidden-matrix NS maintenance."""

from learning.dare_agent import DAREAgent
from learning.dare_multi_agent import DAREMultiAgent
from learning.dare_multi4_ns_model import (
    DAREMulti4NSModel, HIDDEN_WIDTHS, MULTI4_NS_SCHEMA_VERSION,
)


class DAREMulti4NSAgent(DAREMultiAgent):
    """Reuse summed BCE, four mapped-reward mean, and one shared calibration."""

    def _build_model(self, config):
        self._model = DAREMulti4NSModel(config["model"], self._env)

    def _build_optimizer(self, config):
        super()._build_optimizer(config)
        self._install_disc_ns_hook()

    def _install_disc_ns_hook(self):
        """Run once after only the discriminator's underlying optimizer step.

        PyTorch post-step hooks finish before MPOptimizer.step performs its
        distributed parameter-synchronization check or returns to the caller.
        The hook itself is recreated when an agent is constructed, while its
        maintenance count lives in the serialized model state.
        """
        if getattr(self, "_ns_step_hook_handle", None) is not None:
            return
        self._ns_step_hook_handle = (
            self._disc_optimizer._optimizer.register_step_post_hook(
                self._after_disc_optimizer_step))

    def _after_disc_optimizer_step(self, optimizer, args, kwargs):
        self._model.maintain_hidden_orthogonality_()

    def _update_disc(self, batch_size, steps):
        info = super()._update_disc(batch_size, steps)
        # Report the completed count, rather than averaging pre-step counters
        # together with the minibatch loss measurements.
        info["disc_ns_update_count"] = self._model.get_ns_update_count()
        return info

    def _get_checkpoint_extra_state(self):
        state = DAREAgent._get_checkpoint_extra_state(self)
        state["multi4_ns"] = {
            "version": MULTI4_NS_SCHEMA_VERSION,
            "groups": self._model.get_disc_error_groups(),
            "hidden_widths": HIDDEN_WIDTHS,
            "hidden_constraint": "column_orthogonal_init_then_single_fp32_ns_post_step",
            "head_constraint": "spectral_norm",
            "reward_aggregation": "mean_of_mapped_rewards",
            "loss_aggregation": "sum_of_independent_losses",
            "calibration_gap_raw": self._calibration_gap_raw,
            "calibration_gaps_raw": self._calibration_gaps_raw,
            "ns_update_count": int(self._model.get_ns_update_count().item()),
        }
        return state

    def _load_checkpoint_extra_state(self, state):
        saved = state.get("multi4_ns", {})
        groups = tuple((name, tuple(indices)) for name, indices in saved.get("groups", ()))
        if (saved.get("version") != MULTI4_NS_SCHEMA_VERSION
                or groups != self._model.get_disc_error_groups()
                or tuple(saved.get("hidden_widths", ())) != HIDDEN_WIDTHS
                or saved.get("hidden_constraint")
                != "column_orthogonal_init_then_single_fp32_ns_post_step"
                or saved.get("head_constraint") != "spectral_norm"
                or saved.get("reward_aggregation") != "mean_of_mapped_rewards"
                or saved.get("loss_aggregation") != "sum_of_independent_losses"
                or saved.get("ns_update_count")
                != int(self._model.get_ns_update_count().item())):
            raise ValueError("Cannot resume with different four-discriminator NS semantics")
        DAREAgent._load_checkpoint_extra_state(self, state)
        self._calibration_gap_raw = saved["calibration_gap_raw"]
        self._calibration_gaps_raw = saved["calibration_gaps_raw"]
