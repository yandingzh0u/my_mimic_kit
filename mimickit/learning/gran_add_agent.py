import learning.add_agent as add_agent
import learning.gran_add_model as gran_add_model


class GraNADDAgent(add_agent.ADDAgent):
    """ADD with no GP and a scale-separated gradient-normalized head."""

    def _load_params(self, config):
        if "disc_grad_penalty" in config:
            raise ValueError(
                "GraN-ADD removes disc_grad_penalty; delete it from config")
        compatibility_config = dict(config)
        compatibility_config["disc_grad_penalty"] = 0.0
        super()._load_params(compatibility_config)
        return

    def _build_model(self, config):
        self._model = gran_add_model.GraNADDModel(
            config["model"], self._env)
        return

    def _compute_disc_loss(self, batch):
        self._model.begin_gran_diagnostics()
        disc_info = super()._compute_disc_loss(batch)
        disc_info.update(self._model.end_gran_diagnostics())
        return disc_info
