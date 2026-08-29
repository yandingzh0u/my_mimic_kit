import learning.amp_model as amp_model


class ADDModel(amp_model.AMPModel):
    """Vanilla ADD model operating directly on normalized differentials."""

    def __init__(self, config, env):
        super().__init__(config, env)
        return
