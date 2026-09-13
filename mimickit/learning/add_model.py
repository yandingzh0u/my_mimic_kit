import learning.amp_model as amp_model


class ADDModel(amp_model.AMPModel):
    """Official ADD model: the standard dense AMP discriminator."""

    def __init__(self, config, env):
        super().__init__(config, env)
