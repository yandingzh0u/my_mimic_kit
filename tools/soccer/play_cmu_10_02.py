"""Play the archived CMU 10_02 checkpoint with its original action guide."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mimickit"))

import numpy as np
import torch

import run
from envs import soccer_env
from envs.char_env import CharEnv


class CMUSoccerReplayEnv(soccer_env.SoccerEnv):
    """Keep the legacy controller local to playback of this saved policy."""

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        config = env_config.copy()
        self._residual_limit = float(config.pop("residual_limit_rad"))
        with np.load(config.pop("physical_reference"), allow_pickle=False) as ref:
            self._reference_fps = int(ref["fps"])
            self._guide = {name: torch.as_tensor(ref[name], device=device)
                           for name in ("commands", "root_vel", "root_ang_vel", "dof_vel")}
        if self._reference_fps != engine_config["control_freq"]:
            raise ValueError("Replay control frequency must match the archived guide")
        super().__init__(config, engine_config, num_envs, device, visualize, record_video)

    def _reference_frame(self):
        return (self._get_motion_times() * self._reference_fps).round().long()

    def _ref_state_init(self, env_ids):
        super()._ref_state_init(env_ids)
        frames = self._reference_frame()[env_ids]
        char_id = self._get_char_id()
        for name in ("root_vel", "root_ang_vel", "dof_vel"):
            getattr(self._engine, "set_" + name)(env_ids, char_id, self._guide[name][frames])

    def _apply_action(self, actions):
        low, high = self._action_bound_low, self._action_bound_high
        residual = (2 * (actions - low) / (high - low) - 1).clamp(-1, 1)
        commands = self._guide["commands"]
        frame = self._reference_frame().clamp(0, len(commands) - 1)
        CharEnv._apply_action(self, commands[frame] + self._residual_limit * residual)


def main(argv):
    defaults = ["--arg_file", "args/mira_monotone_humanoid_soccer_cmu_10_02_play_args.txt"]
    if "--arg_file" not in argv:
        argv = [*argv, *defaults]
    if run.load_args(argv).parse_string("mode") != "test":
        raise ValueError("This entry point is for playback only")
    original = soccer_env.SoccerEnv
    soccer_env.SoccerEnv = CMUSoccerReplayEnv
    try:
        run.main(argv)
    finally:
        soccer_env.SoccerEnv = original


if __name__ == "__main__":
    main(sys.argv)
