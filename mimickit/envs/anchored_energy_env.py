import torch

import envs.aligned_add_env as aligned_add_env


REFERENCE_PHI_STD_FLOOR = 1e-4


def build_uniform_reference_schedule(num_motions, num_samples, device):
    """Build a deterministic, uniformly spaced phase grid per motion."""
    num_motions = int(num_motions)
    num_samples = int(num_samples)
    if num_motions <= 0:
        raise ValueError("num_motions must be positive")
    if num_samples < num_motions:
        raise ValueError(
            "reference_phi_stats_samples must cover every reference motion")

    samples_per_motion = [num_samples // num_motions] * num_motions
    for motion_id in range(num_samples % num_motions):
        samples_per_motion[motion_id] += 1

    motion_ids = []
    phases = []
    for motion_id, count in enumerate(samples_per_motion):
        # Midpoints avoid duplicating the first/last pose of wrapped motions
        # while remaining a deterministic uniform quadrature grid on [0, 1].
        motion_ids.append(torch.full(
            [count], motion_id, device=device, dtype=torch.long))
        phases.append(
            (torch.arange(count, device=device, dtype=torch.float32) + 0.5)
            / float(count))

    return torch.cat(motion_ids, dim=0), torch.cat(phases, dim=0)


def compute_reference_phi_stats(reference_phi, std_floor=REFERENCE_PHI_STD_FLOOR):
    """Return deterministic population moments with stable constant axes."""
    if reference_phi.ndim != 2 or reference_phi.shape[0] < 2:
        raise ValueError("reference_phi must contain at least two feature rows")

    mean = torch.mean(reference_phi, dim=0)
    std = torch.std(reference_phi, dim=0, correction=0)
    stable_std = torch.where(
        std < float(std_floor), torch.ones_like(std), std)
    return mean, stable_std


class AnchoredEnergyEnv(aligned_add_env.AlignedADDEnv):
    """AlignedADD with fixed, reference-only feature statistics.

    The policy observation is inherited byte-for-byte from ``AlignedADDEnv``.
    The additional statistics are computed only from the reference motion and
    are exposed to the energy agent through ``get_reference_phi_stats``.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        self._reference_phi_stats_samples = int(
            env_config.get("reference_phi_stats_samples", 4096))
        if self._reference_phi_stats_samples != 4096:
            raise ValueError(
                "the first anchored-energy implementation requires exactly "
                "4096 reference-phi statistics samples")

        super().__init__(
            env_config=env_config,
            engine_config=engine_config,
            num_envs=num_envs,
            device=device,
            visualize=visualize,
            record_video=record_video,
        )
        self._build_reference_phi_stats()

    def get_reference_phi_stats(self):
        """Return fixed reference-only mean/std tensors on the env device."""
        return self._reference_phi_mean, self._reference_phi_std

    def get_reference_phi_mean(self):
        return self._reference_phi_mean

    def get_reference_phi_std(self):
        return self._reference_phi_std

    def _build_reference_phi_stats(self):
        num_motions = self._motion_lib.get_num_motions()
        motion_ids, phases = build_uniform_reference_schedule(
            num_motions=num_motions,
            num_samples=self._reference_phi_stats_samples,
            device=self._device,
        )
        motion_lengths = self._motion_lib.get_motion_lengths()
        motion_times = phases * motion_lengths[motion_ids]

        with torch.no_grad():
            reference_phi = self._compute_disc_obs_demo(
                motion_ids, motion_times)
            expected_dim = self.get_aligned_command_dim()
            if reference_phi.shape != (
                    self._reference_phi_stats_samples, expected_dim):
                raise RuntimeError(
                    "reference phi has shape {}, expected ({}, {})".format(
                        tuple(reference_phi.shape),
                        self._reference_phi_stats_samples,
                        expected_dim,
                    ))
            mean, std = compute_reference_phi_stats(reference_phi)

        self._reference_phi_mean = mean.detach()
        self._reference_phi_std = std.detach()
