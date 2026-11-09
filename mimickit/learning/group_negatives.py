"""Fixed-budget, zero-preserving structured negatives for ADD differences."""

from dataclasses import dataclass

import torch


@dataclass
class GroupNegativeBatch:
    values: torch.Tensor
    group_ids: torch.Tensor
    requested_group_ids: torch.Tensor
    valid: torch.Tensor


class GroupNegativeSampler:
    """Replace half of each source with one uniformly selected error group.

    The 1:1 real/masked split is a fixed sampling rule, not a tunable loss
    coefficient. A zero group falls back to its original real row; an entirely
    zero real row is excluded from negative BCE. Neither case adds a tolerance.
    This helper never owns replay data or normalization statistics.
    """

    VERSION = 1

    def __init__(self, groups, input_dim, device, seed):
        self.groups = tuple((name, tuple(indices)) for name, indices in groups)
        if not self.groups or any(not indices for _, indices in self.groups):
            raise ValueError("Structured negatives require nonempty error groups")
        names = [name for name, _ in self.groups]
        if len(set(names)) != len(names):
            raise ValueError("Structured negative group names must be unique")
        indices = [index for _, group in self.groups for index in group]
        if sorted(indices) != list(range(input_dim)):
            raise ValueError("Error groups must partition the discriminator input")

        self.input_dim = input_dim
        self.device = torch.device(device)
        self._masks = torch.zeros(
            (len(self.groups), input_dim), device=self.device, dtype=torch.bool)
        for group_id, (_, indices) in enumerate(self.groups):
            self._masks[group_id, list(indices)] = True
        # A separate stream leaves policy sampling and initial weights untouched.
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)

    def sample(self, values, source_sizes):
        if (values.ndim != 2 or values.shape[1] != self.input_dim
                or values.device != self._masks.device):
            raise ValueError("Structured negative input has wrong shape or device")
        if (not source_sizes or any(size < 0 for size in source_sizes)
                or sum(source_sizes) != len(values)):
            raise ValueError("Structured negative source sizes do not match input")
        if not torch.isfinite(values).all():
            raise ValueError("Structured negatives require finite real differences")

        requested = torch.full(
            (len(values),), -1, dtype=torch.long, device=values.device)
        selected_rows = []
        begin = 0
        for size in source_sizes:
            rows = torch.randperm(
                size, generator=self._generator, device=values.device)[:size // 2]
            selected_rows.append(rows + begin)
            begin += size
        selected_rows = torch.cat(selected_rows)
        requested[selected_rows] = torch.randint(
            len(self.groups), (len(selected_rows),), generator=self._generator,
            device=values.device)

        selected_values = values[selected_rows]
        masked = torch.where(
            self._masks[requested[selected_rows]], selected_values,
            torch.zeros_like(selected_values))
        nonzero_group = masked.ne(0).any(dim=-1)
        mixed = values.clone()
        mixed[selected_rows] = torch.where(
            nonzero_group.unsqueeze(-1), masked, selected_values)
        applied = torch.full_like(requested, -1)
        applied[selected_rows] = torch.where(
            nonzero_group, requested[selected_rows], -1)
        return GroupNegativeBatch(
            values=mixed, group_ids=applied, requested_group_ids=requested,
            valid=mixed.ne(0).any(dim=-1))

    def state_dict(self):
        return {
            "version": self.VERSION,
            "groups": self.groups,
            "rng_backend": self.device.type,
            "rng_state": self._generator.get_state().clone(),
        }

    def load_state_dict(self, state):
        if (state.get("version") != self.VERSION
                or state.get("groups") != self.groups
                or state.get("rng_backend") != self.device.type):
            raise ValueError("Structured negative sampler state is incompatible")
        self._generator.set_state(state["rng_state"].cpu())
