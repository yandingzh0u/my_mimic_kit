"""Reference-reset coverage at the existing control resolution."""
import math

import torch


def control_cells(duration, timestep):
    if not (math.isfinite(duration) and duration >= 0
            and math.isfinite(timestep) and timestep > 0):
        raise ValueError("Invalid motion duration or control timestep")
    return max(1, math.ceil(math.nextafter(duration / timestep, -math.inf)))


class PhaseCoverage:
    """Visit every control cell in each F consecutive reference resets.

    An affine permutation uses constant memory, including for long motions.
    The jitter keeps starts continuous within cells. This covers reset
    requests; it cannot manufacture observations absent from a later rollout.
    """
    def __init__(self, cells):
        self.cells = int(cells)
        if not 1 <= self.cells <= 2**31 - 1:
            raise ValueError("Unsupported reference-reset resolution")
        self.offset = int(torch.randint(self.cells, ()))
        self.stride = 1
        if self.cells > 1:
            self.stride = int(torch.randint(1, self.cells, ()))
            while math.gcd(self.stride, self.cells) != 1:
                self.stride = (self.stride + 1) % self.cells or 1
        self.cursor = 0

    def draw_times(self, lengths):
        n = lengths.numel()
        positions = (torch.arange(n, device=lengths.device) + self.cursor) % self.cells
        cell = (positions * self.stride + self.offset) % self.cells
        jitter = torch.rand(n, device=lengths.device, dtype=torch.float64)
        times = ((cell + jitter) / self.cells * lengths.double()).to(lengths.dtype)
        # Float32 simulation times can round across a cell boundary. Repair
        # only those entries and verify the actual phase calculation used by
        # the environment, rather than just the intermediate double phase.
        def recovered(t):
            return ((t / lengths).double() * self.cells).long().clamp_max(self.cells - 1)
        bad = recovered(times) != cell
        times[bad] = ((cell[bad].double() + .5) / self.cells * lengths[bad].double()).to(lengths.dtype)
        if not torch.equal(recovered(times), cell):
            raise ValueError("Simulation time precision cannot represent control cells")
        self.cursor = (self.cursor + n) % self.cells
        return times

    def state_dict(self):
        return dict(cells=self.cells, offset=self.offset, stride=self.stride, cursor=self.cursor)

    def load_state_dict(self, state):
        if int(state["cells"]) != self.cells:
            raise ValueError("Reference-reset resolution changed across resume")
        self.offset, self.stride, self.cursor = (int(state[k]) for k in ("offset", "stride", "cursor"))
        if not (0 <= self.offset < self.cells and 0 <= self.cursor < self.cells
                and 1 <= self.stride <= self.cells and math.gcd(self.stride, self.cells) == 1):
            raise ValueError("Invalid reference-reset checkpoint state")
