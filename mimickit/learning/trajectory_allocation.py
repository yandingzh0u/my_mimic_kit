"""Allocate future reference starts using observed episode flow and score moments.

The optimization controls a fixed-conditional, independent-sample second-moment
proxy. It is not a full-network PPO variance or policy-improvement guarantee.
"""

import numpy as np
from scipy.optimize import minimize
import torch


def solve_allocation(visits, second_moments):
    visits = np.asarray(visits, dtype=np.float64)
    moments = np.asarray(second_moments, dtype=np.float64)
    if (visits.ndim != 2 or visits.shape[0] != visits.shape[1]
            or moments.shape != (visits.shape[1],)):
        raise ValueError("Expected a square visit operator and one moment per phase")
    if (not np.isfinite(visits).all() or (visits < 0).any()
            or not np.isfinite(moments).all() or (moments < 0).any()):
        raise ValueError("Visit counts and moments must be finite and nonnegative")
    lengths = visits.sum(1)
    if (lengths <= 0).any() or ((visits.sum(0) == 0) & (moments > 0)).any():
        raise ValueError("Insufficient episodes or unreachable positive-moment phase")
    conditional = visits / lengths[:, None]
    initial = lengths / lengths.sum()
    coefficients = moments / moments.max() if moments.max() > 0 else moments
    if coefficients.sum() > 0:
        coefficients /= coefficients.sum()
    signal = coefficients > 0
    matrix = conditional[:, signal]
    coefficients = coefficients[signal]

    def objective(y):
        p = y @ matrix
        if (p <= 0).any():
            return np.inf
        return float(np.sum(coefficients / p))

    def derivative(y):
        p = np.maximum(y @ matrix, np.finfo(float).eps)
        return -(matrix @ (coefficients / p ** 2))

    y = initial.copy()
    certificate = dict(relative_optimality_gap=0., optimality_gap=0., refinements=0)
    if signal.any():
        fit = minimize(objective, y, jac=derivative, method="SLSQP",
                       bounds=[(0., 1.)] * len(y),
                       constraints={"type": "eq", "fun": lambda x: x.sum() - 1.,
                                    "jac": lambda x: np.ones_like(x)},
                       options={"ftol": 1e-12, "maxiter": 200})
        raw = np.asarray(fit.x)
        if (not np.isfinite(raw).all() or raw.min() < -1e-7
                or abs(raw.sum() - 1.) > 1e-7):
            raise RuntimeError("Allocation solver produced an infeasible point")
        y = np.maximum(raw, 0.)
        y /= y.sum()
        if not np.isfinite(objective(y)):
            raise RuntimeError("Allocation solver removed positive-moment support")
        # SLSQP's stopping rule does not certify the required dual gap. Refine
        # with simplex coordinate exchanges and exact convex line minimization.
        for iteration in range(10001):
            value, gradient = objective(y), derivative(y)
            gap = max(0., float(gradient @ y - gradient.min()))
            relative_gap = gap / max(value, 1.)
            if np.isfinite(relative_gap) and relative_gap <= 1e-6:
                certificate = dict(relative_optimality_gap=relative_gap,
                                   optimality_gap=gap, refinements=iteration,
                                   solver_success=bool(fit.success),
                                   solver_message=str(fit.message))
                break
            receiver = int(gradient.argmin())
            donor = int(np.where(y > 0., gradient, -np.inf).argmax())
            direction = matrix[receiver] - matrix[donor]
            p = y @ matrix
            low, high = 0., float(y[donor])
            for _ in range(60):
                step = (low + high) / 2
                slope = -np.sum(coefficients * direction / (p + step * direction) ** 2)
                if slope < 0:
                    low = step
                else:
                    high = step
            step = (low + high) / 2
            y[receiver] += step
            y[donor] -= step
        else:
            raise RuntimeError("Allocation optimality not certified")
    rates = y / lengths
    result = dict(reset_probabilities=(rates / rates.sum()).tolist(),
                  predicted_occupancy=(y @ conditional).tolist(),
                  baseline_predicted_occupancy=(initial @ conditional).tolist(),
                  normalized_second_moment_objective=objective(y),
                  baseline_second_moment_objective=objective(initial),
                  zero_signal=not signal.any(), numerical_certificate=certificate)
    if result["normalized_second_moment_objective"] > objective(initial) + 1e-7:
        raise RuntimeError("Allocation is worse than the feasible baseline")
    return result


def head_gradient_squared_norm(hidden, actions, means, stddev, advantages):
    """Exact fixed-Gaussian linear mean-head reward-score norm, including bias."""
    scores = (actions - means) / stddev.square()
    return advantages.square() * scores.square().sum(-1) * (1 + hidden.square().sum(-1))


@torch.no_grad()
def actor_moments(agent, bins, require_complete=True):
    sums = torch.zeros(bins, dtype=torch.float64, device=agent._device)
    counts = torch.zeros_like(sums)
    data = {key: agent._exp_buffer.get_data_flat(key)
            for key in ("obs", "action", "adv", "diag_phase")}
    for start in range(0, len(data["obs"]), agent.get_num_envs()):
        batch = {key: value[start:start + agent.get_num_envs()] for key, value in data.items()}
        hidden = agent._model._actor_layers(agent._obs_norm.normalize(batch["obs"]))
        dist = agent._model._action_dist(hidden)
        values = head_gradient_squared_norm(hidden, agent._a_norm.normalize(batch["action"]),
                                             dist.mean, dist.stddev, batch["adv"])
        phase = (batch["diag_phase"] * bins).long().clamp(0, bins - 1)
        sums.scatter_add_(0, phase, values.double())
        counts += torch.bincount(phase, minlength=bins)
    if (require_complete and not (counts > 0).all()) or not torch.isfinite(sums).all():
        raise RuntimeError("Incomplete or non-finite actor moment measurement")
    return (sums / counts.clamp_min(1)).cpu().numpy()


class TrajectoryAllocator:
    """Complete-episode FIFO statistics; partial episodes are never committed."""

    def __init__(self, num_envs, bins, device, seed):
        self.n, self.bins, self.device = num_envs, bins, device
        self.capacity = max(1, num_envs // bins)
        self.history = torch.zeros(bins, self.capacity, bins, device=device)
        self.sizes = np.zeros(bins, dtype=np.int64)
        self.cursors = np.zeros(bins, dtype=np.int64)
        self.visits = torch.zeros(num_envs, bins, device=device)
        self.starts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.pending = []
        self.probabilities = None
        self.updates = 0
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def restart(self, env_ids, phase):
        ids = slice(None) if env_ids is None else env_ids
        self.visits[ids] = 0
        self.starts[ids] = (phase[ids] * self.bins).long().clamp(0, self.bins - 1)

    def record(self, phase):
        bins = (phase * self.bins).long().clamp(0, self.bins - 1)
        self.visits.scatter_add_(1, bins[:, None],
                                 torch.ones(self.n, 1, device=self.device))

    def finish(self, done):
        ended = torch.nonzero(done != 0).flatten()
        if len(ended):
            self.pending.append((self.starts[ended].clone(), self.visits[ended].clone()))

    def flush(self):
        if self.pending:
            starts = torch.cat([x[0] for x in self.pending])
            visits = torch.cat([x[1] for x in self.pending])
            self.pending.clear()
            for i in range(self.bins):
                entries = visits[starts == i]
                total = len(entries)
                if total:
                    keep = min(total, self.capacity)
                    slots = (torch.arange(total - keep, total, device=self.device)
                             + int(self.cursors[i])) % self.capacity
                    self.history[i, slots] = entries[-keep:]
                    self.cursors[i] = (self.cursors[i] + total) % self.capacity
                    self.sizes[i] = min(self.capacity, self.sizes[i] + total)
        if not (self.sizes > 0).all():
            return None
        return torch.stack([self.history[i, :int(n)].mean(0)
                            for i, n in enumerate(self.sizes)]).cpu().numpy()

    def sample_phases(self, n):
        if self.probabilities is None:
            return None
        bins = torch.multinomial(self.probabilities, n, replacement=True,
                                 generator=self.generator)
        offsets = torch.rand(n, device=self.device, generator=self.generator)
        return (bins + offsets) / self.bins

    def update(self, matrix, moments):
        fit = solve_allocation(matrix, moments)
        self.probabilities = torch.tensor(fit["reset_probabilities"],
                                          device=self.device, dtype=torch.float32)
        self.updates += 1
        return fit

    def state_dict(self):
        if self.pending:
            raise RuntimeError("Save allocation state only after flushing complete episodes")
        return dict(history=self.history.cpu(), sizes=self.sizes.copy(),
                    cursors=self.cursors.copy(), updates=self.updates,
                    probabilities=None if self.probabilities is None else self.probabilities.cpu(),
                    generator=self.generator.get_state())

    def load_state_dict(self, state):
        self.history.copy_(state["history"])
        self.sizes[:] = state["sizes"]
        self.cursors[:] = state["cursors"]
        self.updates = int(state["updates"])
        probabilities = state["probabilities"]
        self.probabilities = None if probabilities is None else probabilities.to(self.device)
        self.generator.set_state(state["generator"].cpu())
        # Like the simulator, unfinished episodes restart after checkpoint load.
        self.visits.zero_()
        self.pending.clear()
