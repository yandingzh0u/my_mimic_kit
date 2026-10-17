"""A BCE-trained cost with certified responses to every error group."""
import torch
import torch.nn.functional as F


def certified_weight(weight):
    # ||W||_2 <= sqrt(||W||_1 ||W||_inf), including rectangular matrices.
    bound = (weight.abs().sum(-1).amax()
             * weight.abs().sum(-2).amax()).clamp_min(1.).sqrt()
    return weight / bound


class ConvexPotentialBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(width, width))
        self.bias = torch.nn.Parameter(torch.zeros(width))
        torch.nn.init.kaiming_uniform_(self.weight, a=5 ** .5)

    def forward(self, inputs):
        weight = certified_weight(self.weight)
        hidden = F.relu(F.linear(inputs, weight, self.bias))
        return inputs - 2 * F.linear(hidden, weight.T)


class MonotoneGroupCost(torch.nn.Module):
    """Learn a bounded correction without suppressing an entire error group.

    Inputs to g are normalized group RMS magnitudes, not a quality target.
    For G>1, g is 1-Lipschitz in L2 and the cost has strictly positive partial
    derivatives in every group magnitude. This prior discards direction and
    limits learned preferences; it does not certify physical task quality.
    """
    def __init__(self, groups, width, depth):
        super().__init__()
        if len(groups) <= 1 or any(not ids for _, ids in groups):
            raise ValueError("Monotone DARE requires multiple nonempty error groups")
        if width < 1 or depth < 3:
            raise ValueError("Discriminator width must be positive and depth >= 3")
        self.groups = tuple((name, tuple(ids)) for name, ids in groups)
        self.width = width
        for i, (_, ids) in enumerate(self.groups):
            self.register_buffer(f'indices_{i}', torch.tensor(ids, dtype=torch.long))
        self.embedding = torch.nn.Linear(len(groups), width)
        torch.nn.init.zeros_(self.embedding.bias)
        self.blocks = torch.nn.Sequential(*[
            ConvexPotentialBlock(width) for _ in range(depth - 2)])
        self.projection = torch.nn.Parameter(torch.empty(1, width))
        torch.nn.init.normal_(self.projection)

    def magnitudes(self, diff):
        return torch.stack([
            diff.index_select(-1, getattr(self, f'indices_{i}')).norm(dim=-1)
            / len(ids) ** .5 for i, (_, ids) in enumerate(self.groups)], -1)

    def correction(self, error):
        weight = self.embedding.weight / self.embedding.weight.norm().clamp_min(1.)
        hidden = F.linear(error, weight, self.embedding.bias)
        hidden = self.blocks(hidden)
        projection = self.projection / self.projection.norm().clamp_min(1.)
        return F.linear(hidden, projection)

    def cost(self, error):
        n = len(self.groups) ** .5
        values = self.correction(torch.cat((torch.zeros_like(error[:1]), error)))
        return (error.sum(-1, keepdim=True) + (values[1:] - values[:1]) / n) / (n + 1 / n)

    def forward(self, diff):
        return self.cost(self.magnitudes(diff))

    def slope_bounds(self):
        n = len(self.groups) ** .5
        denominator = n + 1 / n
        return (1 - 1 / n) / denominator, (1 + 1 / n) / denominator


class AnchorLogit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, cost):
        return self.bias - cost
