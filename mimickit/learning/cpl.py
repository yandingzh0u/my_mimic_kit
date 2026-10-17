"""Full-coordinate semantic encoders and shared convex-potential layers."""
import torch
import torch.nn.functional as F


class PairSort(torch.nn.Module):
    """1-Lipschitz pairwise GroupSort; no norm loss at differentiable points."""
    def forward(self, inputs):
        paired = inputs.shape[-1] // 2 * 2
        values = inputs[..., :paired].reshape(*inputs.shape[:-1], -1, 2)
        ordered = values.sort(dim=-1).values.flatten(-2)
        return torch.cat((ordered, inputs[..., paired:]), dim=-1)


def spectral_linear(in_features, out_features):
    linear = torch.nn.Linear(in_features, out_features)
    torch.nn.init.zeros_(linear.bias)
    return torch.nn.utils.parametrizations.spectral_norm(linear)


class ConvexPotentialBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = spectral_linear(width, width)

    def forward(self, inputs):
        weight = self.linear.weight
        hidden = F.relu(F.linear(inputs, weight, self.linear.bias))
        return inputs - 2 * F.linear(hidden, weight.T)


class GroupSeparableDiscLayers(torch.nn.Module):
    """Keep each residual coordinate; use GroupSort at shared fusion only.

    The encoder budget is twice the trunk width, as in the original 1024/512
    discriminator. Group encoders retain their original ReLU. No pooled RMS,
    fixed tracking cost, group weights or monotonic-response claim is added.
    Spectral normalization uses power iteration, not a certified norm bound.
    """
    def __init__(self, groups, width, depth):
        super().__init__()
        if width < 1 or depth < 3 or not groups or any(not ids for _, ids in groups):
            raise ValueError('CPL requires positive width, depth >= 3 and nonempty groups')
        self.groups = tuple((name, tuple(ids)) for name, ids in groups)
        self.width = width
        self.group_width = 2 * width // len(groups)
        if self.group_width < 1:
            raise ValueError('Encoder budget is smaller than the group count')
        self.encoders = torch.nn.ModuleList()
        for index, (_, ids) in enumerate(self.groups):
            self.register_buffer(f'group_indices_{index}', torch.tensor(ids, dtype=torch.long))
            self.encoders.append(torch.nn.Sequential(
                spectral_linear(len(ids), self.group_width), torch.nn.ReLU()))
        self.trunk = torch.nn.Sequential(
            spectral_linear(self.group_width * len(groups), width), PairSort(),
            *[ConvexPotentialBlock(width) for _ in range(depth - 3)])

    def forward(self, diff):
        encoded = [encoder(diff.index_select(-1, getattr(self, f'group_indices_{index}')))
                   for index, encoder in enumerate(self.encoders)]
        return self.trunk(torch.cat(encoded, dim=-1))
