"""Physical tracking errors for a free box and the hands that carry it."""

from itertools import permutations, product

import torch
from torch import nn

from util.torch_util import quat_to_matrix


def _box_surface_distance(points, half_size):
    """Signed distance from points in box coordinates to the surface."""
    overhang = points.abs() - half_size
    return overhang.clamp_min(0).norm(dim=-1) + overhang.amax(dim=-1).clamp_max(0)


class CarryGeometry(nn.Module):
    def __init__(self, box_size):
        super().__init__()
        size_tensor = torch.as_tensor(box_size).detach().cpu().reshape(3)
        self.register_buffer("_half_size", size_tensor / 2, persistent=False)
        size = size_tensor.tolist()
        symmetries = []
        for axes in permutations(range(3)):
            if [size[axis] for axis in axes] != size:
                continue
            inversions = sum(axes[i] > axes[j] for i in range(3) for j in range(i + 1, 3))
            parity = (-1) ** inversions
            for signs in product((1, -1), repeat=3):
                if signs[0] * signs[1] * signs[2] != parity:
                    continue
                rotation = torch.zeros(3, 3)
                rotation[range(3), axes] = torch.tensor(signs, dtype=rotation.dtype)
                symmetries.append(rotation)
        self.register_buffer("_box_symmetries", torch.stack(symmetries), persistent=False)

    def align_box_rotation(self, actual_quat, reference_quat):
        """Use the equivalent actual frame with the smallest angle to the reference."""
        actual = quat_to_matrix(actual_quat)
        reference = quat_to_matrix(reference_quat)
        relative = reference.transpose(-1, -2) @ actual
        symmetry = self._box_symmetries.to(relative)
        # For a rotation C, maximizing tr(relative @ C.T) minimizes the angle.
        best = (relative.flatten(-2) @ symmetry.flatten(-2).T).argmax(-1)
        alignment = symmetry[best].transpose(-1, -2)
        return actual @ alignment, reference, relative @ alignment

    def forward(self, actual_box_pose, reference_box_pose, actual_hand_pos, reference_hand_pos):
        """Return center distance, angle and left/right box-relative hand errors.

        Poses use XYZ and XYZW. Hand centers are paired by anatomical side in
        world coordinates. The same box frame matches the pose and both hands.
        """
        actual_center, actual_quat = actual_box_pose[..., :3], actual_box_pose[..., 3:]
        ref_center, ref_quat = reference_box_pose[..., :3], reference_box_pose[..., 3:]
        actual_rot, ref_rot, relative = self.align_box_rotation(actual_quat, ref_quat)

        # Use atan2 to retain small angles that would round to zero with acos.
        skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                            relative[..., 0, 2] - relative[..., 2, 0],
                            relative[..., 1, 0] - relative[..., 0, 1]), dim=-1)
        trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
        angle = torch.atan2(skew.norm(dim=-1), trace - 1)

        actual_local = (actual_hand_pos - actual_center.unsqueeze(-2)) @ actual_rot
        ref_local = (reference_hand_pos - ref_center.unsqueeze(-2)) @ ref_rot
        # A tangential slide preserves the signed surface distance, while a
        # separation or penetration changes it. The fixed hand radius cancels.
        surface_delta = (_box_surface_distance(actual_local, self._half_size)
                         - _box_surface_distance(ref_local, self._half_size))
        hand_error = torch.cat((actual_local - ref_local, surface_delta.unsqueeze(-1)), dim=-1).norm(dim=-1)
        center_error = (actual_center - ref_center).norm(dim=-1, keepdim=True)
        return torch.cat((center_error, angle.unsqueeze(-1), hand_error), dim=-1)
