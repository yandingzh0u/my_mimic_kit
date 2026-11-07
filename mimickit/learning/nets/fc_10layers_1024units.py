import numpy as np
import torch


def build_net(input_dict, activation):
    """Ten-affine-layer MLP used only for the discriminator depth stress test.

    DARE uses the remaining eight widths for its shared fusion/CPL trunk;
    its coordinate-token frontend and scalar head are built by DAREModel.
    """
    layer_sizes = [1024] + [512] * 8

    input_dim = np.sum([np.prod(curr_input.shape)
                        for curr_input in input_dict.values()])

    in_size = input_dim
    layers = []
    for out_size in layer_sizes:
        curr_layer = torch.nn.Linear(in_size, out_size)
        torch.nn.init.zeros_(curr_layer.bias)
        layers.append(curr_layer)
        layers.append(activation())
        in_size = out_size

    return torch.nn.Sequential(*layers), {}
