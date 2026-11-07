import numpy as np
import torch


def build_net(input_dict, activation):
    """Ten-affine-layer MLP used only for the discriminator depth stress test.

    DARE uses the remaining eight widths for one fusion layer and seven CPL
    blocks; its learnable semiorthogonal CReLU stem and scalar head are built by
    DAREModel.  Constructing this template preserves historical RNG draws.
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
