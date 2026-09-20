import numpy as np
import torch


def build_net(input_dict, activation):
    """Ten-affine-layer MLP used only for the discriminator depth stress test.

    DARE replaces the first affine layer by seven semantic encoders and keeps
    the remaining eight affine layers as the shared trunk.  Its scalar output
    head is added by DAREModel, so one input-to-logit path contains ten Linear
    modules in total: 1 encoder + 8 trunk layers + 1 output layer.
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
