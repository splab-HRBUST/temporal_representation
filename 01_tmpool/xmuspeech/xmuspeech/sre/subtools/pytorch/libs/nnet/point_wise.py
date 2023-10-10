import torch
import torch.nn as nn
from .temp_botneck import temporal_bottleneck_layer


def nonlinear_map(inputs):
    x = inputs
    x = torch.sqrt(x.clamp(min=0.0))

    return x


def point_wise_map(inputs):
    x = inputs

    x_pos = torch.clamp(x.clone(), min=0.0)
    x_neg = torch.clamp(x.clone(), max=0.0)

    x_pos = torch.sqrt(x_pos)
    x_neg = torch.sqrt(-x_neg)

    return torch.cat((x_pos, x_neg), dim=1)




# x = torch.randn(512, 768, 13)
# x = temporal_bottleneck_layer(x)

# x = torch.ones(1,4,4) * 3
# print(x)
# x = point_wise_map(x)
# print(x)