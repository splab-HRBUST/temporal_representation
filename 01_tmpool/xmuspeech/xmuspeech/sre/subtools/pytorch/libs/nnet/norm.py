import torch

def l2_norm(inputs):
    x = inputs
    norm = torch.norm(x, dim=1, p=2, keepdim=True)
    x = x / norm
    return x