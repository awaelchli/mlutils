import torch
from torch.nn import functional as F


def divergence(flow):
    """ Computes the divergence of the vector field given by the optical flow. """
    # flow: (B, 2, H, W)
    flow_padded = F.pad(flow, (1, 1, 1, 1), mode='replicate')

    dx = flow_padded[:, 0, 1:-1, 2:] - flow[:, 0]
    dy = flow_padded[:, 1, 2:, 1:-1] - flow[:, 1]
    div = dx + dy  # (B, H, W)
    return div


def const_flow(height, width, dx, dy, device='cpu'):
    x = torch.full((height, width), dx, device=device)
    y = torch.full((height, width), dy, device=device)
    # output: (2, H, W)
    return torch.stack((x, y), 0)


def truncate_flow(flow, thresh=1.0):
    """ Sets all flow vectors with a norm < thres to zero. """
    # flow: (B, 2, H, W) or (2, H, W)
    assert flow.ndimension() in (3, 4)
    if flow.ndimension == 3:
        norm_dim = 0
    else:
        norm_dim = 1

    norm = torch.norm(flow, p=2, dim=norm_dim, keepdim=True).expand_as(flow)
    flow = torch.where(norm <= thresh, torch.zeros_like(flow), flow)
    return flow