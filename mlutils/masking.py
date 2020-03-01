import torch


def mask_points(points, mask):
    # points: (B, N, M) where M is the point dimension (usually 2 or 3)
    # mask: (B, N)
    point_dim = points.size(2)
    return points * mask.type(points.type()).unsqueeze(2).repeat(1, 1, point_dim)

