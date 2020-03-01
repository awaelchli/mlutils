import torch
from torch import Tensor
import numpy as np


def epe(flow: Tensor, target: Tensor, dim: int = 1, average: bool = True, squared: bool = False) -> Tensor:
    assert 0 <= dim <= flow.ndimension() == target.ndimension()
    errors = torch.sum((flow - target) ** 2, dim=dim)
    if not squared:
        errors = errors.sqrt()
    if average:
        errors = errors.mean()
    return errors


def angular_error(flow: Tensor, target: Tensor, dim: int = 1, average: bool = True, degrees: bool = False) -> Tensor:
    assert 0 <= dim <= flow.ndimension() == target.ndimension()
    n1 = torch.norm(flow, p=2, dim=dim)
    n2 = torch.norm(target, p=2, dim=dim)
    eps = 0.000000001
    cosine = torch.sum(flow * target, dim=dim) / (n1 * n2 + eps)
    cosine = torch.clamp(cosine, -1.0, 1.0)
    angle = torch.acos(cosine)
    if average:
        angle = angle.mean()

    if degrees:
        angle = angle * 180 / np.pi

    return angle
