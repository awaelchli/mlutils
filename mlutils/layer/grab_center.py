import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GrabCenter(nn.Module):

    def __init__(self, mode: str = 'bilinear'):
        super().__init__()
        self.__mode = mode

    def forward(self, x: Tensor) -> Tensor:
        return grab_center(x, self.__mode)


def grab_center(tensor: Tensor, mode: str = 'bilinear') -> Tensor:
    """
    Interpolates and returns the center value in spatial dimensions Y and X, the last two
    dimensions of the tensor.
    """
    b, _, h, w = tensor.shape
    grid = tensor.new_zeros((b, 1, 1, 2))
    center = F.grid_sample(tensor, grid, mode=mode)
    center = center.flatten(1)
    return center
