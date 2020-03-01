import torch.nn as nn
from torch import Tensor


class BlockReshape(nn.Module):

    def __init__(self, scale_factor: int = 1):
        super().__init__()
        self.__scale_factor = scale_factor

    @property
    def scale_factor(self) -> int:
        return self.__scale_factor

    def forward(self, x: Tensor) -> Tensor:
        return block_reshape(x, self.scale_factor)


def block_reshape(x: Tensor, scale_factor: int = 1) -> Tensor:
    """
    Block-wise reshaping.

    :param x: Tensor of size (..., s * s * C, H, W) to be reshaped.
    :param scale_factor: The factor by which the spatial dimensions increase.
    :return: Tensor of size (..., C, s * H, s * W)
    """
    assert x.ndimension() >= 3
    assert scale_factor > 0
    *dims, inp_channels, h, w, = x.shape
    s = scale_factor
    c = inp_channels // (s ** 2)
    assert (s ** 2) * c == inp_channels, 'number of input channels must be divisible by (scale_factor ** 2)'

    x = x.view(*dims, c, s * s, h, w)
    x = x.permute(*range(len(dims) + 1), -2, -1, -3)
    x = x.reshape(*dims, c, h, w, s, s)
    x = x.transpose(-3, -2)
    x = x.reshape(*dims, c, h * s, h * s)
    return x
