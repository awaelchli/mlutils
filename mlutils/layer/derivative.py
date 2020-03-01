import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PartialDerivative(nn.Module):

    def __init__(self, dim: int = 0):
        super().__init__()
        self.__dim = dim

    @property
    def dim(self):
        return self.__dim

    def forward(self, x):
        return partial_derivative(x, dim=self.dim)


def partial_derivative(tensor: Tensor, dim: int = 0):
    assert dim in (0, 1)
    b, c, h, w = tensor.shape
    kernel = tensor.new_zeros((c, c, 2))
    for i in range(c):
        kernel[i, i, 0] = -1
        kernel[i, i, 1] = 1

    if dim == 0:
        kernel = kernel.view(c, c, 2, 1)
    if dim == 1:
        kernel = kernel.view(c, c, 1, 2)

    derivative = F.conv2d(tensor, kernel)
    return derivative
