import torch
from mlutils.layer.derivative import partial_derivative


def test_dx():
    tensor = torch.tensor([
        [1,  2,  3.5],
        [1, -1, -1.1],
    ]).view(1, 1, 2, 3).float()
    expected = torch.tensor([
        [1, 1.5],
        [-2, -0.1],
    ]).view(1, 1, 2, 2).float()
    result = partial_derivative(tensor, 1)
    assert torch.allclose(result, expected)


def test_dy():
    tensor = torch.tensor([
        [1, -1, -0.5],
        [-3, 2, -0.7],
    ]).view(1, 1, 2, 3).float()
    expected = torch.tensor([
        [-4, 3, -0.2],
    ]).view(1, 1, 1, 3).float()
    result = partial_derivative(tensor, 0)
    assert torch.allclose(result, expected)


def test_multiple_channels_dx():
    tensor = torch.tensor([
        [[1, 2, 3.5],
         [1, -1, -1.1]],

        [[3.5, 2, 1],
         [-1, 1, 1.1]],
    ]).view(1, 2, 2, 3).float()
    expected = torch.tensor([
        [[1, 1.5],
         [-2, -0.1]],

        [[-1.5, -1],
         [2, 0.1]],
    ]).view(1, 2, 2, 2).float()
    result = partial_derivative(tensor, 1)
    assert torch.allclose(result, expected)
