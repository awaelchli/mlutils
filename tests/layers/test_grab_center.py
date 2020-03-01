import torch
from mlutils.layer import grab_center


def test_grab_center_odd_size():
    tensor = torch.tensor([
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],

        [[1, 2, 3],
         [4, 8, 6],
         [7, 8, 9]]
    ]).unsqueeze(0).float()
    expected = torch.tensor([
        [5., 8.],
    ])
    result = grab_center(tensor)
    assert torch.all(result == expected)


def test_grab_center_even_size():
    tensor = torch.tensor([
        [[1, 2, 3, 4],
         [5, 6, 7, 8]],

        [[-1, 0, -3, -4],
         [-5, -6, -1, -8]]
    ]).unsqueeze(0).float()
    expected = torch.tensor([
        [4.5, -2.5],
    ])
    result = grab_center(tensor)
    assert torch.allclose(result, expected)


def test_grab_center_batch():
    tensor = torch.tensor([
        [[[1.1]]],
        [[[2.2]]],
        [[[3.3]]],
        [[[4.4]]],
    ]).float()
    expected = torch.tensor([
        [1.1],
        [2.2],
        [3.3],
        [4.4]
    ])
    result = grab_center(tensor)
    assert torch.all(result == expected)
