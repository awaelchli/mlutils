import torch
from mlutils.layer.block_reshape import block_reshape, BlockReshape


example_input = torch.tensor([
    [[0.0, 0.1],
     [0.2, 0.3]],

    [[1.0, 1.1],
     [1.2, 1.3]],

    [[2.0, 2.1],
     [2.2, 2.3]],

    [[3.0, 3.1],
     [3.2, 3.3]],
])

example_target = torch.tensor([
    [[0.0, 1.0, 0.1, 1.1],
     [2.0, 3.0, 2.1, 3.1],
     [0.2, 1.2, 0.3, 1.3],
     [2.2, 3.2, 2.3, 3.3]]
])


def test_single_element():
    tensor = torch.tensor([
        [[1]]
    ])
    result = block_reshape(tensor)
    assert torch.all(tensor == result)


def test_single_output_channel():
    result = block_reshape(example_input, 2)
    assert torch.all(result == example_target)

    module = BlockReshape(2)
    result = module(example_input)
    assert torch.all(result == example_target)


def test_multiple_output_channels():
    tensor = example_input.repeat(2, 1, 1)
    target = example_target.repeat(2, 1, 1)
    result = block_reshape(tensor, 2)
    assert torch.all(result == target)


def test_multiple_batch_and_output_channels():
    tensor = example_input.unsqueeze(0).repeat(5, 2, 1, 1)
    target = example_target.unsqueeze(0).repeat(5, 2, 1, 1)
    result = block_reshape(tensor, 2)
    assert torch.all(result == target)
