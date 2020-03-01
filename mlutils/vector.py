import torch


def hom(x):
    """ Appends the homogeneous coordinate to the end of the vector. """
    # x: (..., D)
    # output: (..., D+1)
    ones = torch.ones_like(x[..., 0].unsqueeze(-1))
    return torch.cat((x, ones), -1)


if __name__ == '__main__':
    vec = torch.rand(2)
    print(hom(vec))

    vec = torch.rand(2, 3, 4)
    print(hom(vec.view(2, 12)))
