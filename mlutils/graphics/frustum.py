import torch


class Frustum(object):

    def __init__(self, fov=60.0, near=None, far=None):
        self._fov = fov

    def project(self, points):
        x, y, z = torch.split(points, 1, -1)
        x = x / z
        y = y / z
        return torch.cat((x, y), -1)


if __name__ == '__main__':
    f = Frustum()
    p = torch.Tensor([[3., 2., 1.],
                      [1., 2., 3.],
                      [2., 4., 4.]])

    p2 = f.project(p)
    print(p2)
    print(p)
