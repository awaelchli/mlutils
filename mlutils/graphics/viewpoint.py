import torch
import torch.nn.functional as F
from numpy import pi

import mlutils.optical_flow.transform

EPSILON = 1e-16


class Viewpoint(object):

    def __init__(self, eye=(0, 0, 1), look_at=(0, 0, 0), up=None):
        self._eye = torch.as_tensor(eye).float()
        self._look_at = torch.as_tensor(look_at).float()
        if up is not None:
            self._up = torch.as_tensor(up).float()
        else:
            self._up = torch.Tensor([0, 1, 0]).to(self.eye.device)
        assert self.eye.device == self.look_at.device == self.up.device
        self._axes = self._get_axes()

    @property
    def eye(self):
        return self._eye

    @eye.setter
    def eye(self, x):
        self._eye = x
        self._axes = self._get_axes()

    @property
    def look_at(self):
        return self._look_at

    @look_at.setter
    def look_at(self, x):
        self._look_at = x
        self._axes = self._get_axes()

    @property
    def up(self):
        return self._up

    @up.setter
    def up(self, x):
        self._up = x
        self._axes = self._get_axes()

    @property
    def x_axis(self):
        return self._axes[0]

    @property
    def y_axis(self):
        return self._axes[1]

    @property
    def z_axis(self):
        return self._axes[2]

    @property
    def rotation(self):
        return torch.stack(self._axes, 1)

    @property
    def position(self):
        return self.eye

    def _get_axes(self):
        eye, at, up = self.eye, self.look_at, self.up
        z_axis = mlutils.optical_flow.transform.normalize(at - eye, dim=0)
        x_axis = mlutils.optical_flow.transform.normalize(torch.cross(up, z_axis), dim=0)
        y_axis = mlutils.optical_flow.transform.normalize(torch.cross(z_axis, x_axis), dim=0)
        # this is a right-handed coordinate system (z-axis pointing towards observer)
        return x_axis, y_axis, -z_axis

    def world2cam(self, points):
        assert points.size(1) == 3
        # create rotation matrix: [3, 3]
        r = torch.stack(self._get_axes(), dim=1).transpose(0, 1)  # camera-to-world rotation
        points = points - self.eye
        points = torch.matmul(points, r.transpose(0, 1))  # = r * points
        return points

    def cam2world(self, points):
        assert points.size(1) == 3
        # rotation matrix is defined by the axes of the camera
        r = torch.stack(self._get_axes(), dim=1)
        points = torch.matmul(points, r.transpose(0, 1))  # = r * points
        points += self.eye
        return points

    def longitude(self):
        axis = self.z_axis
        return torch.atan2(axis[1], axis[0])

    def latitude(self):
        axis = self.z_axis
        theta = torch.acos(axis[2] / (torch.norm(axis) + EPSILON))
        return theta - pi / 2

    def __call__(self, points):
        """ Transforms a batch of points of size N x 3 to the coordinate system of the viewpoint. """
        return self.world2cam(points)

    def __repr__(self):
        return f'Viewpoint with eye = {self.eye}, look_at = {self.look_at}, up = {self.up}'


class ViewpointSampler(object):
    """ Abstract class for sampling viewpoints. """

    def __init__(self, device='cpu', *args, **kwargs):
        self._device = device

    @property
    def device(self):
        return self._device

    def sample(self):
        pass


class FixedLookSampler(ViewpointSampler):
    """ Uniformly samples viewpoints on a sphere with fixed look-at point at zero. """

    def __init__(self, radius=1.0, device='cpu'):
        super(FixedLookSampler, self).__init__(device)
        self.radius = radius

    def sample(self):
        theta = torch.rand(1, 1, device=self.device) * 2 * pi
        z = torch.rand(1, 1, device=self.device) * 2 - 1
        x = torch.sqrt(1 - z ** 2) * torch.cos(theta)
        y = torch.sqrt(1 - z ** 2) * torch.sin(theta)
        eye = torch.cat((x, y, z)).view(-1)
        eye *= self.radius
        look_at = torch.zeros_like(eye)
        up = torch.Tensor([0.0, 1.0, 0.0]).to(eye.device)
        return Viewpoint(eye, look_at, up)


class SingleViewSampler(ViewpointSampler):
    """ Always samples the same constant viewpoint. """

    def __init__(self, device='cpu'):
        super(SingleViewSampler, self).__init__(device)

    def sample(self):
        eye = torch.zeros(3, device=self.device)
        look_at = torch.Tensor([0, 0, 1.0]).to(self.device)
        up = torch.Tensor([0.0, 1.0, 0.0]).to(self.device)
        return Viewpoint(eye, look_at, up)


def select_visible_points(points, mask):
    return torch.masked_select(points, mask.view(-1, 1)).view(-1, 3)


if __name__ == '__main__':
    sampler = FixedLookSampler(10.0, device='cuda')
    pts = torch.Tensor([
        [0, 0, 0]
    ]).to('cuda')
    view = sampler.sample()
    print(view)
    transformed = view.world2cam(pts)
    print(transformed)
