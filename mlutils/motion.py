import torch
from numpy import pi
from mlutils.parametric import Sinusoids, Sphere
from mlutils.utils import Viewpoint


def generate_motion(order, num_samples, offset=0.0):
    max = pi / order
    coeff = torch.Tensor(order).uniform_(-max, max)
    period = 2 * num_samples
    f = Sinusoids(coeff, period)
    motion = torch.stack([f(t + offset) for t in range(num_samples)])
    return motion


def generate_angular_motion(order, num_samples, num_angles=3, offset=None):
    if offset is None:
        offset = torch.zeros(num_angles)
    elif offset == 'random':
        offset = torch.Tensor(num_angles).uniform_(-pi, pi)
    else:
        offset = torch.as_tensor(offset)
    assert len(offset) == num_angles
    angles = [generate_motion(order, num_samples, offset[i]) for i in range(num_angles)]
    angles = torch.stack(angles, 1)
    # angles: (num_samples, 3)
    return angles


def generate_motion_on_sphere(order, num_samples, radius=1.0, random_offset=True):
    offset = 'random' if random_offset else None
    angles = generate_angular_motion(order, num_samples, num_angles=2, offset=offset)  # (N, 2) - longitude and latitude
    sphere = Sphere(radius)
    return torch.stack([sphere(long, lat) for long, lat in angles])


def viewpoint_motion_on_sphere(num_views, radius=1.0):
    """ Samples a sequence of viewpoints on the surface of the sphere looking towards the center. """
    positions = generate_motion_on_sphere(
        order=4,
        num_samples=num_views,
        radius=radius,
        random_offset=True
    )
    viewpoints = [Viewpoint(p, look_at=[0, 0, 0]) for p in positions]
    return viewpoints
