import numpy as np
import torch
from transforms3d.taitbryan import euler2mat


def subsample(points, n):
    indices = np.random.choice(len(points), n, replace=True)
    return points[indices]


def normalize(points):
    """ Normalizes the point cloud such that it fits in the unit sphere placed at the origin.
        Input shape: N x 3
    """
    centroid = torch.mean(points, 0, keepdim=True)
    points = points - centroid
    m = torch.max(torch.sum(points ** 2, 1))
    points = points / torch.sqrt(m)
    return points


def rotate(points, range=None):
    """ Rotates the points around the origin with a uniformly sampled rotation matrix.
        :param range: The range/extent of rotation angle for the Z-, Y-, and X-axis rotations.
        The default range is the full set of Euler angles, i.e.,
        [0, 2pi) for Z- and X-axis, and [0, pi) for the Y-axis.
        The rotation will be applied always in the order Z first, then Y, and finally X.
        :param points: N x 3 tensor
    """
    range = (2 * np.pi, np.pi, 2 * np.pi) if range is None else range
    assert type(range) == tuple and len(range) == 3
    az = np.random.uniform(0, range[0], 1)
    ay = np.random.uniform(0, range[1], 1)
    ax = np.random.uniform(0, range[2], 1)
    # Static frame rotation around z-axis, then y-axis, then x-axis (Tait-Bryan).
    mat = euler2mat(az, ay, ax)
    mat = torch.as_tensor(mat, dtype=points.dtype, device=points.device)
    points = torch.matmul(points, mat.transpose(0, 1))
    return points, mat


def jitter(points, mean=0.0, std=0.01):
    """ Adds Gaussian noise to the points.
        Input: shape: N x 3
    """
    noise = torch.empty_like(points).normal_(mean, std)
    return points + noise
