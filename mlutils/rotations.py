"""
Code taken from:    https://github.com/zycliao/pytorch-rotation
Accessed:           Dec. 12, 2018

"""
import transforms3d
import torch
import numpy as np

EPS = 1e-8


def expmap2rotmat(r):
    """
        :param r: Axis-angle, Nx3
        :return: Rotation matrix, Nx3x3
    """
    dev = r.device
    assert r.shape[1] == 3
    bs = r.shape[0]
    theta = torch.sqrt(torch.sum(torch.pow(r, 2), 1, keepdim=True))
    cos_theta = torch.cos(theta).unsqueeze(-1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    eye = torch.unsqueeze(torch.eye(3, device=dev), 0).repeat(bs, 1, 1)
    norm_r = r / (theta + EPS)
    r_1 = torch.unsqueeze(norm_r, 2)  # N, 3, 1
    r_2 = torch.unsqueeze(norm_r, 1)  # N, 1, 3
    zero_col = torch.zeros(bs, 1, device=dev)
    skew_sym = torch.cat([zero_col, -norm_r[:, 2:3], norm_r[:, 1:2], norm_r[:, 2:3], zero_col,
                          -norm_r[:, 0:1], -norm_r[:, 1:2], norm_r[:, 0:1], zero_col], 1)
    skew_sym = skew_sym.contiguous().view(bs, 3, 3)
    R = cos_theta*eye + (1-cos_theta)*torch.bmm(r_1, r_2) + sin_theta*skew_sym
    return R


def rotmat2expmap(r):
    """
        :param r: Rotation matrix, Nx3x3
        :return: r: Axis-angle, Nx3
    """
    assert r.shape[1] == r.shape[2] == 3
    x = (r - r.permute(0, 2, 1)) / 2.
    x = x.contiguous().view(-1, 9)
    sintheta_r = x[:, [7, 2, 3]]
    sintheta = torch.sqrt(torch.sum(torch.pow(sintheta_r, 2), 1, keepdim=True))
    r_norm = sintheta_r / (sintheta+EPS)
    theta = torch.asin(torch.clamp(sintheta, min=-1., max=1.))
    a = theta * r_norm
    return a


def quat2expmap(q):
    """
        :param q: quaternion, Nx4
        :return: r: Axis-angle, Nx3
    """
    assert q.shape[1] == 4
    cos_theta_2 = torch.clamp(q[:, 0: 1], min=-1., max=1.)
    theta = torch.acos(cos_theta_2)*2
    sin_theta_2 = torch.sqrt(1-torch.pow(cos_theta_2, 2))
    r = theta * q[:, 1:4] / (sin_theta_2 + EPS)
    return r


def expmap2quat(r):
    """
        :param r: Axis-angle, Nx3
        :return: q: quaternion, Nx4
    """
    assert r.shape[1] == 3
    theta = torch.sqrt(torch.sum(torch.pow(r, 2), 1, keepdim=True))
    unit_r = r / (theta + EPS)
    theta_2 = theta / 2.
    cos_theta_2 = torch.cos(theta_2)
    sin_theta_2 = torch.sin(theta_2)
    q = torch.cat((cos_theta_2, unit_r*sin_theta_2), 1)
    return q


def rotmat2quat(r):
    """ Converts a batch of rotation matrices to quaternions.
        This implementation does not support CUDA tensors, converts torch tensors to numpy and does not
        parallelize in the batch dimension.

        :param r: The batch of rotation matrices of size B x 3 x 3
        :return: A batch of quaternions of size B x 4
    """
    q = np.stack([transforms3d.quaternions.mat2quat(matrix.numpy()) for matrix in r])
    return torch.from_numpy(q).float()


def quat2rotmat(q):
    """ Converts unit quaternions to rotation matrices.

        :param q: Batch of unit quaternions with size B x 4
        :returns Batch of rotation matrices of size B x 3 x 3

        Code adapted from:
        https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotmat = torch.stack([
        w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
        2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
        2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2
    ], dim=1).view(-1, 3, 3)
    return rotmat




if __name__ == '__main__':
    q = torch.rand(1, 4)
    # a = np.pi/2
    # q = torch.Tensor([
    #     [np.cos(a/2),
    #      np.sin(a/2) * 0,
    #      np.sin(a/2) * 0,
    #      np.sin(a/2) * 1]
    # ])
    q /= torch.norm(q, dim=1, keepdim=True)
    rotmat = quat2rotmat(q)

    # x = torch.mm(rotmat.view(3, 3), rotmat.view(3, 3).transpose(0, 1))
    # print(x)
    print(rotmat2quat(quat2rotmat(q)))
    print(q)
    # print(rotmat)
