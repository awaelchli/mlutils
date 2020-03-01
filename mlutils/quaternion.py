import torch


def qconj(q):
    """ Computes the conjugate of a batch of quaternions. """
    c = torch.ones_like(q)
    c[..., 1:4] = -1.0
    return q * c


def qrot(v, q):
    """ Rotates a batch of vectors by a batch of quaternions. """
    u = torch.zeros_like(q)
    u[..., 1:4] = v
    return qmul(q, qmul(u, qconj(q)))[..., 1:]


def qmul(q1, q0):
    """ Batch-wise quaternion multiplication (Hamilton product) """
    w0, x0, y0, z0 = torch.split(q0, 1, dim=-1)
    w1, x1, y1, z1 = torch.split(q1, 1, dim=-1)

    # Hamilton product
    w2 = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
    x2 = x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0
    y2 = -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0
    z2 = x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0

    q3 = torch.cat((w2, x2, y2, z2), -1)
    return q3


def qfix(q):
    """ Flips the sign of the quaternion (w, x, y, z) if w is negative.
        :param q: The batch of quaternions of shape (B, 4)
    """
    cond = q[:, 0] < 0
    mask = cond.view(-1, 1).repeat(1, 4)
    q = torch.where(mask, -q, q)
    return q


def qunit(q):
    """ Projects the batch of quaternions to unit-quaternions. """
    return q / torch.norm(q, dim=-1, keepdim=True)


if __name__ == '__main__':
    p = torch.rand(10, 3)
    quat = torch.rand(10, 4)
    out = qrot(p, quat)
    print(out.shape)

    p = torch.Tensor([1., 2., 3., 4.]).unsqueeze(0).repeat(3, 1)
    quat = torch.Tensor([5., 6., 7., 8.]).unsqueeze(0).repeat(3, 1)
    print(qmul(p, quat))
    # output: -60.,  12.,  30.,  24.

    p = torch.Tensor([1., 0., 1., 0.]).unsqueeze(0).repeat(3, 1)
    quat = torch.Tensor([1., .5, .5, .75]).unsqueeze(0).repeat(3, 1)
    print(qmul(p, quat))
    # output: 0.5, 1.25, 1.5, 0.25

    vec = torch.Tensor([1., 2., 3.]).unsqueeze(0).repeat(3, 1)
    quat = torch.Tensor([4., 5., 6., 7.]).unsqueeze(0).repeat(3, 1)
    print(qrot(vec, quat))
    # output: [318., 204., 282.]

    vec = torch.Tensor([1., 1., 1.]).unsqueeze(0).repeat(3, 1)
    quat = torch.Tensor([1., 1., 1., 1.]).unsqueeze(0).repeat(3, 1)
    print(qrot(vec, quat))
    # output: 4, 4, 4, 4

    p = torch.Tensor([1., 0., 0., 0.]).unsqueeze(0).repeat(2, 1)
    vec = torch.Tensor([.1, .2, .3]).unsqueeze(0).repeat(2, 1)
    print(qrot(vec, p))  # should not rotate

    quat = torch.Tensor([
        [-0.5, 1, 0, -1],
        [0, 0.1, -1, 1],
        [0.4, 1, 1, 1],
    ])
    print(qfix(quat))
    # output: should flip first row
