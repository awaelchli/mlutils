import torch


def cross_product_matrix(vectors):
    """ Constructs the cross-product matrices for a batch of vectors.
        The vectors can have an arbitrary number of dimensions with the last one of size 3, i.e., (..., 3).
        The output will then be of size (..., 3, 3), i.e. a 3 x 3 matrix per vector.
    """
    # vector: (..., 3)
    # output: (..., 3, 3)
    assert vectors.size(-1) == 3
    sizes = vectors.shape
    t1, t2, t3 = torch.split(vectors, 1, dim=-1)
    t1, t2, t3 = t1.squeeze(-1), t2.squeeze(-1), t3.squeeze(-1)
    tx = torch.zeros(sizes + (3, ), dtype=vectors.dtype, device=vectors.device)
    tx[..., 2, 1] = t1
    tx[..., 1, 2] = -t1
    tx[..., 2, 0] = -t2
    tx[..., 0, 2] = t2
    tx[..., 1, 0] = t3
    tx[..., 0, 1] = -t3
    return tx


def essential_matrix(r, t):
    """ Construct the essential matrices for a batch of rotation matrices and a batch of translation vectors.
        The rotations r and t are expected to have a shape (B, 3, 3) and (B, 3) respectively,
        where B is the batch size.
    """
    # r, rotation matrix: (B, 3, 3)
    # t, translation: (B, 3)
    assert r.dim() == 3 and t.dim() == 2
    assert r.shape[-2:] == (3, 3) and t.size(-1) == 3
    assert r.size(0) == t.size(0)
    tx = cross_product_matrix(t)
    e = torch.bmm(tx, r)
    return e


if __name__ == '__main__':
    vec = torch.Tensor([1, 2, 3])
    mat = cross_product_matrix(vec)
    print(mat)

    # essential matrix
    rot = torch.Tensor([
        [0.5, 0.0, -1.5],
        [1.2, 0.3, -0.1],
        [2.2, -0.2, 1.9]
    ])
    mat = essential_matrix(rot.unsqueeze(0), vec.unsqueeze(0))
    print(mat)
