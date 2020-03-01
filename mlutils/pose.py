import torch


def relative_to_first_pose_matrix(matrices):
    """ Makes all pose matrices in the batch relative to the first pose.
        Each matrix in the batch is expected to be a 3 x 4 torch tensor encoding rotation in the first 3 x 3 block
        and translation in the last column.
        Given a pair of matrices, the formulas to convert the poses is as follows:
        Given: R(1->world), R(2->world), t1, t2
        R(2->1) = R(world->1) * R(2->world)
                = R(1->world)^(-1) * R(2->world)
        t(2->1) = R(1->world)^(-1) * (t2 - t1)
    """
    n = len(matrices)
    r = matrices[:, :, 0:3]
    t = matrices[:, :, 3:4]

    rot1_inv = r[0:1].expand(n, -1, -1).transpose(1, 2)  # For rotations: inverse = transpose
    t1 = t[0:1]

    r_rel = torch.bmm(rot1_inv, r)
    t_rel = torch.bmm(rot1_inv, t - t1)

    rel_matrices = torch.cat((r_rel, t_rel), -1)
    return rel_matrices
