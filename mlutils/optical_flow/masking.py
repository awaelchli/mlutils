from math import sqrt

import torch
from mlutils.optical_flow.misc import divergence
from mlutils.optical_flow.transform import warp_grid


def valid_warp_mask(flow):
    # (normalized) flow: (B, 2, H, W)
    flow = flow.permute(0, 2, 3, 1)
    grid = warp_grid(flow)  # (B, H, W, 2)
    valid = (grid >= -1.) & (grid <= 1.)
    valid = valid[:, :, :, 0] & valid[:, :, :, 1]
    valid = valid.unsqueeze(1)
    return valid  # (B, 1, H, W)


def count_valid(flow):
    """ Counts the number of valid optical flow vectors per sample in the batch. """
    # (normalized) flow: (B, 2, H, W)
    valid = valid_warp_mask(flow).float()
    return valid.flatten(1).sum(dim=1).mean()


def boundary_mask(image, boundary=1):
    """ Creates a broadcastable mask (FloatTensor) with shape (1, 1, H, W) with the height and width of the given image
        that has shape (B, C, H, W). The boundary is set to zero, and the inside contains ones.
    """
    _, _, h, w = image.shape
    mask = torch.ones(1, 1, (h - 2 * boundary), (w - 2 * boundary), device=image.device)
    paddings = (boundary, boundary, boundary, boundary)
    mask = F.pad(mask, paddings, mode='constant', value=0.)
    return mask


def masked_criterion(criterion, flow, warped, target, fraction=0.9):
    """ Applies a dynamic mask to the warping and ground truth before the criterion.
        The mask is computed based on the valid elements in the flow field.

        :param criterion: A function or torch.nn criterion that takes two arbitrarily shaped tensors of the same size
        and computes the error.

        :param fraction: The fraction of max. invalid flows that should be ignored, relative to the number of pixels.
        The function will select at most M invalid pixels that are removed from the error computation, where
        M = fraction * number of pixels.

        :param flow: The optical flow of size (B, 2, H, W) used for warping. It is expected that most values are in
        the range [-1, 1], and the ones that are not will be considered as invalid.

        :param warped: An image batch of size (B, 3, H, W) warped by the flow field.

        :param target: The target image batch.
    """
    b, _, h, w = flow.shape
    m = int(h * w * fraction)

    # get valid elements of flow grid
    valid = valid_warp_mask(flow)
    valid = valid.view(b, -1)

    # sample a permutation
    weights = torch.ones(h * w, device=flow.device)
    perm = torch.multinomial(weights, h * w, replacement=True)

    # shuffle pixels (same permutation for each sample in batch)
    valid_shuffled = torch.index_select(valid, 1, perm)
    invalid_shuffled = ~valid_shuffled

    # pick at most m invalid elements
    sums = torch.cumsum(invalid_shuffled, dim=1)
    invalid_m = torch.where(sums <= m, invalid_shuffled, torch.zeros_like(invalid_shuffled))

    # "at most m invalid elements" is equivalent to "at least m valid elements".
    valid_m = ~invalid_m
    valid_m = valid_m.view(b, 1, -1)

    # shuffle pixels in the warped- and ground-truth image the same way the flow was shuffled
    warped = warped.view(b, 3, -1)
    target = target.view(b, 3, -1)
    warped = torch.index_select(warped, 2, perm)
    target = torch.index_select(target, 2, perm)

    # apply loss function only on the masked elements
    warp_reduced = torch.masked_select(warped, valid_m)
    gt_reduced = torch.masked_select(target, valid_m)

    return criterion(warp_reduced, gt_reduced)


def flow_fold_mask(flow, threshold=1.0):
    # flow: (B, 2, H, W)
    flow = F.pad(flow, (1, 1, 1, 1))

    c = flow[:, :, 1:-1, 1:-1]
    r = flow[:, :, 1:-1, 2:]
    l = flow[:, :, 1:-1, :-2]
    t = flow[:, :, :-2, 1:-1]
    b = flow[:, :, 2:, 1:-1]

    # corners
    tl = flow[:, :, :-2, :-2]
    tr = flow[:, :, :-2, 2:]
    bl = flow[:, :, 2:, :-2]
    br = flow[:, :, 2:, 2:]

    # concatenate (call it neighbors of c)
    # n: (B, 2, 8, H, W)
    n = torch.stack((r, l, t, b, tl, tr, bl, br), 2)

    # center to ...
    s = 1.0 / sqrt(2.0)
    dirs = torch.Tensor([
        [1.0, 0.0],     # r
        [-1.0, 0.0],    # l
        [0.0, -1.0],    # t
        [0.0, 1.0],     # b
        [-s, -s],       # tl
        [s, -s],        # tr
        [-s, s],        # bl
        [s, s],         # br
    ]).to(flow.device)

    # make dirs broadcastable in B, H, W dimensions
    dirs = dirs.transpose(0, 1).view(1, 2, -1, 1, 1)
    ref = c.unsqueeze(2)

    # dot product between neighbor flow and dir
    dot1 = torch.sum(n * dirs, dim=1)

    # dot product between center flow and dir
    dot2 = torch.sum(ref * dirs, dim=1)

    diff = torch.abs(dot1 - dot2)
    mask = diff > threshold
    mask = mask[:, 0] | mask[:, 1] | mask[:, 2] | mask[:, 3] | mask[:, 4] | mask[:, 5] | mask[:, 6] | mask[:, 7]
    return mask.unsqueeze(1)


def div_mask(flow, threshold=1.0):
    div = divergence(flow)
    # mask = torch.abs(div) > threshold
    mask = div < -threshold  # only when divergence negative, there is occlusion from frame t -> t+1
    return mask.unsqueeze(1)