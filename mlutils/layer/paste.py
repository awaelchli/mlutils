from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor, LongTensor


class Paste(nn.Module):

    def forward(self, *args, **kwargs):
        return paste(*args, **kwargs)


def paste(background: Tensor, patch: Tensor, x: LongTensor, y: LongTensor, mask: Optional[Tensor] = None):
    """
    Pastes the given patch into the background image tensor at the specified location.
    Optionally a mask of the same size as the patch can be passed in to blend the
    pasted contents with the background.

    :param background: A batch of image tensors of shape (B, C, H, W) that represent the background
    :param patch: A batch of image tensors of shape (B, C, h, w) which values get pasted into the background
    :param x: The horizontal integer coordinates relative to the top left corner of the background image.
        This tensor must be a one-dimensional tensor of shape (B, ).
    :param y: The vertical integer coordinates relative to the top left corner of the background image.
        This tensor must be a one-dimensional tensor of shape (B, ).
    :param mask: A mask of the same size as the patch that is used to blend foreground and background values.
        It is optional and defaults to ones (all is foreground).
    :return: The composite tensor of background and foreground values of shape (B, C, H, W).

    Note:
        1.  The X- and Y-coordinates can exceed the range of the background image (negative and positive).
            The background will be dynamically padded and cropped again after pasting such that the
            contents can go over the borders of the background image.
        2.  Currently it only supports integer locations.
        3.  All tensors must be on the same device.
    """
    # background: (B, C, H, W)
    # patch, mask: (B, C, h, w)
    # x, y: (B, )
    b, c, H, W = background.shape
    _, _, h, w = patch.shape
    mask = torch.ones_like(patch) if mask is None else mask
    device = background.device
    assert b == patch.size(0) == mask.size(0)
    assert b == x.size(0) == y.size(0)
    assert c == patch.size(1) == mask.size(1)
    assert h == mask.size(-2)
    assert w == mask.size(-1)
    assert 1 == x.ndimension() == y.ndimension()
    assert device == patch.device == x.device == y.device == mask.device
    x = x.long()
    y = y.long()

    # dynamically pad background for patches that go over borders
    left = min(x.min().abs().item(), 0)
    top = min(y.min().abs().item(), 0)
    right = max(x.max().item() + w - W, 0)
    bottom = max(y.max().item() + h - H, 0)
    background = nn.functional.pad(background, pad=[left, right, top, bottom])

    # generate indices
    gridb, gridc, gridy, gridx = torch.meshgrid(
        torch.arange(b, device=device),
        torch.arange(c, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device)
    )
    x = x.view(b, 1, 1, 1).repeat(1, c, h, w)
    y = y.view(b, 1, 1, 1).repeat(1, c, h, w)
    x = x + gridx + left
    y = y + gridy + top

    # we need to ignore negative indices, or pasted conent will be rolled to the other side
    mask = mask * (x >= 0) * (y >= 0)
    # paste
    background[(gridb, gridc, y, x)] = mask * patch + (1 - mask) * background[(gridb, gridc, y, x)]
    # crop away the padded regions
    background = background[..., top:(top + H), left:(left + W)]
    return background
