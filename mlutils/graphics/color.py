import torch
from torch import Tensor


def hsv_to_rgb(image: Tensor) -> Tensor:
    r"""Convert an HSV image to RGB.
    The image data is assumed to be in the range of (0, 1).
    The code was taken and adapted from Kornia.
    See: https://github.com/kornia/kornia

    Args:
        image: RGB image to be converted to HSV.

    Returns:
        HSV version of the image.
    """

    assert torch.is_tensor(image)
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    h = image[..., 0, :, :]
    s = image[..., 1, :, :]
    v = image[..., 2, :, :]

    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    out = torch.stack([hi, hi, hi], dim=-3) % 6

    out[out == 0] = torch.stack((v, t, p), dim=-3)[out == 0]
    out[out == 1] = torch.stack((q, v, p), dim=-3)[out == 1]
    out[out == 2] = torch.stack((p, v, t), dim=-3)[out == 2]
    out[out == 3] = torch.stack((p, q, v), dim=-3)[out == 3]
    out[out == 4] = torch.stack((t, p, v), dim=-3)[out == 4]
    out[out == 5] = torch.stack((v, p, q), dim=-3)[out == 5]

    return out


def rgb_to_hsv(image: Tensor) -> Tensor:
    r"""Convert an RGB image to HSV.
    The code was taken and adapted from Kornia.
    See: https://github.com/kornia/kornia

    Args:
        image: RGB image to be converted to HSV.

    Returns:
        HSV version of the image.
    """

    assert torch.is_tensor(image)
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    maxc = image.max(-3)[0]
    minc = image.min(-3)[0]

    v = maxc  # brightness

    deltac = maxc - minc
    s = deltac / v  # saturation

    # avoid division by zero
    deltac = torch.where(
        deltac == 0, torch.ones_like(deltac), deltac
    )

    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac

    maxg = g == maxc
    maxr = r == maxc

    h = 4.0 + gc - rc
    h[maxg] = 2.0 + rc[maxg] - bc[maxg]
    h[maxr] = bc[maxr] - gc[maxr]
    h[minc == maxc] = 0.0

    h = (h / 6.0) % 1.0

    return torch.stack([h, s, v], dim=-3)
