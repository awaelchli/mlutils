import torch
import numpy as np
from matplotlib.colors import hsv_to_rgb


def flow2rgb(flow, max_norm=1., invert_y=True):
    """
    Map optical flow to color image.
    The color hue is determined by the angle to the X-axis and the norm of the flow determines the saturation.
    White represents zero optical flow.

    :param flow: A torch.Tensor or numpy.ndarray of shape (B, 2, H, W). The components flow[:, 0] and flow[:, 1] are
    the X- and Y-coordinates of the optical flow, respectively.
    :param max_norm: The maximum norm of optical flow to be clipped. Default: 1.
    The optical flows that have a norm greater than max_value will be clipped for visualization.
    :param invert_y: Default: True. By default the optical flow is expected to be in a coordinate system with the
    Y axis pointing downwards. For intuitive visualization, the Y-axis is inverted.
    :return: A numpy.ndarray of shape (B, 3, H, W)
    """
    flow = flow.clone()
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    assert isinstance(flow, np.ndarray)

    # flow[:, :, 0, 0] = max_norm
    # flow[:, :, 0, 1] = -max_norm

    if invert_y:
        flow[:, 1] *= -1

    dx, dy = flow[:, 0], flow[:, 1]

    angle = np.arctan2(dy, dx)
    angle = np.where(angle < 0, 2 * np.pi + angle, angle)
    scale = np.sqrt(dx ** 2 + dy ** 2) / max_norm

    h = angle / (2 * np.pi)
    s = np.clip(scale, 0, 1)
    v = np.ones_like(s)

    hsv = np.stack((h, s, v), -1)
    rgb = hsv_to_rgb(hsv)
    rgb = np.transpose(rgb, (0, 3, 1, 2))
    return rgb
