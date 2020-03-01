import torch
import random


class RandomCropper:
    """ This cropper can be applied multiple times to different data yielding identical crops. """

    def __init__(self, original_size, crop_size, border=(0.0, 0.0)):
        self.original_size = original_size
        self.crop_size = crop_size
        self.border = border
        self.grid = None
        self.sample()

    def sample(self):
        h, w = self.original_size
        height, width = self.crop_size

        # meshgrid for center crop
        range_x = torch.linspace((w - width) / 2, (w + width) / 2, width)
        range_y = torch.linspace((h - height) / 2, (h + height) / 2, height)
        grid_y, grid_x = torch.meshgrid(range_y, range_x)

        # grid has shape (B, H, W, 2)
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

        # sample shift
        rx, ry = (w - width) / 2., (h - height) / 2.
        dx = (2. * random.random() - 1.) * (rx - self.border[1])
        dy = (2. * random.random() - 1.) * (ry - self.border[0])

        # shift the center (same for the whole batch)
        grid[..., 0] += dx
        grid[..., 1] += dy

        # normalize
        grid[..., 0] = 2. * grid[..., 0] / w - 1.
        grid[..., 1] = 2. * grid[..., 1] / h - 1.
        self.grid = grid

    def __call__(self, batch, mode='bilinear'):
        ndim = batch.ndim
        assert ndim in (3, 4)  # (B, C, H, W) or (C, H, W)
        batch = batch.unsqueeze(0) if ndim == 3 else batch
        grid = self.grid.repeat(batch.shape[0], 1, 1, 1).to(batch.device)
        crop = torch.nn.functional.grid_sample(batch, grid, mode=mode)
        crop = crop.squeeze(0) if ndim == 3 else crop
        return crop


def random_crop_batch(batch, size, border=(0.0, 0.0), return_grid=False):
    """ Applies a random crop to a whole batch of tensors with shape (B, C, H, W).
        The same crop is chosen across the entire batch.
        :param size: A tuple specifying the height and width of the crop.
        :param border: A tuple specifying the relative border size for each dimension (height and width).
        The random crop will stay within this border.
        :param return_grid: If True, returns the grid used for sampling the crop (unnormalized).
    """
    b, _, h, w = batch.shape
    height, width = size

    # meshgrid for center crop
    range_x = torch.linspace((w - width) / 2, (w + width) / 2, width, device=batch.device)
    range_y = torch.linspace((h - height) / 2, (h + height) / 2, height, device=batch.device)
    grid_y, grid_x = torch.meshgrid(range_y, range_x)

    # grid has shape (B, H, W, 2)
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)

    # sample shift
    rx, ry = (w - width) / 2., (h - height) / 2.
    dx = (2. * random.random() - 1.) * (rx - border[1])
    dy = (2. * random.random() - 1.) * (ry - border[0])

    # shift the center (same for the whole batch)
    grid[..., 0] += dx
    grid[..., 1] += dy

    # normalize
    grid[..., 0] = 2. * grid[..., 0] / w - 1.
    grid[..., 1] = 2. * grid[..., 1] / h - 1.

    # sample
    crop = torch.nn.functional.grid_sample(batch, grid)
    if return_grid:
        return crop, grid
    else:
        return crop


def shift_center_crop(batch, size, rng):
    b, _, h, w = batch.shape
    height, width = size

    assert 0 <= rng <= 1.0

    # meshgrid for center crop
    range_x = torch.linspace((w - width) / 2, (w + width) / 2, width, device=batch.device)
    range_y = torch.linspace((h - height) / 2, (h + height) / 2, height, device=batch.device)
    grid_y, grid_x = torch.meshgrid(range_y, range_x)

    # grid has shape (B, H, W, 2)
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)

    # center crop
    gridc = grid.clone()
    gridc[..., 0] = 2. * grid[..., 0] / w - 1.
    gridc[..., 1] = 2. * grid[..., 1] / h - 1.
    center = torch.nn.functional.grid_sample(batch, gridc)

    # random shift (optical flow)
    dx = (2. * random.random() - 1.) * width * rng
    dy = (2. * random.random() - 1.) * height * rng

    # shift the center (same for the whole batch)
    grid[..., 0] += dx
    grid[..., 1] += dy

    # normalize
    grid[..., 0] = 2. * grid[..., 0] / w - 1.
    grid[..., 1] = 2. * grid[..., 1] / h - 1.

    # shifted crop
    shifted = torch.nn.functional.grid_sample(batch, grid)

    return shifted, center, dx, dy


def random_rotate_batch(batch):
    assert batch.dim() >= 2

    def rot90(x):
        return x.transpose(-2, -1).flip(-2)

    i = random.choice(range(4))

    if i == 0:
        rotated = batch
    elif i == 1:
        rotated = rot90(batch)
    elif i == 2:
        rotated = rot90(rot90(batch))
    else:
        rotated = rot90(rot90(rot90(batch)))

    return rotated
