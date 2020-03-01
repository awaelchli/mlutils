import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class TransformDataset(Dataset):

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.transform(*self.dataset[item])


class FlowNetTransform(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x1, x2, flow, *args):
        pass


class Compose(FlowNetTransform):

    def __init__(self, transforms=None):
        super(FlowNetTransform, self).__init__()
        for t in transforms:
            assert isinstance(t, FlowNetTransform)
        self.transforms = transforms or []

    def __call__(self, x1, x2, flow, *args):
        outputs = x1, x2, flow, *args
        for transform in self.transforms:
            outputs = transform(*outputs)
        return outputs


class ArrayToTensor(FlowNetTransform):
    """ Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W). """

    def __call__(self, *args):
        tensors = []
        for array in args:
            assert (isinstance(array, np.ndarray))
            array = np.transpose(array, (2, 0, 1))
            tensor = torch.from_numpy(array)
            tensors.append(tensor)
        return tensors


class Normalize(FlowNetTransform):
    """ Abstract class for normalization. """

    def __init__(self, mean: list, std: list):
        super(Normalize, self).__init__()
        assert len(mean) == len(std)
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def __call__(self, *args):
        pass


class NormalizeImages(Normalize):
    """ Normalize images with mean, std. """

    def __init__(self, mean: list, std: list):
        super(NormalizeImages, self).__init__(mean, std)
        assert len(mean) == len(std) == 3

    def __call__(self, x1, x2, flow, *args):
        return (self.normalize(x1), self.normalize(x2), flow, *args)


class NormalizeFlow(Normalize):
    """ Normalize the flow with mean, std. """

    def __init__(self, mean: list, std: list):
        super(NormalizeFlow, self).__init__(mean, std)
        assert len(mean) == len(std) == 2

    def __call__(self, x1, x2, flow, *args):
        return (x1, x2, self.normalize(flow), *args)


class NormalizeMultiple(Normalize):
    """ Normalize the flow with mean, std. """

    def __init__(self, mean: list, std: list, indices=None):
        super(NormalizeMultiple, self).__init__(mean, std)
        self.indices = indices

    def __call__(self, *args):
        indices = self.indices
        if indices is None:
            indices = range(len(args))
        else:
            assert isinstance(self.indices, (tuple, list)) and max(self.indices) < len(args)

        arrays = [array for array in args]
        for i in indices:
            arrays[i] = self.normalize(arrays[i])
        return arrays


class CenterCrop(FlowNetTransform):
    """ Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """

    def __init__(self, size):
        super(CenterCrop, self).__init__()
        if isinstance(size, (float, int)):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1, img2, flow, *args):
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
        x2 = int(round((w2 - tw) / 2.))
        y2 = int(round((h2 - th) / 2.))

        img1 = img1[y1:(y1 + th), x1:(x1 + tw)]
        img2 = img2[y2:(y2 + th), x2:(x2 + tw)]
        flow = flow[y1:(y1 + th), x1:(x1 + tw)]
        return (img1, img2, flow, *args)


class Scale(FlowNetTransform):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size, order=2):
        super(FlowNetTransform, self).__init__()
        self.size = size
        self.order = order

    def __call__(self, img1, img2, flow, *args):
        h, w, _ = img1.shape
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return (img1, img2, flow, *args)
        if w < h:
            ratio = self.size / w
        else:
            ratio = self.size / h

        img1 = np.clip(ndimage.interpolation.zoom(img1, (ratio, ratio, 1), order=self.order), 0., 1.)
        img2 = np.clip(ndimage.interpolation.zoom(img2, (ratio, ratio, 1), order=self.order), 0., 1.)

        flow = ndimage.interpolation.zoom(flow, (ratio, ratio, 1), order=self.order)
        flow *= ratio
        return (img1, img2, flow, *args)


class RandomCrop(FlowNetTransform):
    """ Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, indices=(0, 1, 2)):
        super(RandomCrop, self).__init__()
        if isinstance(size, (float, int)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.indices = tuple(set(indices))  # ignore index repetitions

    def __call__(self, *args):
        assert max(self.indices) < len(args)
        cropped = [arg for arg in args]
        for i in self.indices:
            array = args[i]
            h, w, _ = array.shape
            th, tw = self.size
            if w == tw and h == th:
                continue
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            cropped[i] = array[y1:(y1 + th), x1:(x1 + tw)]
        return tuple(cropped)


class RandomSizedCrop(FlowNetTransform):

    def __init__(self, min_size, max_size, indices):
        super(RandomSizedCrop, self).__init__()
        self.min_size = self.__normalize_size(min_size)
        self.max_size = self.__normalize_size(max_size)
        assert 1 <= self.min_size[0] <= self.max_size[0]
        assert 1 <= self.min_size[1] <= self.max_size[1]
        self.indices = indices

    @staticmethod
    def __normalize_size(size):
        if isinstance(size, (float, int)):
            return int(size), int(size)
        else:
            return size

    def __call__(self, *args):
        size_h = random.randint(self.min_size[0], self.max_size[0])
        size_w = random.randint(self.min_size[1], self.max_size[1])
        size = (size_h, size_w)
        cropper = RandomCrop(size, indices=self.indices)
        return cropper(*args)


class RandomHorizontalFlip(FlowNetTransform):
    """ Randomly horizontally flips the given PIL.Image with a probability of 0.5. """

    def __call__(self, img1, img2, flow, *args):
        if random.random() < 0.5:
            img1 = np.copy(np.fliplr(img1))
            img2 = np.copy(np.fliplr(img2))
            flow = np.copy(np.fliplr(flow))
            flow[:, :, 0] *= -1
        return (img1, img2, flow, *args)


class RandomVerticalFlip(FlowNetTransform):
    """ Randomly horizontally flips the given PIL.Image with a probability of 0.5. """

    def __call__(self, img1, img2, flow, *args):
        if random.random() < 0.5:
            img1 = np.copy(np.flipud(img1))
            img2 = np.copy(np.flipud(img2))
            flow = np.copy(np.flipud(flow))
            flow[:, :, 1] *= -1
        return (img1, img2, flow, *args)


class RandomRotate(FlowNetTransform):
    """ Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        super(RandomRotate, self).__init__()
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, img1, img2, flow, *args):
        applied_angle = random.uniform(-self.angle, self.angle)
        diff = random.uniform(-self.diff_angle, self.diff_angle)
        angle1 = applied_angle - diff / 2
        angle2 = applied_angle + diff / 2
        angle1_rad = angle1 * np.pi / 180

        h, w, _ = flow.shape

        def rotate_flow(i, j, k):
            return -k * (j - w / 2) * (diff * np.pi / 180) + (1 - k) * (i - h / 2) * (diff * np.pi / 180)

        rotate_flow_map = np.fromfunction(rotate_flow, flow.shape)
        flow += rotate_flow_map

        img1 = ndimage.interpolation.rotate(img1, angle1, reshape=self.reshape, order=self.order)
        img2 = ndimage.interpolation.rotate(img2, angle2, reshape=self.reshape, order=self.order)
        flow = ndimage.interpolation.rotate(flow, angle1, reshape=self.reshape, order=self.order)
        # flow vectors must be rotated too! careful about Y flow which is upside down
        flow_ = np.copy(flow)
        flow[:, :, 0] = np.cos(angle1_rad) * flow_[:, :, 0] + np.sin(angle1_rad) * flow_[:, :, 1]
        flow[:, :, 1] = -np.sin(angle1_rad) * flow_[:, :, 0] + np.cos(angle1_rad) * flow_[:, :, 1]
        return (img1, img2, flow, *args)


class RandomTranslate(FlowNetTransform):

    def __init__(self, translation):
        super(RandomTranslate, self).__init__()
        if isinstance(translation, (float, int)):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, img1, img2, flow, *args):
        h, w, _ = img1.shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return (img1, img2, flow, *args)
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1, x2, x3, x4 = max(0, tw), min(w + tw, w), max(0, -tw), min(w - tw, w)
        y1, y2, y3, y4 = max(0, th), min(h + th, h), max(0, -th), min(h - th, h)

        img1 = img1[y1:y2, x1:x2]
        img2 = img2[y3:y4, x3:x4]
        flow = flow[y1:y2, x1:x2]
        flow[:, :, 0] += tw
        flow[:, :, 1] += th

        return (img1, img2, flow, *args)


class RandomColorWarp(FlowNetTransform):

    def __init__(self, mean_range=0, std_range=0):
        super(RandomColorWarp, self).__init__()
        self.mean_range = mean_range
        self.std_range = std_range

    def __call__(self, img1, img2, flow, *args):
        random_std = np.random.uniform(-self.std_range, self.std_range, 3)
        random_mean = np.random.uniform(-self.mean_range, self.mean_range, 3)
        random_order = np.random.permutation(3)

        img1 *= (1 + random_std)
        img1 += random_mean

        img2 *= (1 + random_std)
        img2 += random_mean

        img1 = img1[:, :, random_order]
        img2 = img2[:, :, random_order]

        return (img1, img2, flow, *args)


class ConstantFlow(FlowNetTransform):

    def __init__(self, dxdy=0):
        super(ConstantFlow, self).__init__()
        if isinstance(dxdy, (tuple, list)):
            assert len(dxdy) == 2
            self.dx, self.dy = (int(dxdy[0]), int(dxdy[1]))
        else:
            assert isinstance(dxdy, (int, float))
            self.dx = self.dy = int(dxdy)

    def __call__(self, img1, img2, flow, *args):
        flow[:, :, 0] = self.dx
        flow[:, :, 1] = self.dy
        return (img1, img2, flow, *args)
