import numpy as np


__all__ = [
    'read_flo'
]


TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'


def read_flo(filename):
    """ Read optical flow from file, return (U,V) tuple.
        Original code by Deqing Sun, adapted from Daniel Scharstein.
        Check for endianness, based on Daniel Scharstein's optical flow code.
        Using little-endian architecture, these two should be equal.
    """
    f = open(filename, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT, check)
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' flow_read:: Wrong input size (width = {0}, height = {1}).'.format(width, height)
    tmp = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width * 2))
    u = tmp[:, np.arange(width) * 2]
    v = tmp[:, np.arange(width) * 2 + 1]
    uv = np.stack((u, v), 2)
    return uv
