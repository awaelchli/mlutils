from math import log10


def mse2psnr(mse):
    psnr = 10. * log10(1. / (mse + 1e-16))
    return psnr
