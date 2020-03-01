from typing import Union

from torch import Tensor


class AverageMeter(object):
    """ Computes and stores the average and current value. """

    def __init__(self):
        self.__reset()

    def __reset(self):
        self.__sum = 0
        self.__count = 0

    @property
    def sum(self) -> Union[Tensor, float]:
        return self.__sum

    @property
    def count(self) -> int:
        return self.__count

    @property
    def avg(self) -> Union[Tensor, float]:
        return self.sum / self.count

    def reset(self):
        self.__reset()

    def update(self, val: Union[Tensor, float], n: int = 1):
        self.__sum += val * n
        self.__count += n
