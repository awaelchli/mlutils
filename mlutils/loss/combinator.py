from typing import Union

from torch import Tensor


class LossCombinator(object):

    def __init__(self):
        super().__init__()
        self.__reset()

    def __reset(self):
        self.__parts = []

    def reset(self):
        self.__reset()

    def add(self, value: Union[Tensor, float], weight: float = 1.0):
        self.__parts.append((value, weight))

    def total(self, normalize: bool = False) -> Union[Tensor, float]:
        total = sum(value * weight for value, weight in self.__parts)
        weights = sum(weight for _, weight in self.__parts)
        if normalize and weights != 0:
            total = total / weights
        return total
