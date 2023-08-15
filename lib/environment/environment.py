import abc
import numpy as np
from typing import Tuple

from lib.vector import Vector


class Environment(metaclass=abc.ABCMeta):
    def __init__(self, n: int, m: int, sigma: float):
        self.n = n
        self.m = m
        self.sigma = sigma

    @abc.abstractmethod
    def pull(self, i: int) -> Vector:
        """
        Arguments:
            int: arm to pull, 0-based
        Returns:
            np.ndarray: vector sampled from enviroment
                corresponding to the given arm
        """
        pass