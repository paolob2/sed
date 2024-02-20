import abc
from math import log, sqrt


class Bound(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def compute(self, n: int, m: int, t: int, p: int, sigma: float) -> float:
        """
        Arguments:
            int: number of arms
            int: vector length
            int: current time epoch
            int: pulls count for the current arm
            float: sub-gaussian parameter
        Returns:
            float: bound according to the given parameters
        """
        pass

    @abc.abstractmethod
    def name(self) -> str:
        """
        Returns:
            str: a name describing the bound
        """


class Vector_Bound(Bound):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def compute_vector(self, n: int, m: int, t: int, pi: int, pj: int, R: float) -> float:
        """
        Computes the bound for the L2-norm of the error vector
        rather than the bound for the single error element
        """
        pass


class SubG_Bound(Vector_Bound):
    def __init__(self, delta: float, lip_k: float = 1):
        self.delta = delta
        self.lip_k = lip_k

    def compute(self, n: int, m: int, t: int, p: int, R: float) -> float:
        return R * sqrt(2 * m * log(4 * n * m * t * t / self.delta) / p)

    def compute_vector(self, n: int, m: int, t: int, pi: int, pj: int, R: float) -> float:
        return sqrt(m) * self.lip_k * (self.compute(n, m, t, pi, R) + self.compute(n, m, t, pj, R))

    def name(self) -> str:
        return "SubGaussian bound"


class SubExp_Bound(Vector_Bound):
    def __init__(self, delta: float, lip_k: float = 1):
        self.delta = delta
        self.lip_k = lip_k

    def compute(self, n: int, m: int, t: int, p: int, R: float) -> float:
        # the constants here have been slightly corrected since the paper experiments
        return sqrt(16 * R * R * max(m, log(2 * n * t * t / self.delta)) / p)

    def compute_vector(self, n: int, m: int, t: int, pi: int, pj: int, R: float) -> float:
        return self.lip_k * (self.compute(n, m, t, pi, R) + self.compute(n, m, t, pj, R))

    def name(self) -> str:
        return "SubExponential bound"
