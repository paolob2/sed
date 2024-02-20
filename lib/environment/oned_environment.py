import abc
import numpy as np
from typing import List, Tuple

from lib.environment.environment import Environment
from lib.vector import Vector


class OneDEnvironment(Environment):
    def __init__(self, mu: List[float], sigma: float):
        super().__init__(len(mu), 1, sigma)
        self.mu = mu

    @abc.abstractmethod
    def pull_scalar(self, i: int) -> float:
        pass

    def pull(self, i: int) -> Vector:
        return Vector(np.array([self.pull_scalar(i)]))


class GaussianOneDEnvironment(OneDEnvironment):
    def __init__(self, mu: List[float], variance_range: Tuple[float, float]):
        super().__init__(mu, np.sqrt(variance_range[1]))
        self.sigmas = np.random.uniform(np.sqrt(variance_range[0]), np.sqrt(variance_range[1]), len(mu))

    def pull_scalar(self, i: int) -> float:
        return np.random.normal(self.mu[i], self.sigmas[i])


class BernoulliOneDEnvironment(OneDEnvironment):
    def __init__(self, mu: List[float]):
        super().__init__(mu, 0.5)

    def pull_scalar(self, i: int) -> float:
        if np.random.rand() < self.mu[i]:
            return 1
        else:
            return 0


class PairedOneDEnvironment(OneDEnvironment):
    def __init__(self, base_environment: OneDEnvironment):
        self.base_environment = base_environment
        self.n = base_environment.n * (base_environment.n - 1)
        self.m = 1
        self.sigma = 2 * base_environment.sigma

    def pull_scalar(self, i: int) -> float:
        n = self.base_environment.n
        u, v = self.to_pair(i, n)
        return self.base_environment.pull_scalar(u) - self.base_environment.pull_scalar(v)

    def to_pair(self, i: int, n: int) -> Tuple[int, int]:
        u = i // (n - 1)
        v = i % (n - 1)
        if v >= u:
            v += 1
        return u, v


class PairedVectorEnvironment(PairedOneDEnvironment):
    def __init__(self, base_environment: Environment):
        self.base_environment = base_environment
        self.n = base_environment.n * (base_environment.n - 1) // 2
        self.m = 1
        self.sigma = 2 * np.sqrt(2 * base_environment.m) * base_environment.sigma
        self.pairs = []
        for i in range(base_environment.n):
            for j in range(i + 1, base_environment.n):
                self.pairs.append((i, j))

    def pull_scalar(self, i: int) -> float:
        n = self.base_environment.n
        u, v = self.to_pair(i, n)
        return self.base_environment.pull(u).diff(self.base_environment.pull(v))

    def to_pair(self, i: int, n: int) -> Tuple[int, int]:
        return self.pairs[i]
