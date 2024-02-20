import numpy as np
from typing import Any, List, Tuple

from scipy.stats import norm

from lib.vector import Vector


def argmax(lst: List[Any]) -> int:
    return lst.index(max(lst))


def argmin(lst: List[Any]) -> int:
    return lst.index(min(lst))


def U(p: float, sigma_squared: float, delta: float, epsilon: float) -> float:
    return (1 + np.sqrt(epsilon)) * np.sqrt(
        2 * sigma_squared * (1 + epsilon) / p * np.log(np.log((1 + epsilon) * p) / delta)
    )


def lil_delta(delta: float, epsilon: float) -> float:
    return np.log(1 + epsilon) * pow(delta * epsilon / (2 + epsilon), 1 / (1 + epsilon))


def get_confidence_range(var: float, n: int, confidence: float):
    return norm.ppf(1 - (1 - confidence) / 2) * np.sqrt(var) / np.sqrt(n)


def compute_env_info(means: List[Vector]) -> Tuple[Tuple[int, int, float], float]:
    """
    Returns:
        (int, int, float): the indices of the best pair and the respective score
        (float): the minimum difference between the best score and the score of any other pair
    """
    n = len(means)
    opt = (0, 0, 0.0)
    delta_min = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            v = means[i].diff(means[j])
            if v > opt[2]:
                delta_min = v - opt[2]
                opt = (i, j, v)
            elif opt[2] - v < delta_min:
                delta_min = opt[2] - v
    return opt, delta_min
