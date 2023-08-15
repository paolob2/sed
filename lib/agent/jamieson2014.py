import math
from typing import List, Optional, Tuple

from lib.agent.agent import OneDAgent
from lib.environment.oned_environment import OneDEnvironment
from lib.utils import lil_delta, U, argmax

class Jamieson2014(OneDAgent):
    def __init__(self, confidence: float, epsilon: float):
        super().__init__(confidence)
        self.epsilon = epsilon
        self.lil_delta = lil_delta(confidence, epsilon)

    def best_arm(self, environment: OneDEnvironment) -> Tuple[Optional[int], List[List[int]]]:
        n = environment.n
        sigma_squared = environment.sigma ** 2
        means = [environment.pull_scalar(i) for i in range(n)]
        pulls = [[i] for i in range(n)]
        pull_count = [1 for _ in range(n)]
        ucb = [means[i] + U(pull_count[i], sigma_squared, self.lil_delta / n, self.epsilon) for i in range(n)]
        t = n
        while True:
            i = argmax(means)
            ucb_i = ucb[i]
            ucb[i] = -math.inf
            j = argmax(ucb)
            ucb[i] = ucb_i
            if means[i] - U(pull_count[i], sigma_squared, self.lil_delta / n, self.epsilon) > ucb[j]:
                return i, pulls
            if pull_count[j] < pull_count[i]:
                i = j
            pull = environment.pull_scalar(i)
            means[i] = (means[i] * pull_count[i] + pull) / (pull_count[i] + 1)
            pulls.append([i])
            pull_count[i] += 1
            ucb[i] = means[i] + U(pull_count[i], sigma_squared, self.lil_delta / n, self.epsilon)
            t += 1

    def name(self) -> str:
        return "Jamieson2014"
