from math import inf, log, sqrt
from typing import List, Optional, Tuple

from lib.agent.agent import OneDAgent
from lib.environment.oned_environment import OneDEnvironment

class EvenDar2002(OneDAgent):
    def best_arm(self, environment: OneDEnvironment) -> Tuple[Optional[int], List[List[int]]]:
        n = environment.n
        sigma = environment.sigma
        means = [environment.pull_scalar(i) for i in range(n)]
        pulls = [[i] for i in range(n)]
        pull_count = [1 for _ in range(n)]
        active = [True for _ in range(n)]
        t = 1
        active_count = n
        while True:
            rad_t = sqrt(2 * sigma * sigma * log(4 * n * t * t / self.confidence) / t)
            best = max([means[i] for i in range(n) if active[i]])

            for i in range(n):
                if not active[i]:
                    continue
                if best - means[i] >= 2 * rad_t:
                    active[i] = False
                    active_count -= 1
                    continue

            for i in range(n):
                if not active[i]:
                    continue
                if active_count == 1:
                    return i, pulls
                pull = environment.pull_scalar(i)
                means[i] = (means[i] * pull_count[i] + pull) / (pull_count[i] + 1)
                pulls.append([i])
                pull_count[i] += 1

            t += 1

    def name(self) -> str:
        return "Evendar2002"

