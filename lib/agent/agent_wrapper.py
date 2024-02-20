from typing import List, Tuple

from lib.agent.agent import Agent, OneDAgent
from lib.environment.environment import Environment
from lib.environment.generator import Generator
from lib.environment.oned_environment import (
    OneDEnvironment,
    PairedOneDEnvironment,
    PairedVectorEnvironment,
)
from lib.utils import compute_env_info


def get_pulls_count(pulls: List[List[int]]) -> int:
    return sum([len(epoch) for epoch in pulls])


def get_rounds_count(pulls: List[List[int]]) -> int:
    return len(pulls)


class AgentStats:
    def __init__(self):
        # each element in the following lists corresponds to a different run
        self.error_deltas: List[float] = []
        self.rounds_count: List[int] = []
        self.pulls_count: List[int] = []

    def update(
        self,
        best: Tuple[int, int],
        pulls: List[List[int]],
        true_best: Tuple[int, int],
        delta_min: float,
    ):
        if best != true_best:
            self.error_deltas.append(delta_min)
        self.rounds_count.append(get_rounds_count(pulls))
        self.pulls_count.append(get_pulls_count(pulls))


class AgentWrapper:
    """
    Utility class to extract relevant metrics from the results of agent.best_arm
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        self.stats = [AgentStats()]

    def run(self, environment: Environment):
        best, pulls = self.agent.best_arm(environment)
        assert best is not None and best[0] < best[1]
        if isinstance(environment, OneDEnvironment):
            opt = (0, environment.n - 1)
            delta_min = min(
                environment.mu[1] - environment.mu[0],
                environment.mu[-1] - environment.mu[-2],
            )
        else:
            opt, delta_min = compute_env_info(environment.means)
        self.stats[0].update(best, pulls, opt[0:2], delta_min)


class PairedAgentWrapper(AgentWrapper):
    def __init__(self, agent: OneDAgent):
        super().__init__(agent)

    def run_paired(self, environment: PairedOneDEnvironment, n: int) -> Tuple[Tuple[int, int], List[List[int]]]:
        best, pulls = self.agent.best_arm(environment)
        assert best is not None
        best = environment.to_pair(best, n)
        assert best[0] < best[1]
        for j in range(len(pulls)):
            converted_pulls = []
            for paired_pull in pulls[j]:
                converted_pulls.extend(list(environment.to_pair(paired_pull, n)))
            pulls[j] = converted_pulls
        return best, pulls

    def run(self, environment: OneDEnvironment | Generator):
        if isinstance(environment, OneDEnvironment):
            best, pulls = self.run_paired(PairedOneDEnvironment(environment), environment.n)
            opt = (0, environment.n - 1)
            delta_min = min(
                environment.mu[1] - environment.mu[0],
                environment.mu[-1] - environment.mu[-2],
            )
        else:
            best, pulls = self.run_paired(PairedVectorEnvironment(environment), environment.n)
            opt, delta_min = compute_env_info(environment.means)
        self.stats[0].update(best, pulls, opt[0:2], delta_min)


class MaxMinAgentWrapper(AgentWrapper):
    def __init__(self, agent: OneDAgent):
        self.agent = agent
        self.stats = [AgentStats(), AgentStats()]

    def run(self, environment: OneDEnvironment):
        max_best, max_pulls = self.agent.best_arm(environment)
        assert max_best is not None
        self.stats[0].update(max_best, max_pulls, 0, environment.mu[0] - environment.mu[1])
        environment.mu = [-mu for mu in environment.mu]
        min_best, min_pulls = self.agent.best_arm(environment)
        assert min_best is not None
        self.stats[1].update(
            min_best,
            min_pulls,
            environment.n - 1,
            environment.mu[-1] - environment.mu[-2],
        )
        environment.mu = [-mu for mu in environment.mu]
