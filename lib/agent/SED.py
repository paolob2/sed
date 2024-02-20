from collections import OrderedDict
from typing import List, Optional, Tuple

from lib.agent.agent import Agent
from lib.agent.bound import Vector_Bound
from lib.environment.environment import Environment
from lib.vector import VectorSamples


class SED(Agent):
    def __init__(
        self,
        bound: Vector_Bound,
        max_iterations: Optional[int] = None,
        l2_opt: bool = True,
    ):
        super().__init__()
        self.bound = bound
        self.max_iterations = max_iterations
        self.l2_opt = l2_opt

    def best_arm(self, environment: Environment) -> Tuple[Optional[Tuple[int, int]], List[List[int]]]:
        n = environment.n
        m = environment.m
        sigma = environment.sigma

        pulls = [[i for i in range(n)]]
        vector_samples: List[VectorSamples] = [VectorSamples(environment.pull(i)) for i in range(n)]

        C: list[Tuple[int, int, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                C.append((i, j, vector_samples[i].avg.diff(vector_samples[j].avg)))

        t = 2
        while True:
            best = max(C, key=lambda pair: pair[2])

            # number of samples is t-1 for each arm still in C
            score_bound = self.bound.compute_vector(n, m, t, t - 1, t - 1, sigma)

            def disjoint(pairA: Tuple[int, int], pairB: Tuple[int, int]) -> bool:
                return pairA[0] != pairB[0] and pairA[0] != pairB[1] and pairA[1] != pairB[0] and pairA[1] != pairB[1]

            if self.l2_opt:
                # slight optimization if the score is the l2 distance:
                # (recall that score_bound is the 2 U_t term from the paper)
                # it can be proven that if a pair of arms (i, j) shares one
                # arm with the current best pair (i', j'), we can eliminate
                # pair (i, j) if the score gap is at least 3 U_t (rather than 4 U_t)
                # (the paper experiments do not include this optimization)
                C = [
                    pair
                    for pair in C
                    if (
                        (disjoint(pair[:2], best[:2]) and pair[2] + score_bound > best[2] - score_bound)
                        or (not disjoint(pair[:2], best[:2]) and pair[2] + score_bound > best[2] - score_bound / 2)
                    )
                ]
            else:
                C = [pair for pair in C if pair[2] + score_bound > best[2] - score_bound]

            if len(C) == 1:
                return C[0][0:2], pulls

            if t == self.max_iterations:
                print("interrupting", self.name())
                return None, pulls

            to_pull: List[int] = []
            for pair in C:
                to_pull.append(pair[0])
                to_pull.append(pair[1])

            # sort and remove duplicates
            to_pull = list(OrderedDict.fromkeys(to_pull))

            for arm in to_pull:
                vector_samples[arm].add_vector(environment.pull(arm))

            # update empirical averages
            C = [
                (
                    pair[0],
                    pair[1],
                    vector_samples[pair[0]].avg.diff(vector_samples[pair[1]].avg),
                )
                for pair in C
            ]

            pulls.append(to_pull)

            t += 1

    def name(self) -> str:
        return "SED (" + self.bound.name() + ")" + (" (l2 opt)" if self.l2_opt else "")
