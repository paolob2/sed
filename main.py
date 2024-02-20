from argparse import ArgumentParser
import csv
import numpy as np
from typing import Callable, List

from lib.agent.agent import Agent, OneDAgent
from lib.agent.agent_wrapper import AgentWrapper, PairedAgentWrapper, MaxMinAgentWrapper
from lib.agent.evendar2002 import EvenDar2002
from lib.agent.jamieson2014 import Jamieson2014
from lib.agent.SED import SED
from lib.agent.bound import SubExp_Bound
from lib.environment.environment import Environment
from lib.environment.oned_environment import (
    GaussianOneDEnvironment,
    OneDEnvironment,
)
from lib.environment.generator import Generator, ImageGenerator
from lib.utils import get_confidence_range, compute_env_info
from lib.vector import Image

# gen choices
equalIntervals = "equalIntervals"
concentratedMid = "concentratedMid"
concentratedMinMax = "concentratedMinMax"
random = "random"

# score choices
l1 = "L1Distance"
l2 = "L2Distance"


def build_random_nd_environment(args) -> Environment:
    ok = False
    while not ok:
        images = [Image((args.image_size, args.image_size)) for _ in range(args.number_of_arms)]
        _, delta_min = compute_env_info(images)
        if delta_min >= args.delta_min:
            ok = True
    return ImageGenerator(images, (args.min_variance, args.max_variance))


def build_1d_environment(args) -> OneDEnvironment:
    # the means mu will always be sorted descendingly for convenience
    n = args.number_of_arms
    if args.generator == equalIntervals:
        mu = [(n - i) / n for i in range(n)]
    elif args.generator == concentratedMid:
        mu = [0.5 for i in range(n)]
        mu[0] = 1
        mu[-1] = 0
    elif args.generator == concentratedMinMax:
        mu = [(n - 1) / n for _ in range(n // 2)]
        if n % 2 == 1:
            mu.append(0.5)
        mu.extend([1 / n for _ in range(n // 2)])
        mu[0] = 1
        mu[-1] = 0
    elif args.generator == random:
        # TODO delta_min not currently checked
        mu = (np.random.rand(n)).tolist()
        mu.sort(reverse=True)
        print(mu)
    else:
        raise ValueError("unspecified environment")
    return GaussianOneDEnvironment(mu, (args.min_variance, args.max_variance))


def build_nd_environment(args) -> Generator:
    if args.generator == random:
        return build_random_nd_environment(args)
    else:
        raise ValueError("invalid generator for n-d environment")


def main(args):
    if args.score_fun == l1:
        Image.score_function = Image.L1_score
    elif args.score_fun == l2:
        Image.score_function = Image.L2_score
    else:
        raise ValueError("unspecified score function")

    if args.min_variance is None:
        args.min_variance = args.max_variance

    classic_algorithms: List[Callable[[float], OneDAgent]] = [
        lambda delta: Jamieson2014(delta, args.lil_epsilon),
        lambda delta: EvenDar2002(delta),
    ]
    vector_algorithms: List[Callable[[float], Agent]] = [
        lambda delta: SED(SubExp_Bound(delta)),
        lambda delta: SED(SubExp_Bound(delta), None, False),
    ]

    agent_wrappers = []

    for algorithm in classic_algorithms:
        agent_wrappers.append(PairedAgentWrapper(algorithm(args.delta)))
    for algorithm in vector_algorithms:
        agent_wrappers.append(AgentWrapper(algorithm(args.delta)))
    if args.image_size == 1:
        for algorithm in classic_algorithms:
            agent_wrappers.append(MaxMinAgentWrapper(algorithm(args.delta / 2)))

    for round in range(args.rounds):
        print(round + 1)
        if args.image_size == 1:
            env = build_1d_environment(args)
        else:
            env = build_nd_environment(args)

        for agent in agent_wrappers:
            agent.run(env)

    with open(args.output, "w") as f:
        writer = csv.writer(f)
        header = [
            "Name",
            "Number of errors",
            "Average delta_min during errors",
            "Avg number of rounds",
            "Var number of rounds",
            "90% confidence for avg_rounds",
            "Avg number of pulls",
            "Var number of pulls",
            "90% confidence for avg_pulls",
        ]
        writer.writerow(header)
        for agent in agent_wrappers:
            for stats in agent.stats:
                row = [
                    agent.agent.name(),
                    len(stats.error_deltas),
                    np.mean(stats.error_deltas),
                    np.mean(stats.rounds_count),
                    np.var(stats.rounds_count),
                    get_confidence_range(np.var(stats.rounds_count), args.rounds, 0.9),
                    np.mean(stats.pulls_count),
                    np.var(stats.pulls_count),
                    get_confidence_range(np.var(stats.pulls_count), args.rounds, 0.9),
                ]
                writer.writerow(row)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--generator",
        choices=[equalIntervals, concentratedMid, concentratedMinMax, random],
        default=random,
    )
    arg_parser.add_argument("-n", "--number-of-arms", type=int, required=True)
    arg_parser.add_argument("-s", "--image-size", type=int, required=True)
    arg_parser.add_argument("--delta-min", type=float, default=0.1)
    arg_parser.add_argument("--score-fun", choices=[l1, l2], default=l2)
    arg_parser.add_argument("--min-variance", type=float, default=None)
    arg_parser.add_argument("--max-variance", type=float, default=1)
    arg_parser.add_argument("--delta", type=float, default=0.1)
    arg_parser.add_argument("--lil-epsilon", type=float, default=0.01)
    arg_parser.add_argument("--max-iterations", type=int, default=1e6)
    arg_parser.add_argument("--rounds", type=int, default=10)
    arg_parser.add_argument("--output", type=str, default="result.csv")
    args = arg_parser.parse_args()
    print(args)
    main(args)
