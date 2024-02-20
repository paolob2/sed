import abc
from typing import List, Optional, Tuple

from lib.environment.environment import Environment
from lib.environment.oned_environment import OneDEnvironment


class Agent(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def best_arm(self, environment: Environment) -> Tuple[Optional[Tuple[int, int]], List[List[int]]]:
        """
        Returns:
            (int, int): indices of the best pair or None if the agent failed
            list[list[int]]: list of arms pulled during each epoch
        """
        pass

    @abc.abstractmethod
    def name(self) -> str:
        """
        Returns:
            str: name of this agent
        """
        pass


class OneDAgent(Agent):
    def __init__(self, confidence: float):
        self.confidence = confidence

    @abc.abstractmethod
    def best_arm(self, environment: OneDEnvironment) -> Tuple[Optional[int], List[List[int]]]:
        """
        Returns:
            (int): index of the best arm or None if the agent failed
            list[list[int]]: list of arms pulled during each round
        """
        pass
