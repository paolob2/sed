import numpy as np
from typing import List, Tuple

from lib.environment.environment import Environment
from lib.vector import Image, Vector


class Generator(Environment):
    """
    Generates Vector samples from a normal distribution
    """

    def __init__(self, means: List[Vector], variance_range: Tuple[float, float]):
        super().__init__(len(means), means[0].data.size, np.sqrt(variance_range[1]))
        for image in means:
            image.assert_compatible(means[0])
        self.means = means
        self.variances = [
            np.random.uniform(variance_range[0], variance_range[1], self.means[i].data.shape) for i in range(self.n)
        ]

    def pull(self, i: int) -> Vector:
        # use a normal rather than a truncated normal as it is up to 100x faster
        return Vector(np.random.normal(self.means[i].data, np.sqrt(self.variances[i]), self.means[i].data.shape))


class ImageGenerator(Generator):
    def __init__(self, means: List[Image], variance_range: Tuple[float, float]):
        super().__init__(means, variance_range)

    def pull_image(self, i: int) -> Image:
        # use a normal rather than a truncated normal as it is up to 100x faster
        return Image(
            self.means[i].data.shape,
            np.random.normal(self.means[i].data, np.sqrt(self.variances[i]), self.means[i].data.shape),
        )
