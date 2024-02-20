from typing import Optional, Tuple

import numpy as np


def random_image(shape: Tuple[int, int]) -> np.ndarray:
    return np.random.uniform(0, 1, shape)


class Vector:
    L1_score = lambda x: np.sum(np.abs(x))
    L2_score = lambda x: np.linalg.norm(x)
    score_function = L2_score

    def __init__(self, data: np.ndarray):
        self.data = data

    def assert_compatible(self, other: "Vector"):
        if self.data.shape != other.data.shape:
            raise ValueError(
                "non matching Vector shapes: ",
                self.data.shape,
                " and ",
                other.data.shape,
            )

    def diff(self, other: "Vector") -> float:
        self.assert_compatible(other)
        return Vector.score_function(np.reshape(self.data - other.data, -1))


class VectorSamples:
    def __init__(self, first_sample: Vector, keep_samples: bool = False):
        self.n_samples = 1
        self.avg = first_sample
        self.keep_samples = keep_samples
        if self.keep_samples:
            self.samples = [first_sample]

    def add_vector(self, vector: Vector):
        self.avg.assert_compatible(vector)
        self.avg.data = (self.avg.data * self.n_samples + vector.data) / (self.n_samples + 1)
        self.n_samples += 1
        if self.keep_samples:
            self.samples.append(vector)


class Image(Vector):
    def __init__(self, shape: Tuple[int, int], pixels: Optional[np.ndarray] = None):
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("invalid shape ", shape)
        if pixels is None:
            pixels = random_image(shape)
        if shape != pixels.shape:
            raise ValueError("shape is ", shape, ", image shape is ", pixels.shape)
        self.data = pixels

    def serialize(self) -> Vector:
        return Vector(np.reshape(self.data, -1))
