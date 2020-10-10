from typing import Optional, Iterable

import numpy as np

from active_labeling.active_learning.sampling import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(self, seed: Optional[int] = None):
        self._seed = seed

    def sample(self, data: np.ndarray, sample_size: int) -> Iterable[int]:
        if self._seed:
            np.random.seed(self._seed)

        indices = np.arange(len(data))
        return np.random.choice(indices, size=sample_size, replace=False)
