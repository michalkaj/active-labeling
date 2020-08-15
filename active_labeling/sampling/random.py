from typing import Optional, Iterable

import numpy as np

from active_labeling.loading.sample import Sample
from active_labeling.sampling.base import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(self, data: np.ndarray, seed: Optional[int] = None):
        super().__init__(data)
        self._seed = seed

    def sample(self, sample_size: int) -> Iterable[Sample]:
        if self._seed:
            np.random.seed(self._seed)

        indices = np.arange(len(self._data))
        return np.random.choice(indices, size=sample_size, replace=False)
