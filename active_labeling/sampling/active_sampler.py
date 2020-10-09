from typing import Iterable

import numpy as np
from modAL import ActiveLearner

from active_labeling.sampling.base import BaseSampler
import math


class ActiveSampler(BaseSampler):
    def __init__(self, learner: ActiveLearner):
        self._learner = learner

    def sample(self, data: np.ndarray, batch_size: int, pool_size: float = 1.)\
            -> Iterable[int]:
        pooled_indices = self._reduce_dataset_size(data, pool_size)
        indices, _ = self._learner.query(data[pooled_indices], n_instances=batch_size)
        return pooled_indices[indices]

    def _reduce_dataset_size(self, data: np.ndarray, pool_size: float) -> np.ndarray:
        indices = np.arange(len(data))

        if math.isclose(pool_size, 1., abs_tol=0.01):
            return indices

        pool_size = int(len(data) * pool_size)
        return np.random.choice(indices, pool_size, replace=False)
