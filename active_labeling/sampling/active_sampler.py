from typing import Iterable

import numpy as np
from modAL import ActiveLearner

from active_labeling.sampling.base import BaseSampler


class ActiveSampler(BaseSampler):
    def __init__(self, learner: ActiveLearner):
        self._learner = learner

    def sample(self, data: np.ndarray, sample_size: int, pool_size: float = 1.)\
            -> Iterable[int]:
        pooled_data = self._reduce_dataset_size(data, pool_size)
        indices, _ = self._learner.query(pooled_data, n_instances=sample_size)
        return indices

    def _reduce_dataset_size(self, data: np.ndarray, pool_size: float) -> np.ndarray:
        if pool_size == 1.:
            return data

        pool_size = np.floor(len(data) * pool_size)
        indices = np.arange(len(data))
        pool_indices = np.random.choice(indices, pool_size, replace=False)
        return data[pool_indices]
