from typing import Iterable

import numpy as np
from modAL import ActiveLearner

from active_labeling.sampling.base import BaseSampler


class ActiveSampler(BaseSampler):
    def __init__(self, data: np.ndarray, learner: ActiveLearner):
        super().__init__(data)
        self._learner = learner

    def sample(self, sample_size: int) -> Iterable[int]:
        indices, _ = self._learner.query(self._data, n_instances=sample_size)
        return indices
