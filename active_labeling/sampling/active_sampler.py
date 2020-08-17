from typing import Iterable

import numpy as np
from modAL import ActiveLearner

from active_labeling.sampling.base import BaseSampler


class ActiveSampler(BaseSampler):
    def __init__(self, learner: ActiveLearner):
        self._learner = learner

    def sample(self, data: np.ndarray, sample_size: int) -> Iterable[int]:
        indices, _ = self._learner.query(data, n_instances=sample_size)
        return indices
