from typing import Iterable

import numpy as np
from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

from active_labeling.sampling.base import BaseSampler

_DEFAULT_ESTIMATOR = RandomForestClassifier()

class ActiveSampler(BaseSampler):
    def __init__(self, data: np.ndarray, estimator: BaseEstimator = _DEFAULT_ESTIMATOR):
        super().__init__(data)
        self._learner = ActiveLearner(
            estimator=estimator,
            query_strategy=uncertainty_sampling
        )

    def query(self, sample_size: int) -> Iterable[int]:
        indices, _ = self._learner.query(self._data, n_instances=sample_size)
        return indices
