import numpy as np

from active_labeling.active_learning.dataset import ActiveDataset
from active_labeling.active_learning.sampling.acquisition.base import BaseQuery


class RandomQuery(BaseQuery):
    def _compute_scores(self, dataset: ActiveDataset) -> np.ndarray:
        return np.random.rand(len(dataset))
