from pathlib import Path
from typing import Sequence

from active_labeling.active_learning.dataset import ActiveDataset
from active_labeling.active_learning.sampling.acquisition.base import BaseQuery
from active_labeling.active_learning.sampling.base import BaseSampler


class ActiveSampler(BaseSampler):
    def __init__(self, query: BaseQuery, pool_size_reduction: float = 0.2):
        self._query = query
        self._pool_size_reduction = pool_size_reduction

    def sample(self,
               active_dataset: ActiveDataset,
               sample_size: int) -> Sequence[Path]:
        # with Reducer(active_dataset, self._pool_size_reduction) as reduced_dataset:
        indices = self._query(active_dataset.evaluate(), sample_size)
        return active_dataset.get_examples(indices)
