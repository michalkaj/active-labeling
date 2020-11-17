from pathlib import Path
from typing import Sequence

import numpy as np

from active_labeling.active_learning.dataset import ActiveDataset, divide_pool
from active_labeling.active_learning.sampling.acquisition.base import BaseQuery


class Sampler:
    def __init__(self, query: BaseQuery, pool_size_reduction: float = 1.):
        self._query = query
        self._pool_size_reduction = pool_size_reduction

    def sample(self,
               active_dataset: ActiveDataset,
               sample_size: int) -> Sequence[Path]:
        with _Reducer(active_dataset, self._pool_size_reduction) as reduced_dataset:
            indices = self._query(reduced_dataset.evaluate(), sample_size)
            return reduced_dataset.get_examples(indices)


class _Reducer:
    def __init__(self, dataset: ActiveDataset, dataset_frac: float = 1.):
        self._dataset = dataset
        self._dataset_frac = dataset_frac
        self.__container = None

    def __enter__(self):
        length = int(len(self._dataset._not_labeled_pool) * self._dataset_frac)
        reduced_paths = np.random.choice(self._dataset._not_labeled_pool, size=length, replace=False)
        self._dataset._not_labeled_pool, self.__container = divide_pool(
            self._dataset._not_labeled_pool, set(reduced_paths))
        return self._dataset

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dataset._not_labeled_pool = self._dataset._not_labeled_pool + self.__container
        self.__container = None
