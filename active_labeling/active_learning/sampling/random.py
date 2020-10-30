from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from active_labeling.active_learning.training import ActiveDataset
from active_labeling.active_learning.sampling.base import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(self, seed: Optional[int] = None):
        self._seed = seed

    def sample(self,
               active_dataset: ActiveDataset,
               batch_size: int) -> Sequence[Path]:
        if self._seed:
            np.random.seed(self._seed)
        pool = active_dataset.not_labeled_pool
        indices = np.arange(len(pool))
        np.random.shuffle(indices)
        return [pool[i] for i in indices[:batch_size]]
