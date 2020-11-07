from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from active_labeling.active_learning.sampling.base import BaseSampler
from active_labeling.active_learning.training.dataset import ActiveDataset


class RandomSampler(BaseSampler):
    def sample(self,
               active_dataset: ActiveDataset,
               batch_size: int) -> Sequence[Path]:
        pool = active_dataset.not_labeled_pool
        indices = np.arange(len(pool))
        np.random.shuffle(indices)
        indices = indices[:batch_size]
        return [pool[i] for i in indices]
