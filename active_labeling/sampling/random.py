from typing import Optional, Sequence, Any

import numpy as np

from active_labeling.loading.sample import Sample
from active_labeling.sampling.base import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self._seed = seed

    def sample(self, items: Sequence[Sample], sample_size: int) -> Sequence[Sample]:
        if self._seed:
            np.random.seed(self._seed)

        indices = np.arange(len(items))
        np.random.shuffle(indices)
        return [items[idx] for idx in indices[:sample_size]]
