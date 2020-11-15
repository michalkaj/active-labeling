import abc
from typing import Sequence

import numpy as np
import torch

from active_labeling.active_learning.dataset import ActiveDataset


class BaseQuery(abc.ABC):
    def __init__(self, shuffle_prob: float = 0.):
        self._shuffle_prob = shuffle_prob

    def __call__(self, dataset: ActiveDataset, acquisition_size: int) -> Sequence[int]:
        scores = torch.from_numpy(self._compute_scores(dataset))
        indices_to_shuffle = torch.nonzero(
            torch.rand(len(scores)) < self._shuffle_prob, as_tuple=False,
        )
        shuffle_mask = torch.randperm(len(indices_to_shuffle))
        scores[indices_to_shuffle] = scores[indices_to_shuffle[shuffle_mask]]
        return torch.topk(scores, k=acquisition_size).indices


    @abc.abstractmethod
    def _compute_scores(self, dataset: ActiveDataset) -> np.ndarray:
        pass
