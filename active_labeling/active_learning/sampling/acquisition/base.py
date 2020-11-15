import abc
from typing import Sequence, Callable

import numpy as np
import torch
from torch import Tensor

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


class ActiveQuery(BaseQuery):
    def __init__(self,
                 predict_func: Callable[[ActiveDataset], Tensor],
                 shuffle_prob: float=0.):
        super().__init__(shuffle_prob)
        self._predict_func = predict_func

    def _compute_scores(self, dataset: ActiveDataset) -> np.ndarray:
        logits = self._predict_func(dataset).numpy()
        return self._compute_scores_from_logits(logits)

    @abc.abstractmethod
    def _compute_scores_from_logits(self, logits: np.ndarray) -> np.ndarray:
        pass
