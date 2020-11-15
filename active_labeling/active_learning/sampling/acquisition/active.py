import abc
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from active_labeling.active_learning.dataset import ActiveDataset
from active_labeling.active_learning.sampling.acquisition.base import BaseQuery


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


class BALDQuery(ActiveQuery):
    def _compute_scores_from_logits(self, logits: np.ndarray) -> np.ndarray:
        logits = torch.from_numpy(logits)
        probs = F.softmax(logits, dim=-1)  # [n, sample_size, classes]
        expected_entropy = -_x_log_x(probs).sum(dim=-1).mean(dim=1)  # [n]
        expected_prob = probs.mean(dim=1)  # [n, classes]
        expected_entropy_prob = -_x_log_x(expected_prob).sum(dim=1)  # [n]
        return (expected_entropy_prob - expected_entropy).numpy()


def _x_log_x(x: torch.Tensor):
    result = torch.zeros_like(x)
    return result.where(x == 0, x * torch.log(x))


class UncertaintyQuery(ActiveQuery):
    def _compute_scores_from_logits(self, logits: np.ndarray) -> np.ndarray:
        logits = torch.from_numpy(logits)
        probs = F.softmax(logits, dim=-1).numpy()
        return -np.amax(probs, axis=-1)
