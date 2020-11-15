import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from active_labeling.active_learning.sampling.acquisition.base import ActiveQuery


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

