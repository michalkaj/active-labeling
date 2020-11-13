from typing import Tuple

import torch
import torch.nn.functional as F


def BALD(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=-1)  # [n, sample_size, classes]
    expected_entropy = -_x_log_x(probs).sum(dim=-1).mean(dim=1)  # [n]
    expected_prob = probs.mean(dim=1)  # [n, classes]
    expected_entropy_prob = -_x_log_x(expected_prob).sum(dim=1)  # [n]

    bald_scores = expected_entropy_prob - expected_entropy
    return bald_scores


def _x_log_x(x: torch.Tensor):
    result = torch.zeros_like(x)
    return result.where(x == 0, x * torch.log(x))
