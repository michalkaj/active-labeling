from typing import Tuple

import torch
import torch.nn.functional as F


def BALD(logits: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Logits: [n, sample_size, classes]
    probs = F.softmax(logits, dim=-1)
    scores_N = _entropy(probs) - _conditional_entropy(probs)
    batch_size = min(batch_size, len(logits))
    return torch.topk(scores_N, batch_size)


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    mean_probs = probs.mean(dim=1)  # [N, C]
    nats = mean_probs * torch.log(mean_probs)
    nats[mean_probs == 0] = 0.
    return -nats.sum(dim=1)  # [N]


def _conditional_entropy(probs: torch.Tensor) -> torch.Tensor:
    _, K, _ = probs.shape
    nats = probs * torch.log(probs)
    nats[probs == 0] = 0.
    return -nats.sum(dim=(1, 2)) / K  # [N}
