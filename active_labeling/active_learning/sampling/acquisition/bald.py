from typing import Tuple

import torch
import torch.nn.functional as F


def BALD(logits: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ I(y;ω | x, Dtrain) = H(y | x, Dtrain) − Ep(ω | Dtrain)[H(y | x, ω, Dtrain)]
        1. The first term looks for images that have high entropy in the average output
        2. The second term penalizes images where many of the sampled models are not confident about.
         This keeps only images where the models disagree on.
    """
    probs = F.softmax(logits, dim=-1)  # [n, sample_size, classes]
    expected_entropy = -_x_log_x(probs).sum(dim=-1).mean(dim=1)  # [n]
    expected_prob = probs.mean(dim=1)  # [n, classes]
    expected_entropy_prob = -_x_log_x(expected_prob).sum(dim=1)  # [n]

    bald_scores = expected_entropy_prob - expected_entropy
    _, indices = torch.topk(bald_scores, batch_size)
    return indices

def BALD2(predictions, batch_size):
    predictions = predictions.permute((0, 2, 1))
    import numpy as np
    from scipy.special import xlogy
    predictions = F.softmax(predictions, 1).numpy()

    expected_entropy = - np.mean(np.sum(xlogy(predictions, predictions), axis=1),
                                 axis=-1)  # [batch size, ...]
    expected_p = np.mean(predictions, axis=-1)  # [batch_size, n_classes, ...]
    entropy_expected_p = - np.sum(xlogy(expected_p, expected_p),
                                  axis=1)  # [batch size, ...]

    bald_acq = entropy_expected_p - expected_entropy
    return bald_acq

def _x_log_x(x: torch.Tensor):
    result = torch.zeros_like(x)
    return result.where(x == 0, x * torch.log(x))

# def BALD(logits: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
#     # Logits: [n, sample_size, classes]
#     probs = F.softmax(logits, dim=-1)
#     scores_N = _entropy(probs) - _conditional_entropy(probs)
#     batch_size = min(batch_size, len(logits))
#     return torch.topk(scores_N, batch_size)
#
#
# def _entropy(probs: torch.Tensor) -> torch.Tensor:
#     mean_probs = probs.mean(dim=1)  # [N, C]
#     nats = mean_probs * torch.log(mean_probs)
#     nats[mean_probs == 0] = 0.
#     return -nats.sum(dim=1)  # [N]
#
#
# def _conditional_entropy(probs: torch.Tensor) -> torch.Tensor:
#     _, K, _ = probs.shape
#     nats = probs * torch.log(probs)
#     nats[probs == 0] = 0.
#     return -nats.sum(dim=(1, 2)) / K  # [N}
