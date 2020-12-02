import unittest
from unittest.mock import Mock

import numpy as np
import torch

from active_labeling.active_learning.sampling.acquisition.active import BALDQuery, UncertaintyQuery


class TestBALDQuery(unittest.TestCase):
    def test_compute_scores(self):
        length = 10
        uncertain_logits = torch.rand(length, 20, 10)
        certain_logits = torch.randn(length, 20, 10) * 0.01
        certain_logits[0, :, 0] = 1
        certain_logits[1, :, 1] = 1
        certain_logits[2, :, 2] = 1
        all_logits = torch.cat((uncertain_logits, certain_logits))

        scores = BALDQuery(Mock())._compute_scores_from_logits(all_logits.numpy())

        self.assertGreater(scores[:length].mean(), scores[length:].mean())


class TestUncertaintyQuery(unittest.TestCase):
    def test_compute_scores(self):
        logits = np.array([
            [0, 1, 4, 1, 0],
            [0, 0.3, 0.5, 0.4, 0],
            [0, 1, 1, 1, 0],
            [0, 3, 2, 1, 0],
        ], dtype=np.float)

        scores = UncertaintyQuery(Mock())._compute_scores_from_logits(logits)

        self.assertSequenceEqual(
            [0, 3, 2, 1],
            list(np.argsort(scores)),
        )
