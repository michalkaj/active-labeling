import unittest

import torch

from active_labeling.active_learning.sampling.acquisition.queries import BALD


class TestBald(unittest.TestCase):
    def test_uncertain(self):
        length = 10
        uncertain_logits = torch.rand(length, 20, 10)
        certain_logits = torch.randn(length, 20, 10) * 0.01
        certain_logits[0, :, 0] = 1
        certain_logits[1, :, 1] = 1
        certain_logits[2, :, 2] = 1
        all_logits = torch.cat((uncertain_logits, certain_logits))

        scores = BALD(all_logits)

        self.assertGreater(scores[:length].mean(), scores[length:].mean())
