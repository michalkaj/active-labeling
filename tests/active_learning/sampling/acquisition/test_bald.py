import unittest

import torch

from active_labeling.active_learning.sampling.acquisition.bald import BALD


class TestBald(unittest.TestCase):
    def test_uncertain(self):
        uncertain_logits = torch.rand(10, 20, 10)
        certain_logits = torch.randn(10, 20, 10) * 0.01
        certain_logits[0, :, 0] = 1
        certain_logits[1, :, 1] = 1
        certain_logits[2, :, 2] = 1
        all_logits = torch.cat((uncertain_logits, certain_logits))

        indices = BALD(all_logits, 10)

        indices_set = {i.item() for i in indices}
        self.assertSetEqual(set(range(10)), indices_set)
