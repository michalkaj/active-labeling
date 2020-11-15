import unittest
from unittest.mock import Mock, MagicMock, patch

import torch

from active_labeling.active_learning.sampling import sampler
from active_labeling.active_learning.sampling.sampler import Sampler


class TestActiveSampler(unittest.TestCase):
    def test_sample(self):
        model = Mock()
        acqusition_return_value = MagicMock(spec=torch.Tensor)
        acquisition_func = MagicMock(return_value=acqusition_return_value)
        bayesian_sample_size = 32
        sampler = Sampler()
        active_dataset = MagicMock()
        reduced_dataset = MagicMock()

        batch_size = 10
        indices = [0, 5, 9]

        with patch.object(torch, 'topk', return_value=(Mock(), indices)) as topk, \
                patch.object(sampler.Reducer, '__enter__', return_value=reduced_dataset):
            samples = sampler.sample(active_dataset, batch_size)

            topk.assert_called_with(acqusition_return_value, k=batch_size)
        self.assertSequenceEqual(
            [reduced_dataset._not_labeled_pool.__getitem__() for _ in indices], samples)
