import unittest
from unittest.mock import Mock, MagicMock

from active_labeling.active_learning.sampling.random import RandomSampler


class TestRandom(unittest.TestCase):
    def test_sample(self):
        sampler = RandomSampler()
        pool = MagicMock()
        pool.__len__.return_value = 10
        dataset = MagicMock(not_labeled_pool=pool)
        sample_size = 10
        samples = sampler.sample(dataset, sample_size)

        self.assertEqual(sample_size, len(samples))
