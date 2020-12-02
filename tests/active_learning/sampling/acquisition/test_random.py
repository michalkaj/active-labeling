import unittest
from unittest.mock import MagicMock, patch

from active_labeling.active_learning.sampling.acquisition.random import RandomQuery
from active_labeling.active_learning.sampling.acquisition.random import np


class TestRandom(unittest.TestCase):
    def test_sample(self):
        sampler = RandomQuery()
        dataset = MagicMock()

        with patch.object(np.random, 'rand') as rand:
            sampler._compute_scores(dataset)

        rand.assert_called_with(len(dataset))


