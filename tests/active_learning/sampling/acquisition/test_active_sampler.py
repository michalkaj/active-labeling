import unittest
from unittest.mock import Mock, MagicMock, patch

from active_labeling.active_learning.sampling import sampler as sampler_module
from active_labeling.active_learning.sampling.sampler import Sampler


class TestActiveSampler(unittest.TestCase):
    def test_sample(self):
        indices = Mock()
        query = Mock(return_value=indices)
        sampler = Sampler(query)
        active_dataset = MagicMock()
        reduced_dataset = MagicMock()
        batch_size = 10

        with patch.object(sampler_module._Reducer, '__enter__', return_value=reduced_dataset):
            sampler.sample(active_dataset, batch_size)

        query.assert_called_with(reduced_dataset.evaluate(), batch_size)
        reduced_dataset.get_examples.assert_called_with(indices)
