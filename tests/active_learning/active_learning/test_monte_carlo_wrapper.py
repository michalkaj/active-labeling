import unittest
from unittest.mock import Mock, patch

import torch

import active_labeling.active_learning.models.monte_carlo_wrapper as mcw
from active_labeling.active_learning.models.monte_carlo_wrapper import MonteCarloWrapper


class TestModelWrapper(unittest.TestCase):
    def test_forward(self):
        model_output = Mock(spec=torch.Tensor)
        model = Mock(return_value=model_output)
        sample_size = 10
        dropouts = [Mock()]
        with patch.object(MonteCarloWrapper, '_get_dropouts', return_values=dropouts), \
                patch.object(mcw.torch, 'stack') as stack:
            model_wrapper = MonteCarloWrapper(model, sample_size)

            model_wrapper.forward(Mock())

            stack.assert_called_with([model_output] * sample_size, dim=1)
