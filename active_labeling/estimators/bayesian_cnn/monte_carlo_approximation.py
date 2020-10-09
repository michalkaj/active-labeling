from typing import Sequence

import torch
from torch import nn


def monte_carlo_posterior_approximator(wrapped_model_class):
    class MonteCarloWrapper:
        def __init__(self, *args, **kwargs):
            self._wrapped = wrapped_model_class(*args, **kwargs)
            self._dropouts = list(self._get_dropouts(self._wrapped))

        def _get_dropouts(self, module) -> Sequence[nn.Module]:
            for layer in module.children():
                if isinstance(layer, (nn.Dropout2d, nn.Dropout)):
                    yield layer
                else:
                    yield from self._get_dropouts(layer)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def __call__(self, *args, **kwargs) -> Sequence[torch.Tensor]:
            sample_size = kwargs.pop('sample_size')

            if sample_size is None:
                raise ValueError("sample_size has to be set for a MC approximation")

            # Make sure that all dropouts are enabled
            for dropout in self._dropouts:
                dropout.train()

            return [self._wrapped(*args, **kwargs) for _ in range(sample_size)]

    return MonteCarloWrapper
