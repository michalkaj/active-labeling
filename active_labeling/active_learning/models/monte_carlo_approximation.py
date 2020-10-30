import torch
from copy import deepcopy
from torch import nn
from typing import Sequence


class MonteCarloWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self._wrapped = model
        self._dropouts = list(self._get_dropouts(self._wrapped))
        self._wrapped_initial_state_dict = deepcopy(model.state_dict())

    def _get_dropouts(self, module) -> Sequence[nn.Module]:
        for layer in module.children():
            if isinstance(layer, (nn.Dropout2d, nn.Dropout)):
                yield layer
            else:
                yield from self._get_dropouts(layer)

    def reset_weights(self):
        self._wrapped.load_state_dict(deepcopy(self._wrapped_initial_state_dict))

    def forward(self, *args, **kwargs) -> Sequence[torch.Tensor]:
        # Make sure that all dropouts are enabled
        for dropout in self._dropouts:
            dropout.train()

        sample_size = kwargs.pop('sample_size', None)
        if sample_size is None:
            return self._wrapped(*args, **kwargs)

        return [self._wrapped(*args, **kwargs) for _ in range(sample_size)]