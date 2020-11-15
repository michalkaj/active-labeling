from copy import deepcopy
from typing import Sequence

import torch
from torch import nn

BAYESIAN_SAMPLE_DIM = 1


class ActiveWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self._wrapped = model
        self._wrapped_initial_state_dict = deepcopy(model.state_dict())

    def reset_weights(self, only_fc=False):
        if only_fc:
            def reset(layer):
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
            self._wrapped.apply(reset)
        else:
            self._wrapped.load_state_dict(deepcopy(self._wrapped_initial_state_dict))


class MonteCarloWrapper(ActiveWrapper):
    def __init__(self, model: nn.Module, sample_size: int):
        super().__init__(model)
        self._dropouts = list(self._get_dropouts(self._wrapped))
        self.sample_size = sample_size

    def _get_dropouts(self, module) -> Sequence[nn.Module]:
        for layer in module.children():
            if isinstance(layer, (nn.Dropout2d, nn.Dropout)):
                yield layer
            else:
                yield from self._get_dropouts(layer)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        sample_size = kwargs.pop('sample_size', self.sample_size)
        deterministic = kwargs.pop('deterministic', False)

        if deterministic:
            return self._wrapped(*args, **kwargs)

        # Make sure that all dropouts are enabled
        for dropout in self._dropouts:
            dropout.train()

        return torch.stack(
            [self._wrapped(*args, **kwargs) for _ in range(sample_size)], dim=BAYESIAN_SAMPLE_DIM)
