import math
from typing import Iterable, Callable

import numpy as np
import torch
from toma import toma
from torch import nn
from tqdm.auto import tqdm

from active_labeling.active_learning.sampling.acquisition.bald import BALD
from active_labeling.active_learning.sampling.base import BaseSampler

_INITIAL_STEP = 512


class ActiveSampler(BaseSampler):
    def __init__(self,
                 model: nn.Module,
                 acquisition_func: Callable,
                 num_classes: int,
                 bayesian_sample_size: int = 20):
        self._model = model
        self._acquisition_func = acquisition_func
        self._bayesian_sample_size = bayesian_sample_size
        self._num_classes = num_classes

    def sample(self,
               data: np.ndarray,
               batch_size: int,
               pool_size: float = 0.05) -> Iterable[int]:
        reduced = self._reduce_dataset_size(data, pool_size)
        reduced_data = data[reduced]
        logits = self._compute_logits(reduced_data)
        _, indices = BALD(logits, batch_size)
        return reduced[indices]

    def _compute_logits(self, data: np.ndarray) -> torch.Tensor:
        data_tensor = torch.from_numpy(data)
        logits = torch.empty(
            len(data_tensor),
            self._bayesian_sample_size,
            self._num_classes,
            dtype=torch.float32
        )

        progress_bar = tqdm(
            initial=0,
            total=len(data),
            desc="Computing logits"
        )

        @toma.execute.chunked(data_tensor, initial_step=_INITIAL_STEP)
        def compute(batch: torch.Tensor, start: int, end: int):
            posterior_sample = self._model(batch, sample_size=self._bayesian_sample_size)
            logits[start: end] = torch.stack(posterior_sample, dim=1)
            progress_bar.update(end)

        return logits

    def _reduce_dataset_size(self, data: np.ndarray, pool_size: float) -> np.ndarray:
        indices = np.arange(len(data))

        if math.isclose(pool_size, 1., abs_tol=0.01):
            return indices

        pool_size = int(len(data) * pool_size)
        return np.random.choice(indices, pool_size, replace=False)
