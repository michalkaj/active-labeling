import math
from pathlib import Path
from typing import Iterable, Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from active_labeling.active_learning.learners.training.dataset import ActiveDataset, Reducer
from active_labeling.active_learning.sampling.acquisition.bald import BALD
from active_labeling.active_learning.sampling.base import BaseSampler
from active_labeling.config import ActiveLearningConfig


class ActiveSampler(BaseSampler):
    def __init__(self,
                 model: nn.Module,
                 acquisition_func: Callable,
                 config: ActiveLearningConfig):
        self._model = model
        self._acquisition_func = acquisition_func
        self._config = config

    def sample(self,
               active_dataset: ActiveDataset,
               batch_size: int) -> Sequence[Path]:

        with Reducer(active_dataset, self._config.pool_size) as (reduced_dataset, indices_subset):
            logits = self._compute_logits(reduced_dataset)
        _, indices = self._acquisition_func(logits, batch_size)
        indices = indices_subset[indices]
        return [active_dataset.not_labeled_pool[i] for i in indices]

    def _compute_logits(self, dataset: ActiveDataset) -> torch.Tensor:
        dataset = dataset.evaluate()
        self._model.to(self._config.device)
        self._model.eval()

        logits = torch.empty(
            len(dataset),
            self._config.bayesian_sample_size,
            len(self._config.labels),
            dtype=torch.float32
        )


        start = 0
        dataloader = DataLoader(
            dataset,
            **self._config.dataloader_kwargs
        )

        for batch in dataloader:
            images = batch['image'].to(self._config.device)
            end = min(start + len(images), len(dataset))
            with torch.no_grad():
                posterior_sample = self._model(images, sample_size=self._config.bayesian_sample_size)
            logits[start: end] = torch.stack(posterior_sample, dim=1).detach().cpu()

            start = end

        return logits

    def _reduce_dataset_size(self, data: np.ndarray, pool_size: float) -> np.ndarray:
        indices = np.arange(len(data))

        if math.isclose(pool_size, 1., abs_tol=0.01):
            return indices

        pool_size = int(len(data) * pool_size)
        return np.random.choice(indices, pool_size, replace=False)
