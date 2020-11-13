from pathlib import Path
from typing import Callable, Sequence, Optional, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from active_labeling.active_learning.sampling.base import BaseSampler
from active_labeling.active_learning.training.dataset import ActiveDataset, Reducer


class ActiveSampler(BaseSampler):
    def __init__(self,
                 model: nn.Module,
                 acquisition_func: Callable,
                 bayesian_sample_size: int,
                 pool_size_reduction: float = 0.2,
                 device: torch.device = torch.device('cpu'),
                 dataloader_kwargs: Optional[Dict[str, Any]] = None,
                 num_classes: int = 10,
                 shuffle_prob: float = 0.1,
                 ):
        self._model = model
        self._acquisition_func = acquisition_func
        self._bayesian_sample_size = bayesian_sample_size
        self._pool_size_reduction = pool_size_reduction
        self._device = device
        self._dataloader_kwargs = dataloader_kwargs or {}
        self._num_classes = num_classes
        self._shuffle_prob = shuffle_prob

    def sample(self,
               active_dataset: ActiveDataset,
               sample_size: int) -> Sequence[Path]:
        with Reducer(active_dataset, self._pool_size_reduction) as reduced_dataset:
            logits = self._compute_logits(reduced_dataset)
            scores = self._acquisition_func(logits)
            indices = self._get_indices_to_query(scores, sample_size)
            paths_to_query = [reduced_dataset.not_labeled_pool[i] for i in indices]
        return paths_to_query

    def _compute_logits(self, dataset: ActiveDataset) -> torch.Tensor:
        dataset = dataset.evaluate()
        self._model.to(self._device)
        self._model.eval()

        logits = torch.empty(
            len(dataset),
            self._bayesian_sample_size,
            self._num_classes,
            dtype=torch.float32
        )

        start = 0
        dataloader = DataLoader(
            dataset,
            **self._dataloader_kwargs
        )

        for batch in tqdm(dataloader):
            images = batch['image'].to(self._device)
            end = min(start + len(images), len(dataset))
            with torch.no_grad():
                posterior_sample = self._model(images)
            logits[start: end] = posterior_sample.detach().cpu()

            start = end

        return logits

    def _get_indices_to_query(self, scores, acquisition_batch_size):
        shuffle_indices = torch.nonzero(
            torch.rand(len(scores)) < self._shuffle_prob,
            as_tuple=False
        )
        scores[shuffle_indices] = scores[np.random.permutation(shuffle_indices)]
        _, indices = torch.topk(scores, k=acquisition_batch_size)
        return indices
