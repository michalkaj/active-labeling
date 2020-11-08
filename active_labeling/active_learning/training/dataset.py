from itertools import compress
from pathlib import Path
from typing import Sequence, Dict, Optional, Callable, Union

import PIL
import numpy as np
import torch
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def _divide_pool(pool: Sequence, mask: Sequence[bool]):
    labeled_pool = list(compress(pool, mask))
    not_labeled_pool = list(compress(pool, np.logical_not(mask)))

    assert len(labeled_pool) + len(not_labeled_pool) == len(pool)

    return labeled_pool, not_labeled_pool


class ActiveDataset(Dataset):
    def __init__(self,
                 pool: Sequence[Path],
                 labels: Dict[Path, int],
                 train: bool = True,
                 transform: Optional[Callable[[Image], Tensor]] = None,
                 target_transform: Optional[Callable[[str], int]] = None):
        self.labels = {path: label for path, label in labels.items()}

        labeled_mask = [path in self.labels for path in pool]
        self._labeled_pool, self.not_labeled_pool = _divide_pool(pool, labeled_mask)
        self._train = train
        self._transform = transform if transform else ToTensor()
        self._target_transform = target_transform or (lambda x: x)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, Optional[int]]]:
        path = self._get_example(index)
        image = PIL.Image.open(path)
        image = self._transform(image)

        return {'image': image, 'label': self._get_label(path)}

    def _get_example(self, index: int) -> Path:
        pool = self._labeled_pool if self._train else self.not_labeled_pool
        return pool[index]

    def _get_label(self, path):
        if not self._train:
            return -1
        label = self.labels[path]
        return self._target_transform(label)

    def __len__(self):
        return len(self._labeled_pool if self._train else self.not_labeled_pool)

    def add_labels(self, labels: Dict[Path, str]) -> None:
        self._validate(labels)

        labeled_mask = [path in labels for path in self.not_labeled_pool]
        labeled, not_labeled = _divide_pool(self.not_labeled_pool, labeled_mask)

        self._labeled_pool.extend(labeled)
        self._not_labeled_pool = not_labeled

        self.labels.update(labels)

    def _validate(self, labels: Dict[Path, str]):
        intersecting_keys = set(labels.keys()).intersection(set(self.labels.keys()))
        if intersecting_keys:
            raise ValueError(f"Some of the samples are already labeled: {intersecting_keys}")

    def train(self) -> 'ActiveDataset':
        self._train = True
        return self

    def evaluate(self) -> 'ActiveDataset':
        self._train = False
        return self


class Reducer:
    def __init__(self, dataset: ActiveDataset, dataset_frac: float = 1.):
        self._dataset = dataset
        self._dataset_frac = dataset_frac
        self.__container = None

    def __enter__(self):
        length = len(self._dataset.not_labeled_pool)
        indices = np.random.permutation(length)
        subset_indices = indices[:int(self._dataset_frac * length)]
        mask = np.zeros(length, dtype=np.bool)
        mask[subset_indices] = 1
        self._dataset.not_labeled_pool, self.__container = _divide_pool(
            self._dataset.not_labeled_pool, mask)
        return self._dataset, subset_indices

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dataset.not_labeled_pool = self._dataset.not_labeled_pool + self.__container
        self.__container = None
