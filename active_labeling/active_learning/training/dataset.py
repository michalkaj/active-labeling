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


def _divide_pool(pool: Sequence, keys):
    first_part, second_part = [], []
    for item in pool:
        if item in keys:
            first_part.append(item)
        else:
            second_part.append(item)
    return first_part, second_part


class ActiveDataset(Dataset):
    def __init__(self,
                 pool: Sequence[Path],
                 labels: Dict[Path, int],
                 train: bool = True,
                 transform: Optional[Callable[[Image], Tensor]] = None,
                 target_transform: Optional[Callable[[str], int]] = None):
        self.labels = {path: label for path, label in labels.items()}

        self._labeled_pool, self.not_labeled_pool = _divide_pool(pool, self.labels)
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

        labeled, not_labeled = _divide_pool(self.not_labeled_pool, labels)

        self._labeled_pool.extend(labeled)
        self.not_labeled_pool = not_labeled

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
        length = int(len(self._dataset.not_labeled_pool) * self._dataset_frac)
        reduced_paths = np.random.choice(self._dataset.not_labeled_pool, size=length, replace=False)
        self._dataset.not_labeled_pool, self.__container = _divide_pool(
            self._dataset.not_labeled_pool, set(reduced_paths))
        return self._dataset

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dataset.not_labeled_pool = self._dataset.not_labeled_pool + self.__container
        self.__container = None
