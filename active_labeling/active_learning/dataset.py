import abc
from pathlib import Path
from typing import Sequence, Dict, Optional, Callable, Union, Hashable

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


class ActiveDataset(Dataset, abc.ABC):
    def __init__(self,
                 pool: Sequence[Hashable],
                 labels: Dict[Hashable, int],
                 train_transform: Optional[Callable[[Image], Tensor]] = None,
                 evaluate_transform: Optional[Callable[[Image], Tensor]] = None,
                 target_transform: Optional[Callable[[str], int]] = None):
        self.labels = labels
        self._labeled_pool, self._not_labeled_pool = _divide_pool(pool, self.labels)
        self._train_transform = train_transform or ToTensor()
        self._evaluate_transform = evaluate_transform or self._train_transform
        self._target_transform = target_transform or (lambda x: x)
        self._train = True

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, Optional[int]]]:
        pool = self._labeled_pool if self._train else self._not_labeled_pool
        key = pool[index]
        image = self._get_example(key)
        label = self._get_label(key)

        image = self._train_transform(image) if self._train else self._evaluate_transform(image)

        return {'image': image, 'label': label}

    @abc.abstractmethod
    def _get_example(self, index: int) -> np.ndarray:
        pass

    def _get_label(self, key: Hashable):
        if not self._train:
            return -1
        label = self.labels[key]
        return self._target_transform(label)

    def __len__(self):
        return len(self._labeled_pool if self._train else self._not_labeled_pool)

    def add_labels(self, labels: Dict[Hashable, str]) -> None:
        self._validate(labels)

        labeled, not_labeled = _divide_pool(self._not_labeled_pool, labels)

        self._labeled_pool.extend(labeled)
        self._not_labeled_pool = not_labeled

        self.labels.update(labels)

    def get_examples(self, indices: Sequence[int]) -> Sequence:
        return [self._not_labeled_pool[i] for i in indices]


    def _validate(self, labels: Dict[Hashable, str]):
        intersecting_keys = set(labels.keys()).intersection(set(self.labels.keys()))
        if intersecting_keys:
            raise ValueError(f"Some of the samples are already labeled: {intersecting_keys}")

    def train(self) -> 'ActiveDataset':
        self._train = True
        return self

    def evaluate(self) -> 'ActiveDataset':
        self._train = False
        return self


class FileDataset(ActiveDataset):
    def __init__(self,
                 pool: Sequence[Path],
                 labels: Dict[Path, int],
                 train_transform: Optional[Callable[[Image], Tensor]] = None,
                 evaluate_transform: Optional[Callable[[Image], Tensor]] = None,
                 target_transform: Optional[Callable[[str], int]] = None):
        super().__init__(
            pool=pool,
            labels=labels,
            train_transform=train_transform,
            evaluate_transform=evaluate_transform,
            target_transform=target_transform,
        )

    def _get_example(self, key: Path) -> Image:
        return PIL.Image.open(key)


class InMemoryDataset(ActiveDataset):
    def __init__(self,
                 pool: np.ndarray,
                 labels: Dict[int, int],
                 train_transform: Optional[Callable[[Image], Tensor]] = None,
                 evaluate_transform: Optional[Callable[[Image], Tensor]] = None,
                 target_transform: Optional[Callable[[str], int]] = None):
        self._numpy_pool = pool
        super().__init__(
            pool=np.arange(len(pool)),
            labels=labels,
            train_transform=train_transform,
            evaluate_transform=evaluate_transform,
            target_transform=target_transform,
        )

    def _get_example(self, key: int) -> Image:
        array = self._numpy_pool[key]
        return PIL.Image.fromarray(array)


class Reducer:
    def __init__(self, dataset: ActiveDataset, dataset_frac: float = 1.):
        self._dataset = dataset
        self._dataset_frac = dataset_frac
        self.__container = None

    def __enter__(self):
        length = int(len(self._dataset._not_labeled_pool) * self._dataset_frac)
        reduced_paths = np.random.choice(self._dataset._not_labeled_pool, size=length, replace=False)
        self._dataset._not_labeled_pool, self.__container = _divide_pool(
            self._dataset._not_labeled_pool, set(reduced_paths))
        return self._dataset

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dataset._not_labeled_pool = self._dataset._not_labeled_pool + self.__container
        self.__container = None
