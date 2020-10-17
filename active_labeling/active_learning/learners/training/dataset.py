from itertools import compress
from pathlib import Path
from typing import Sequence, Dict, Optional, Callable, Tuple, Union

import PIL
import numpy as np
import torch
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class ActiveDataset(Dataset):
    def __init__(self,
                 pool: Sequence[Path],
                 labels: Dict[Union[str, Path], int],
                 train: bool = True,
                 transform: Optional[Callable[[Image], Tensor]] = None):
        self.labels = {Path(path): label for path, label in labels.items()}
        self._labeled_pool, self.not_labeled_pool = self._divide_pool(pool, self.labels)
        self._train = train
        self._transform = transform if transform else ToTensor()
        self._dataset_fraction = 1.

    @staticmethod
    def _divide_pool(pool: Sequence, labels: Dict[Path, int]):
        labeled_mask = [path in labels for path in pool]
        labeled_pool = list(compress(pool, labeled_mask))
        not_labeled_pool = list(compress(pool, np.logical_not(labeled_mask)))

        assert len(labeled_pool) + len(not_labeled_pool) == len(pool)

        return labeled_pool, not_labeled_pool

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, Optional[int]]]:
        path = self._get_example(index)
        image = PIL.Image.open(path)
        image = self._transform(image)

        label = {'label': self.labels[path]} if self._train else {}
        return {'image': image, **label}

    def _get_example(self, index: int) -> Path:
        pool = self._labeled_pool if self._train else self.not_labeled_pool
        return pool[index]

    def __len__(self):
        return len(self._labeled_pool) if self._train else len(self.not_labeled_pool)

    # def __call__(self, dataset_fraction: float = 0.1):
    #     self._dataset_fraction = dataset_fraction
    #
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #

    def add_labels(self, labels: Dict[Path, int]) -> None:
        labeled, not_labeled = self._divide_pool(self.not_labeled_pool, labels)

        self._labeled_pool.extend(labeled)
        self._not_labeled_pool = not_labeled

        self.labels.update(labels)

    def train(self) -> 'ActiveDataset':
        self._train = True
        return self

    def evaluate(self) -> 'ActiveDataset':
        self._train = False
        return self

