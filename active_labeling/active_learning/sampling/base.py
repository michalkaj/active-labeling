import abc
from typing import Iterable

from active_labeling.active_learning.dataset import ActiveDataset


class BaseSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, active_dataset: ActiveDataset, sample_size: int) -> Iterable[int]:
        pass
