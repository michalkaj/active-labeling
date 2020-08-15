import abc
from typing import Iterable

import numpy as np


class BaseSampler(abc.ABC):
    def __init__(self, data: np.ndarray):
        self._data = data

    @abc.abstractmethod
    def query(self, sample_size: int) -> Iterable[int]:
        pass
