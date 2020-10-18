import abc
from typing import Iterable

import numpy as np


class BaseSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, data: np.ndarray, sample_size: int) -> Iterable[int]:
        pass
