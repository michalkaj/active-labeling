import abc
from typing import Sequence

from active_labeling.loading.sample import Sample


class BaseSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, items: Sequence[Sample], sample_size: int) -> Sequence[Sample]:
        pass
