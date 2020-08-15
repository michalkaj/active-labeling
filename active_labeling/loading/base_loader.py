import abc
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Tuple, Iterable, List

import numpy as np

from active_labeling.loading.sample import Sample


class BaseLoader(abc.ABC):
    def __init__(self, extensions: Tuple[str, ...] = None, recursive: bool = False):
        self._extensions = extensions
        self._recursive = recursive

    def load(self, dir_path: Path) -> Tuple[np.ndarray, List[Sample]]:
        samples = self._read_samples(dir_path)
        return self._transform(samples)

    def _read_samples(self, dir_path: Path) -> Iterable[Sample]:
        glob = partial(dir_path.rglob) if self._recursive else partial(dir_path.glob)
        sample_paths = chain.from_iterable(glob(f'*.{ext}') for ext in self._extensions)
        return (Sample.from_path(sample_path) for sample_path in sample_paths)

    @abc.abstractmethod
    def _transform(self, samples: Iterable[Sample]) -> Tuple[np.ndarray, List[Sample]]:
        pass
