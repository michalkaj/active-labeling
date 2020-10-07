import abc
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Iterable, Set, Generic, TypeVar, Dict

T = TypeVar('T')

class BaseDataLoader(abc.ABC, Generic[T]):
    def __init__(self, extensions: Set[str], recursive: bool = True):
        self._recursive = recursive
        self._extensions = extensions

    def load(self, dir_path: Path) -> Dict[Path, T]:
        paths = self._discover_paths(dir_path)
        return {path: self._load_file(path) for path in paths}

    def _discover_paths(self, dir_path: Path) -> Iterable[Path]:
        glob = partial(dir_path.rglob) if self._recursive else partial(dir_path.glob)
        return chain.from_iterable(glob(f'*.{ext}') for ext in self._extensions)

    @abc.abstractmethod
    def _load_file(self, path: Path) -> T:
        pass
