from functools import partial
from itertools import chain
from pathlib import Path
from typing import Tuple, Optional, Iterable

from active_labeling.loading.sample import Sample


class Loader:
    def __init__(self, extensions: Optional[Tuple[str, ...]] = None, recursive: bool = False):
        self._extensions = extensions
        self._recursive = recursive

    def load(self, dir_path: Path) -> Iterable[Sample]:
        glob = partial(dir_path.rglob) if self._recursive else partial(dir_path.glob)
        if self._extensions:
            sample_paths = chain.from_iterable(glob(f'*.{ext}') for ext in self._extensions)
        else:
            sample_paths = (pth for pth in glob('*') if pth.is_file())
        return (Sample.from_path(sample_path) for sample_path in sample_paths)
