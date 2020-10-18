import base64
import json
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Dict, Any
from typing import Set, List


def path_to_base64(path: Path) -> str:
    with path.open('rb') as file:
        b64_file = base64.b64encode(file.read())
    return b64_file.decode('utf-8')


def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open('r') as file:
        return json.load(file)


def discover_paths(dir_path: Path, extensions: Set[str], recursive: bool = True) -> List[Path]:
    glob = partial(dir_path.rglob) if recursive else partial(dir_path.glob)
    paths = chain.from_iterable(glob(f'*{ext}') for ext in extensions)
    return list(paths)
