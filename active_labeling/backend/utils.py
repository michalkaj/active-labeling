import base64
from pathlib import Path


def path_to_base64(path: Path) -> str:
    with path.open('rb') as file:
        b64_file = base64.b64encode(file.read())
    return b64_file.decode('utf-8')