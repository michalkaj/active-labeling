import base64
import json
from pathlib import Path
from typing import Dict, Any


def path_to_base64(path: Path) -> str:
    with path.open('rb') as file:
        b64_file = base64.b64encode(file.read())
    return decode(b64_file)

def decode(b: bytes):
    return b.decode('utf-8')

def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open('r') as file:
        return json.load(file)
