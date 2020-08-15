import base64
from pathlib import Path

IMAGE_EXTENSIONS = ('jpg', 'jpeg', 'png')


def _get_image_type(extension: str) -> str:
    if extension in IMAGE_EXTENSIONS:
        return 'image'
    else:
        return 'unknown'


class Sample:
    def __init__(self, path: Path, name: str):
        self.path = path
        self.name = name

    @classmethod
    def from_path(cls, path: Path):
        return cls(path, path.name)

    def to_dict(self):
        with self.path.open('rb') as file:
            b64_file = base64.b64encode(file.read())
        extension = self.path.suffix[1:]
        return {
            'base64_file': b64_file.decode('utf-8'),
            'name': self.name,
            'extension': extension,
            'type': _get_image_type(extension)
        }

    def __repr__(self):
        return f"Sample(name={self.name}, path={self.path})"
