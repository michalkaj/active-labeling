from pathlib import Path
from typing import Any, Dict, Optional, Sequence

IMAGE_EXTENSIONS = ('jpg', 'jpeg', 'png')

def _get_image_type(extension: str) -> str:
    if extension in IMAGE_EXTENSIONS:
        return 'image'
    else:
        return 'unknown'


class Sample:
    def __init__(self, path: Path, labels: Optional[Sequence[str]] = None):
        self.path = path
        self.name = path.name
        self.labels = labels or []

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]):
        labels = dictionary['labels']
        if len(labels):
            labels = [labels[0]]
        return cls(Path(dictionary['path']),
                   labels)

    def to_dict(self):
        return {
            'name': self.name,
            'path': str(self.path),
            'labels': self.labels
        }

    def __repr__(self):
        return f"Sample(name={self.name}, path={self.path})"
