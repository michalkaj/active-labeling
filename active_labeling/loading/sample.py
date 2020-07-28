from pathlib import Path


class Sample:
    def __init__(self, path: Path, name: str, extension: str):
        self.path = path
        self.name = name
        self.extension = extension

    @classmethod
    def from_path(cls, path: Path):
        return cls(path, path.name, path.suffix)

    def __repr__(self):
        return f"Sample(name={self.name}, path={self.path})"
