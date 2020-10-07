from dataclasses import dataclass
from pathlib import Path
from typing import Set, Optional, Dict, Any


@dataclass
class ActiveLearningConfig:
    server_url: str
    labels: Set[str]
    unlabeled_data_path: Path
    batch_size: int = 10
    pool_size: float = 1.
    labels_file: Optional[Path] = None
    validation_data_path: Optional[Path] = None
    validation_labels_file_path: Optional[Path] = None
    dataset_name: str = 'dataset'

    def as_dict(self) -> Dict[str, Any]:
        return vars(self)
