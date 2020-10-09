from dataclasses import dataclass
from pathlib import Path
from typing import Set, Optional, Callable, Dict

import numpy as np


@dataclass
class ActiveLearningConfig:
    server_url: str
    labels: Set[str]
    unlabeled_data_path: Path
    batch_size: int = 10
    pool_size: float = 1.
    dataset_name: str = 'dataset'
    labels_file: Optional[Path] = None
    validation_data_path: Optional[Path] = None
    validation_labels_file_path: Optional[Path] = None
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None
