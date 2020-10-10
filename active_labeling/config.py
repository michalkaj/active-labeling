from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Optional, Callable, Dict, Any
from pytorch_lightning.metrics import Metric, Accuracy

import numpy as np


@dataclass
class ActiveLearningConfig:
    server_url: str
    labels: Set[str]
    unlabeled_data_path: Path
    validation_data_path: Optional[Path] = None
    validation_labels_path: Optional[Path] = None
    early_stopping_metric: str = 'accuracy'
    metrics: Dict[str, Metric] = field(default_factory={'accuracy': Accuracy})
    batch_size: int = 10
    pool_size: float = 1.
    dataset_name: str = 'dataset'
    labels_file: Optional[Path] = None
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)
