from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

import torch
from ordered_set import OrderedSet
from pytorch_lightning.metrics import Metric


@dataclass
class LearningConfig:
    data_root: Path
    labels: OrderedSet[str]
    early_stopping_metric: str
    batch_size: int = 10
    pool_size: float = 0.1
    epochs: int = 10
    bayesian_sample_size: int = 20
    initial_training_set_size: int = 100
    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)
    device: torch.device = torch.device('cpu')
