from dataclasses import dataclass, field
from typing import Set, Dict, Any

import torch
from ordered_set import OrderedSet
from pytorch_lightning.metrics import Metric


@dataclass
class ActiveLearningConfig:
    labels: OrderedSet[str]
    metrics: Dict[str, Metric]
    early_stopping_metric: str
    batch_size: int = 10
    pool_size: float = 0.1
    bayesian_sample_size: int = 20
    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)
    device: torch.device = torch.device('cpu')
