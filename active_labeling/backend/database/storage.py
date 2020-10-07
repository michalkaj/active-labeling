import copy
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np

from active_labeling.config import ActiveLearningConfig


class Storage:
    def __init__(self,
                 unlabeled_data: Dict[Path, np.ndarray],
                 config: ActiveLearningConfig,
                 data_labels: Optional[Dict[Path, str]] = None,
                 validation_data: Optional[Dict[Path, np.ndarray]] = None,
                 validation_labels: Optional[Dict[Path, str]] = None):
        self.unlabeled_data = unlabeled_data
        self.labels = data_labels or {}
        self.config = config
        self.metrics = {}
        self.validation_data = validation_data
        self.validation_labels = validation_labels


class StorageHandler:
    def __init__(self, storage: Storage):
        self._storage = storage

    def get_unlabeled_data(self):
        unlabeled_images = set(self._storage.unlabeled_data) - set(self._storage.labels)
        return {k: self._storage.unlabeled_data[k] for k in unlabeled_images}

    def get_labeled_data(self):
        return {k: (self._storage.unlabeled_data[k], self._storage.labels[k])
                for k in self._storage.labels}

    def annotate(self, labeled_samples):
        self._storage.labels.update(labeled_samples)

    def get_config(self) -> ActiveLearningConfig:
        return copy.deepcopy(self._storage.config)

    def save_config(self, config: ActiveLearningConfig) -> None:
        self._storage.config = config

    def save_metric(self, name, value):
        self._storage.metrics.setdefault(name, []).append(value)
