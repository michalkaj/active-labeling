import copy
from typing import Dict, Optional, Any, Tuple

import numpy as np

from active_labeling.config import ActiveLearningConfig


class Storage:
    def __init__(self,
                 unlabeled_data: Dict[str, np.ndarray],
                 config: ActiveLearningConfig,
                 data_labels: Optional[Dict[str, str]] = None,
                 validation_data: Optional[Dict[str, np.ndarray]] = None,
                 validation_labels: Optional[Dict[str, str]] = None):
        self.unlabeled_data = unlabeled_data
        self.labels = data_labels or {}
        self.config = config
        self.metrics = {}
        self.validation_data = validation_data
        self.validation_labels = validation_labels


class StorageHandler:
    def __init__(self, storage: Storage):
        self._storage = storage

    def get_unlabeled_data(self) -> Dict[str, np.ndarray]:
        unlabeled_images = set(self._storage.unlabeled_data) - set(self._storage.labels)
        return {k: self._storage.unlabeled_data[k] for k in unlabeled_images}

    def get_labeled_samples(self) -> Dict[str, Tuple[np.ndarray, str]]:
        return {
            path: (self._storage.unlabeled_data[path], label) for path, label in self._storage.labels.items()
        }

    def get_validation_samples(self) -> Dict[str, Tuple[np.ndarray, str]]:
        return {
            path: (self._storage.validation_data[path], label) for path, label in
            self._storage.validation_labels.items()
        }

    def annotate(self, labeled_samples: Dict[str, str]) -> None:
        self._storage.labels.update(labeled_samples)

    def get_config(self) -> ActiveLearningConfig:
        return copy.deepcopy(self._storage.config)

    def save_config(self, config: ActiveLearningConfig) -> None:
        self._storage.config = config

    def get_metrics(self):
        return copy.deepcopy(self._storage.metrics)

    def save_metric(self, name: str, value: Any) -> None:
        self._storage.metrics.setdefault(name, []).append(value)
