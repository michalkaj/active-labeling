from typing import Dict, Any

from flask_restful import Resource

from active_labeling.active_learning.training.dataset import ActiveDataset
from active_labeling.config import LearningConfig


class Metrics(Resource):
    endpoint = '/metrics'
    @classmethod
    def instantiate(cls,
                    config: LearningConfig,
                    metrics: Dict[str, Any],
                    active_dataset: ActiveDataset):
        cls._config = config
        cls._metrics = metrics
        cls._active_dataset = active_dataset
        return cls

    def get(self):
        metrics = list(self._metrics.items())
        return {
            'metrics': metrics,
            'label_frequencies': self._get_label_frequencies()
        }

    def _get_label_frequencies(self) -> Dict[str, int]:
        counts = {label: 0 for label in self._config.labels}
        for label in self._active_dataset.labels.values():
            counts[label] += 1
        return counts
