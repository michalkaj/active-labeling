from typing import Dict, Any

from flask_restful import Resource

from active_labeling.active_learning.learners.training.dataset import ActiveDataset
from active_labeling.backend.loggers import get_logger
from active_labeling.config import ActiveLearningConfig

_LOGGER = get_logger(__name__)


class Annotations(Resource):
    endpoint = '/annotations'

    @classmethod
    def instantiate(cls, config: ActiveLearningConfig, active_dataset: ActiveDataset):
        cls._config = config
        cls._active_dataset = active_dataset
        return cls

    def get(self) -> Dict[str, Any]:
        mapping = {i: label for i, label in enumerate(self._config.labels)}
        labels = {str(path): mapping[label_int] for path, label_int
                  in self._active_dataset.labels.items()}
        return {
            'labels': list(self._config.labels),
            'annotations': labels,
        }
