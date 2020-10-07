from typing import Dict

from flask_restful import Resource

from active_labeling.backend.database.storage import StorageHandler
from active_labeling.backend.loggers import get_logger

_LOGGER = get_logger(__name__)


class Metrics(Resource):
    endpoint = '/metrics'
    @classmethod
    def instantiate(cls, storage_handler: StorageHandler):
        cls._storage_handler = storage_handler
        return cls

    def get(self):
        metrics = list(self._db_connection.get_metrics())
        return {
            'metrics': metrics,
            'label_frequencies': self._get_label_frequencies()
        }

    def _get_label_frequencies(self) -> Dict[str, int]:
        _, samples = self._storage_handler.get_labeled_data()
        config = self._storage_handler.get_config()
        frequencies = {label: 0 for label in config['allowed_labels']}

        labels = (label for sample in samples for label in sample.labels)
        for label in labels:
            frequencies[label] += 1

        return frequencies

