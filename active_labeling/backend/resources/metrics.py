from typing import Dict

from flask_restful import Resource

from active_labeling.backend.database.base import BaseDatabaseConnection
from active_labeling.backend.loggers import get_logger

_LOGGER = get_logger(__name__)


class Metrics(Resource):
    endpoint = '/metrics'
    @classmethod
    def instantiate(cls, db_connection: BaseDatabaseConnection):
        cls._db_connection = db_connection
        return cls

    def get(self):
        metrics = list(self._db_connection.get_metrics())
        return {
            'metrics': metrics,
            'label_frequencies': self._get_label_frequencies()
        }

    def _get_label_frequencies(self) -> Dict[str, int]:
        _, samples = self._db_connection.get_annotated_samples()
        config = self._db_connection.get_config()
        frequencies = {label: 0 for label in config['allowed_labels']}

        labels = (label for sample in samples for label in sample.labels)
        for label in labels:
            frequencies[label] += 1

        print(frequencies)
        return frequencies

