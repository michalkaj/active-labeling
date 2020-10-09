from collections import Counter
from operator import itemgetter
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
        metrics = list(self._storage_handler.get_metrics().items())
        return {
            'metrics': metrics,
            'label_frequencies': self._get_label_frequencies()
        }

    def _get_label_frequencies(self) -> Dict[str, int]:
        labels = map(itemgetter(1), self._storage_handler.get_labeled_samples().values())
        all_labels = self._storage_handler.get_config().labels
        counts =  {**dict(zip(all_labels, [0] * len(all_labels))), **Counter(labels)}
        return dict(counts)
