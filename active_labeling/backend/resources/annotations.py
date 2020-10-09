from typing import Dict, Any

from flask_restful import Resource

from active_labeling.backend.database.storage import StorageHandler
from active_labeling.backend.loggers import get_logger

_LOGGER = get_logger(__name__)


class Annotations(Resource):
    endpoint = '/annotations'

    @classmethod
    def instantiate(cls, storage_handler: StorageHandler):
        cls._storage_handler = storage_handler
        return cls

    def get(self) -> Dict[str, Any]:
        labeled_samples = {name: label for name, (_, label)
                           in self._storage_handler.get_labeled_samples().items()}
        config = self._storage_handler.get_config()
        return {
            'dataset_name': config.dataset_name,
            'labels': list(config.labels),
            'annotations': labeled_samples,
        }
