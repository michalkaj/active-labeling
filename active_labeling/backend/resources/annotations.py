from typing import List, Dict

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

    def post(self) -> Dict[str, Dict[str, str]]:
        labeled_samples = self._storage_handler.get_labeled_data()
        return {'samples': labeled_samples}
