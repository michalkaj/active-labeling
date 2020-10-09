from pathlib import Path

from flask_restful import Resource, reqparse

from active_labeling.backend.database.storage import StorageHandler
from active_labeling.backend.loggers import get_logger

_LOGGER = get_logger(__name__)


class Annotate(Resource):
    endpoint = '/annotate'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()
        self._parser.add_argument('samples', type=list, location='json')

    @classmethod
    def instantiate(cls, storage_handler: StorageHandler):
        cls._storage_handler = storage_handler
        return cls

    def post(self):
        args = self._parser.parse_args()
        samples_json = args['samples']
        samples = {s['path']: s['label'] for s in samples_json}
        self._storage_handler.annotate(samples)
        return 200
