import json
from pathlib import Path
from typing import Any

from flask_restful import Resource, reqparse

from active_labeling.backend.database.storage import StorageHandler
from active_labeling.backend.loggers import get_logger

_LOGGER = get_logger(__name__)


class Config(Resource):
    endpoint = '/config'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()
        self._parser.add_argument('labels', type=list, location='json')
        self._parser.add_argument('multiclass', type=str, location='json')

    @classmethod
    def instantiate(cls, storage_handler: StorageHandler):
        cls._storage_handler = storage_handler
        return cls

    def get(self):
        config = self._storage_handler.get_config()
        return {
            'server_url': config.server_url,
            'labels': list(config.labels),
            'batch_size': config.batch_size,
            'pool_size': config.pool_size,
            'dataset_name': config.dataset_name,
        }
