import json
from pathlib import Path
from typing import Optional

from flask_restful import Resource, reqparse

from active_labeling.backend.database.base import BaseDatabaseConnection
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
    def instantiate(cls, config_path: Optional[Path], db_connection: BaseDatabaseConnection):
        cls._db_connection = db_connection
        config_path = config_path or Path('.')
        config = cls._load_config(config_path)
        cls._db_connection.save_config(config)
        return cls

    @classmethod
    def _load_config(cls, config_path: Path):
        with config_path.open('r') as file:
            config = json.load(file)
        return config

    def get(self):
        return self._db_connection.get_config()

    def post(self):
        config  = self._parser.parse_args()
        self._db_connection.save_config(config)


