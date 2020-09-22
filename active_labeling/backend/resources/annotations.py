from typing import List, Dict

from flask_restful import Resource

from active_labeling.backend.database.base import BaseDatabaseConnection
from active_labeling.backend.loggers import get_logger
from active_labeling.loading.sample import Sample

_LOGGER = get_logger(__name__)


class Annotations(Resource):
    endpoint = '/annotations'

    @classmethod
    def instantiate(cls, db_connection: BaseDatabaseConnection):
        cls._db_connection = db_connection
        return cls

    def post(self) -> Dict[str, List[Sample]]:
        _, samples = self._db_connection.get_annotated_samples()
        return {'samples': [s.to_dict() for s in samples]}
