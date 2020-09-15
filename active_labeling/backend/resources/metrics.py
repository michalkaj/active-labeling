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
        return {'metrics': metrics}
