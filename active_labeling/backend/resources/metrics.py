from flask_restful import Resource, reqparse
from modAL import ActiveLearner

from active_labeling.backend.database.base import BaseDatabaseConnection
from active_labeling.backend.loggers import get_logger
from active_labeling.loading.sample import Sample

_LOGGER = get_logger(__name__)


class Metrics(Resource):
    endpoint = '/progress'
    @classmethod
    def instantiate(cls, db_connection: BaseDatabaseConnection):
        cls._db_connection = db_connection
        return cls

    def get(self):
        metrics = list(self._db_connection.get_metrics())
        return {'metrics': metrics}
