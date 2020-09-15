from flask_restful import Resource, reqparse
from modAL import ActiveLearner

from active_labeling.backend.database.base import BaseDatabaseConnection
from active_labeling.backend.loggers import get_logger
from active_labeling.loading.sample import Sample

_LOGGER = get_logger(__name__)


class Annotate(Resource):
    endpoint = '/annotate'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()
        self._parser.add_argument('samples', type=list, location='json')

    @classmethod
    def instantiate(cls, learner: ActiveLearner,  db_connection: BaseDatabaseConnection):
        cls._learner = learner
        cls._db_connection = db_connection
        return cls

    def post(self):
        args = self._parser.parse_args()
        samples_json = args['samples']
        samples = map(Sample.from_dict, samples_json)
        self._db_connection.annotate_samples(samples)
