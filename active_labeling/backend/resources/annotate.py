from flask_restful import Resource, reqparse
from modAL import ActiveLearner

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
    def instantiate(cls, storage_handler: StorageHandler, learner: ActiveLearner):
        cls._learner = learner
        cls._storage_handler = storage_handler
        return cls

    def post(self):
        args = self._parser.parse_args()
        samples_json = args['samples']
        samples = map(Sample.from_dict, samples_json)
        self._storage_handler.annotate(samples)
