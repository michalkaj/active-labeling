from flask_restful import Resource, reqparse
from modAL import ActiveLearner
from redis import Redis

from active_labeling.backend.loggers import get_logger
from active_labeling.settings import ANNOTATED, NOT_ANNOTATED

_LOGGER = get_logger(__name__)


class Annotate(Resource):
    endpoint = '/annotate'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()
        self._parser.add_argument('samples', type=list, location='json')

    @classmethod
    def instantiate(cls, learner: ActiveLearner, redis: Redis):
        cls._learner = learner
        cls._redis = redis
        return cls

    def post(self):
        args = self._parser.parse_args()
        _LOGGER.debug(args)

        for sample in args['samples']:
            self._redis.sadd(ANNOTATED, sample['index'])
            self._redis.srem(NOT_ANNOTATED, sample['index'])
            self._redis.hset(sample['index'], 'label', sample['label'])
