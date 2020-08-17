import numpy as np
from flask_restful import Resource, reqparse
from modAL import ActiveLearner
from redis import Redis
from active_labeling.backend.loggers import get_logger
from active_labeling.settings import ANNOTATED

_LOGGER = get_logger(__name__)


class Teach(Resource):
    endpoint = '/teach'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()

    @classmethod
    def instantiate(cls, data: np.ndarray, learner: ActiveLearner, redis: Redis):
        cls._data = data
        cls._learner = learner
        cls._redis = redis
        return cls

    def get(self):
        sample_indices = np.fromiter(self._redis.smembers(ANNOTATED), np.uint32)
        labels = (self._redis.hget(str(index), 'label') for index in sample_indices)

        x_train = self._data[sample_indices]
        y_train = np.fromiter(labels, dtype=np.uint8)

        _LOGGER.debug((sample_indices, y_train))
        self._learner.teach(x_train, y_train)
        return 'OK'