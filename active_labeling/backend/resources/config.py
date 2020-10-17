from flask_restful import Resource, reqparse

from active_labeling.backend.loggers import get_logger
from active_labeling.config import ActiveLearningConfig

_LOGGER = get_logger(__name__)


class Config(Resource):
    endpoint = '/config'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()
        self._parser.add_argument('labels', type=list, location='json')
        self._parser.add_argument('multiclass', type=str, location='json')

    @classmethod
    def instantiate(cls, config: ActiveLearningConfig):
        cls._config = config
        return cls

    def get(self):
        config = self._config
        return {
            'labels': list(config.labels),
            'batch_size': config.batch_size,
            'pool_size': config.pool_size,
        }
