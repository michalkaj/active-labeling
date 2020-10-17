from pathlib import Path
from typing import Dict

from flask_restful import Resource, reqparse
from torch import nn

from active_labeling.active_learning.learners.training.dataset import ActiveDataset
from active_labeling.active_learning.sampling.acquisition.bald import BALD
from active_labeling.active_learning.sampling.active_sampler import ActiveSampler
from active_labeling.backend.file_utils import path_to_base64
from active_labeling.backend.loggers import get_logger
from active_labeling.config import ActiveLearningConfig
from active_labeling.settings import DEFAULT_BATCH_SIZE, DEFAULT_POOL_SIZE

_LOGGER = get_logger(__name__)


class Query(Resource):
    endpoint = '/query'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()
        self._parser.add_argument('batch_size', type=int, default=DEFAULT_BATCH_SIZE)
        self._parser.add_argument('pool_size', type=float, default=DEFAULT_POOL_SIZE)

    @classmethod
    def instantiate(cls, config: ActiveLearningConfig, learner: nn.Module, dataset: ActiveDataset):
        cls._config = config
        cls._sampler = ActiveSampler(learner, BALD, config)
        cls._dataset = dataset
        return cls

    def get(self):
        args = self._parser.parse_args()
        batch_size = args['batch_size']
        paths_to_query = self._sampler.sample(self._dataset, batch_size)
        return {'samples': [self._prepare_sample(path) for path in paths_to_query]}

    def _prepare_sample(self, path: Path) -> Dict[str, str]:
        return {
            'path': path,
            'name': str(Path(path).stem),
            'base64_file': path_to_base64(path)
        }
