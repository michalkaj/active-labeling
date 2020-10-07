from pathlib import Path
from typing import Iterable, Dict, Any

import numpy as np
from flask_restful import Resource, reqparse
from modAL import ActiveLearner

from active_labeling.backend.database.storage import StorageHandler
from active_labeling.backend.loggers import get_logger
from active_labeling.backend.utils import path_to_base64
from active_labeling.sampling.active_sampler import ActiveSampler
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
    def instantiate(cls, storage_handler: StorageHandler, learner: ActiveLearner):
        cls._sampler = ActiveSampler(learner)
        cls._storage_handler = storage_handler
        return cls

    def get(self):
        samples = self._storage_handler.get_unlabeled_data()
        data_x = np.stack(list(samples.values()))

        args = self._parser.parse_args()
        indices_to_query = self._sampler.sample(data_x, **args)

        return {'samples': list(self._prepare_samples(indices_to_query, samples))}

    def _prepare_samples(self, indices: Iterable[int], samples: Dict[str, np.ndarray])\
            -> Iterable[Dict[str, str]]:
        image_paths = list(samples.keys())
        for index in indices:
            path = image_paths[index]
            yield path_to_base64(Path(path))
