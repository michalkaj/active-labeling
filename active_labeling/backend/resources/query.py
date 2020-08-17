from pathlib import Path
from typing import Iterable, Dict

import numpy as np
from flask import request
from flask_restful import Resource
from modAL import ActiveLearner
from redis import Redis

from active_labeling.backend.loggers import get_logger
from active_labeling.backend.utils import path_to_base64
from active_labeling.sampling.active_sampler import ActiveSampler
from active_labeling.settings import NOT_ANNOTATED, DEFAULT_BATCH_SIZE

_SAMPLE_COLS = ('name', 'path', 'type')
_LOGGER = get_logger(__name__)


class Query(Resource):
    endpoint = '/query'

    @classmethod
    def instantiate(cls, data: np.ndarray,  learner: ActiveLearner, redis: Redis):
        cls._data = data
        cls._sampler = ActiveSampler(learner)
        cls._redis = redis
        return cls

    def get(self):
        not_annotated_indices = np.fromiter(self._redis.smembers(NOT_ANNOTATED), np.uint32)
        data = self._data[not_annotated_indices]

        batch_size = int(request.args.get('batch_size', DEFAULT_BATCH_SIZE))
        indices_to_query = self._sampler.sample(data, batch_size)
        return {'samples': list(self._get_instances(not_annotated_indices[indices_to_query]))}

    def _get_instances(self, indices: Iterable[int]) -> Iterable[Dict[str, str]]:
        samples = (self._redis.hmget(str(index), *_SAMPLE_COLS)
                   for index in indices)
        _LOGGER.debug(samples)

        for sample in samples:
            decoded_sample = (s.decode('utf-8') for s in sample)
            sample = dict(zip(_SAMPLE_COLS, decoded_sample))
            sample['base64_file'] = path_to_base64(Path(sample['path']))
            yield sample
