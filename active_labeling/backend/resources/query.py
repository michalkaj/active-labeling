from pathlib import Path
from typing import Iterable, Dict, Callable, Sequence

import numpy as np
from flask import request
from flask_restful import Resource
from modAL import ActiveLearner

from active_labeling.backend.database.base import BaseDatabaseConnection
from active_labeling.backend.loggers import get_logger
from active_labeling.backend.utils import path_to_base64
from active_labeling.loading.sample import Sample
from active_labeling.sampling.active_sampler import ActiveSampler
from active_labeling.settings import DEFAULT_BATCH_SIZE

_LOGGER = get_logger(__name__)


class Query(Resource):
    endpoint = '/query'

    @classmethod
    def instantiate(cls, data: np.ndarray,
                    learner: ActiveLearner, db_connection: BaseDatabaseConnection):
        cls._data = data
        cls._sampler = ActiveSampler(learner)
        cls._db_connection = db_connection
        return cls

    def get(self):
        indices, samples = self._db_connection.get_not_annotated_samples()
        data_x = self._data[indices]

        batch_size = int(request.args.get('batch_size', DEFAULT_BATCH_SIZE))
        indices_to_query = self._sampler.sample(data_x, batch_size)

        return {'samples': list(self._prepare_samples(indices_to_query, samples))}

    def _prepare_samples(self, indices: Iterable[int], samples: Sequence[Sample])\
            -> Iterable[Dict[str, str]]:
        for index in indices:
            sample = samples[index]
            sample_dict = sample.to_dict()
            sample_dict['base64_file'] = path_to_base64(Path(sample.path))
            yield sample_dict
