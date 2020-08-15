from typing import List

import numpy as np
from flask import request
from flask_restful import Resource
from modAL import ActiveLearner

from active_labeling.loading.sample import Sample
from active_labeling.sampling.active_sampler import ActiveSampler
from active_labeling.settings import DEFAULT_BATCH_SIZE


class Query(Resource):
    endpoint = '/query'

    @classmethod
    def instantiate(cls, data: np.ndarray, samples: List[Sample], learner: ActiveLearner):
        cls._data = data
        cls._samples = samples
        cls._sampler = ActiveSampler(cls._data, learner)
        return cls

    def get(self):
        batch_size = int(request.args.get('batch_size', DEFAULT_BATCH_SIZE))
        indices = self._sampler.sample(batch_size)
        instances = (self._samples[i] for i in indices)
        return {
            'samples': [sample.to_dict() for sample in instances]
        }
