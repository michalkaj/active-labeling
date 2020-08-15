from pathlib import Path
from typing import List

import numpy as np
from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api

from active_labeling.loading.image_loader import ImageLoader
from active_labeling.loading.sample import Sample
from active_labeling.sampling.active_sampler import ActiveSampler
from active_labeling.settings import DEFAULT_BATCH_SIZE


class Query(Resource):
    endpoint = '/query'

    @classmethod
    def instantiate(cls, data: np.ndarray, samples: List[Sample]):
        cls._data = data
        cls._samples = samples
        cls._sampler = ActiveSampler(cls._data)
        return cls

    def get(self):
        batch_size = int(request.args.get('batch_size', '0'))
        indices = self._sampler.query(batch_size or DEFAULT_BATCH_SIZE)
        instances = (self._samples[i] for i in indices)
        return {
            'samples': [sample.to_dict() for sample in instances]
        }


class ActiveLearning:
    def __init__(self, filedir_path: Path):
        self._app = Flask(__name__)
        CORS(self._app)
        self._api = Api(self._app)

        loader = ImageLoader(recursive=True)
        data, samples = loader.load(filedir_path)
        query = Query.instantiate(data, samples)
        self._api.add_resource(query, query.endpoint)

    def run(self) -> None:
        self._app.run()
