from pathlib import Path

import numpy as np
from flask import Flask
from flask_cors import CORS
from flask_restful import Api
from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from redis import Redis
from sklearn.base import BaseEstimator

from active_labeling.backend.resources.annotate import Annotate
from active_labeling.backend.resources.query import Query
from active_labeling.backend.resources.teach import Teach
from active_labeling.loading.image_loader import ImageLoader
from active_labeling.settings import NOT_ANNOTATED


class ActiveLearning:
    def __init__(self, filedir_path: Path, estimator: BaseEstimator):
        self._app = Flask(__name__)
        CORS(self._app)
        self._redis = Redis()

        self._api = Api(self._app)

        self._learner = ActiveLearner(
            estimator=estimator,
            query_strategy=uncertainty_sampling
        )
        data = self._load_data(filedir_path)
        self.init_resources(data)

    def init_resources(self, data):
        query = Query.instantiate(data, self._learner, self._redis)
        teach = Teach.instantiate(data, self._learner, self._redis)
        annotate = Annotate.instantiate(self._learner, self._redis)
        self._api.add_resource(annotate, annotate.endpoint)
        self._api.add_resource(teach, teach.endpoint)
        self._api.add_resource(query, query.endpoint)

    def _load_data(self, filedir_path: Path) -> np.ndarray:
        loader = ImageLoader(recursive=True)
        data, samples = loader.load(filedir_path)
        for index, sample in enumerate(samples):
            self._redis.sadd(NOT_ANNOTATED, index)
            self._redis.hmset(index, sample.to_dict())
        return data

    def run(self) -> None:
        self._app.run()
