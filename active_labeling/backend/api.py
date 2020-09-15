from pathlib import Path
from typing import Optional, Iterable, Callable, List

import numpy as np
from flask import Flask
from flask_cors import CORS
from flask_restful import Api
from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.base import BaseEstimator

from active_labeling.backend.database.redis_db import RedisConnection
from active_labeling.backend.resources.annotate import Annotate
from active_labeling.backend.resources.config import Config
from active_labeling.backend.resources.metrics import Metrics
from active_labeling.backend.resources.query import Query
from active_labeling.backend.resources.teach import Teach
from active_labeling.loading.base_loader import SampleLoader
from active_labeling.loading.sample import Sample


class ActiveLearning:
    def __init__(self, filedir_path: Path, estimator: BaseEstimator,
                 transform_samples: Callable[[Iterable[Sample]], np.ndarray],
                 config_path: Optional[Path] = None):
        self._app = Flask(__name__)
        CORS(self._app)
        self._db_connection = RedisConnection()
        self._api = Api(self._app)

        self._learner = ActiveLearner(
            estimator=estimator,
            query_strategy=uncertainty_sampling
        )

        self._transform_samples = transform_samples
        data = self._prepare_data(filedir_path)
        self._init_resources(data, config_path)

    def run(self) -> None:
        self._app.run()

    def _init_resources(self, data: np.ndarray, config_path: Optional[Path]):
        query = Query.instantiate(data, self._learner, self._db_connection)
        teach = Teach.instantiate(data, self._learner, self._db_connection)
        annotate = Annotate.instantiate(self._learner, self._db_connection)
        config = Config.instantiate(config_path, self._db_connection)
        metrics = Metrics.instantiate(self._db_connection)
        self._api.add_resource(annotate, annotate.endpoint)
        self._api.add_resource(teach, teach.endpoint)
        self._api.add_resource(query, query.endpoint)
        self._api.add_resource(config, config.endpoint)
        self._api.add_resource(metrics, metrics.endpoint)

    def _prepare_data(self, filedir_path: Path) -> np.ndarray:
        # Load
        loader = SampleLoader(('png', 'jpg', 'jpeg'), recursive=True)
        samples = list(loader.load(filedir_path))
        self._db_connection.save_samples(samples)

        # Transform
        return self._transform_samples(samples)



