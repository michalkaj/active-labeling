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
from active_labeling.backend.resources.annotations import Annotations
from active_labeling.backend.resources.config import Config
from active_labeling.backend.resources.metrics import Metrics
from active_labeling.backend.resources.query import Query
from active_labeling.backend.resources.teach import Teach
from active_labeling.loading.sample import Sample


class ActiveLearning:
    def __init__(self, unlabeled_data: List[Path],
                 estimator: BaseEstimator,
                 transform_samples: Callable[[Iterable[Path]], np.ndarray],
                 config_path: Optional[Path] = None):
        self._app = Flask(__name__)
        CORS(self._app)
        self._db_connection = RedisConnection()
        self._api = Api(self._app)

        self._learner = ActiveLearner(
            estimator=estimator,
            query_strategy=uncertainty_sampling
        )

        # TODO: Decouple data initialization from resource initialization
        data = transform_samples(unlabeled_data)
        unlabeled_samples = [Sample(p) for p in unlabeled_data]
        self._db_connection.save_samples(unlabeled_samples)

        self._init_resources(data, config_path)

    def run(self) -> None:
        self._app.run()

    def _init_resources(self, data: np.ndarray, config_path: Optional[Path]) -> None:
        resources = (
            Query.instantiate(data, self._learner, self._db_connection),
            Teach.instantiate(data, self._learner, self._db_connection),
            Annotate.instantiate(self._learner, self._db_connection),
            Config.instantiate(config_path, self._db_connection),
            Metrics.instantiate(self._db_connection),
            Annotations.instantiate(self._db_connection)
        )

        for resource in resources:
            self._api.add_resource(resource, resource.endpoint)
