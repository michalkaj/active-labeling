from numbers import Number
from typing import Iterable, Callable, Optional

import numpy as np
from flask_restful import Resource, reqparse
from modAL import ActiveLearner
from sklearn.metrics import accuracy_score

from active_labeling.backend.database.base import BaseDatabaseConnection
from active_labeling.backend.loggers import get_logger
from active_labeling.loading.sample import Sample

_LOGGER = get_logger(__name__)


class Teach(Resource):
    endpoint = '/teach'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()

    @classmethod
    def instantiate(cls, data: np.ndarray,
                    learner: ActiveLearner, db_connection: BaseDatabaseConnection,
                    metrics: Optional[Iterable[Callable[[np.ndarray, np.ndarray], Number]]] = None):
        cls._data = data
        cls._learner = learner
        cls._db_connection = db_connection
        cls._metrics = metrics or [accuracy_score]
        return cls

    def get(self):
        indices, samples = self._db_connection.get_annotated_samples()
        if not samples:
            return

        y_train = self._transform_labels(samples)
        x_train = self._data[indices]
        self._learner.teach(x_train, y_train)
        self._compute_metrics(x_train, y_train)
        return 'OK'

    def _transform_labels(self, samples: Iterable[Sample]) -> np.array:
        config = self._db_connection.get_config()
        allowed_labels = {label: index for index, label in enumerate(config['allowed_labels'])}
        _LOGGER.debug(allowed_labels)
        return np.array(
            [np.array([allowed_labels[label] for label in sample.labels]) for sample in samples],
            dtype=np.uint32
        ).flatten()

    def _compute_metrics(self, x: np.ndarray, y_true: np.ndarray) -> None:
        y_pred = self._learner.predict(x)
        for metric in self._metrics:
            result = metric(y_true, y_pred)
            _LOGGER.debug(f'{metric.__name__} = {result}, size: {len(x)}')
            self._db_connection.save_metric(metric.__name__, result, len(x))
