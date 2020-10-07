from numbers import Number
from typing import Iterable, Callable, Optional

import numpy as np
from flask_restful import Resource, reqparse
from modAL import ActiveLearner
from sklearn.metrics import accuracy_score

from active_labeling.backend.database.storage import StorageHandler
from active_labeling.backend.loggers import get_logger
from active_labeling.loading.sample import Sample

_LOGGER = get_logger(__name__)


class Teach(Resource):
    endpoint = '/teach'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()

    @classmethod
    def instantiate(cls,
                    storage_handler: StorageHandler,
                    learner: ActiveLearner,
                    metrics: Optional[Iterable[Callable[[np.ndarray, np.ndarray], Number]]] = None):
        cls._storage_handler = storage_handler
        cls._learner = learner
        cls._metrics = metrics or [accuracy_score]
        return cls

    def get(self):
        labeled_samples = self._storage_handler.get_labeled_data()
        if not labeled_samples:
            return  # TODO

        y_train = self._transform_labels(label for _, label in labeled_samples)
        x_train = np.stack(image for image, _ in labeled_samples)
        self._learner.teach(x_train, y_train)
        self._compute_metrics(x_train, y_train)
        return 'OK'

    def _transform_labels(self, samples: Iterable[Sample]) -> np.array:
        config = self._storage_handler.get_config()
        allowed_labels = {label: index for index, label in enumerate(config['allowed_labels'])}
        return np.array([allowed_labels[sample] for sample in samples])

    def _compute_metrics(self, x: np.ndarray, y_true: np.ndarray) -> None:
        y_pred = self._learner.predict(x)
        for metric in self._metrics:
            result = metric(y_true, y_pred)
            self._storage_handler.save_metric(
                name=metric.__name__,
                value={'result': result, 'step': len(x)}
            )
