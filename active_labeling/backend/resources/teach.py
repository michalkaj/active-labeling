from typing import Iterable, Sequence

import numpy as np
from flask_restful import Resource, reqparse
from modAL import ActiveLearner

from active_labeling.backend.database.storage import StorageHandler
from active_labeling.backend.loggers import get_logger

_LOGGER = get_logger(__name__)


class Teach(Resource):
    endpoint = '/teach'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()

    @classmethod
    def instantiate(cls,
                    storage_handler: StorageHandler,
                    learner: ActiveLearner):
        cls._storage_handler = storage_handler
        cls._learner = learner
        return cls

    def get(self):
        labeled_samples = self._storage_handler.get_labeled_samples()
        if not labeled_samples:
            return  # TODO

        images, labels = zip(*labeled_samples.values())
        y_train = self._transform_labels(labels)
        x_train = self._transform_images(images)

        self._learner.teach(x_train, y_train)
        self._compute_metrics()
        return 200

    def _transform_labels(self, labels: Iterable[str]) -> np.array:
        config = self._storage_handler.get_config()
        allowed_labels = {label: index for index, label in enumerate(config.labels)}
        return np.array([allowed_labels[label] for label in labels])

    def _transform_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        images_array = np.stack(images)
        config = self._storage_handler.get_config()
        if config.transform is None:
            return images_array
        else:
            return config.transform(images_array)

    def _compute_metrics(self) -> None:
        config = self._storage_handler.get_config()
        if config.metrics is None:
            return

        valid_samples = self._storage_handler.get_validation_samples()
        if not valid_samples:
            return

        images, labels = zip(*valid_samples.values())
        y_valid = self._transform_labels(labels)
        x_valid = self._transform_images(images)

        y_pred = self._learner.predict(x_valid)

        for name, metric in config.metrics.items():
            result = metric(y_valid, y_pred)
            self._storage_handler.save_metric(
                name=name,
                value={
                    'metric_value': result,
                    'num_samples': len(self._storage_handler.get_labeled_samples())
                }
            )
            print(self._storage_handler.get_metrics())
