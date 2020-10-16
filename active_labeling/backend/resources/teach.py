from http import HTTPStatus
from typing import Iterable, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from flask_restful import Resource, reqparse
from pytorch_lightning.metrics import Accuracy
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from active_labeling.active_learning.learners.training.training_system import TrainingSystem
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
                    learner: nn.Module):
        cls._storage_handler = storage_handler
        cls._config = storage_handler.get_config()
        cls._learner = learner
        cls._training_system = TrainingSystem(learner, metrics={'accuracy': Accuracy()})
        return cls

    def get(self):
        labeled_samples = self._storage_handler.get_labeled_samples()
        valid_samples = self._storage_handler.get_validation_samples()
        if not labeled_samples or not valid_samples:
            raise ValueError("Both labeled and validation samples are required")

        train_loader = self._create_loader(labeled_samples.values())
        valid_loader = self._create_loader(valid_samples.values())

        self._learner.init_weights()
        trainer = pl.Trainer(
            early_stop_callback=EarlyStopping(
                monitor=self._config.early_stopping_metric,
                min_delta=0.05,
            ),
            **self._config.trainer_kwargs,
        )
        trainer.fit(
            model=self._training_system,
            train_dataloader=train_loader,
            val_dataloaders=valid_loader,
        )

        self._save_metrics()
        return HTTPStatus.OK

    def _transform_labels(self, labels: Iterable[str]) -> np.array:
        allowed_labels = {label: index for index, label in enumerate(self._config.labels)}
        return np.array([allowed_labels[label] for label in labels])

    def _transform_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        images_array = np.stack(images)
        if self._config.transform is None:
            return images_array
        else:
            return self._config.transform(images_array)

    def _create_loader(self, samples: Iterable[Tuple[np.ndarray, str]]):
        images, labels = zip(*samples)
        x = self._transform_images(images)
        y = self._transform_labels(labels)
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        return DataLoader(
            dataset,
            **self._config.dataloader_kwargs
        )

    def _save_metrics(self) -> None:
        for name, metric in self._training_system.metrics.items():
            self._storage_handler.save_metric(
                name=name,
                value={
                    'metric_value': metric.compute().item(),
                    'num_samples': len(self._storage_handler.get_labeled_samples())
                }
            )
            print(self._storage_handler.get_metrics())
