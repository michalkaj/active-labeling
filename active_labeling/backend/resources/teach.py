from http import HTTPStatus

import pytorch_lightning as pl
from flask_restful import Resource, reqparse
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.metrics import Accuracy
from torch import nn
from torch.utils.data import DataLoader, Dataset

from active_labeling.active_learning.learners.training.dataset import ActiveDataset
from active_labeling.active_learning.learners.training.training_system import TrainingSystem
from active_labeling.backend.loggers import get_logger
from active_labeling.config import ActiveLearningConfig

_LOGGER = get_logger(__name__)


class Teach(Resource):
    endpoint = '/teach'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()

    @classmethod
    def instantiate(cls,
                    config: ActiveLearningConfig,
                    learner: nn.Module,
                    active_dataset: ActiveDataset,
                    valid_dataset: Dataset):
        cls._config = config
        cls._active_dataset = active_dataset
        cls._valid_dataset = valid_dataset
        cls._learner = learner
        cls._training_system = TrainingSystem(learner, metrics={'accuracy': Accuracy()})
        return cls

    def get(self):
        self._learner.init_weights()
        train_loader = DataLoader(self._active_dataset.train(), **self._config.dataloader_kwargs)
        valid_loader = DataLoader(self._valid_dataset, **self._config.dataloader_kwargs)

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
