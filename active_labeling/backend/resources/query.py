from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
from flask_restful import Resource, reqparse
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.utils.data import Dataset, DataLoader

from active_labeling.active_learning.models.monte_carlo_approximation import \
    MonteCarloWrapper
from active_labeling.active_learning.training import ActiveDataset
from active_labeling.active_learning.training.training_system import TrainingSystem
from active_labeling.active_learning.sampling.acquisition.bald import BALD
from active_labeling.active_learning.sampling.active_sampler import ActiveSampler
from active_labeling.backend.file_utils import path_to_base64
from active_labeling.backend.loggers import get_logger
from active_labeling.config import ActiveLearningConfig
from active_labeling.settings import DEFAULT_BATCH_SIZE, DEFAULT_POOL_SIZE

_LOGGER = get_logger(__name__)


class Query(Resource):
    endpoint = '/query'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()
        self._parser.add_argument('batch_size', type=int, default=DEFAULT_BATCH_SIZE)
        self._parser.add_argument('pool_size', type=float, default=DEFAULT_POOL_SIZE)

    @classmethod
    def instantiate(cls,
                    config: ActiveLearningConfig,
                    learner: MonteCarloWrapper,
                    active_dataset: ActiveDataset,
                    valid_dataset: Dataset,
                    metrics: Dict[str, List[Dict]]
                    ):
        cls._config = config
        cls._learner = learner
        cls._sampler = ActiveSampler(learner, BALD, config)
        cls._active_dataset = active_dataset
        cls._valid_dataset = valid_dataset
        cls._training_system = TrainingSystem(learner, metrics=config.metrics)
        cls._metrics = metrics
        return cls

    def get(self):
        self._teach()
        args = self._parser.parse_args()
        batch_size = args['batch_size']
        paths_to_query = self._sampler.sample(self._active_dataset, batch_size)
        return {'samples': [self._prepare_sample(path) for path in paths_to_query]}

    def _teach(self):
        self._learner.reset_weights()
        train_loader = DataLoader(self._active_dataset.train(), **self._config.dataloader_kwargs,
                                  shuffle=True)
        valid_loader = DataLoader(self._valid_dataset, **self._config.dataloader_kwargs)

        trainer = pl.Trainer(
            callbacks=[EarlyStopping(
                monitor=self._config.early_stopping_metric,
                min_delta=0.05,
            )],
            max_epochs=3,
            **self._config.trainer_kwargs,
        )
        trainer.fit(
            model=self._training_system,
            train_dataloader=train_loader,
            val_dataloaders=valid_loader,
        )

        self._save_metrics()

    def _save_metrics(self) -> None:
        for name, metric in self._training_system.metrics.items():
            self._metrics.setdefault(name, []).append({
                'metric_value': metric.compute().item(),
                'num_samples': len(self._active_dataset)
            })
            print(self._metrics)

    def _prepare_sample(self, path: Path) -> Dict[str, str]:
        return {
            'path': str(path),
            'name': str(Path(path).stem),
            'base64_file': path_to_base64(path)
        }
