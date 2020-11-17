from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
from flask_restful import Resource, reqparse
from pytorch_lightning.metrics import Metric
from torch.utils.data import Dataset, DataLoader

from active_labeling.active_learning.dataset import ActiveDataset
from active_labeling.active_learning.modeling.training_system import TrainingSystem
from active_labeling.active_learning.modeling.wrappers import \
    MonteCarloWrapper
from active_labeling.active_learning.sampling.acquisition.random import RandomQuery
from active_labeling.active_learning.sampling.sampler import Sampler
from active_labeling.backend.file_utils import path_to_base64
from active_labeling.config import LearningConfig
from active_labeling.settings import DEFAULT_BATCH_SIZE, DEFAULT_POOL_SIZE


class Query(Resource):
    endpoint = '/query'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()
        self._parser.add_argument('batch_size', type=int, default=DEFAULT_BATCH_SIZE)
        self._parser.add_argument('pool_size', type=float, default=DEFAULT_POOL_SIZE)
        self._paths_to_query_cache = None

    @classmethod
    def instantiate(cls,
                    config: LearningConfig,
                    learner: MonteCarloWrapper,
                    sampler: Sampler,
                    active_dataset: ActiveDataset,
                    valid_dataset: Dataset,
                    metrics: Dict[str, List[Dict]],
                    batch_cache: List[Path],
                    ):
        cls._config = config
        cls._learner = learner
        cls._active_sampler = sampler
        cls._random_sampler = Sampler(RandomQuery())
        cls._train_dataset = active_dataset
        cls._valid_dataset = valid_dataset
        cls._metrics = metrics
        cls._trainer = pl.Trainer(
            # callbacks=[EarlyStopping(
            #     monitor=config.early_stopping_metric,
            #     min_delta=0.01,
            # )],
            **config.trainer_kwargs,
            weights_summary=None,
            num_sanity_val_steps=0,
            max_epochs=config.epochs,
            check_val_every_n_epoch=config.epochs,
        )
        cls._batch_cache = batch_cache
        return cls

    def get(self):
        args = self._parser.parse_args()
        batch_size = args['batch_size']

        if self._batch_cache:
            paths_to_query = self._batch_cache
        else:
            paths_to_query = self._sample_paths(batch_size)
            self._batch_cache.extend(paths_to_query)

        return {'samples': [self._prepare_sample(path) for path in paths_to_query]}

    def _sample_paths(self, batch_size: int):
        if len(self._train_dataset.train()) < self._config.initial_training_set_size:
            return self._random_sampler.sample(self._train_dataset.evaluate(), batch_size)
        else:
            self._teach()
            return self._active_sampler.sample(self._train_dataset.evaluate(), batch_size)

    def _teach(self):
        train_loader = DataLoader(
            self._train_dataset.train(), **self._config.dataloader_kwargs, shuffle=True)
        valid_loader = DataLoader(self._valid_dataset, **self._config.dataloader_kwargs)

        self._learner.reset_weights()
        training_system = TrainingSystem(self._learner)
        self._trainer.fit(
            model=training_system,
            train_dataloader=train_loader,
            val_dataloaders=valid_loader,
        )

        self._save_metrics(training_system.metrics)

    def _save_metrics(self, metrics: Dict[str, Metric]) -> None:
        for name, metric in metrics.items():
            self._metrics.setdefault(name, []).append({
                'metric_value': metric.compute().item(),
                'num_samples': len(self._train_dataset)
            })
            print(self._metrics)

    def _prepare_sample(self, path: Path) -> Dict[str, str]:
        return {
            'path': str(path),
            'name': str(Path(path).stem),
            'base64_file': path_to_base64(path)
        }
