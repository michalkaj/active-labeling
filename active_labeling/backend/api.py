from flask import Flask
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
from flask_restful import Api
from torch import nn
from torch.utils.data import Dataset

from active_labeling.active_learning.training.dataset import ActiveDataset
from active_labeling.backend.resources.annotations import Annotations
from active_labeling.backend.resources.config import Config
from active_labeling.backend.resources.metrics import Metrics
from active_labeling.backend.resources.query import Query
from active_labeling.config import LearningConfig


class ActiveLearningAPI:
    def __init__(self,
                 learner: nn.Module,
                 active_dataset: ActiveDataset,
                 valid_dataset: Dataset,
                 config: LearningConfig):
        self._app = Flask(__name__)
        self._api = Api(self._app)
        CORS(self._app)

        self._init_resources(config, learner, active_dataset, valid_dataset)

    def _init_resources(self,
                        config: LearningConfig,
                        learner: nn.Module,
                        active_dataset: ActiveDataset,
                        valid_dataset: Dataset) -> None:
        metrics = {}
        batch_cache = []
        resources = (
            Query.instantiate(config, learner, active_dataset, valid_dataset, metrics, batch_cache),
            Config.instantiate(config),
            Metrics.instantiate(config, metrics, active_dataset),
            Annotations.instantiate(config, active_dataset, batch_cache),
        )

        for resource in resources:
            self._api.add_resource(resource, resource.endpoint)

    def run(self, ngrok=False) -> None:
        if ngrok:
            run_with_ngrok(self._app)
        self._app.run()
