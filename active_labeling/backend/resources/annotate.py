from pathlib import Path
from typing import Dict

from flask_restful import Resource, reqparse

from active_labeling.active_learning.learners.training.dataset import ActiveDataset
from active_labeling.backend.loggers import get_logger
from active_labeling.config import ActiveLearningConfig

_LOGGER = get_logger(__name__)


class Annotate(Resource):
    endpoint = '/annotate'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()
        self._parser.add_argument('samples', type=list, location='json')

    @classmethod
    def instantiate(cls, config: ActiveLearningConfig, active_dataset: ActiveDataset):
        cls._config = config
        cls._active_dataset = active_dataset
        return cls

    def post(self):
        args = self._parser.parse_args()
        samples_json = args['samples']
        annotations = {Path(s['path']): s['label'] for s in samples_json}
        self._active_dataset.add_labels(annotations)
        return 200
