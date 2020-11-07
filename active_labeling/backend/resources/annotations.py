from http import HTTPStatus
from pathlib import Path
from typing import Dict, Any, List

from flask_restful import Resource, reqparse

from active_labeling.active_learning.training.dataset import ActiveDataset
from active_labeling.config import LearningConfig


class Annotations(Resource):
    endpoint = '/annotations'

    def __init__(self):
        super().__init__()
        self._parser = reqparse.RequestParser()
        self._parser.add_argument('samples', type=list, location='json')

    @classmethod
    def instantiate(cls, config: LearningConfig, active_dataset: ActiveDataset,
                    batch_cache: List[Path]):
        cls._config = config
        cls._active_dataset = active_dataset
        cls._batch_cache = batch_cache
        return cls

    def get(self) -> Dict[str, Any]:
        labels = {str(path): label for path, label in self._active_dataset.labels.items()}
        return {
            'labels': list(self._config.labels),
            'annotations': labels,
        }

    def post(self):
        args = self._parser.parse_args()
        samples_json = args['samples']
        annotations = {(self._config.data_root / Path(s['path'])): s['label'] for s in samples_json}
        self._active_dataset.add_labels(annotations)
        self._batch_cache.clear()
        return HTTPStatus.OK
