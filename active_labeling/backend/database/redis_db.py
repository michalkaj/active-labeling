import json
from numbers import Number
from typing import Dict, Any, Iterable, Tuple, List

import numpy as np
from redis import Redis

from active_labeling.backend.database.base import BaseDatabaseConnection
from active_labeling.backend.utils import decode
from active_labeling.loading.sample import Sample

NOT_ANNOTATED = 'to_annotate'
ANNOTATED = 'annotated'
PATH_TO_INDEX = 'path_to_index:'
SAMPLE_PREFIX = 'sample:'
METRIC_PREFIX = 'metric'
METRICS = 'metrics'


class RedisConnection(BaseDatabaseConnection):
    def __init__(self, **redis_kwargs):
        self._redis = Redis(**redis_kwargs)
        self._redis.delete(ANNOTATED)
        self._redis.delete(NOT_ANNOTATED)

    def save_samples(self, samples: Iterable[Sample]):
        self._redis.delete(ANNOTATED)
        for index, sample in enumerate(samples):
            path = str(sample.path)
            self._redis.sadd(NOT_ANNOTATED, path)
            self._redis.set(SAMPLE_PREFIX + path,
                            json.dumps({'index': index, 'sample': sample.to_dict()}))

    def _get_samples(self, paths: Iterable[bytes]) -> Iterable[Tuple[int, Sample]]:
        for path in paths:
            path = path.decode('utf-8')
            index_sample_bytes = self._redis.get(SAMPLE_PREFIX + path)
            index_sample_dict = json.loads(index_sample_bytes)
            index, sample_dict = index_sample_dict['index'], index_sample_dict['sample']
            yield index, Sample.from_dict(sample_dict)

    def get_annotated_samples(self) -> Tuple[List[int], List[Sample]]:
        annotated = self._redis.smembers(ANNOTATED)
        indices, samples = zip(*self._get_samples(annotated))
        return list(indices), list(samples)

    def get_not_annotated_samples(self) -> Tuple[List[int], List[Sample]]:
        not_annotated = self._redis.smembers(NOT_ANNOTATED)
        indices, samples = zip(*self._get_samples(not_annotated))
        return list(indices), list(samples)

    def annotate_samples(self, samples: Iterable[Sample]):
        for sample in samples:
            path = str(sample.path)
            self._redis.sadd(ANNOTATED, path)
            self._redis.srem(NOT_ANNOTATED, path)
            index_sample_bytes = self._redis.get(SAMPLE_PREFIX + path)
            index_sample_dict = json.loads(index_sample_bytes)
            index, _ = index_sample_dict['index'], index_sample_dict['sample']
            self._redis.set(SAMPLE_PREFIX + path,
                            json.dumps({'index': index, 'sample': sample.to_dict()}))

    def get_indices(self, sample_names: Iterable[str]) -> np.ndarray:
        return np.fromiter(
            (self._redis.get(PATH_TO_INDEX + name) for name in sample_names),
            dtype=np.uint32
        )

    def get_config(self) -> Dict[str, Any]:
        config_string = self._redis.get('config').decode('utf-8')
        return json.loads(config_string)

    def save_config(self, config: Dict[str, Any]):
        config_json = json.dumps(config)
        self._redis.set('config', config_json)

    def save_metric(self, metric_name: str, metric_value: Number, num_samples: int):
        self._redis.sadd(METRICS, metric_name)
        self._redis.lpush(f'{METRIC_PREFIX}:{metric_name}:num_samples', num_samples)
        self._redis.lpush(f'{METRIC_PREFIX}:{metric_name}:metric_value', metric_value)

    def get_metrics(self) -> Iterable[Dict]:
        metrics = self._redis.smembers(METRICS)
        for metric_name in metrics:
            metric_name = decode(metric_name)
            num_samples = self._redis.lrange(
                f'{METRIC_PREFIX}:{metric_name}:num_samples', 0, -1)
            metric_value = self._redis.lrange(
                f'{METRIC_PREFIX}:{metric_name}:metric_value', 0, -1)
            yield {
                'metric_name': metric_name,
                'num_samples': list(reversed(map(int, num_samples))),
                'metric_value': list(reversed(map(float, metric_value)))
            }