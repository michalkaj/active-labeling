import abc
from numbers import Number
from typing import Dict, Any, Iterable, Tuple, List

from active_labeling.loading.sample import Sample


class BaseDatabaseConnection(abc.ABC):
    @abc.abstractmethod
    def get_annotated_samples(self) -> Tuple[List[int], List[Sample]]:
        pass

    @abc.abstractmethod
    def get_not_annotated_samples(self) -> Tuple[List[int], List[Sample]]:
        pass

    @abc.abstractmethod
    def save_samples(self, samples: List[Sample]):
        pass

    @abc.abstractmethod
    def annotate_samples(self, samples: Iterable[Sample]):
        pass

    @abc.abstractmethod
    def get_indices(self, sample_names: List[str]):
        pass

    @abc.abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def save_config(self, config: Dict[str, Any]):
        pass

    @abc.abstractmethod
    def save_metric(self, metric_name: str, metric_value: Number, num_samples: int):
        pass

    @abc.abstractmethod
    def get_metrics(self) -> List[Dict]:
        pass