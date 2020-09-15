from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from active_labeling.backend.api import ActiveLearning
from active_labeling.loading.sample import Sample
from active_labeling.settings import DEFAULT_ESTIMATOR

def transform_images(samples: Iterable[Sample]) -> np.ndarray:
    return np.stack(
        [np.array(Image.open(sample.path)).flatten() for sample in samples]
    )


if __name__ == '__main__':
    path = Path('/home/michal/projects/thesis/mnist-csv-png/test')
    active_learning = ActiveLearning(
        path,
        DEFAULT_ESTIMATOR(),
        transform_images,
        config_path=Path('../active-config.json'))
    active_learning.run()