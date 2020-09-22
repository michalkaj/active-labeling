from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

from active_labeling.backend.api import ActiveLearning
from active_labeling.loading.base_loader import SampleLoader


def transform_images(paths: Iterable[Path]) -> np.ndarray:
    return np.stack(
        [np.array(Image.open(path)).flatten() for path in paths]
    )


if __name__ == '__main__':
    path = Path('/media/data/data/cifar/test')
    loader = SampleLoader(('png', 'jpg', 'jpeg'), recursive=True)
    unlabeled_data = list(loader.load(path))

    active_learning = ActiveLearning(
        unlabeled_data,
        RandomForestClassifier(),
        transform_images,
        config_path=Path('../active-config.json'))

    active_learning.run()