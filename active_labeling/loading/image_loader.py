from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

from active_labeling.loading.base_loader import BaseLoader
from active_labeling.loading.sample import Sample, IMAGE_EXTENSIONS


class ImageLoader(BaseLoader):
    def __init__(self, recursive: bool = False):
        super().__init__(IMAGE_EXTENSIONS, recursive)

    def _transform(self, samples: Iterable[Sample]) -> Tuple[np.ndarray, List[Sample]]:
        samples_list = list(samples)
        image_array = np.array(
            [np.array(Image.open(sample.path)) for sample in samples_list],
            dtype=np.uint8
        )
        return image_array, samples_list





