from pathlib import Path

import numpy as np
from PIL import Image

from active_labeling.loading.base_loader import BaseDataLoader

IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}


class ImageLoader(BaseDataLoader[np.ndarray]):
    def __init__(self, recurive: bool = True):
        super().__init__(IMAGE_EXTENSIONS, recurive)

    def _load_file(self, path: Path) -> np.ndarray:
        image = Image.open(path)
        return np.array(image)
