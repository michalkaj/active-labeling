from pathlib import Path

from active_labeling.backend.api import ActiveLearning
from active_labeling.settings import DEFAULT_ESTIMATOR

if __name__ == '__main__':
    path = Path('/home/michal/projects/thesis/mnist-csv-png/test')
    active_learning = ActiveLearning(path, DEFAULT_ESTIMATOR())
    active_learning.run()