from pathlib import Path

from active_labeling.backend.api import ActiveLearning

if __name__ == '__main__':
    path = Path('/home/michal/projects/thesis/mnist-csv-png/test')
    active_learning = ActiveLearning(path)
    active_learning.run()