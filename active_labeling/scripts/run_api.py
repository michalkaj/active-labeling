from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from active_labeling.backend.api import ActiveLearning
from active_labeling.config import ActiveLearningConfig
from active_labeling.loading.image_loader import ImageLoader

if __name__ == '__main__':
    active_learning = ActiveLearning(
        estimator=RandomForestClassifier(),
        config=ActiveLearningConfig(
            server_url='http://127.0.0.1:5000/',
            labels={'cat', 'dog'},
            unlabeled_data_path=Path('/media/data/data/cifar/test')
    ),
        data_loader=ImageLoader(),
    )
    active_learning.run()