import json
from pathlib import Path
from random import shuffle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from active_labeling.backend.api import ActiveLearning
from active_labeling.config import ActiveLearningConfig
from active_labeling.loading.image_loader import ImageLoader

if __name__ == '__main__':
    config = ActiveLearningConfig(
        server_url='http://127.0.0.1:5000/',
        labels={'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                'truck'},
        unlabeled_data_path=Path('/media/data/data/cifar/train'),
        labels_file=Path('./train_labels.json'),
        transform=lambda arrays: arrays.reshape(len(arrays), -1),
        validation_data_path=Path('/media/data/data/cifar/test'),
        validation_labels_file_path=Path('./valid_labels.json'),
        pool_size=0.1,
        metrics={
            'accuracy': accuracy_score
        }
    )
    active_learning = ActiveLearning(
        estimator=RandomForestClassifier(),
        config=config,
        data_loader=ImageLoader(),
    )
    active_learning.run()

    #
    # js = Path('./train_labels.json')
    # unlabeled_data_path = Path('/media/data/data/cifar/train')
    # paths = list(unlabeled_data_path.rglob('*.png'))
    # shuffle(paths)
    # di = {'labels': set(), 'annotations': {}}
    # for p in paths[:1000]:
    #     name = str(p.relative_to(unlabeled_data_path))
    #     label = p.parent.stem
    #     di['labels'].add(label)
    #     di['annotations'][name] = label
    #
    # di['labels'] = list(di['labels'])
    # with js.open('wt') as f:
    #     json.dump(di, f, indent=4)

