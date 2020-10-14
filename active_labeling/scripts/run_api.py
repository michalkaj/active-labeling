import json
from pathlib import Path
from random import shuffle

import numpy as np

from pytorch_lightning.metrics import Accuracy
from active_labeling.active_learning.learners.bayesian_cnn.base_model import ConvNet
from active_labeling.backend.api import ActiveLearning
from active_labeling.config import ActiveLearningConfig
from active_labeling.loading.image_loader import ImageLoader

def transform(arrays):
    arrays = np.transpose(arrays, (0, 3, 1, 2)) / 255.
    return arrays.astype(np.float32)

if __name__ == '__main__':
    config = ActiveLearningConfig(
        labels={'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                'truck'},
        unlabeled_data_path=Path('/media/data/data/cifar/train'),
        labels_file=Path('./train_labels.json'),
        transform=transform,
        validation_data_path=Path('/media/data/data/cifar/test'),
        validation_labels_path=Path('./valid_labels.json'),
        pool_size=0.1,
        metrics={'accuracy': Accuracy()}
    )
    bayesian_cnn = ConvNet(
        num_classes=len(config.labels),
        conv_channel_dimensions=(3, 32, 64, 128),
        conv_dropout_prob=0.1,
        mlp_dimensions=(2048, 128),
        mlp_dropout_prob=0.1,
    )
    active_learning = ActiveLearning(
        learner=bayesian_cnn,
        config=config,
        data_loader=ImageLoader(),
    )
    active_learning.run()

    #
    # js = Path('./train_labels.json')
    # test_data_path = Path('/media/data/data/cifar/train')
    # paths = list(test_data_path.rglob('*.png'))
    # shuffle(paths)
    # di = {'labels': set(), 'annotations': {}}
    #
    # for p in paths[:2000]:
    #     name = str(p.relative_to(test_data_path))
    #     label = p.parent.stem
    #     di['labels'].add(label)
    #     di['annotations'][name] = label
    #
    # di['labels'] = list(di['labels'])
    # with js.open('wt') as f:
    #     json.dump(di, f, indent=4)
    #
    #
    # di = {'labels': set(), 'annotations': {}}
    # js = Path('./valid_labels.json')
    # for p in paths[2000:4000]:
    #     name = str(p.relative_to(test_data_path))
    #     label = p.parent.stem
    #     di['labels'].add(label)
    #     di['annotations'][name] = label
    #
    # di['labels'] = list(di['labels'])
    # with js.open('wt') as f:
    #     json.dump(di, f, indent=4)
    #
