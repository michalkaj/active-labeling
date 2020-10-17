import json
from pathlib import Path
from random import shuffle

import torchvision.transforms as tvt
from pytorch_lightning.metrics import Accuracy

from active_labeling.active_learning.learners.bayesian_cnn.base_model import ConvNet
from active_labeling.active_learning.learners.training.dataset import ActiveDataset
from active_labeling.backend.api import ActiveLearning
from active_labeling.backend.file_utils import load_json_file, discover_paths
from active_labeling.config import ActiveLearningConfig


def get_dataset(data_path: Path, labels_path: Path) -> ActiveDataset:
    image_paths = discover_paths(data_path, {'png', 'jpg'})
    labels_json = load_json_file(labels_path)

    transform = tvt.Compose([
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return ActiveDataset(
        image_paths,
        labels_json['annotations'],
        train=True,
        transform=transform,
    )


if __name__ == '__main__':
    config = ActiveLearningConfig(
        labels={'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                'truck'},
        metrics={'accuracy': Accuracy()},
        early_stopping_metric='accuracy',
        pool_size=0.1,
    )

    bayesian_cnn = ConvNet(
        num_classes=len(config.labels),
        conv_channel_dimensions=(3, 32, 64, 128),
        conv_dropout_prob=0.1,
        mlp_dimensions=(2048, 128),
        mlp_dropout_prob=0.1,
    )

    active_dataset = get_dataset(Path('/media/data/data/cifar/train'), Path('train_labels.json'))
    valid_dataset = get_dataset(Path('/media/data/data/cifar/train'), Path('valid_labels.json'))

    active_learning = ActiveLearning(
        learner=bayesian_cnn,
        active_dataset=active_dataset,
        valid_dataset=valid_dataset,
        config=config,
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
    #     name = str(p)
    #     label = p.parent.stem
    #     di['labels'].add(label)
    #     di['annotations'][name] = label
    #
    # di['labels'] = list(di['labels'])
    # with js.open('wt') as f:
    #     json.dump(di, f, indent=4)
    #
    #
    # test_data_path = Path('/media/data/data/cifar/test')
    # paths = list(test_data_path.rglob('*.png'))
    # shuffle(paths)
    # di = {'labels': set(), 'annotations': {}}
    # js = Path('./valid_labels.json')
    # for p in paths[:2000]:
    #     name = str(p)
    #     label = p.parent.stem
    #     di['labels'].add(label)
    #     di['annotations'][name] = label
    #
    # di['labels'] = list(di['labels'])
    # with js.open('wt') as f:
    #     json.dump(di, f, indent=4)
    # #
