from pathlib import Path

import torchvision.transforms as tvt
from ordered_set import OrderedSet
from pytorch_lightning.metrics import Accuracy

from active_labeling.active_learning.models.base_model import ConvNet
from active_labeling.active_learning.models.monte_carlo_approximation import \
    MonteCarloWrapper
from active_labeling.active_learning.training.dataset import ActiveDataset
from active_labeling.backend.api import ActiveLearningAPI
from active_labeling.backend.file_utils import load_json_file, discover_paths
from active_labeling.config import LearningConfig


def get_dataset(data_path: Path, labels_path: Path, all_labels: OrderedSet[str]) -> ActiveDataset:
    image_paths = discover_paths(data_path, {'png', 'jpg'})
    labels_json = load_json_file(labels_path)
    annotations = {(data_path / path): label for path, label in labels_json['annotations'].items()}

    transform = tvt.Compose([
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return ActiveDataset(
        image_paths,
        annotations,
        all_labels,
        train=True,
        transform=transform,
    )


if __name__ == '__main__':
    data_path = Path('/media/data/data/cifar/train')

    config = LearningConfig(
        data_root=data_path,
        labels=OrderedSet(('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                           'horse', 'ship', 'truck')),
        metrics={'accuracy': Accuracy()},
        early_stopping_metric='accuracy',
        pool_size=0.1,
        dataloader_kwargs={'batch_size': 16},
        initial_training_set_size=230,
    )

    cnn = ConvNet(
        num_classes=len(config.labels),
        conv_channel_dimensions=(3, 32, 64, 128),
        conv_dropout_prob=0.1,
        mlp_dimensions=(2048, 128),
        mlp_dropout_prob=0.1,
    )
    bayesian_cnn = MonteCarloWrapper(cnn, config.bayesian_sample_size)

    active_dataset = get_dataset(data_path, Path('initial_labels2.json'), config.labels)
    valid_dataset = get_dataset(Path('/media/data/data/cifar/train'),
                                Path('valid_labels.json'), config.labels)

    active_learning = ActiveLearningAPI(
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
    # test_data_path = Path('/media/data/data/cifar/test')
    # paths = list(test_data_path.rglob('*.png'))
    # shuffle(paths)
    # di = {'labels': set(), 'annotations': {}}
    # js = Path('./valid_labels.json')
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
