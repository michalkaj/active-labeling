from pathlib import Path

import torchvision.transforms as tvt
from ordered_set import OrderedSet
from torch import nn
from torchvision.transforms import transforms

from active_labeling.active_learning.dataset import ActiveDataset, FileDataset
from active_labeling.active_learning.modeling.prediction import predict
from active_labeling.active_learning.modeling.wrappers import \
    ActiveWrapper
from active_labeling.active_learning.sampling.acquisition.active import UncertaintyQuery
from active_labeling.active_learning.sampling.sampler import Sampler
from active_labeling.backend.api import ActiveLearningAPI
from active_labeling.backend.file_utils import load_json_file, discover_paths
from active_labeling.config import LearningConfig


def get_dataset(data_path: Path, labels_path: Path, all_labels: OrderedSet[str]) -> ActiveDataset:
    label_to_ind = {l: i for i, l in enumerate(all_labels)}
    image_paths = discover_paths(data_path, {'png', 'jpg'})
    labels_json = load_json_file(labels_path)
    annotations = {(data_path / path): label for path, label in labels_json['annotations'].items()}

    transform = tvt.Compose([
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return FileDataset(
        image_paths,
        annotations,
        train_transform=transform,
        target_transform=lambda l: label_to_ind[l],
    )

if __name__ == '__main__':
    data_path = Path('/media/data/data/cifar/train')

    config = LearningConfig(
        data_root=data_path,
        labels=OrderedSet(('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                           'horse', 'ship', 'truck')),
        early_stopping_metric='val_accuracy',
        pool_size=0.2,
        dataloader_kwargs={'batch_size': 16},
        initial_training_set_size=220,
        epochs=2,
    )

    cnn = ActiveWrapper(nn.Sequential(
        nn.Conv2d(3, 32, 3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3),
        nn.MaxPool2d(2),
        nn.Flatten(start_dim=1),
        nn.Linear(1152, 10)),
    )
    sampler = Sampler(
        query=UncertaintyQuery(
            predict_func=lambda dataset: predict(cnn, dataset, config.dataloader_kwargs),
            shuffle_prob=0.,
        ),
        pool_size_reduction=config.pool_size,
    )

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_dataset = get_dataset(data_path, Path('initial_labels2.json'), config.labels)
    test_dataset = get_dataset(
        Path('/media/data/data/cifar/train'),
        Path('valid_labels.json'), config.labels
    )

    active_learning = ActiveLearningAPI(
        learner=cnn,
        sampler=sampler,
        active_dataset=train_dataset,
        valid_dataset=test_dataset,
        config=config,
    )
    active_learning.run()
