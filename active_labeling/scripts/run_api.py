from pathlib import Path

import torchvision.transforms as tvt
from ordered_set import OrderedSet
from pytorch_lightning.metrics import Accuracy

from active_labeling.active_learning.models.monte_carlo_wrapper import \
    MonteCarloWrapper
from active_labeling.active_learning.models.pretrained import get_pretrained_model
from active_labeling.active_learning.training.dataset import ActiveDataset, FileDataset
from active_labeling.backend.api import ActiveLearningAPI
from active_labeling.backend.file_utils import load_json_file, discover_paths
from active_labeling.config import LearningConfig


def get_dataset(data_path: Path, labels_path: Path, all_labels: OrderedSet[str]) -> ActiveDataset:
    label_to_ind = {l: i for i, l in enumerate(all_labels)}
    image_paths = discover_paths(data_path, {'png', 'jpg'})[:100]
    labels_json = load_json_file(labels_path)
    annotations = {(data_path / path): label for path, label in labels_json['annotations'].items()}
    annotations = {}

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
        metrics={'accuracy': Accuracy()},
        early_stopping_metric='accuracy',
        pool_size=0.1,
        dataloader_kwargs={'batch_size': 16},
        initial_training_set_size=60,
    )

    cnn = get_pretrained_model()
    bayesian_cnn = MonteCarloWrapper(cnn, config.bayesian_sample_size)
    bayesian_cnn.reset_weights(True)

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
