from typing import Sequence

from torch import nn
from torchvision.models.resnet import resnet18


def get_pretrained_model(get_model_func=resnet18,
                         dropout_prob: float = 0.5,
                         num_classes: int = 10,
                         mlp_sizes: Sequence[int] = (512, 512)) -> nn.Module:
    model = get_model_func(pretrained=True)
    in_out_features = zip(mlp_sizes, mlp_sizes[1:])
    fc = nn.Sequential(
        *(_mlp_block(in_f, out_f, dropout_prob) for in_f, out_f in in_out_features),
        nn.Linear(mlp_sizes[-1], num_classes),
    )
    model.fc = fc
    return model

def _mlp_block(in_features: int, out_features: int, dropout_prob: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.Dropout(p=dropout_prob),
    )