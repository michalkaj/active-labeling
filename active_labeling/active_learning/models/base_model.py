import torch
from torch import nn
from torch.nn.init import kaiming_normal_
from typing import Sequence


class ConvNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 conv_channel_dimensions: Sequence[int] = (3, 32, 64),
                 conv_dropout_prob: float = 0.1,
                 mlp_dimensions: Sequence[int] = (512, 512),
                 mlp_dropout_prob: float = 0.1,
                 pool_every: int = 1,
                 ):
        super().__init__()
        self._feature_extractor = self._create_conv_layers(
            conv_channel_dimensions, conv_dropout_prob, pool_every)
        self._classifier = self._create_mlp_layers(num_classes, mlp_dimensions, mlp_dropout_prob)
        self.init_weights()

    def init_weights(self):
        def init(module: nn.Module):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                kaiming_normal_(module.weight)
        self.apply(init)

    @staticmethod
    def _create_conv_layers(dimensions: Sequence[int], dropout_prob: float,
                            pool_every: int) -> nn.Module:
        layers = []
        for i, (in_channels, out_channels) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(_conv_block(in_channels, out_channels, dropout_prob=dropout_prob))
            if (i + 1) % pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    @staticmethod
    def _create_mlp_layers(num_classes:int, dimensions: Sequence[int], dropout_prob: float) \
            -> nn.Module:
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Dropout(p=dropout_prob))

        layers.append(nn.Linear(dimensions[-1], num_classes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._feature_extractor(x)
        features = torch.flatten(features, 1)
        return self._classifier(features)


def _conv_block(in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1,
                dropout_prob: float = 0.1) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        nn.ReLU(),
        nn.Dropout2d(p=dropout_prob),
    )
