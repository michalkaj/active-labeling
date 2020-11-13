import unittest

from torch import nn
from torchvision.models import resnet18

from active_labeling.active_learning.models.pretrained import get_pretrained_model


class TestGetPretrained(unittest.TestCase):
    def test_get_pretrained_model(self):
        model = get_pretrained_model(
            get_model_func=resnet18,
            mlp_sizes=(512, 256, 128),
        )

        def _count_dropouts(layer, count):
            if isinstance(layer, nn.Dropout):
                count[0] += 1
        count = [0]
        model.apply(lambda m: _count_dropouts(m, count))

        self.assertEqual(2, count[0])
