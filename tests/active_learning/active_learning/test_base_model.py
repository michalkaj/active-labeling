import unittest

from active_labeling.active_learning.models.base_model import ConvNet


class TestConvNet(unittest.TestCase):
    def test_init(self):
        num_classes = 10
        conv_sizes = (32, 64, 128)
        mlp_sizes = (32, 32)

        model = ConvNet(
            num_classes=num_classes,
            conv_channel_dimensions=conv_sizes,
            mlp_dimensions=mlp_sizes
        )

        self.assertEqual(4, len(model._feature_extractor))
        self.assertEqual(3, len(model._classifier))
        self.assertEqual(10, model._classifier[-1].out_features)
