import unittest
from unittest.mock import Mock, MagicMock

from active_labeling.active_learning.training.training_system import TrainingSystem


class TestTrainingSystem(unittest.TestCase):
    def test_steps(self):
        model = Mock()
        metrics = MagicMock()
        loss = MagicMock()
        training_system = TrainingSystem(model, metrics=metrics, loss=loss)

        training_system.training_step(MagicMock(), Mock())
        training_system.validation_step(MagicMock(), Mock())
        training_system.validation_epoch_end(MagicMock())
