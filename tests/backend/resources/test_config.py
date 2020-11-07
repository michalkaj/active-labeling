import unittest
from unittest.mock import Mock, MagicMock

from active_labeling.backend.api import ActiveLearningAPI


class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._config = MagicMock()
        cls._app = ActiveLearningAPI(
            learner=Mock(),
            active_dataset=Mock(),
            valid_dataset=Mock(),
            config=cls._config,
        )._app.test_client()

    def test_get_config(self):
        self.assertTrue(True)  # TODO
