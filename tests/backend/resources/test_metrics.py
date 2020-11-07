import unittest
from http import HTTPStatus
from unittest.mock import Mock, MagicMock

from active_labeling.backend.api import ActiveLearningAPI


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        label = 'test_label'
        cls._app = ActiveLearningAPI(
            learner=Mock(),
            active_dataset=MagicMock(labels={'test_path': label}),
            valid_dataset=Mock(),
            config=MagicMock(labels={label}),
        )._app.test_client()

    def test_get_config(self):
        response = self._app.get('/metrics')

        self.assertEqual(HTTPStatus.OK, response.status_code)
