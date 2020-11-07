import json
import unittest
from http import HTTPStatus
from unittest.mock import Mock, MagicMock

from active_labeling.backend.api import ActiveLearningAPI


class TestAnnotations(unittest.TestCase):
    def setUp(self) -> None:
        self.tearDown()

        active_dataset = MagicMock(labels={'test_path': 'test_label'})
        self._app = ActiveLearningAPI(
            learner=Mock(),
            active_dataset=active_dataset,
            valid_dataset=Mock(),
            config=MagicMock(labels={'test_label'}),
        )._app.test_client()

    def test_add_annotations(self):

        annotations = json.dumps({'samples': [
            {'path': 'test_name', 'label': 'test_label'},
            {'path': 'test_name2', 'label': 'test_label'},
        ]})

        response = self._app.post(
            '/annotations',
            headers={"Content-Type": "application/json"},
            data=annotations
        )

        self.assertEqual(HTTPStatus.OK, response.status_code)

    def test_get_annotations(self):
        response = self._app.get('/annotations')

        self.assertEqual(HTTPStatus.OK, response.status_code)
        self.assertDictEqual(
            {'labels': ['test_label'], 'annotations': {'test_path': 'test_label'}},
            response.json
        )

        self._app = None
