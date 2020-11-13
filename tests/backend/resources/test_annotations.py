import json
import unittest
from http import HTTPStatus
from unittest.mock import Mock, MagicMock

import flask_restful
from flask import Flask

from active_labeling.backend.resources.annotations import Annotations


class TestAnnotations(unittest.TestCase):
    def setUp(self) -> None:
        self._app = Flask(__name__)
        self._api = flask_restful.Api(self._app)

    def test_add_annotations(self):
        resource = Annotations.instantiate(MagicMock(), Mock(), Mock())
        self._api.add_resource(resource, '/annotations', endpoint=Annotations.endpoint)
        annotations = json.dumps({'samples': [
            {'path': 'test_name', 'label': 'test_label'},
            {'path': 'test_name2', 'label': 'test_label'},
        ]})

        response = self._app.test_client().post(
                '/annotations',
                headers={"Content-Type": "application/json"},
                data=annotations
            )

        self.assertEqual(HTTPStatus.OK, response.status_code)

    def test_get_annotations(self):
        labels = ['label1', 'label2']
        annots = {'test': 'label'}
        active_dataset = Mock(labels=annots)
        resource = Annotations.instantiate(MagicMock(labels=labels), active_dataset, Mock())
        self._api.add_resource(resource, '/annotations', endpoint=Annotations.endpoint)

        response = self._app.test_client().get('/annotations')

        self.assertEqual(HTTPStatus.OK, response.status_code)
        self.assertDictEqual(
            {'annotations': annots, 'labels': labels},
            response.json
        )
