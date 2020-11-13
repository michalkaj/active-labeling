import unittest
from http import HTTPStatus
from unittest.mock import MagicMock

import flask_restful
from flask import Flask

from active_labeling.backend.resources.metrics import Metrics


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self._app = Flask(__name__)
        self._api = flask_restful.Api(self._app)

    def test_get_metric(self):
        metrics = {}
        resource = Metrics.instantiate(MagicMock(), metrics, MagicMock())
        self._api.add_resource(resource, '/metrics', endpoint=Metrics.endpoint)

        response = self._app.test_client().get('/metrics')

        self.assertEqual(HTTPStatus.OK, response.status_code)
