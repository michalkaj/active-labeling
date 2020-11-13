import unittest
from unittest.mock import MagicMock

import flask_restful
from flask import Flask

from active_labeling.backend.resources.config import Config


class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._app = Flask(__name__)
        self._api = flask_restful.Api(self._app)

    def test_get_config(self):
        labels = ['1', '2', '3']
        batch_size = 16
        pool_size = 0.1
        resource = Config.instantiate(MagicMock(
            labels=labels, batch_size=batch_size, pool_size=pool_size,
        ))
        self._api.add_resource(resource, '/config', endpoint=Config.endpoint)

        response = self._app.test_client().get('/config')

        self.assertDictEqual(
            {'labels': labels, 'batch_size': batch_size, 'pool_size': pool_size},
            response.json,
        )
