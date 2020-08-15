from pathlib import Path

from flask import Flask
from flask_cors import CORS
from flask_restful import Api
from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.base import BaseEstimator

from active_labeling.backend.resources import Query
from active_labeling.loading.image_loader import ImageLoader


class ActiveLearning:
    def __init__(self, filedir_path: Path, estimator: BaseEstimator):
        self._app = Flask(__name__)
        CORS(self._app)
        self._api = Api(self._app)

        self._learner = ActiveLearner(
            estimator=estimator,
            query_strategy=uncertainty_sampling
        )

        loader = ImageLoader(recursive=True)
        data, samples = loader.load(filedir_path)
        query = Query.instantiate(data, samples)
        self._api.add_resource(query, query.endpoint)

    def run(self) -> None:
        self._app.run()
