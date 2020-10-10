from sklearn.base import BaseEstimator


class ActiveEstimator:
    def __init__(self, estimator: BaseEstimator):
        self._estimator = estimator
