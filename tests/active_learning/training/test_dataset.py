import math
import unittest
from unittest.mock import Mock

from ordered_set import OrderedSet

from active_labeling.active_learning.training.dataset import ActiveDataset, Reducer


class TestActiveDataset(unittest.TestCase):
    def test_add_labels(self):
        paths = [Mock() for _ in range(10)]
        labels = {path: Mock() for path in paths[:5]}
        dataset = ActiveDataset(paths, labels, OrderedSet(labels.values()))
        new_labels = {paths[-1]: Mock(), paths[6]: Mock()}

        dataset.add_labels(new_labels)

        self.assertDictEqual({**labels, **new_labels}, dataset.labels)

    def test_add_existing_labels(self):
        paths = [Mock() for _ in range(10)]
        labels = {path: Mock() for path in paths[:5]}
        dataset = ActiveDataset(paths, labels, OrderedSet(labels.values()))
        new_labels = {paths[0]: Mock(), paths[6]: Mock()}

        with self.assertRaises(ValueError):
            dataset.add_labels(new_labels)

    def test_lengths(self):
        paths = [Mock() for _ in range(10)]
        labels = {path: Mock() for path in paths[:3]}
        dataset = ActiveDataset(paths, labels, OrderedSet(labels.values()))

        self.assertEqual(3, len(dataset.train()))
        self.assertEqual(7, len(dataset.evaluate()))


class TestReducer(unittest.TestCase):
    def test_reduce(self):
        paths = [Mock() for _ in range(1000)]
        labels = {path: Mock() for path in paths[:5]}
        dataset = ActiveDataset(paths, labels, OrderedSet(labels.values()))

        with Reducer(dataset, 0.1) as test_dataset:
            self.assertTrue(math.isclose(100, len(test_dataset.evaluate()), abs_tol=10))

        self.assertEqual(995, len(test_dataset.evaluate()))
