import unittest
from unittest.mock import Mock, MagicMock, patch

from active_labeling.backend.api import ActiveLearningAPI


class TestAPI(unittest.TestCase):
    def test_run(self):
        api = ActiveLearningAPI(
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            MagicMock(),
        )

        with patch.object(api, '_app') as app:
            api.run()

            app.run.assert_called_once()
