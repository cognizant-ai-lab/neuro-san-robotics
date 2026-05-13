import unittest
from unittest.mock import patch

from coded_tools.unigo2 import robot_macros


class RobotMacrosDeferredActionTests(unittest.TestCase):
    def setUp(self):
        robot_macros.clear_deferred_actions()

    def tearDown(self):
        robot_macros.clear_deferred_actions()

    def test_clear_deferred_actions_returns_count_and_empties_queue(self):
        robot_macros.queue_deferred_action("dance", {})
        robot_macros.queue_deferred_action("content", {})

        self.assertEqual(robot_macros.clear_deferred_actions(), 2)
        self.assertEqual(robot_macros.clear_deferred_actions(), 0)

        with patch.object(robot_macros, "Go2Macros") as go2_cls:
            results = robot_macros.execute_deferred_actions()

        self.assertEqual(results, [])
        go2_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
