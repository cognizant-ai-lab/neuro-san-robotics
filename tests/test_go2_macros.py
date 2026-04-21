import os
import unittest
from unittest.mock import patch

from coded_tools.unigo2.go2_macros import Go2Macros


class _FakeCli:
    def __init__(self):
        self.calls = []

    def FreeAvoid(self, flag):
        self.calls.append(("FreeAvoid", flag))

    def Move(self, vx, vy, vyaw):
        self.calls.append(("Move", vx, vy, vyaw))

    def StopMove(self):
        self.calls.append(("StopMove",))

    def FreeWalk(self):
        self.calls.append(("FreeWalk",))


class Go2MacrosTests(unittest.TestCase):
    def test_move_enables_collision_avoidance_by_default(self):
        go2 = Go2Macros(use_robot=False)
        go2.cli = _FakeCli()

        with patch.dict(os.environ, {}, clear=False):
            go2.move(vx=0.15, vy=0.05, vyaw=0.1)

        self.assertEqual(
            go2.cli.calls,
            [
                ("FreeAvoid", True),
                ("Move", 0.15, 0.05, 0.1),
            ],
        )

    def test_move_can_disable_collision_avoidance_via_env_flag(self):
        go2 = Go2Macros(use_robot=False)
        go2.cli = _FakeCli()

        with patch.dict(os.environ, {"GO2_ENABLE_COLLISION_AVOIDANCE": "0"}, clear=False):
            go2.move(vx=0.12, vy=0.0, vyaw=0.0)

        self.assertEqual(
            go2.cli.calls,
            [
                ("Move", 0.12, 0.0, 0.0),
            ],
        )

    def test_step_forward_enables_collision_avoidance_before_moving(self):
        go2 = Go2Macros(use_robot=False)
        go2.cli = _FakeCli()

        with patch.dict(os.environ, {}, clear=False):
            with patch("coded_tools.unigo2.go2_macros.time.sleep", return_value=None):
                go2.step_forward(vx=0.2, t=0.5)

        self.assertEqual(
            go2.cli.calls,
            [
                ("FreeAvoid", True),
                ("Move", 0.2, 0.0, 0.0),
                ("StopMove",),
            ],
        )

    def test_free_walk_enables_collision_avoidance_before_gait_change(self):
        go2 = Go2Macros(use_robot=False)
        go2.cli = _FakeCli()

        with patch.dict(os.environ, {}, clear=False):
            go2.free_walk()

        self.assertEqual(
            go2.cli.calls,
            [
                ("FreeAvoid", True),
                ("FreeWalk",),
            ],
        )


if __name__ == "__main__":
    unittest.main()
