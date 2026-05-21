import unittest
from unittest.mock import MagicMock, patch

from coded_tools.unigo2 import go2_macros


class FakeSportClient:
    instances = []

    def __init__(self):
        self.timeout = None
        self.initialized = False
        self.calls = []
        FakeSportClient.instances.append(self)

    def SetTimeout(self, timeout):
        self.timeout = timeout

    def Init(self):
        self.initialized = True

    def RecoveryStand(self):
        self.calls.append(("RecoveryStand",))
        return 0

    def BalanceStand(self):
        self.calls.append(("BalanceStand",))
        return 0

    def Move(self, vx=0.0, vy=0.0, vyaw=0.0):
        self.calls.append(("Move", vx, vy, vyaw))
        return 0

    def StopMove(self):
        self.calls.append(("StopMove",))
        return 0

    def FreeAvoid(self, flag):
        self.calls.append(("FreeAvoid", flag))
        return 0

    def Dance1(self):
        self.calls.append(("Dance1",))
        return 0


class FakeSportClientModule:
    SportClient = FakeSportClient


def reset_robot_init_state():
    go2_macros._ROBOT_INIT_STATE.update(
        {
            "attempted": False,
            "available": True,
            "error": None,
            "reported_disabled": False,
            "client": None,
            "channel_initialized": False,
            "last_failure_at": 0.0,
        }
    )
    FakeSportClient.instances.clear()


class Go2MacrosInitializationTests(unittest.TestCase):
    def setUp(self):
        reset_robot_init_state()

    def tearDown(self):
        reset_robot_init_state()

    def test_reuses_sport_client_and_channel_factory(self):
        channel_init = MagicMock()

        with (
            patch.object(go2_macros, "ChannelFactoryInitialize", channel_init),
            patch.object(go2_macros, "sport_client", FakeSportClientModule),
        ):
            first = go2_macros.Go2Macros()
            second = go2_macros.Go2Macros()

        self.assertTrue(first.available)
        self.assertTrue(second.available)
        self.assertIs(first.cli, second.cli)
        self.assertEqual(channel_init.call_count, 1)
        self.assertEqual(len(FakeSportClient.instances), 1)
        self.assertEqual(first.cli.timeout, 10.0)
        self.assertTrue(first.cli.initialized)
        self.assertIn(("FreeAvoid", False), first.cli.calls)
        self.assertEqual(first._move_log_interval_s, -1.0)

    def test_can_skip_startup_free_avoid_configuration(self):
        with (
            patch.dict(go2_macros.os.environ, {"GO2_DISABLE_FREE_AVOID_ON_INIT": "0"}),
            patch.object(go2_macros, "ChannelFactoryInitialize", MagicMock()),
            patch.object(go2_macros, "sport_client", FakeSportClientModule),
        ):
            bot = go2_macros.Go2Macros()

        self.assertTrue(bot.available)
        client = FakeSportClient.instances[0]
        self.assertNotIn(("FreeAvoid", False), client.calls)

    def test_channel_init_failure_does_not_build_client_with_none_participant(self):
        channel_init = MagicMock(side_effect=Exception("channel factory init error."))

        with (
            patch.object(go2_macros, "ChannelFactoryInitialize", channel_init),
            patch.object(go2_macros, "sport_client", FakeSportClientModule),
            patch.object(go2_macros.traceback, "print_exc"),
        ):
            bot = go2_macros.Go2Macros()

        self.assertFalse(bot.available)
        self.assertIsNone(bot.cli)
        self.assertIsNone(go2_macros._ROBOT_INIT_STATE["client"])
        self.assertFalse(go2_macros._ROBOT_INIT_STATE["channel_initialized"])
        self.assertEqual(channel_init.call_count, 1)
        self.assertEqual(len(FakeSportClient.instances), 0)

    def test_retries_after_channel_init_failure_when_retry_window_has_passed(self):
        channel_init = MagicMock(side_effect=[Exception("domain error"), None])

        with (
            patch.dict(go2_macros.os.environ, {"GO2_INIT_RETRY_SECONDS": "0"}),
            patch.object(go2_macros, "ChannelFactoryInitialize", channel_init),
            patch.object(go2_macros, "sport_client", FakeSportClientModule),
            patch.object(go2_macros.traceback, "print_exc"),
        ):
            first = go2_macros.Go2Macros()
            second = go2_macros.Go2Macros()

        self.assertFalse(first.available)
        self.assertTrue(second.available)
        self.assertIs(second.cli, go2_macros._ROBOT_INIT_STATE["client"])
        self.assertTrue(go2_macros._ROBOT_INIT_STATE["channel_initialized"])
        self.assertEqual(channel_init.call_count, 2)
        self.assertEqual(len(FakeSportClient.instances), 1)

    def test_step_forward_uses_repeated_known_good_move_commands(self):
        with (
            patch.object(go2_macros, "ChannelFactoryInitialize", MagicMock()),
            patch.object(go2_macros, "sport_client", FakeSportClientModule),
            patch.object(go2_macros.time, "sleep"),
        ):
            bot = go2_macros.Go2Macros()
            bot.step_forward()

        client = FakeSportClient.instances[0]
        move_calls = [call for call in client.calls if call[0] == "Move"]
        self.assertEqual(move_calls[0], ("Move", 0.45, 0.0, 0.0))
        self.assertEqual(len(move_calls), 4)
        self.assertIn(("RecoveryStand",), client.calls)
        self.assertIn(("BalanceStand",), client.calls)
        self.assertEqual(client.calls[-1], ("StopMove",))

    def test_dance_uses_sdk_special_motion_by_default(self):
        with (
            patch.object(go2_macros, "ChannelFactoryInitialize", MagicMock()),
            patch.object(go2_macros, "sport_client", FakeSportClientModule),
            patch.object(go2_macros.time, "sleep"),
        ):
            bot = go2_macros.Go2Macros()
            bot.dance1()

        client = FakeSportClient.instances[0]
        self.assertIn(("Dance1",), client.calls)
        self.assertFalse([call for call in client.calls if call[0] == "Move"])

    def test_dance_can_disable_sdk_special_motion_with_env_override(self):
        with (
            patch.dict(go2_macros.os.environ, {"GO2_USE_SDK_SPECIAL_MOTIONS": "0"}),
            patch.object(go2_macros, "ChannelFactoryInitialize", MagicMock()),
            patch.object(go2_macros, "sport_client", FakeSportClientModule),
            patch.object(go2_macros.time, "sleep"),
        ):
            bot = go2_macros.Go2Macros()
            bot.dance1()

        client = FakeSportClient.instances[0]
        self.assertNotIn(("Dance1",), client.calls)
        self.assertGreaterEqual(
            len([call for call in client.calls if call[0] == "Move"]),
            10,
        )


if __name__ == "__main__":
    unittest.main()
