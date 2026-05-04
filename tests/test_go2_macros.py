import unittest
from unittest.mock import MagicMock, patch

from coded_tools.unigo2 import go2_macros


class FakeSportClient:
    instances = []

    def __init__(self):
        self.timeout = None
        self.initialized = False
        FakeSportClient.instances.append(self)

    def SetTimeout(self, timeout):
        self.timeout = timeout

    def Init(self):
        self.initialized = True


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


if __name__ == "__main__":
    unittest.main()
