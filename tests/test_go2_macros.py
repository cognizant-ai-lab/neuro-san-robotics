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

    def test_reuses_existing_dds_participant_after_channel_init_error(self):
        channel_init = MagicMock(side_effect=Exception("channel factory init error."))

        with (
            patch.object(go2_macros, "ChannelFactoryInitialize", channel_init),
            patch.object(go2_macros, "sport_client", FakeSportClientModule),
        ):
            bot = go2_macros.Go2Macros()

        self.assertTrue(bot.available)
        self.assertIs(bot.cli, go2_macros._ROBOT_INIT_STATE["client"])
        self.assertTrue(go2_macros._ROBOT_INIT_STATE["channel_initialized"])
        self.assertEqual(channel_init.call_count, 1)
        self.assertEqual(len(FakeSportClient.instances), 1)


if __name__ == "__main__":
    unittest.main()
