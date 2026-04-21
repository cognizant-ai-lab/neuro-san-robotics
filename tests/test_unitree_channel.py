import os
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from coded_tools.unigo2.unitree_channel import initialize_unitree_channel
from coded_tools.unigo2.unitree_channel import reset_unitree_channel_state
from coded_tools.unigo2.unitree_channel import resolve_unitree_interface
from coded_tools.unigo2.unitree_channel import _CYCLONEDDS_CONFIG


class UnitreeChannelTests(unittest.TestCase):
    def tearDown(self):
        reset_unitree_channel_state()

    def test_resolve_unitree_interface_prefers_explicit_default(self):
        with patch.dict(os.environ, {"CYCLONEDDS_NETWORK_INTERFACE": "wlan0"}, clear=True):
            self.assertEqual(resolve_unitree_interface("eth0"), "eth0")

    def test_initialize_unitree_channel_uses_interface_once_and_reuses_state(self):
        factory = Mock()

        with patch.dict(os.environ, {}, clear=True):
            state = initialize_unitree_channel(factory, "eth0")
            reused_state = initialize_unitree_channel(factory, "eth0")
            self.assertEqual(os.environ["CYCLONEDDS_NETWORK_INTERFACE"], "eth0")
            self.assertEqual(os.environ["CYCLONEDDS_URI"], _CYCLONEDDS_CONFIG.resolve().as_uri())

        factory.assert_called_once_with(0, "eth0")
        self.assertEqual(state["ifname"], "eth0")
        self.assertEqual(reused_state["ifname"], "eth0")

    def test_initialize_unitree_channel_rejects_conflicting_interface(self):
        factory = Mock()
        initialize_unitree_channel(factory, "eth0")

        with self.assertRaises(RuntimeError):
            initialize_unitree_channel(factory, "wlan0")


if __name__ == "__main__":
    unittest.main()
