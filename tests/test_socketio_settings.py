import os
import unittest
from unittest.mock import patch

from apps.conscious_assistant.socketio_settings import socketio_client_transports
from apps.conscious_assistant.socketio_settings import socketio_server_options


class SocketIoSettingsTests(unittest.TestCase):
    def test_defaults_to_polling_only_without_upgrades(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(socketio_client_transports(), ["polling"])
            self.assertEqual(socketio_server_options()["allow_upgrades"], False)
            self.assertEqual(socketio_server_options()["always_connect"], True)

    def test_enabling_upgrades_restores_polling_then_websocket(self):
        with patch.dict(os.environ, {"CONSCIOUS_SOCKETIO_ALLOW_UPGRADES": "1"}, clear=True):
            self.assertEqual(socketio_client_transports(), ["polling", "websocket"])
            self.assertEqual(socketio_server_options()["allow_upgrades"], True)

    def test_explicit_client_transport_override_is_respected(self):
        env = {
            "CONSCIOUS_SOCKETIO_ALLOW_UPGRADES": "0",
            "CONSCIOUS_SOCKETIO_TRANSPORTS": "websocket,polling",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(socketio_client_transports(), ["websocket", "polling"])


if __name__ == "__main__":
    unittest.main()
