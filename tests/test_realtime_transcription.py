import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from apps.conscious_assistant import interface_flask
from apps.conscious_assistant import realtime_transcription
from coded_tools.unigo2 import agent_events


ROOT = Path(__file__).resolve().parents[1]


class RealtimeTranscriptionTests(unittest.TestCase):
    def test_session_is_transcription_only_and_tuned_for_room_audio(self):
        config = realtime_transcription.transcription_session_config("test-model")

        self.assertEqual(config["type"], "transcription")
        audio_input = config["audio"]["input"]
        self.assertEqual(audio_input["transcription"]["model"], "test-model")
        self.assertEqual(audio_input["transcription"]["language"], "en")
        self.assertEqual(audio_input["noise_reduction"]["type"], "far_field")
        self.assertEqual(audio_input["turn_detection"]["type"], "server_vad")

    def test_session_request_sends_transcription_session_as_json(self):
        response = MagicMock()
        response.status = 201
        response.headers.get_content_type.return_value = "application/json"
        response.headers.get.return_value = "req_test"
        response.read.return_value = b'{"value":"ephemeral-key"}'
        context = MagicMock()
        context.__enter__.return_value = response

        with patch.object(realtime_transcription, "urlopen", return_value=context) as open_url:
            result = realtime_transcription.request_realtime_session(
                "secret-key",
                "test-model",
            )

        self.assertEqual(result.status, 201)
        self.assertEqual(result.content_type, "application/json")
        self.assertEqual(result.payload, b'{"value":"ephemeral-key"}')
        self.assertEqual(result.request_id, "req_test")
        self.assertFalse(result.failed)
        upstream_request = open_url.call_args.args[0]
        self.assertEqual(upstream_request.get_header("Authorization"), "Bearer secret-key")
        self.assertEqual(upstream_request.get_header("Content-type"), "application/json")
        self.assertIn(b'"session"', upstream_request.data)
        self.assertIn(b'"model": "test-model"', upstream_request.data)
        self.assertNotIn(b"secret-key", upstream_request.data)

    def test_session_request_retries_a_gateway_timeout_once(self):
        timeout = realtime_transcription.RealtimeSessionResponse(
            504, "text/plain", b"error code: 504", "req_timeout",
        )
        success = realtime_transcription.RealtimeSessionResponse(
            200, "application/json", b'{"value":"key"}', "req_success",
        )

        with (
            patch.object(
                realtime_transcription,
                "_post_session_request",
                side_effect=[timeout, success],
            ) as send,
            patch.object(realtime_transcription.time, "sleep") as sleep,
        ):
            result = realtime_transcription.request_realtime_session(
                "secret-key",
                "test-model",
            )

        self.assertEqual(result, success)
        self.assertEqual(send.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_failed_and_transient_classify_upstream_statuses(self):
        def response(status):
            return realtime_transcription.RealtimeSessionResponse(
                status, "application/json", b"{}", "req_test",
            )

        self.assertFalse(response(200).failed)
        self.assertTrue(response(401).failed)
        self.assertFalse(response(401).transient)
        self.assertTrue(response(503).transient)

    def test_token_route_keeps_an_upstream_error_body_out_of_the_log_and_response(self):
        """A failure may report status and request id, never the upstream body."""
        failure = realtime_transcription.RealtimeSessionResponse(
            401,
            "application/json",
            b'{"error":{"message":"Incorrect API key sk-live-do-not-log"}}',
            "req_denied",
        )

        with patch.object(interface_flask, "request_realtime_session", return_value=failure):
            with self.assertLogs(level="ERROR") as captured:
                with interface_flask.app.test_client() as client:
                    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-live-do-not-log"}):
                        response = client.post("/api/realtime/transcription-token")

        logged = "\n".join(captured.output)
        self.assertNotIn("sk-live-do-not-log", logged)
        self.assertNotIn("Incorrect API key", logged)
        self.assertIn("req_denied", logged)
        self.assertIn("401", logged)

        self.assertEqual(response.status_code, 401)
        self.assertNotIn(b"sk-live-do-not-log", response.data)
        self.assertEqual(response.get_json(), {"error": "Could not start realtime transcription"})

    def test_token_route_forwards_the_credential_to_the_browser_only(self):
        """The happy path must still hand the browser its ephemeral key verbatim."""
        success = realtime_transcription.RealtimeSessionResponse(
            200, "application/json", b'{"value":"ek_browser_token"}', "req_ok",
        )

        with patch.object(interface_flask, "request_realtime_session", return_value=success):
            with self.assertLogs(level="INFO") as captured:
                with interface_flask.app.test_client() as client:
                    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-live-do-not-log"}):
                        response = client.post("/api/realtime/transcription-token")

        self.assertNotIn("ek_browser_token", "\n".join(captured.output))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'{"value":"ek_browser_token"}')
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_agent_output_carries_promoted_ambient_speech(self):
        with patch.object(agent_events, "_post_json") as post:
            delivered = agent_events.publish_ui_output(
                thought="They addressed me.",
                say="Hello!",
                heard="CAIL-E, come here.",
            )

        self.assertTrue(delivered)
        payload = post.call_args.args[1]
        self.assertEqual(payload["heard"], "CAIL-E, come here.")

    def test_browser_streams_ambient_audio_and_keeps_raw_text_in_thoughts(self):
        browser_source = (
            ROOT / "apps" / "conscious_assistant" / "templates" / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn("new RTCPeerConnection()", browser_source)
        self.assertIn("/api/realtime/transcription-token", browser_source)
        self.assertIn("https://api.openai.com/v1/realtime/calls", browser_source)
        self.assertIn(
            "conversation.item.input_audio_transcription.completed",
            browser_source,
        )
        self.assertIn("'assistant-thoughts'", browser_source)
        self.assertIn("'Heard: ' + data.data", browser_source)
        self.assertNotIn("AMBIENT_CHUNK_MS", browser_source)
        self.assertNotIn("ambientRecorder", browser_source)


if __name__ == "__main__":
    unittest.main()
