import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from apps.conscious_assistant import realtime_transcription
from coded_tools.unigo2 import agent_events


ROOT = Path(__file__).resolve().parents[1]


class RealtimeTranscriptionTests(unittest.TestCase):
    def test_session_is_transcription_only_and_tuned_for_room_audio(self):
        config = realtime_transcription.transcription_session_config("test-model")

        self.assertEqual(config["type"], "transcription")
        audio_input = config["audio"]["input"]
        self.assertEqual(audio_input["transcription"]["model"], "test-model")
        self.assertEqual(audio_input["noise_reduction"]["type"], "far_field")
        self.assertEqual(audio_input["turn_detection"]["type"], "server_vad")

    def test_realtime_call_sends_sdp_and_session_as_multipart(self):
        response = MagicMock()
        response.status = 201
        response.headers.get_content_type.return_value = "application/sdp"
        response.read.return_value = b"answer-sdp"
        context = MagicMock()
        context.__enter__.return_value = response

        with patch.object(realtime_transcription, "urlopen", return_value=context) as open_url:
            result = realtime_transcription.create_realtime_call(
                b"offer-sdp",
                "secret-key",
                "test-model",
            )

        self.assertEqual(result, (201, "application/sdp", b"answer-sdp"))
        upstream_request = open_url.call_args.args[0]
        self.assertEqual(upstream_request.get_header("Authorization"), "Bearer secret-key")
        self.assertIn(b'name="sdp"', upstream_request.data)
        self.assertIn(b"offer-sdp", upstream_request.data)
        self.assertIn(b'name="session"', upstream_request.data)
        self.assertIn(b'"model": "test-model"', upstream_request.data)
        self.assertNotIn(b"secret-key", upstream_request.data)

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
