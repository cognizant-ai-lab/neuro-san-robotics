import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from apps.conscious_assistant import realtime_transcription
from coded_tools.unigo2 import agent_events


ROOT = Path(__file__).resolve().parents[1]


class RealtimeTranscriptionTests(unittest.TestCase):
    def test_session_is_transcription_only_and_defaults_to_a_room_mic(self):
        with patch.dict(os.environ, {}, clear=False):
            for name in (
                "CONSCIOUS_MIC_NOISE_REDUCTION",
                "CONSCIOUS_VAD_THRESHOLD",
                "CONSCIOUS_VAD_PREFIX_PADDING_MS",
                "CONSCIOUS_VAD_SILENCE_MS",
            ):
                os.environ.pop(name, None)
            config = realtime_transcription.transcription_session_config("test-model")

        self.assertEqual(config["type"], "transcription")
        audio_input = config["audio"]["input"]
        self.assertEqual(audio_input["transcription"]["model"], "test-model")
        self.assertEqual(audio_input["transcription"]["language"], "en")
        # near_field on a mic that is not at the speaker's mouth treats their
        # voice as distant noise, which no VAD threshold can compensate for.
        self.assertEqual(audio_input["noise_reduction"]["type"], "far_field")
        self.assertEqual(audio_input["turn_detection"]["type"], "server_vad")

    def test_a_worn_mic_can_ask_for_the_near_field_profile(self):
        with patch.dict(os.environ, {"CONSCIOUS_MIC_NOISE_REDUCTION": "near_field"}):
            config = realtime_transcription.transcription_session_config("test-model")

        self.assertEqual(
            config["audio"]["input"]["noise_reduction"]["type"], "near_field"
        )

    def test_noise_reduction_can_be_turned_off_entirely(self):
        # Neither profile means "leave it alone", and both shape the audio
        # before the VAD ever sees it, so the field has to be absent.
        with patch.dict(os.environ, {"CONSCIOUS_MIC_NOISE_REDUCTION": "off"}):
            audio_input = realtime_transcription.transcription_session_config(
                "test-model"
            )["audio"]["input"]

        self.assertNotIn("noise_reduction", audio_input)
        self.assertEqual(audio_input["turn_detection"]["type"], "server_vad")

    def test_capture_thresholds_come_from_the_environment(self):
        with patch.dict(os.environ, {
            "CONSCIOUS_VAD_THRESHOLD": "0.7",
            "CONSCIOUS_VAD_PREFIX_PADDING_MS": "250",
            "CONSCIOUS_VAD_SILENCE_MS": "500",
        }):
            turn = realtime_transcription.transcription_session_config(
                "test-model"
            )["audio"]["input"]["turn_detection"]

        self.assertEqual(turn["threshold"], 0.7)
        self.assertEqual(turn["prefix_padding_ms"], 250)
        self.assertEqual(turn["silence_duration_ms"], 500)

    def test_unusable_capture_settings_fall_back_to_defaults(self):
        with patch.dict(os.environ, {
            "CONSCIOUS_MIC_NOISE_REDUCTION": "nonsense",
            "CONSCIOUS_VAD_THRESHOLD": "loud",
        }):
            audio_input = realtime_transcription.transcription_session_config(
                "test-model"
            )["audio"]["input"]

        self.assertEqual(audio_input["noise_reduction"]["type"], "far_field")
        self.assertEqual(audio_input["turn_detection"]["threshold"], 0.45)

    def test_client_secret_request_sends_transcription_session_as_json(self):
        response = MagicMock()
        response.status = 201
        response.headers.get_content_type.return_value = "application/json"
        response.headers.get.return_value = "req_test"
        response.read.return_value = b'{"value":"ephemeral-key"}'
        context = MagicMock()
        context.__enter__.return_value = response

        with patch.object(realtime_transcription, "urlopen", return_value=context) as open_url:
            result = realtime_transcription.create_realtime_client_secret(
                "secret-key",
                "test-model",
            )

        self.assertEqual(
            result,
            (201, "application/json", b'{"value":"ephemeral-key"}', "req_test"),
        )
        upstream_request = open_url.call_args.args[0]
        self.assertEqual(upstream_request.get_header("Authorization"), "Bearer secret-key")
        self.assertEqual(upstream_request.get_header("Content-type"), "application/json")
        self.assertIn(b'"session"', upstream_request.data)
        self.assertIn(b'"model": "test-model"', upstream_request.data)
        self.assertNotIn(b"secret-key", upstream_request.data)

    def test_client_secret_retries_a_gateway_timeout_once(self):
        timeout = (504, "text/plain", b"error code: 504", "req_timeout")
        success = (200, "application/json", b'{"value":"key"}', "req_success")

        with (
            patch.object(
                realtime_transcription,
                "_send_client_secret_request",
                side_effect=[timeout, success],
            ) as send,
            patch.object(realtime_transcription.time, "sleep") as sleep,
        ):
            result = realtime_transcription.create_realtime_client_secret(
                "secret-key",
                "test-model",
            )

        self.assertEqual(result, success)
        self.assertEqual(send.call_count, 2)
        sleep.assert_called_once_with(0.5)

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
