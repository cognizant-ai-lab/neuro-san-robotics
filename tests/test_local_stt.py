"""Offline speech-to-text fallback and how the transcribe route chooses."""

import io
import os
import unittest
from unittest.mock import patch

from apps.conscious_assistant import interface_flask
from coded_tools.unigo2 import local_stt


STT_VARS = ("GO2_STT_ENGINE", "GO2_STT_MODEL", "GO2_STT_DEVICE", "GO2_STT_LANGUAGE")


def stt_env(**overrides):
    """An environment with every STT variable under test control."""
    env = {name: "" for name in STT_VARS}
    env.update(overrides)
    return env


def upload():
    """A recording large enough to clear the route's minimum-size guard."""
    return {"audio": (io.BytesIO(b"\x00" * 4000), "speech.webm")}


class ConfigurationTests(unittest.TestCase):
    def test_auto_is_the_default(self):
        with patch.dict(os.environ, stt_env()):
            self.assertEqual(local_stt.engine(), "auto")

    def test_an_unrecognised_value_falls_back_to_auto(self):
        with patch.dict(os.environ, stt_env(GO2_STT_ENGINE="wat")):
            self.assertEqual(local_stt.engine(), "auto")

    def test_engine_choices_are_honoured(self):
        for value in ("auto", "openai", "local"):
            with patch.dict(os.environ, stt_env(GO2_STT_ENGINE=value)):
                self.assertEqual(local_stt.engine(), value)

    def test_it_stays_off_the_gpu_unless_asked(self):
        """The Orin is already carrying YOLO and deepface."""
        with patch.dict(os.environ, stt_env()):
            self.assertEqual(local_stt.device(), "cpu")
        with patch.dict(os.environ, stt_env(GO2_STT_DEVICE="cuda")):
            self.assertEqual(local_stt.device(), "cuda")

    def test_model_size_defaults_to_base(self):
        with patch.dict(os.environ, stt_env()):
            self.assertEqual(local_stt.model_size(), "base")
        with patch.dict(os.environ, stt_env(GO2_STT_MODEL="small")):
            self.assertEqual(local_stt.model_size(), "small")


class RouteSelectionTests(unittest.TestCase):
    """Which recogniser /api/transcribe reaches for, and when."""

    def post(self):
        with interface_flask.app.test_client() as client:
            return client.post("/api/transcribe", data=upload(),
                               content_type="multipart/form-data")

    def test_local_mode_never_calls_the_hosted_api(self):
        """A region with no hosted recogniser must not wait on a doomed call."""
        env = stt_env(GO2_STT_ENGINE="local")
        env["OPENAI_API_KEY"] = ""
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "transcribe", return_value="hello there") as local:
                with patch.object(interface_flask.openai_provider, "create_client") as hosted:
                    response = self.post()
        self.assertEqual(response.get_json(), {"text": "hello there"})
        local.assert_called_once()
        hosted.assert_not_called()

    def test_local_mode_needs_no_hosted_credentials(self):
        """The 503 credential gate must not turn a local-only robot away."""
        env = stt_env(GO2_STT_ENGINE="local")
        env["OPENAI_API_KEY"] = ""
        env["AZURE_OPENAI_API_KEY"] = ""
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "transcribe", return_value="no key needed"):
                response = self.post()
        self.assertEqual(response.status_code, 200)

    def test_auto_prefers_the_hosted_recogniser(self):
        env = stt_env(GO2_STT_ENGINE="auto")
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "transcribe") as local:
                with patch.object(interface_flask.openai_provider, "create_client") as factory:
                    factory.return_value.audio.transcriptions.create.return_value.text = "hosted"
                    response = self.post()
        self.assertEqual(response.get_json(), {"text": "hosted"})
        local.assert_not_called()

    def test_auto_falls_back_when_the_hosted_call_fails(self):
        env = stt_env(GO2_STT_ENGINE="auto")
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "available", return_value=True):
                with patch.object(local_stt, "transcribe", return_value="rescued") as local:
                    with patch.object(interface_flask.openai_provider, "create_client",
                                      side_effect=RuntimeError("region has no whisper")):
                        response = self.post()
        self.assertEqual(response.get_json(), {"text": "rescued"})
        local.assert_called_once()

    def test_openai_mode_does_not_fall_back(self):
        """Naming the hosted engine means you want to hear when it breaks."""
        env = stt_env(GO2_STT_ENGINE="openai")
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "available", return_value=True):
                with patch.object(local_stt, "transcribe") as local:
                    with patch.object(interface_flask.openai_provider, "create_client",
                                      side_effect=RuntimeError("upstream down")):
                        response = self.post()
        self.assertEqual(response.status_code, 500)
        local.assert_not_called()

    def test_an_existing_robot_behaves_exactly_as_before(self):
        """
        No new variables set and no weights cached: the hosted path runs and
        nothing else happens. This is the upgrade case for robots already in
        the field, which must not change behaviour.
        """
        env = stt_env()
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "is_model_cached", return_value=False):
                with patch.object(local_stt, "transcribe") as local:
                    with patch.object(interface_flask.openai_provider,
                                      "create_client") as factory:
                        factory.return_value.audio.transcriptions.create.return_value.text = "as before"
                        response = self.post()
        self.assertEqual(response.get_json(), {"text": "as before"})
        local.assert_not_called()

    def test_auto_never_downloads_weights_mid_request(self):
        """
        A hosted failure on a robot that has never used the fallback must not
        stall the request behind a few hundred megabytes of model download.
        """
        env = stt_env(GO2_STT_ENGINE="auto")
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "is_model_cached", return_value=False):
                with patch.object(local_stt, "transcribe") as local:
                    with patch.object(interface_flask.openai_provider, "create_client",
                                      side_effect=RuntimeError("upstream down")):
                        response = self.post()
        self.assertEqual(response.status_code, 500)
        local.assert_not_called()

    def test_explicit_local_may_download_because_it_was_asked_for(self):
        env = stt_env(GO2_STT_ENGINE="local")
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "is_model_cached", return_value=False):
                with patch.object(local_stt, "transcribe", return_value="downloaded") as local:
                    response = self.post()
        self.assertEqual(response.get_json(), {"text": "downloaded"})
        local.assert_called_once()

    def test_a_hosted_failure_with_no_local_model_still_reports_cleanly(self):
        env = stt_env(GO2_STT_ENGINE="auto")
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "available", return_value=False):
                with patch.object(interface_flask.openai_provider, "create_client",
                                  side_effect=RuntimeError("upstream down")):
                    response = self.post()
        self.assertEqual(response.status_code, 500)
        # Still no traceback on the public surface, per the earlier hardening.
        self.assertEqual(response.get_json(), {"error": "Transcription failed"})


if __name__ == "__main__":
    unittest.main()
