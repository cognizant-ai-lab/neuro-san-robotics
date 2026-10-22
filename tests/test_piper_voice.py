import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coded_tools.unigo2 import piper_voice


PIPER_VARS = ("GO2_PIPER_VOICE", "GO2_PIPER_DIR", "GO2_PIPER_MODEL", "GO2_PIPER_CONFIG")


def piper_env(**overrides):
    """An environment with every Piper variable cleared, then overrides applied."""
    env = {name: "" for name in PIPER_VARS}
    env.update(overrides)
    return env


def clear_piper_env(env):
    """patch.dict cannot unset a key, so empty strings stand in for 'unset'."""
    return patch.dict(os.environ, env)


class VoiceNameTests(unittest.TestCase):
    def test_defaults_to_the_voice_the_robot_shipped_with(self):
        with clear_piper_env(piper_env()):
            self.assertEqual(piper_voice.voice_name(), "en_GB-cori-high")

    def test_env_var_picks_another_voice(self):
        with clear_piper_env(piper_env(GO2_PIPER_VOICE="en_GB-southern_english_female-low")):
            self.assertEqual(piper_voice.voice_name(), "en_GB-southern_english_female-low")


class PathTests(unittest.TestCase):
    def test_paths_derive_from_voice_and_directory(self):
        env = piper_env(GO2_PIPER_VOICE="en_GB-southern_english_female-low",
                        GO2_PIPER_DIR="/voices")
        with clear_piper_env(env):
            self.assertEqual(piper_voice.model_path(),
                             "/voices/en_GB-southern_english_female-low.onnx")
            self.assertEqual(piper_voice.config_path(),
                             "/voices/en_GB-southern_english_female-low.onnx.json")

    def test_directory_defaults_to_piper_models_under_home(self):
        with clear_piper_env(piper_env()):
            expected = os.path.expanduser("~/piper_models/en_GB-cori-high.onnx")
            self.assertEqual(piper_voice.model_path(), expected)

    def test_explicit_model_and_config_paths_win(self):
        env = piper_env(GO2_PIPER_VOICE="en_GB-cori-high",
                        GO2_PIPER_MODEL="/elsewhere/voice.onnx",
                        GO2_PIPER_CONFIG="/elsewhere/voice.json")
        with clear_piper_env(env):
            self.assertEqual(piper_voice.model_path(), "/elsewhere/voice.onnx")
            self.assertEqual(piper_voice.config_path(), "/elsewhere/voice.json")


class DownloadUrlTests(unittest.TestCase):
    def test_url_follows_the_hugging_face_layout(self):
        base = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
        self.assertEqual(
            piper_voice.download_url("en_GB-cori-high", "onnx"),
            f"{base}/en/en_GB/cori/high/en_GB-cori-high.onnx",
        )

    def test_speaker_names_may_contain_underscores(self):
        base = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
        self.assertEqual(
            piper_voice.download_url("en_GB-southern_english_female-low", "onnx.json"),
            f"{base}/en/en_GB/southern_english_female/low/"
            "en_GB-southern_english_female-low.onnx.json",
        )

    def test_a_name_that_is_not_lang_speaker_quality_is_rejected(self):
        with self.assertRaises(ValueError):
            piper_voice.download_url("cori", "onnx")


class SampleRateTests(unittest.TestCase):
    def write_config(self, payload) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".onnx.json", delete=False)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        json.dump(payload, handle)
        handle.close()
        return handle.name

    def test_sample_rate_comes_from_the_voice_config(self):
        config = self.write_config({"audio": {"sample_rate": 16000, "quality": "low"}})
        self.assertEqual(piper_voice.sample_rate(config), 16000)

    def test_missing_config_falls_back_to_the_rate_every_shipped_voice_used(self):
        self.assertEqual(piper_voice.sample_rate("/nonexistent/voice.onnx.json"), 22050)

    def test_config_without_a_rate_falls_back_too(self):
        config = self.write_config({"audio": {}})
        self.assertEqual(piper_voice.sample_rate(config), 22050)

    def test_default_argument_reads_the_configured_voice(self):
        config = self.write_config({"audio": {"sample_rate": 16000}})
        with clear_piper_env(piper_env(GO2_PIPER_CONFIG=config)):
            self.assertEqual(piper_voice.sample_rate(), 16000)


if __name__ == "__main__":
    unittest.main()
