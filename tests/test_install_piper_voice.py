import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


def load_installer():
    spec = importlib.util.spec_from_file_location(
        "install_piper_voice", REPO / "scripts" / "install_piper_voice.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InstallerFollowsTheConfiguredVoiceTests(unittest.TestCase):
    """
    The installer used to carry its own copy of the voice name and URL. It now
    asks piper_voice, so swapping voices is an env change and nothing else.
    """

    def setUp(self):
        self.installer = load_installer()

    def test_downloads_are_planned_for_the_configured_voice(self):
        env = {"GO2_PIPER_VOICE": "en_GB-southern_english_female-low",
               "GO2_PIPER_DIR": "/voices", "GO2_PIPER_MODEL": "", "GO2_PIPER_CONFIG": ""}
        with patch.dict(os.environ, env):
            plan = self.installer.downloads()

        base = ("https://huggingface.co/rhasspy/piper-voices/resolve/main"
                "/en/en_GB/southern_english_female/low/en_GB-southern_english_female-low")
        self.assertEqual(plan, [
            (f"{base}.onnx", Path("/voices/en_GB-southern_english_female-low.onnx")),
            (f"{base}.onnx.json", Path("/voices/en_GB-southern_english_female-low.onnx.json")),
        ])

    def test_voice_flag_overrides_the_environment(self):
        env = {"GO2_PIPER_VOICE": "en_GB-cori-high",
               "GO2_PIPER_DIR": "/voices", "GO2_PIPER_MODEL": "", "GO2_PIPER_CONFIG": ""}
        with patch.dict(os.environ, env), \
                patch.object(self.installer.sys, "argv",
                             ["install_piper_voice.py", "--check",
                              "--voice", "en_GB-southern_english_female-low"]), \
                patch("builtins.print"):
            self.installer.main()
            self.assertEqual(os.environ["GO2_PIPER_VOICE"],
                             "en_GB-southern_english_female-low")


if __name__ == "__main__":
    unittest.main()
