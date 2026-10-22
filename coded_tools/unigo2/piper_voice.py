"""
Where the offline Piper voice lives, and what it sounds like.

One place answers four questions that used to be answered separately, and
inconsistently, by tts_go2.py, the voice installer and the env scripts:
which voice, where its two files are, where to fetch them from, and what
sample rate it produces. The last one matters most. Piper voices come in
tiers that do not share a sample rate (low is 16 kHz, medium and high are
22.05 kHz), and aplay plays raw PCM at whatever rate it is told, so a voice
swap that leaves a hard-coded 22050 behind comes out fast and high-pitched.

Environment:
  GO2_PIPER_VOICE   voice name as Piper publishes it, "<lang>-<speaker>-<quality>".
                    Default "en_GB-cori-high".
  GO2_PIPER_DIR     directory holding the .onnx and .onnx.json. Default ~/piper_models
  GO2_PIPER_MODEL   explicit path to the .onnx; overrides the two above
  GO2_PIPER_CONFIG  explicit path to the .onnx.json; overrides the two above
"""

import json
import logging
import os
from typing import Optional

DEFAULT_VOICE = "en_GB-cori-high"
DEFAULT_DIR = "~/piper_models"

# Every voice the robot has shipped with so far was a 22.05 kHz tier. Used
# only when the config cannot be read, so a missing file degrades to the old
# behaviour rather than to silence.
FALLBACK_SAMPLE_RATE = 22050

# Piper's voices are published on Hugging Face by the Rhasspy project.
HF_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def voice_name() -> str:
    """The configured voice, e.g. "en_GB-cori-high"."""
    return _env("GO2_PIPER_VOICE") or DEFAULT_VOICE


def voice_dir() -> str:
    """The directory the voice files live in, with ~ expanded."""
    return os.path.expanduser(_env("GO2_PIPER_DIR") or DEFAULT_DIR)


def model_path() -> str:
    """Path to the .onnx model."""
    return _env("GO2_PIPER_MODEL") or os.path.join(voice_dir(), f"{voice_name()}.onnx")


def config_path() -> str:
    """Path to the .onnx.json config beside the model."""
    return _env("GO2_PIPER_CONFIG") or os.path.join(voice_dir(), f"{voice_name()}.onnx.json")


def download_url(name: str, extension: str) -> str:
    """
    Where Hugging Face serves one file of a voice.

    Piper names voices "<lang>-<speaker>-<quality>" and files them under
    "<family>/<lang>/<speaker>/<quality>/", where family is the part of the
    language code before the underscore. Speaker names use underscores, never
    hyphens, so splitting on "-" is unambiguous.
    """
    parts = name.split("-")
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"Piper voice {name!r} is not of the form <lang>-<speaker>-<quality>, "
            f"e.g. en_GB-cori-high"
        )
    lang, speaker, quality = parts
    family = lang.split("_")[0]
    return f"{HF_BASE_URL}/{family}/{lang}/{speaker}/{quality}/{name}.{extension}"


def sample_rate(config: Optional[str] = None) -> int:
    """
    The rate, in Hz, of the raw PCM Piper emits for this voice.

    Read from the voice's own config, so it is right for whatever voice is
    installed. Falls back to 22050 if the file is missing or malformed, which
    is what every voice before this module assumed anyway.
    """
    path = config or config_path()
    try:
        with open(path, encoding="utf-8") as handle:
            rate = json.load(handle)["audio"]["sample_rate"]
        return int(rate)
    except (OSError, ValueError, KeyError, TypeError):
        logging.warning("Could not read sample rate from %s; assuming %d Hz",
                        path, FALLBACK_SAMPLE_RATE)
        return FALLBACK_SAMPLE_RATE
