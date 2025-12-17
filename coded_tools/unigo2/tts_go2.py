"""
go2_tts.py — Offline TTS for Unitree Go2 EDU speaker (Linux/Jetson) and macOS.

Linux / Unitree Go2:
- PRIMARY: Piper TTS (neural, offline, natural voice)
- FALLBACK: espeak-ng -> ALSA via aplay

macOS:
- pyttsx3 (NSSpeechSynthesizer backend)

Tested with Piper CLI requiring:
  piper -m MODEL -c CONFIG --output-raw | aplay
"""

import argparse
import logging
import os
import platform
import shutil
import subprocess
from typing import Any, Dict

from neuro_san.interfaces.coded_tool import CodedTool


# ---------------------------------------------------------------------
# Configuration (override via env vars if needed)
# ---------------------------------------------------------------------

PIPER_MODEL = os.environ.get(
    "GO2_PIPER_MODEL",
    "/home/unitree/piper_models/en_US-amy-medium.onnx",
)

PIPER_CONFIG = os.environ.get(
    "GO2_PIPER_CONFIG",
    "/home/unitree/piper_models/en_US-amy-medium.onnx.json",
)

DEFAULT_ALSA_DEVICE = os.environ.get("GO2_TTS_DEVICE", "plughw:0,0")

# ALSA mixer control name for volume (common names: "Master", "PCM", "Speaker")
# Set via env var if the default doesn't work on your hardware
ALSA_MIXER_CONTROL = os.environ.get("GO2_ALSA_MIXER", "Master")

# Default volume percentage to set before each playback (0-100)
DEFAULT_VOLUME_PERCENT = int(os.environ.get("GO2_TTS_VOLUME", "100"))


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _set_alsa_volume(volume_percent: int = DEFAULT_VOLUME_PERCENT) -> None:
    """
    Set ALSA mixer volume before playback to ensure consistent volume.
    
    This helps prevent volume drift that can occur on some systems where
    other processes may adjust mixer levels between playbacks.
    
    Args:
        volume_percent: Volume level 0-100 (default from GO2_TTS_VOLUME env var)
    """
    if not _has("amixer"):
        logging.debug("amixer not found, skipping volume set")
        return
    
    volume_percent = max(0, min(100, volume_percent))
    
    # Try common mixer control names
    controls_to_try = [ALSA_MIXER_CONTROL]
    if ALSA_MIXER_CONTROL != "Master":
        controls_to_try.append("Master")
    if ALSA_MIXER_CONTROL != "PCM":
        controls_to_try.append("PCM")
    
    for control in controls_to_try:
        try:
            result = subprocess.run(
                ["amixer", "set", control, f"{volume_percent}%"],
                capture_output=True,
                timeout=2,
            )
            if result.returncode == 0:
                logging.debug("Set %s volume to %d%%", control, volume_percent)
                return
        except Exception as e:
            logging.debug("Failed to set %s volume: %s", control, e)
            continue
    
    logging.debug("Could not set ALSA volume (tried: %s)", controls_to_try)


# ---------------------------------------------------------------------
# Linux: Piper TTS (primary)
# ---------------------------------------------------------------------

def _linux_say_via_piper(
    text: str,
    volume: float = 1.0,
    alsa_device: str | None = None,
) -> None:
    if not _has("piper"):
        raise RuntimeError("piper binary not found on PATH")

    if not os.path.isfile(PIPER_MODEL) or not os.path.isfile(PIPER_CONFIG):
        raise RuntimeError(
            f"Piper model/config not found:\n"
            f"  MODEL={PIPER_MODEL}\n"
            f"  CONFIG={PIPER_CONFIG}"
        )

    device = alsa_device or DEFAULT_ALSA_DEVICE

    # Set ALSA volume before playback to ensure consistent volume
    # This prevents volume drift that can occur on some systems
    volume_percent = int(volume * DEFAULT_VOLUME_PERCENT)
    _set_alsa_volume(volume_percent)

    piper_cmd = [
        "piper",
        "-m", PIPER_MODEL,
        "-c", PIPER_CONFIG,
        "--output-raw",
    ]

    aplay_cmd = [
        "aplay",
        "-D", device,
        "-r", "22050",
        "-f", "S16_LE",
        "-c", "1",  # Mono output (Piper outputs mono audio)
        "-t", "raw",
    ]

    logging.info("GO2_TTS: Piper -> aplay (%s) at %d%% volume", device, volume_percent)

    # IMPORTANT: newline + communicate()
    piper_proc = subprocess.Popen(
        piper_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    piper_input = (text.strip() + "\n").encode("utf-8")

    stdout, stderr = piper_proc.communicate(input=piper_input)

    if piper_proc.returncode != 0:
        raise RuntimeError(
            f"Piper failed (rc={piper_proc.returncode}): {stderr.decode(errors='ignore')}"
        )

    subprocess.run(
        aplay_cmd,
        input=stdout,
        check=True,
    )


# ---------------------------------------------------------------------
# Linux: eSpeak fallback
# ---------------------------------------------------------------------

def _linux_say_via_espeak(
    text: str,
    rate: int = 150,
    volume: float = 1.0,
    voice: str = "en-us+f3",
    alsa_device: str | None = None,
) -> None:
    if not _has("espeak-ng"):
        raise RuntimeError("espeak-ng not installed")

    amp = max(0, min(200, int(volume * 200)))
    device = alsa_device or DEFAULT_ALSA_DEVICE

    espeak_cmd = [
        "espeak-ng",
        "--stdout",
        "-s", str(rate),
        "-a", str(amp),
        "-v", voice,
        text,
    ]

    aplay_cmd = ["aplay", "-D", device]

    logging.warning("GO2_TTS: Falling back to espeak-ng")

    p1 = subprocess.Popen(espeak_cmd, stdout=subprocess.PIPE)
    try:
        subprocess.run(aplay_cmd, stdin=p1.stdout, check=True)
    finally:
        if p1.stdout:
            p1.stdout.close()
        p1.wait()


# ---------------------------------------------------------------------
# macOS: pyttsx3
# ---------------------------------------------------------------------

def _mac_say_via_pyttsx3(
    text: str,
    rate: int,
    volume: float,
    voice: str,
) -> None:
    try:
        import pyttsx3  # type: ignore
    except ImportError as e:
        raise RuntimeError("pyttsx3 not installed") from e

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", max(0.0, min(1.0, volume)))

    for v in engine.getProperty("voices"):
        if voice.lower() in (v.name.lower() + " " + v.id.lower()):
            engine.setProperty("voice", v.id)
            break

    engine.say(text)
    engine.runAndWait()


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def say(
    text: str,
    rate: int = 150,
    volume: float = 1.0,
    voice: str = "en-us+f3",
    alsa_device: str | None = None,
) -> None:
    system = platform.system()

    if system == "Linux":
        try:
            _linux_say_via_piper(text, volume=volume, alsa_device=alsa_device)
            return
        except Exception as e:
            logging.exception("Piper failed, falling back to espeak-ng")
            _linux_say_via_espeak(
                text,
                rate=rate,
                volume=volume,
                voice=voice,
                alsa_device=alsa_device,
            )
            return

    if system == "Darwin":
        _mac_say_via_pyttsx3(text, rate, volume, voice)
        return

    raise RuntimeError(f"TTS not supported on OS={system}")


# ---------------------------------------------------------------------
# CodedTool wrapper
# ---------------------------------------------------------------------

class Go2TTSTool(CodedTool):
    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return "Missing required 'text'"

        try:
            say(
                text=text,
                rate=int(args.get("rate", 150)),
                volume=float(args.get("volume", 1.0)),
                voice=args.get("voice", "en-us+f3"),
                alsa_device=args.get("alsa_device"),
            )
            return f"TTS OK: {text}"
        except Exception as e:
            logging.exception("TTS failed")
            return f"TTS error: {e}"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser("Offline TTS for Unitree Go2")
    parser.add_argument("text", nargs="*", help="Text to speak")
    args = parser.parse_args()

    text = " ".join(args.text) if args.text else "Hello from Unitree!"
    say(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
