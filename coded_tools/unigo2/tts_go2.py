"""
go2_tts.py — Offline TTS for Unitree Go2 EDU speaker (Linux/Jetson) and macOS.

Linux / Unitree Go2:
- Uses espeak-ng -> ALSA via aplay, forced to an ALSA device (default hw:0,0).
- This matches the working bash script: aplay -D hw:0,0 <file>

macOS:
- Uses pyttsx3 (NSSpeechSynthesizer backend).

Install (Linux / Go2):

  sudo apt-get update
  sudo apt-get install -y python3-pip espeak-ng alsa-utils
  pip3 install --user pyttsx3

Usage:
  python3 go2_tts.py "Hello from Unitree!"
  python3 go2_tts.py -r 170 -v 0.9 -V en-us "Starting mission."
"""

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict

from neuro_san.interfaces.coded_tool import CodedTool


def _has(cmd: str) -> bool:
    """Return True if `cmd` is available on PATH."""
    return shutil.which(cmd) is not None


def _linux_say_via_espeak_aplay(
    text: str,
    rate: int = 180,
    volume: float = 1.0,
    voice: str = "en-us",
    alsa_device: str | None = None,
) -> None:
    """
    Linux path: espeak-ng --stdout | aplay -D <alsa_device>

    - This matches your working bash script path.
    - Default ALSA device is hw:0,0 (USB speaker on Unitree Go2).
    """

    if not _has("espeak-ng"):
        raise RuntimeError(
            "espeak-ng not found on PATH. Install with:\n"
            "  sudo apt-get update && sudo apt-get install -y espeak-ng alsa-utils"
        )

    if alsa_device is None:
        # Environment override, otherwise default to the working Go2 device.
        alsa_device = os.environ.get("GO2_TTS_DEVICE", "hw:0,0")

    # Map volume (0.0–1.0) to espeak-ng amplitude (0–200)
    amp = max(0, min(200, int(round(float(volume) * 200))))

    # Build espeak-ng command
    espeak_cmd = [
        "espeak-ng",
        "--stdout",               # write WAV to stdout
        "-s", str(int(rate)),     # speed
        "-a", str(amp),           # amplitude
    ]
    if voice:
        espeak_cmd += ["-v", voice]
    espeak_cmd.append(text)

    logging.info(
        "GO2_TTS: Linux espeak-ng pipeline: %s | aplay -D %s",
        " ".join(espeak_cmd),
        alsa_device,
    )

    if not _has("aplay"):
        # No aplay: let espeak-ng talk via default ALSA device (less controlled).
        subprocess.run(espeak_cmd, check=True)
        return

    # Pipe espeak-ng audio into aplay on the chosen ALSA device
    aplay_cmd = ["aplay", "-D", alsa_device]

    p1 = subprocess.Popen(espeak_cmd, stdout=subprocess.PIPE)
    try:
        subprocess.run(aplay_cmd, stdin=p1.stdout, check=True)
    finally:
        if p1.stdout:
            p1.stdout.close()
        p1.wait()


def _mac_say_via_pyttsx3(
    text: str,
    rate: int = 180,
    volume: float = 1.0,
    voice: str = "en-us",
) -> None:
    """
    macOS path: use pyttsx3 (NSSpeechSynthesizer backend).
    """
    try:
        import pyttsx3  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pyttsx3 not installed. On macOS, run:\n"
            "  pip3 install --user pyttsx3"
        ) from e

    engine = pyttsx3.init()
    engine.setProperty("rate", int(rate))
    engine.setProperty("volume", max(0.0, min(1.0, float(volume))))

    if voice:
        for v in engine.getProperty("voices"):
            if voice.lower() in (v.name.lower() + " " + v.id.lower()):
                engine.setProperty("voice", v.id)
                break

    logging.info(
        "GO2_TTS: macOS pyttsx3 (rate=%s, volume=%s, voice=%s)", rate, volume, voice
    )
    engine.say(text)
    engine.runAndWait()


def say(
    text: str,
    rate: int = 180,
    volume: float = 1.0,
    voice: str = "en-us",
    alsa_device: str | None = None,
) -> None:
    """
    Cross-platform TTS entrypoint.

    - On macOS: pyttsx3 (system TTS).
    - On Linux (Unitree Go2): espeak-ng -> aplay -> ALSA device (default hw:0,0).
    """

    system = platform.system()

    if system == "Darwin":
        _mac_say_via_pyttsx3(text, rate=rate, volume=volume, voice=voice)
        return

    if system == "Linux":
        _linux_say_via_espeak_aplay(
            text,
            rate=rate,
            volume=volume,
            voice=voice,
            alsa_device=alsa_device,
        )
        return

    # Fallback for any other OS: try pyttsx3 generically
    try:
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        engine.setProperty("rate", int(rate))
        engine.setProperty("volume", max(0.0, min(1.0, float(volume))))
        if voice:
            for v in engine.getProperty("voices"):
                if voice.lower() in (v.name.lower() + " " + v.id.lower()):
                    engine.setProperty("voice", v.id)
                    break
        logging.info(
            "GO2_TTS: generic pyttsx3 fallback (OS=%s, rate=%s, volume=%s, voice=%s)",
            system, rate, volume, voice,
        )
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        raise RuntimeError(
            f"TTS not configured for OS={system}. Install pyttsx3 or espeak-ng."
        ) from e


class Go2TTSTool(CodedTool):
    """
    CodedTool wrapper for Unitree Go2 offline TTS.

    Usage (invoke):
        {
          "action": "say",
          "text": "Hello!",
          "rate": 170,
          "volume": 0.9,
          "voice": "en-us"
        }

    All parameters optional except "text".
    """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return "Missing required 'text' (string) for TTS"

        rate = int(args.get("rate", 180))
        volume = float(args.get("volume", 1.0))
        voice = args.get("voice", "en-us")  # default more standard than en-sc
        alsa_device = args.get("alsa_device") or os.environ.get("GO2_TTS_DEVICE")

        try:
            say(
                text,
                rate=rate,
                volume=volume,
                voice=voice,
                alsa_device=alsa_device,
            )
            logging.info(
                "===== GO2 TTS say (rate=%s, volume=%s, voice=%s, alsa_device=%s) -> %r",
                rate,
                volume,
                voice,
                alsa_device,
                text,
            )
            return f"TTS OK: {text}"
        except Exception as e:
            logging.exception("TTS failed")
            return f"TTS error: {e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "text",
        nargs="*",  # 0 or more words (now optional)
        help="What the Go2 should say. If omitted, defaults to 'Hello from Unitree!'.",
    )
    parser.add_argument(
        "-r", "--rate",
        type=int,
        default=180,
        help="Words per minute (default 180)",
    )
    parser.add_argument(
        "-v", "--volume",
        type=float,
        default=1.0,
        help="Volume 0.0–1.0 (default 1.0)",
    )
    parser.add_argument(
        "-V", "--voice",
        default="en-us",
        help="Voice/language (e.g., en-us, en-gb, fa). "
             "Substring match for pyttsx3 on macOS; espeak-ng voice on Linux.",
    )
    parser.add_argument(
        "-D", "--device",
        default=None,
        help="ALSA device on Linux (e.g., hw:0,0). "
             "If omitted, uses GO2_TTS_DEVICE or hw:0,0.",
    )

    args = parser.parse_args()

    # If user passed words, join them; otherwise use default text
    if args.text:
        text = " ".join(args.text)
    else:
        text = "Hello from Unitree!"

    say(
        text,
        rate=args.rate,
        volume=args.volume,
        voice=args.voice,
        alsa_device=args.device,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
