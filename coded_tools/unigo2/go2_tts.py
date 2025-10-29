"""
go2_tts.py — Offline TTS for Unitree Go2 EDU speaker (Linux/Jetson).
- Prefers pyttsx3 (offline, wraps eSpeak-NG on Linux).
- Falls back to espeak-ng CLI + aplay if pyttsx3 isn't installed.
- Can be imported (call say(...)) or run from CLI.

To install on robot:

sudo apt-get update
sudo apt-get install -y python3-pip espeak-ng alsa-utils
pip3 install --user pyttsx3


Usage:
  python3 go2_tts.py "Hello from Unitree!"
  python3 go2_tts.py -r 170 -v 0.9 -V en-us "Starting mission."
  python3 go2_tts.py -V fa "سلام! من آماده هستم."
"""

# import argparse
import shutil
import subprocess
# import sys
from typing import Any, Dict
import logging
from neuro_san.interfaces.coded_tool import CodedTool

def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def say(text: str, rate: int = 180, volume: float = 1.0, voice: str = "en-us") -> None:
    """
    Speak `text` via the robot's default audio device.
    rate: words per minute (typical 150–200)
    volume: 0.0–1.0 (pyttsx3) / 0–200 espeak-ng amplitude mapped internally
    voice: pyttsx3 voice match (substring) or espeak-ng language code (e.g., en-us, en-gb, fa, de)
    """
    # Try pyttsx3 (offline)
    try:
        import pyttsx3  # pip install pyttsx3
        engine = pyttsx3.init()  # On Linux uses eSpeak-NG backend
        engine.setProperty("rate", int(rate))
        engine.setProperty("volume", max(0.0, min(1.0, float(volume))))
        if voice:
            # Match by substring against installed voices
            for v in engine.getProperty("voices"):
                if voice.lower() in (v.name.lower() + " " + v.id.lower()):
                    engine.setProperty("voice", v.id)
                    break
        engine.say(text)
        engine.runAndWait()
        return
    except Exception:
        pass  # fall back below

    # Fallback: espeak-ng + aplay (both typically available on Jetson/Ubuntu)
    if not _has("espeak-ng"):
        raise RuntimeError(
            "No pyttsx3 and espeak-ng not found. Install one of:\n"
            "  pip3 install pyttsx3\n"
            "  sudo apt-get update && sudo apt-get install -y espeak-ng alsa-utils"
        )

    # Map volume (0.0–1.0) to espeak-ng amplitude (0–200)
    amp = max(0, min(200, int(round(float(volume) * 200))))
    # Play directly (espeak-ng can output to ALSA without a file)
    cmd = [
        "espeak-ng",
        f"-s", str(int(rate)),      # speed (wpm)
        f"-a", str(amp),            # amplitude
        f"-v", voice,               # voice/lang
        text
    ]
    # If aplay exists, espeak-ng will still speak; aplay only needed for WAV route.
    subprocess.run(cmd, check=True)

class Go2TTSTool(CodedTool):
    """
    CodedTool wrapper for Unitree Go2 offline TTS.
    Usage (invoke):
        {"action": "say", "text": "Hello!", "rate": 170, "volume": 0.9, "voice": "en-us"}
    All parameters are optional except "text".
    """
    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:

        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return "Missing required 'text' (string) for TTS"
        rate = int(args.get("rate", 180))
        volume = float(args.get("volume", 1.0))
        voice = args.get("voice", "en-sc")
        try:
            say(text, rate=rate, volume=volume, voice=voice)
            logging.info(f"===== GO2 TTS say (rate={rate}, volume={volume}, voice={voice}) -> {text!r}")
            return f"TTS OK: {text}"
        except Exception as e:
            logging.exception("TTS failed")
            return f"TTS error: {e}"

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("text", help="What the Go2 should say", nargs="+")
#     p.add_argument("-r", "--rate", type=int, default=180, help="Words per minute (default 180)")
#     p.add_argument("-v", "--volume", type=float, default=1.0, help="Volume 0.0–1.0 (default 1.0)")
#     p.add_argument("-V", "--voice", default="en-us",
#                    help="Voice/language (e.g., en-us, en-gb, fa). Substring match for pyttsx3.")
#     args = p.parse_args()
#     say(" ".join(args.text), rate=args.rate, volume=args.volume, voice=args.voice)
#
# if __name__ == "__main__":
#     main()
