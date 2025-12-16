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
    rate: int = 150,
    volume: float = 1.0,
    voice: str = "en-us+f3",
    alsa_device: str | None = None,
) -> None:
    """
    Linux path: espeak-ng --stdout | aplay -D <alsa_device>

    - Uses ALSA plug devices so the USB card can accept mono 22050 Hz.
    - Default device: plughw:0,0 (matches USB speaker on Unitree Go2).
    """

    if not _has("espeak-ng"):
        raise RuntimeError(
            "espeak-ng not found on PATH. Install with:\n"
            "  sudo apt-get update && sudo apt-get install -y espeak-ng alsa-utils"
        )

    # Map volume (0.0–1.0) to espeak-ng amplitude (0–200)
    amp = max(0, min(200, int(round(float(volume) * 200))))

    # Base espeak-ng command (produces WAV on stdout)
    espeak_cmd = [
        "espeak-ng",
        "--stdout",               # write WAV to stdout
        "-s", str(int(rate)),     # speed
        "-a", str(amp),           # amplitude
    ]
    if voice:
        espeak_cmd += ["-v", voice]
    espeak_cmd.append(text)

    if not _has("aplay"):
        # No aplay: let espeak-ng use its default ALSA/Pulse path.
        logging.info("GO2_TTS: aplay not found, running espeak-ng directly: %s", " ".join(espeak_cmd))
        subprocess.run(espeak_cmd, check=True)
        return

    # Build a list of devices to try, in order
    devices_to_try: list[str] = []

    # 1) Explicit function arg
    if alsa_device:
        devices_to_try.append(alsa_device)

    # 2) Environment override
    env_dev = os.environ.get("GO2_TTS_DEVICE")
    if env_dev and env_dev not in devices_to_try:
        devices_to_try.append(env_dev)

    # 3) Best default for Unitree Go2 USB speaker (format-converting)
    if "plughw:0,0" not in devices_to_try:
        devices_to_try.append("plughw:0,0")

    # 4) Last-resort fallbacks (might still fail, but we log them)
    for d in ("hw:0,0", "default"):
        if d not in devices_to_try:
            devices_to_try.append(d)

    last_error: Exception | None = None

    for dev in devices_to_try:
        aplay_cmd = ["aplay", "-D", dev]

        logging.info(
            "GO2_TTS: Linux espeak-ng pipeline: %s | %s",
            " ".join(espeak_cmd),
            " ".join(aplay_cmd),
        )

        # Fresh espeak process for each attempt
        p1 = subprocess.Popen(espeak_cmd, stdout=subprocess.PIPE)
        try:
            subprocess.run(aplay_cmd, stdin=p1.stdout, check=True)
            # Success
            if p1.stdout:
                p1.stdout.close()
            p1.wait()
            return
        except subprocess.CalledProcessError as e:
            last_error = e
            logging.warning("GO2_TTS: aplay failed on device %s: %s", dev, e)
        finally:
            if p1.stdout:
                p1.stdout.close()
            p1.wait()

    # If we get here, all devices failed
    raise RuntimeError(f"TTS playback failed on all ALSA devices tried: {devices_to_try}") from last_error

def _mac_say_via_pyttsx3(
    text: str,
    rate: int = 150,
    volume: float = 1.0,
    voice: str = "en-us+f3",
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
    rate: int = 150,
    volume: float = 1.0,
    voice: str = "en-us+f3",
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

        rate = int(args.get("rate", 150))
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


def list_voices() -> None:
    """
    Print available voices for the current platform.
    - On Linux: uses `espeak-ng --voices`
    - On macOS: uses pyttsx3’s installed voices
    """
    system = platform.system()

    if system == "Linux":
        if shutil.which("espeak-ng"):
            print("Available eSpeak-NG voices (Linux):")
            print("  Common examples: en-us, en-gb, en-sc, de, fr, es, fa, hi")
            print("  Full list:")
            subprocess.run(["espeak-ng", "--voices"], check=False)
        else:
            print("espeak-ng not found. Install with:")
            print("  sudo apt-get update && sudo apt-get install -y espeak-ng")
    elif system == "Darwin":
        try:
            import pyttsx3  # type: ignore
        except ImportError:
            print("pyttsx3 not installed. On macOS, run:")
            print("  pip3 install --user pyttsx3")
            return

        engine = pyttsx3.init()
        print("Available macOS / pyttsx3 voices:")
        for v in engine.getProperty("voices"):
            print(f"  id={v.id!r}, name={v.name!r}, lang={getattr(v, 'languages', '')}")
    else:
        print(f"Voice listing not implemented for OS={system}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline TTS for Unitree Go2 (Linux/espeak-ng) and macOS (pyttsx3).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "text",
        nargs="*",
        help=(
            "What the Go2 should say.\n"
            "If omitted, defaults to: 'Hello from Unitree!'\n"
        ),
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
        default="en-us+f3",
        help=(
            "Voice / language.\n"
            "  Linux (espeak-ng): examples → en-us, en-gb, en-sc, de, fr, es, fa, hi\n"
            "  macOS (pyttsx3)  : matched as a substring of installed voice name/id\n"
            "Default: en-gb+f1"
        ),
    )
    parser.add_argument(
        "-D", "--device",
        default=None,
        help="ALSA device on Linux (e.g., plughw:0,0). "
             "If omitted, uses GO2_TTS_DEVICE or plughw:0,0.",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices for this OS and exit.",
    )

    args = parser.parse_args()

    if args.list_voices:
        list_voices()
        return

    # Default text if none provided
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
    # Male / Female voices on the Unitree (Linux / eSpeak-NG)
    # eSpeak-NG uses voice variants:
    # +m1..+m7 → different male variants
    # +f1..+f4 → different female variants
    # You combine them with the base voice code:
    # en-us+m1 → US English, male 1
    # en-us+m3 → US English, deeper / different male
    # en-us+f1 → US English, female 1
    # en-us+f3 → US English, different female
    main()
