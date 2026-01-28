#!/usr/bin/env python3
"""
go2_tts.py — Offline TTS for Unitree Go2 EDU speaker (Linux/Jetson) and macOS.

Fixes slow Linux latency by **streaming** Piper's raw audio directly into `aplay`
instead of waiting for Piper to finish generating the entire waveform.

Linux / Unitree Go2:
- PRIMARY: Piper TTS (neural, offline, natural voice) streamed -> aplay
- FALLBACK: espeak-ng -> ALSA via aplay

macOS:
- Native 'say' command (blocking, prevents audio overlap)

Env vars:
- GO2_PIPER_MODEL, GO2_PIPER_CONFIG
- GO2_TTS_DEVICE (default: plughw:0,0)
- GO2_ALSA_MIXER (default: Master)
- GO2_TTS_VOLUME (default: 100)
Optional (Linux aplay buffering tweaks; values are in microseconds):
- GO2_APLAY_BUFFER_TIME (e.g., 200000)
- GO2_APLAY_PERIOD_TIME (e.g., 50000)

Piper CLI requirement:
  piper -m MODEL -c CONFIG --output-raw | aplay ...
"""

import argparse
import fcntl
import logging
import os
import platform
import shutil
import subprocess
import threading
from typing import Any, Dict, Optional

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
ALSA_MIXER_CONTROL = os.environ.get("GO2_ALSA_MIXER", "Master")

# Default volume percentage to set before each playback (0-100)
DEFAULT_VOLUME_PERCENT = int(os.environ.get("GO2_TTS_VOLUME", "100"))

# Optional aplay buffering controls (microseconds). Smaller can reduce latency;
# too small can cause underruns on slow CPUs / busy systems.
GO2_APLAY_BUFFER_TIME = os.environ.get("GO2_APLAY_BUFFER_TIME", "").strip()
GO2_APLAY_PERIOD_TIME = os.environ.get("GO2_APLAY_PERIOD_TIME", "").strip()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _read_all(stream) -> bytes:
    """Read all bytes from a file-like stream safely."""
    try:
        if stream is None:
            return b""
        return stream.read() or b""
    except Exception:
        return b""


def _set_alsa_volume(volume_percent: int = DEFAULT_VOLUME_PERCENT) -> None:
    """
    Set ALSA mixer volume before playback to ensure consistent volume.
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
# Linux: Piper TTS (primary) — STREAMED (low latency)
# ---------------------------------------------------------------------

def _linux_say_via_piper(
    text: str,
    volume: float = 1.0,
    alsa_device: Optional[str] = None,
) -> None:
    """
    Stream Piper's raw PCM output directly into aplay to avoid waiting for the
    full waveform to be generated before playback starts.
    """
    if not _has("piper"):
        raise RuntimeError("piper binary not found on PATH")

    if not os.path.isfile(PIPER_MODEL) or not os.path.isfile(PIPER_CONFIG):
        raise RuntimeError(
            "Piper model/config not found:\n"
            f"  MODEL={PIPER_MODEL}\n"
            f"  CONFIG={PIPER_CONFIG}"
        )

    device = alsa_device or DEFAULT_ALSA_DEVICE

    # Set ALSA volume before playback to ensure consistent volume
    volume_percent = int(max(0.0, min(1.0, volume)) * DEFAULT_VOLUME_PERCENT)
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
        "-c", "1",  # Piper outputs mono audio
        "-t", "raw",
        "-q",       # quieter output
    ]

    # Optional latency tuning knobs
    if GO2_APLAY_BUFFER_TIME:
        aplay_cmd.extend(["--buffer-time", GO2_APLAY_BUFFER_TIME])
    if GO2_APLAY_PERIOD_TIME:
        aplay_cmd.extend(["--period-time", GO2_APLAY_PERIOD_TIME])

    logging.info(
        "GO2_TTS: Piper(stream) -> aplay (%s) at %d%% volume", device, volume_percent
    )

    # Start Piper first (we'll feed it text), then start aplay consuming its stdout.
    # bufsize=0 requests unbuffered pipes at the Python layer.
    piper_proc = subprocess.Popen(
        piper_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    # Start aplay immediately, consuming Piper stdout as it is produced.
    aplay_proc = subprocess.Popen(
        aplay_cmd,
        stdin=piper_proc.stdout,  # type: ignore[arg-type]
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    # Important: close our copy of piper stdout so aplay sees EOF properly
    if piper_proc.stdout is not None:
        piper_proc.stdout.close()

    # Capture stderrs in background to avoid deadlocks if buffers fill
    piper_stderr: bytes = b""
    aplay_stderr: bytes = b""

    def _collect_piper_err():
        nonlocal piper_stderr
        piper_stderr = _read_all(piper_proc.stderr)

    def _collect_aplay_err():
        nonlocal aplay_stderr
        aplay_stderr = _read_all(aplay_proc.stderr)

    t1 = threading.Thread(target=_collect_piper_err, daemon=True)
    t2 = threading.Thread(target=_collect_aplay_err, daemon=True)
    t1.start()
    t2.start()

    try:
        # Feed text and close stdin so Piper can start generating immediately.
        if piper_proc.stdin is None:
            raise RuntimeError("Failed to open Piper stdin")

        piper_input = (text.strip() + "\n").encode("utf-8")
        piper_proc.stdin.write(piper_input)
        piper_proc.stdin.flush()
        piper_proc.stdin.close()

        # Wait for aplay to finish playback (it will stop when Piper ends / EOF).
        aplay_rc = aplay_proc.wait()
        piper_rc = piper_proc.wait()

        # Ensure stderr threads have finished
        t1.join(timeout=1)
        t2.join(timeout=1)

        if piper_rc != 0:
            raise RuntimeError(
                f"Piper failed (rc={piper_rc}): {piper_stderr.decode(errors='ignore')}"
            )

        if aplay_rc != 0:
            raise RuntimeError(
                f"aplay failed (rc={aplay_rc}): {aplay_stderr.decode(errors='ignore')}"
            )

    finally:
        # Cleanup: terminate processes if still alive
        for proc in (aplay_proc, piper_proc):
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass


# ---------------------------------------------------------------------
# Linux: eSpeak fallback
# ---------------------------------------------------------------------

def _linux_say_via_espeak(
    text: str,
    rate: int = 150,
    volume: float = 1.0,
    voice: str = "en-us+f3",
    alsa_device: Optional[str] = None,
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

    aplay_cmd = ["aplay", "-D", device, "-q"]

    logging.warning("GO2_TTS: Falling back to espeak-ng")

    p1 = subprocess.Popen(espeak_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        subprocess.run(aplay_cmd, stdin=p1.stdout, check=True)
    finally:
        if p1.stdout:
            p1.stdout.close()
        p1.wait(timeout=10)


# ---------------------------------------------------------------------
# macOS: native 'say' command (preferred - truly blocking)
# ---------------------------------------------------------------------

MAC_VOICE = os.environ.get("GO2_MAC_VOICE", "")


def _mac_say_via_subprocess(
    text: str,
    rate: int = 150,
) -> None:
    cmd = ["say"]
    if MAC_VOICE:
        cmd.extend(["-v", MAC_VOICE])
    cmd.extend(["-r", str(rate)])
    logging.info("GO2_TTS: macOS say command starting (rate=%d)", rate)
    subprocess.run(cmd, input=text, text=True, check=True)
    logging.info("GO2_TTS: macOS say command completed")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

TTS_LOCK_FILE = "/tmp/go2_tts.lock"


def say(
    text: str,
    rate: int = 150,
    volume: float = 1.0,
    voice: str = "en-us+f3",
    alsa_device: Optional[str] = None,
) -> None:
    system = platform.system()

    with open(TTS_LOCK_FILE, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if system == "Linux":
                try:
                    _linux_say_via_piper(text, volume=volume, alsa_device=alsa_device)
                    return
                except Exception:
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
                _mac_say_via_subprocess(text, rate)
                return

            raise RuntimeError(f"TTS not supported on OS={system}")
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


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
