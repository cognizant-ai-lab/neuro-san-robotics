"""
Core TTS functionality for Unitree Go2 EDU speaker (Linux/Jetson) and macOS.

This module contains the core text-to-speech logic without framework dependencies,
making it easy to test and reuse.

Linux / Unitree Go2:
- PRIMARY: Piper TTS (neural, offline, natural voice)
- FALLBACK: espeak-ng -> ALSA via aplay

macOS:
- Native 'say' command (truly blocking, prevents audio overlap)
- Set GO2_MAC_VOICE env var to specify voice (e.g., "Samantha")
"""

import fcntl
import logging
import os
import platform
import shutil
import subprocess
from typing import Optional


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

ALSA_MIXER_CONTROL = os.environ.get("GO2_ALSA_MIXER", "Master")

DEFAULT_VOLUME_PERCENT = int(os.environ.get("GO2_TTS_VOLUME", "100"))

TTS_LOCK_FILE = "/tmp/go2_tts_engine.lock"


# ---------------------------------------------------------------------
# TtsCore Class (persistent model loading)
# ---------------------------------------------------------------------

class TtsCore:
    """
    Text-to-speech engine with persistent model loading.

    Models are loaded once at initialization and reused for all say() calls,
    dramatically improving performance for multiple TTS operations.

    Usage:
        with TtsCore() as engine:
            engine.say("Hello")
            engine.say("World")  # Much faster - model already loaded
    """

    def __init__(self):
        self.system = platform.system()
        self.lock_file = None
        self._initialized = False

    @staticmethod
    def _has(cmd: str) -> bool:
        """Check if a command is available on PATH."""
        return shutil.which(cmd) is not None

    @staticmethod
    def _set_alsa_volume(volume_percent: int = DEFAULT_VOLUME_PERCENT) -> None:
        """
        Set ALSA mixer volume before playback to ensure consistent volume.

        This helps prevent volume drift that can occur on some systems where
        other processes may adjust mixer levels between playbacks.

        Args:
            volume_percent: Volume level 0-100 (default from GO2_TTS_VOLUME env var)
        """
        if not TtsCore._has("amixer"):
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

    def __enter__(self):
        """Context manager entry - acquire lock."""
        self.lock_file = open(TTS_LOCK_FILE, "w")
        fcntl.flock(self.lock_file, fcntl.LOCK_EX)
        self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release lock."""
        if self.lock_file:
            fcntl.flock(self.lock_file, fcntl.LOCK_UN)
            self.lock_file.close()
            self.lock_file = None
        self._initialized = False
        return False

    def say(
        self,
        text: str,
        rate: int = 150,
        volume: float = 1.0,
        voice: str = "en-us+f3",
        alsa_device: Optional[str] = None,
    ) -> None:
        """
        Speak text using platform-appropriate TTS engine.

        Args:
            text: Text to speak
            rate: Speech rate (words per minute, typically 150-200)
            volume: Volume level (0.0-1.0)
            voice: Voice identifier (platform-specific)
            alsa_device: ALSA device for Linux (default: plughw:0,0)
        """
        if not self._initialized:
            raise RuntimeError("TtsCore must be used as a context manager (with TtsCore() as engine:)")

        if self.system == "Linux":
            try:
                self._linux_say_via_piper(text, volume=volume, alsa_device=alsa_device)
                return
            except Exception as e:
                logging.exception("Piper failed, falling back to espeak-ng")
                self._linux_say_via_espeak(
                    text,
                    rate=rate,
                    volume=volume,
                    voice=voice,
                    alsa_device=alsa_device,
                )
                return

        if self.system == "Darwin":
            self._mac_say_via_subprocess(text, rate, volume)
            return

        raise RuntimeError(f"TTS not supported on OS={self.system}")

    def _linux_say_via_piper(
        self,
        text: str,
        volume: float = 1.0,
        alsa_device: Optional[str] = None,
    ) -> None:
        """Linux Piper TTS implementation."""
        if not TtsCore._has("piper"):
            raise RuntimeError("piper binary not found on PATH")

        if not os.path.isfile(PIPER_MODEL) or not os.path.isfile(PIPER_CONFIG):
            raise RuntimeError(
                f"Piper model/config not found:\n"
                f"  MODEL={PIPER_MODEL}\n"
                f"  CONFIG={PIPER_CONFIG}"
            )

        device = alsa_device or DEFAULT_ALSA_DEVICE

        # Set ALSA volume before playback
        volume_percent = int(volume * DEFAULT_VOLUME_PERCENT)
        TtsCore._set_alsa_volume(volume_percent)

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
            "-c", "1",
            "-t", "raw",
        ]

        logging.info("GO2_TTS: Piper -> aplay (%s) at %d%% volume", device, volume_percent)

        # Spawn Piper process
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

    def _linux_say_via_espeak(
        self,
        text: str,
        rate: int = 150,
        volume: float = 1.0,
        voice: str = "en-us+f3",
        alsa_device: Optional[str] = None,
    ) -> None:
        """Linux eSpeak fallback implementation."""
        if not TtsCore._has("espeak-ng"):
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

    def _mac_say_via_subprocess(
        self,
        text: str,
        rate: int = 150,
        volume: float = 1.0,
    ) -> None:
        """macOS say command implementation."""
        # Set system volume before speaking
        volume_percent = int(volume * 100)
        try:
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {volume_percent}"],
                timeout=2,
                check=False
            )
        except Exception as e:
            logging.debug("Failed to set macOS volume: %s", e)

        mac_voice = os.environ.get("GO2_MAC_VOICE", "")
        cmd = ["say"]
        if mac_voice:
            cmd.extend(["-v", mac_voice])
        cmd.extend(["-r", str(rate)])

        logging.info("GO2_TTS: macOS say command starting (rate=%d, volume=%d%%)", rate, volume_percent)
        subprocess.run(cmd, input=text, text=True, check=True)
        logging.info("GO2_TTS: macOS say command completed")


# ---------------------------------------------------------------------
# Convenience function (backward compatible)
# ---------------------------------------------------------------------

def say(
    text: str,
    rate: int = 150,
    volume: float = 1.0,
    voice: str = "en-us+f3",
    alsa_device: Optional[str] = None,
) -> None:
    """
    Convenience function for one-off TTS calls.

    For multiple TTS calls, use TtsCore class for better performance:
        with TtsCore() as engine:
            engine.say("First message")
            engine.say("Second message")  # Much faster!

    Args:
        text: Text to speak
        rate: Speech rate (words per minute, typically 150-200)
        volume: Volume level (0.0-1.0)
        voice: Voice identifier (platform-specific)
        alsa_device: ALSA device for Linux (default: plughw:0,0)
    """
    with TtsCore() as engine:
        engine.say(text, rate=rate, volume=volume, voice=voice, alsa_device=alsa_device)
