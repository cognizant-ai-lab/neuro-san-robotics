"""
go2_tts.py — Offline TTS for Unitree Go2 EDU speaker (Linux/Jetson) and macOS.

Linux / Unitree Go2:
- PRIMARY: Piper TTS (neural, offline, natural voice) with persistent process
- FALLBACK: espeak-ng -> ALSA via aplay

macOS:
- Native 'say' command (truly blocking, prevents audio overlap)
- Set GO2_MAC_VOICE env var to specify voice (e.g., "Samantha")

The Piper process is kept running persistently to avoid model loading latency
on each TTS call. The model is loaded once at startup and reused for all
subsequent speech synthesis requests.
"""

import argparse
import atexit
import fcntl
import json
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
# Linux: Persistent Piper TTS Process
# ---------------------------------------------------------------------
# Keep Piper running persistently to avoid model loading latency.
# The model is loaded once and reused for all TTS requests.

class PersistentPiper:
    """
    Manages a persistent Piper TTS process that keeps the model loaded in memory.
    
    This eliminates the model loading latency (several seconds on Jetson) that
    occurs when spawning a new piper process for each TTS request.
    """
    
    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._initialized = False
    
    def _start_process(self) -> None:
        """Start the persistent piper process with JSON input mode."""
        if not _has("piper"):
            raise RuntimeError("piper binary not found on PATH")
        
        if not os.path.isfile(PIPER_MODEL) or not os.path.isfile(PIPER_CONFIG):
            raise RuntimeError(
                f"Piper model/config not found:\n"
                f"  MODEL={PIPER_MODEL}\n"
                f"  CONFIG={PIPER_CONFIG}"
            )
        
        piper_cmd = [
            "piper",
            "--model", PIPER_MODEL,
            "--config", PIPER_CONFIG,
            "--output-raw",
            "--json-input",
        ]
        
        logging.info("GO2_TTS: Starting persistent Piper process...")
        self._process = subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._initialized = True
        logging.info("GO2_TTS: Persistent Piper process started (PID=%d)", self._process.pid)
    
    def _ensure_running(self) -> None:
        """Ensure the piper process is running, restart if needed."""
        if self._process is None or self._process.poll() is not None:
            if self._process is not None:
                logging.warning("GO2_TTS: Piper process died, restarting...")
            self._start_process()
    
    def synthesize(self, text: str, alsa_device: Optional[str] = None) -> None:
        """
        Synthesize speech from text using the persistent piper process.
        
        Sends text as JSON to piper's stdin and streams the raw audio output
        directly to aplay for immediate playback.
        """
        with self._lock:
            self._ensure_running()
            
            device = alsa_device or DEFAULT_ALSA_DEVICE
            
            aplay_cmd = [
                "aplay",
                "-D", device,
                "-r", "22050",
                "-f", "S16_LE",
                "-c", "1",
                "-t", "raw",
            ]
            
            # Send JSON input to piper
            json_input = json.dumps({"text": text.strip()}) + "\n"
            
            logging.info("GO2_TTS: Synthesizing via persistent Piper...")
            
            try:
                # Write to piper's stdin
                self._process.stdin.write(json_input.encode("utf-8"))
                self._process.stdin.flush()
                
                # Read the raw audio output and pipe to aplay
                # Piper outputs raw 16-bit PCM at 22050 Hz
                # We need to read until we get silence or a reasonable timeout
                aplay_proc = subprocess.Popen(
                    aplay_cmd,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                
                # Read audio data in chunks and write to aplay
                # Piper outputs audio for one sentence then waits for more input
                chunk_size = 4096
                silence_threshold = 0.1  # seconds of silence to detect end
                sample_rate = 22050
                bytes_per_sample = 2
                silence_bytes = int(silence_threshold * sample_rate * bytes_per_sample)
                
                total_bytes = 0
                consecutive_empty = 0
                max_empty_reads = 10
                
                while True:
                    # Non-blocking read with small timeout
                    import select
                    ready, _, _ = select.select([self._process.stdout], [], [], 0.05)
                    
                    if ready:
                        chunk = self._process.stdout.read(chunk_size)
                        if chunk:
                            aplay_proc.stdin.write(chunk)
                            aplay_proc.stdin.flush()
                            total_bytes += len(chunk)
                            consecutive_empty = 0
                        else:
                            consecutive_empty += 1
                    else:
                        consecutive_empty += 1
                    
                    # If we've read some audio and then get empty reads, we're done
                    if total_bytes > 0 and consecutive_empty >= max_empty_reads:
                        break
                    
                    # Safety timeout - if no audio after many empty reads, break
                    if total_bytes == 0 and consecutive_empty >= 100:
                        logging.warning("GO2_TTS: No audio received from Piper")
                        break
                
                # Close aplay's stdin to signal end of audio
                aplay_proc.stdin.close()
                aplay_proc.wait()
                
                logging.info("GO2_TTS: Synthesized %d bytes of audio", total_bytes)
                
            except BrokenPipeError:
                logging.error("GO2_TTS: Piper process pipe broken, will restart on next call")
                self._process = None
                raise RuntimeError("Piper process died unexpectedly")
    
    def shutdown(self) -> None:
        """Shutdown the persistent piper process."""
        with self._lock:
            if self._process is not None:
                logging.info("GO2_TTS: Shutting down persistent Piper process...")
                try:
                    self._process.stdin.close()
                    self._process.terminate()
                    self._process.wait(timeout=5)
                except Exception as e:
                    logging.warning("GO2_TTS: Error shutting down Piper: %s", e)
                    try:
                        self._process.kill()
                    except Exception:
                        pass
                self._process = None
                self._initialized = False


# Global persistent piper instance
_persistent_piper: Optional[PersistentPiper] = None
_persistent_piper_lock = threading.Lock()


def _get_persistent_piper() -> PersistentPiper:
    """Get or create the global persistent piper instance."""
    global _persistent_piper
    with _persistent_piper_lock:
        if _persistent_piper is None:
            _persistent_piper = PersistentPiper()
        return _persistent_piper


def _shutdown_persistent_piper() -> None:
    """Shutdown the persistent piper process on exit."""
    global _persistent_piper
    if _persistent_piper is not None:
        _persistent_piper.shutdown()


# Register shutdown handler
atexit.register(_shutdown_persistent_piper)


def _linux_say_via_piper(
    text: str,
    volume: float = 1.0,
    alsa_device: str | None = None,
) -> None:
    """
    Synthesize speech using the persistent Piper process.
    
    Falls back to spawning a new process if the persistent approach fails.
    """
    # Set ALSA volume before playback
    volume_percent = int(volume * DEFAULT_VOLUME_PERCENT)
    _set_alsa_volume(volume_percent)
    
    try:
        piper = _get_persistent_piper()
        piper.synthesize(text, alsa_device)
    except Exception as e:
        logging.warning("GO2_TTS: Persistent Piper failed (%s), falling back to subprocess", e)
        _linux_say_via_piper_subprocess(text, volume, alsa_device)


def _linux_say_via_piper_subprocess(
    text: str,
    volume: float = 1.0,
    alsa_device: str | None = None,
) -> None:
    """Fallback: spawn a new piper process for each TTS request."""
    if not _has("piper"):
        raise RuntimeError("piper binary not found on PATH")

    if not os.path.isfile(PIPER_MODEL) or not os.path.isfile(PIPER_CONFIG):
        raise RuntimeError(
            f"Piper model/config not found:\n"
            f"  MODEL={PIPER_MODEL}\n"
            f"  CONFIG={PIPER_CONFIG}"
        )

    device = alsa_device or DEFAULT_ALSA_DEVICE

    piper_cmd = [
        "piper",
        "--model", PIPER_MODEL,
        "--config", PIPER_CONFIG,
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

    logging.info("GO2_TTS: Piper subprocess -> aplay (%s)", device)

    # Stream audio directly from piper to aplay
    piper_proc = subprocess.Popen(
        piper_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    aplay_proc = subprocess.Popen(
        aplay_cmd,
        stdin=piper_proc.stdout,
        stderr=subprocess.PIPE,
    )

    piper_proc.stdout.close()
    piper_proc.stdin.write((text.strip() + "\n").encode("utf-8"))
    piper_proc.stdin.close()

    aplay_proc.wait()
    piper_returncode = piper_proc.wait()

    if piper_returncode != 0:
        stderr_output = piper_proc.stderr.read().decode(errors="ignore") if piper_proc.stderr else ""
        raise RuntimeError(
            f"Piper failed (rc={piper_returncode}): {stderr_output}"
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
    alsa_device: str | None = None,
) -> None:
    system = platform.system()

    with open(TTS_LOCK_FILE, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
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
