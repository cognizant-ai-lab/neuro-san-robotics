"""
go2_tts.py — TTS for Unitree Go2 EDU speaker (Linux/Jetson) and macOS.

TTS Engine Priority (configurable via GO2_TTS_ENGINE env var):
1. "openai" - OpenAI TTS API with true streaming (lowest latency, requires internet)
2. "piper" - Piper TTS (neural, offline, British female voice)
3. "espeak" - espeak-ng (offline fallback)
4. "auto" (default) - Try OpenAI first, fall back to Piper, then espeak

Linux / Unitree Go2:
- OpenAI TTS: True streaming via chunk transfer encoding (recommended)
- Piper TTS: Neural, offline, British female voice - en_GB-cori-high
- espeak-ng: Fallback via ALSA

macOS:
- OpenAI TTS: True streaming (recommended)
- Native 'say' command (truly blocking, prevents audio overlap)

Features:
- Text sanitization: Removes symbols/formatting that cause TTS to say "asterisk" etc.
- True streaming: Audio plays as it's generated (OpenAI)
- Chunked TTS: Splits long text for Piper/espeak fallback

Environment Variables:
- GO2_TTS_ENGINE: "openai", "piper", "espeak", or "auto" (default: "auto")
- GO2_TTS_DEVICE: "auto" (default), "usb", "onboard", or an ALSA device name
- GO2_ONBOARD_TTS_DEVICE: Onboard ALSA device (default: "plughw:CARD=APE,DEV=0")
- GO2_TTS_WAV_PATH: Retained onboard TTS WAV (default: "/tmp/go2_tts_last.wav")
- OPENAI_API_KEY: Required for OpenAI TTS
- GO2_OPENAI_VOICE: OpenAI voice (default: "coral")
- GO2_OPENAI_MODEL: OpenAI model (default: "gpt-4o-mini-tts")
- GO2_OPENAI_VOLUME_GAIN: Volume amplification factor (default: "2.0" for 2x louder)
- GO2_OPENAI_TIMEOUT_SECONDS: Max seconds to wait for an OpenAI TTS request
- GO2_TTS_DUCK_VOLUME: Mixer percentage held while ducked for a barge-in (default: "20")

Barge-in:
Playback is interruptible. stop_speaking() terminates whatever is on the
speaker and bumps a generation counter so an utterance still being synthesized
is discarded instead of played late; duck_playback() lowers volume first, which
is reversible when the interruption turns out to be a false trigger.
"""

import argparse
import asyncio
import fcntl
import io
import logging
import os
import platform
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import wave
from typing import Any, Callable, Dict, List, Optional

from neuro_san.interfaces.coded_tool import CodedTool


# ---------------------------------------------------------------------
# Configuration (override via env vars if needed)
# ---------------------------------------------------------------------

# TTS Engine selection: "openai", "piper", "espeak", or "auto"
TTS_ENGINE = os.environ.get("GO2_TTS_ENGINE", "auto").lower()

# OpenAI TTS configuration
OPENAI_VOICE = os.environ.get("GO2_OPENAI_VOICE", "coral")
OPENAI_MODEL = os.environ.get("GO2_OPENAI_MODEL", "gpt-4o-mini-tts")
OPENAI_INSTRUCTIONS = os.environ.get(
    "GO2_OPENAI_INSTRUCTIONS",
    "Speak in a friendly, conversational tone with natural pacing."
)
# Volume gain for OpenAI TTS (1.0 = normal, 2.0 = 2x louder, etc.)
# This applies software amplification to the PCM audio data
OPENAI_VOLUME_GAIN = float(os.environ.get("GO2_OPENAI_VOLUME_GAIN", "3.0"))
OPENAI_PCM_RATE = 24_000
ONBOARD_PCM_RATE = 48_000

# Piper TTS configuration
PIPER_MODEL = os.environ.get(
    "GO2_PIPER_MODEL",
    "/home/unitree/piper_models/en_GB-cori-high.onnx",
)

PIPER_CONFIG = os.environ.get(
    "GO2_PIPER_CONFIG",
    "/home/unitree/piper_models/en_GB-cori-high.onnx.json",
)

ONBOARD_ALSA_DEVICE = os.environ.get(
    "GO2_ONBOARD_TTS_DEVICE",
    "plughw:CARD=APE,DEV=0",
)
DEFAULT_ALSA_DEVICE = os.environ.get("GO2_TTS_DEVICE", "auto")
ONBOARD_WAV_PATH = os.environ.get("GO2_TTS_WAV_PATH", "/tmp/go2_tts_last.wav").strip()

# ALSA mixer control name for volume (common names: "Master", "PCM", "Speaker")
# Set via env var if the default doesn't work on your hardware
ALSA_MIXER_CONTROL = os.environ.get("GO2_ALSA_MIXER", "Master")

# Default volume percentage to set before each playback (0-100)
DEFAULT_VOLUME_PERCENT = int(os.environ.get("GO2_TTS_VOLUME", "100"))

# Lock file for TTS to prevent audio overlap
TTS_LOCK_FILE = "/tmp/go2_tts.lock"
_ONBOARD_SPEAKER_LOCK = threading.Lock()
_ONBOARD_SPEAKER_ENABLED = False
_ONBOARD_SPEAKER_CLIENT: Any = None


def _env_float(name: str, default: float) -> float:
    """Parse float environment variables with a safe fallback."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


OPENAI_TIMEOUT_SECONDS = _env_float("GO2_OPENAI_TIMEOUT_SECONDS", 20.0)

# Volume percentage held while the robot is ducked for a possible barge-in.
DUCK_VOLUME_PERCENT = int(os.environ.get("GO2_TTS_DUCK_VOLUME", "20"))


# ---------------------------------------------------------------------
# Interruptible playback (barge-in support)
# ---------------------------------------------------------------------

class SpeechInterrupted(RuntimeError):
    """
    Raised when playback stopped because a barge-in cancelled it.

    Callers must not treat this as an engine failure: it is the expected
    outcome of stop_speaking() and must never trigger an offline-TTS retry.
    """


class _PlaybackController:
    """Track live audio subprocesses so a barge-in can stop them immediately."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: List[Any] = []
        self._generation = 0
        self._ducked = False

    @property
    def generation(self) -> int:
        """Return the current utterance generation; every cancel bumps it."""
        with self._lock:
            return self._generation

    @property
    def ducked(self) -> bool:
        """Return whether output is currently held at the ducked volume."""
        with self._lock:
            return self._ducked

    def is_active(self) -> bool:
        """Return whether any registered playback process is still running."""
        with self._lock:
            processes = list(self._processes)
        return any(not _process_exited(proc) for proc in processes)

    def register(self, proc: Any) -> None:
        """Track a playback process for the lifetime of its audio."""
        with self._lock:
            self._processes.append(proc)

    def unregister(self, proc: Any) -> None:
        """Stop tracking a playback process that finished on its own."""
        with self._lock:
            if proc in self._processes:
                self._processes.remove(proc)

    def cancel(self) -> tuple:
        """
        Bump the generation and terminate everything currently playing.

        Returns:
            (stopped_audio, was_ducked). The caller restores the mixer, which
            must happen outside this lock because it shells out to amixer.
        """
        with self._lock:
            self._generation += 1
            processes = list(self._processes)
            self._processes.clear()
            was_ducked = self._ducked
            self._ducked = False
        stopped = False
        for proc in processes:
            if not _process_exited(proc):
                stopped = True
            _terminate_process(proc)
        return stopped, was_ducked

    def set_ducked(self, ducked: bool) -> bool:
        """Record the duck state and report whether it actually changed."""
        with self._lock:
            if self._ducked == ducked:
                return False
            self._ducked = ducked
            return True


_PLAYBACK = _PlaybackController()


def _process_exited(proc: Any) -> bool:
    """Return whether a sync or asyncio subprocess has already finished."""
    poll = getattr(proc, "poll", None)
    try:
        if callable(poll):
            return poll() is not None
        return proc.returncode is not None
    except (OSError, ValueError):
        return True


def _terminate_process(proc: Any) -> None:
    """Stop one playback subprocess, sync or asyncio, without raising."""
    try:
        if _process_exited(proc):
            return
        proc.terminate()
    except (OSError, ValueError):
        logging.debug("GO2_TTS: could not terminate playback process", exc_info=True)
        return

    waiter = getattr(proc, "wait", None)
    if waiter is None or asyncio.iscoroutinefunction(waiter):
        # asyncio.subprocess.Process is reaped by the coroutine that owns it.
        return
    try:
        waiter(timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            waiter(timeout=0.5)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            logging.debug("GO2_TTS: playback process ignored SIGKILL", exc_info=True)
    except (OSError, ValueError):
        logging.debug("GO2_TTS: playback process wait failed", exc_info=True)


def stop_speaking() -> bool:
    """
    Cancel any in-flight TTS playback.

    Returns:
        True when audio was actually interrupted, False when nothing was
        playing. The generation is bumped either way, so an utterance that is
        still being synthesized is also discarded before it reaches a speaker.
    """
    stopped, was_ducked = _PLAYBACK.cancel()
    if was_ducked:
        # Leaving the mixer down would mute the next utterance too.
        _apply_mixer_volume(DEFAULT_VOLUME_PERCENT)
    logging.info("GO2_TTS: stop_speaking (interrupted_audio=%s)", stopped)
    return stopped


def playback_generation() -> int:
    """Return the generation a caller should capture before starting playback."""
    return _PLAYBACK.generation


def is_speaking() -> bool:
    """Return whether audio is playing right now."""
    return _PLAYBACK.is_active()


def duck_playback(ducked: bool) -> None:
    """
    Lower or restore output volume mid-utterance.

    Ducking is the first response to a possible barge-in: it is instant and
    reversible, so a false trigger (a cough, or the robot hearing itself) costs
    a brief volume dip instead of a truncated sentence.
    """
    if not _PLAYBACK.set_ducked(ducked):
        return
    logging.info("GO2_TTS: %s playback", "ducking" if ducked else "restoring")
    _apply_mixer_volume(DUCK_VOLUME_PERCENT if ducked else DEFAULT_VOLUME_PERCENT)


def _raise_if_cancelled(generation: int) -> None:
    """Abort the current utterance when a barge-in superseded it."""
    if _PLAYBACK.generation != generation:
        raise SpeechInterrupted("playback cancelled by barge-in")


def _popen_playback(cmd: List[str], **kwargs: Any) -> subprocess.Popen:
    """Start a playback subprocess that stop_speaking() is able to terminate."""
    proc = subprocess.Popen(cmd, **kwargs)  # nosec B603 - fixed audio commands
    _PLAYBACK.register(proc)
    return proc


def _run_playback(
    cmd: List[str],
    *,
    generation: int | None = None,
    capture_output: bool = False,
    check: bool = False,
    input: bytes | str | None = None,  # pylint: disable=redefined-builtin
    stdin: Any = None,
    text: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """
    Run a blocking playback command that a barge-in can interrupt.

    Mirrors the subprocess.run() arguments this module uses, but registers the
    process for cancellation and converts a cancel into SpeechInterrupted so
    callers never mistake a barge-in for an engine failure worth retrying.
    """
    if generation is None:
        generation = _PLAYBACK.generation
    _raise_if_cancelled(generation)

    pipe = subprocess.PIPE
    proc = _popen_playback(
        cmd,
        stdin=pipe if input is not None else stdin,
        stdout=pipe if capture_output else None,
        stderr=pipe if capture_output else None,
        text=text,
    )
    try:
        stdout, stderr = proc.communicate(input=input, timeout=timeout)
    finally:
        _PLAYBACK.unregister(proc)

    _raise_if_cancelled(generation)

    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, stdout, stderr)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


# ---------------------------------------------------------------------
# OpenAI TTS with True Streaming
# ---------------------------------------------------------------------

def _is_openai_available() -> bool:
    """Check if OpenAI TTS is available (API key set)."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def _amplify_pcm_chunk(chunk: bytes, gain: float) -> bytes:
    """
    Amplify a PCM audio chunk by applying a gain factor.

    Args:
        chunk: Raw PCM audio data (16-bit signed little-endian)
        gain: Volume gain factor (1.0 = no change, 2.0 = 2x louder)

    Returns:
        Amplified PCM audio data
    """
    if gain == 1.0:
        return chunk

    import struct

    # PCM is 16-bit signed little-endian, so 2 bytes per sample
    num_samples = len(chunk) // 2
    samples = struct.unpack(f"<{num_samples}h", chunk)

    # Apply gain with clipping to prevent overflow
    amplified = []
    for sample in samples:
        new_sample = int(sample * gain)
        # Clip to 16-bit signed range
        new_sample = max(-32768, min(32767, new_sample))
        amplified.append(new_sample)

    return struct.pack(f"<{num_samples}h", *amplified)


def _is_onboard_audio_device(device: str) -> bool:
    """Return whether an ALSA device targets the Jetson APE speaker route."""
    normalized = device.replace(" ", "").lower()
    onboard = ONBOARD_ALSA_DEVICE.replace(" ", "").lower()
    return normalized == onboard or "card=ape" in normalized


def _prepare_openai_pcm_chunk(chunk: bytes, gain: float, device: str) -> bytes:
    """Amplify OpenAI PCM and upsample it for the fixed-rate APE route."""
    amplified = _amplify_pcm_chunk(chunk, gain)
    if not _is_onboard_audio_device(device):
        return amplified

    # The internal speaker route was verified at 48 kHz. OpenAI supplies
    # 24-kHz PCM, so duplicate each 16-bit sample for a streaming-safe 2x
    # upsample without adding an ffmpeg/sox runtime dependency.
    return b"".join(amplified[index:index + 2] * 2 for index in range(0, len(amplified), 2))


def _pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int) -> bytes:
    """Wrap mono 16-bit PCM in a WAV container for reliable APE playback."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return output.getvalue()


def _play_onboard_wav(pcm_data: bytes, device: str, generation: int | None = None) -> None:
    """Play buffered PCM through the same WAV-file path as the hardware probe."""
    if generation is None:
        generation = _PLAYBACK.generation
    # The APE route renders the whole utterance before any of it is audible, so
    # a barge-in during synthesis must drop it here rather than start playback.
    _raise_if_cancelled(generation)
    wav_data = _pcm_to_wav_bytes(pcm_data, ONBOARD_PCM_RATE)
    temp_path = ""
    retain_wav = bool(ONBOARD_WAV_PATH)
    try:
        if retain_wav:
            temp_path = os.path.abspath(os.path.expanduser(ONBOARD_WAV_PATH))
            parent_dir = os.path.dirname(temp_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(temp_path, "wb") as output_file:
                output_file.write(wav_data)
            logging.info("GO2_TTS: retained onboard WAV at %s", temp_path)
        else:
            with tempfile.NamedTemporaryFile(
                prefix="go2-tts-",
                suffix=".wav",
                delete=False,
            ) as temp_file:
                temp_file.write(wav_data)
                temp_path = temp_file.name

        logging.info(
            "GO2_TTS: playing onboard WAV (%d PCM bytes, %d WAV bytes)",
            len(pcm_data),
            len(wav_data),
        )
        result = _run_playback(
            ["aplay", "-D", device, temp_path],
            generation=generation,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"aplay returned {result.returncode}: "
                f"{result.stderr.decode(errors='ignore')}"
            )
        logging.info("GO2_TTS: onboard WAV playback completed")
    finally:
        if temp_path and not retain_wav:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _ensure_onboard_speaker_enabled(device: str) -> None:
    """Enable the Go2 internal speaker through VUI before APE playback."""
    global _ONBOARD_SPEAKER_CLIENT, _ONBOARD_SPEAKER_ENABLED

    if not _is_onboard_audio_device(device) or _ONBOARD_SPEAKER_ENABLED:
        return

    with _ONBOARD_SPEAKER_LOCK:
        if _ONBOARD_SPEAKER_ENABLED:
            return

        errors: list[Exception] = []
        for root in ("unitree_sdk2_python.unitree_sdk2py", "unitree_sdk2py"):
            try:
                channel_module = __import__(
                    f"{root}.core.channel",
                    fromlist=["ChannelFactoryInitialize"],
                )
                vui_module = __import__(
                    f"{root}.go2.vui.vui_client",
                    fromlist=["VuiClient"],
                )

                def enable_speaker():
                    client = vui_module.VuiClient()
                    client.SetTimeout(3.0)
                    client.Init()
                    result = client.SetSwitch(1)
                    return client, result

                try:
                    client, code = enable_speaker()
                except AttributeError as exc:
                    if "_ref" not in str(exc):
                        raise
                    # The Flask TTS worker may not share the DDS initialization
                    # performed by the agent subprocess. Initialize this process
                    # exactly as the successful standalone speaker probe does.
                    network_interface = (
                        os.environ.get("GO2_NETWORK_INTERFACE")
                        or os.environ.get("CYCLONEDDS_NETWORK_INTERFACE")
                    )
                    if network_interface:
                        channel_module.ChannelFactoryInitialize(0, network_interface)
                    else:
                        channel_module.ChannelFactoryInitialize(0)
                    logging.info(
                        "GO2_TTS: initialized Unitree DDS%s",
                        f" on {network_interface}" if network_interface else "",
                    )
                    client, code = enable_speaker()

                if code != 0:
                    raise RuntimeError(f"VUI SetSwitch failed with code {code}")
                volume_code, volume = client.GetVolume()
                if volume_code == 0:
                    logging.info("GO2_TTS: onboard speaker enabled; VUI volume=%s", volume)
                else:
                    logging.info("GO2_TTS: onboard speaker enabled")
                # Keep the VUI RPC client alive while ALSA uses the APE route.
                # The standalone hardware probe retains this reference through
                # playback; releasing it early leaves the route silent even
                # though aplay accepts every sample.
                _ONBOARD_SPEAKER_CLIENT = client
                _ONBOARD_SPEAKER_ENABLED = True
                return
            except Exception as exc:
                errors.append(exc)

        logging.warning(
            "GO2_TTS: could not enable onboard speaker through VUI: %s",
            "; ".join(str(error) for error in errors),
        )


def _openai_say_streaming(
    text: str,
    voice: str = OPENAI_VOICE,
    model: str = OPENAI_MODEL,
    instructions: str = OPENAI_INSTRUCTIONS,
    alsa_device: str | None = None,
    volume: float = 1.0,
) -> None:
    """
    Speak text using OpenAI TTS with true streaming.

    Audio starts playing as soon as the first chunks arrive from the API,
    providing the lowest possible latency.

    Args:
        text: Text to speak (should already be sanitized)
        voice: OpenAI voice name (default from env var)
        model: OpenAI model (default from env var)
        instructions: Voice instructions for tone/style
        alsa_device: ALSA device for Linux (default from env var)
        volume: Volume level 0.0-1.0 (default 1.0)
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    client = OpenAI(timeout=OPENAI_TIMEOUT_SECONDS, max_retries=0)
    system = platform.system()
    device = alsa_device or _RESOLVED_ALSA_DEVICE
    generation = _PLAYBACK.generation

    # Apply volume gain for louder output
    gain = OPENAI_VOLUME_GAIN

    logging.info(
        "GO2_TTS: OpenAI streaming TTS (voice=%s, model=%s, gain=%.1f)",
        voice, model, gain
    )

    # Set ALSA volume before playback
    if system == "Linux":
        _ensure_onboard_speaker_enabled(device)
        volume_percent = int(volume * DEFAULT_VOLUME_PERCENT)
        _set_alsa_volume(volume_percent)

    # Request PCM format for lowest latency (no decoding overhead)
    # PCM is 24kHz, 16-bit signed, little-endian
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        instructions=instructions,
        response_format="pcm",
    ) as response:
        if system == "Linux":
            output_rate = ONBOARD_PCM_RATE if _is_onboard_audio_device(device) else OPENAI_PCM_RATE
            logging.info(
                "GO2_TTS: streaming PCM to %s at %d Hz",
                device,
                output_rate,
            )
            if _is_onboard_audio_device(device):
                buffered = []
                for chunk in response.iter_bytes(chunk_size=4096):
                    _raise_if_cancelled(generation)
                    buffered.append(_prepare_openai_pcm_chunk(chunk, gain, device))
                pcm_data = b"".join(buffered)
                if not pcm_data:
                    raise RuntimeError("OpenAI TTS returned no PCM audio")
                _play_onboard_wav(pcm_data, device, generation=generation)
                logging.info("GO2_TTS: OpenAI streaming complete")
                return

            # Stream directly to aplay
            aplay_cmd = [
                "aplay",
                "-D", device,
                "-r", str(output_rate),
                "-f", "S16_LE",
                "-c", "1",
                "-t", "raw",
            ]

            aplay_proc = _popen_playback(
                aplay_cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            interrupted = False
            try:
                for chunk in response.iter_bytes(chunk_size=4096):
                    if _PLAYBACK.generation != generation:
                        interrupted = True
                        break
                    if aplay_proc.stdin:
                        # Apply volume gain to PCM audio
                        amplified_chunk = _prepare_openai_pcm_chunk(chunk, gain, device)
                        try:
                            aplay_proc.stdin.write(amplified_chunk)
                        except (BrokenPipeError, ValueError):
                            # stop_speaking() closed the pipe underneath us.
                            interrupted = True
                            break
            finally:
                _PLAYBACK.unregister(aplay_proc)
                try:
                    if aplay_proc.stdin:
                        aplay_proc.stdin.close()
                except (BrokenPipeError, OSError, ValueError):
                    pass
                aplay_proc.wait()

                if aplay_proc.returncode != 0 and not interrupted:
                    stderr = aplay_proc.stderr.read() if aplay_proc.stderr else b""
                    logging.error(
                        "aplay returned %d: %s",
                        aplay_proc.returncode,
                        stderr.decode(errors="ignore")
                    )

            _raise_if_cancelled(generation)

        elif system == "Darwin":
            # On macOS, use afplay with a temp file or ffplay for streaming
            # For simplicity, collect chunks and play via afplay
            import tempfile

            audio_data = b"".join(response.iter_bytes(chunk_size=4096))

            with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            try:
                # Convert raw PCM to playable format and play
                # ffplay can handle raw PCM directly
                if shutil.which("ffplay"):
                    _run_playback(
                        [
                            "ffplay",
                            "-autoexit",
                            "-nodisp",
                            "-f", "s16le",
                            "-ar", "24000",
                            "-ac", "1",
                            temp_path,
                        ],
                        generation=generation,
                        check=True,
                        capture_output=True,
                    )
                else:
                    # Fallback: convert to wav and use afplay
                    wav_path = temp_path + ".wav"
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-f", "s16le",
                            "-ar", "24000",
                            "-ac", "1",
                            "-i", temp_path,
                            "-y",
                            wav_path,
                        ],
                        check=True,
                        capture_output=True,
                    )
                    _run_playback(["afplay", wav_path], generation=generation, check=True)
                    os.unlink(wav_path)
            finally:
                os.unlink(temp_path)

        else:
            raise RuntimeError(f"OpenAI TTS not supported on OS={system}")

    logging.info("GO2_TTS: OpenAI streaming complete")


async def _openai_say_streaming_async(
    text: str,
    voice: str = OPENAI_VOICE,
    model: str = OPENAI_MODEL,
    instructions: str = OPENAI_INSTRUCTIONS,
    alsa_device: str | None = None,
    volume: float = 1.0,
) -> None:
    """
    Async version of OpenAI TTS streaming.

    Uses the async OpenAI client for better integration with async code.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    client = AsyncOpenAI(timeout=OPENAI_TIMEOUT_SECONDS, max_retries=0)
    system = platform.system()
    device = alsa_device or _RESOLVED_ALSA_DEVICE
    generation = _PLAYBACK.generation

    # Apply volume gain for louder output
    gain = OPENAI_VOLUME_GAIN

    logging.info(
        "GO2_TTS: OpenAI async streaming TTS (voice=%s, model=%s, gain=%.1f)",
        voice, model, gain
    )

    # Set ALSA volume before playback
    if system == "Linux":
        _ensure_onboard_speaker_enabled(device)
        volume_percent = int(volume * DEFAULT_VOLUME_PERCENT)
        _set_alsa_volume(volume_percent)

    async with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        instructions=instructions,
        response_format="pcm",
    ) as response:
        if system == "Linux":
            output_rate = ONBOARD_PCM_RATE if _is_onboard_audio_device(device) else OPENAI_PCM_RATE
            logging.info(
                "GO2_TTS: async streaming PCM to %s at %d Hz",
                device,
                output_rate,
            )
            if _is_onboard_audio_device(device):
                chunks = []
                async for chunk in response.iter_bytes(chunk_size=4096):
                    _raise_if_cancelled(generation)
                    chunks.append(_prepare_openai_pcm_chunk(chunk, gain, device))
                pcm_data = b"".join(chunks)
                if not pcm_data:
                    raise RuntimeError("OpenAI TTS returned no PCM audio")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, _play_onboard_wav, pcm_data, device, generation
                )
                logging.info("GO2_TTS: OpenAI async streaming complete")
                return

            aplay_cmd = [
                "aplay",
                "-D", device,
                "-r", str(output_rate),
                "-f", "S16_LE",
                "-c", "1",
                "-t", "raw",
            ]

            aplay_proc = await asyncio.create_subprocess_exec(
                *aplay_cmd,
                stdin=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _PLAYBACK.register(aplay_proc)

            interrupted = False
            try:
                async for chunk in response.iter_bytes(chunk_size=4096):
                    if _PLAYBACK.generation != generation:
                        interrupted = True
                        break
                    if aplay_proc.stdin:
                        # Apply volume gain to PCM audio
                        amplified_chunk = _prepare_openai_pcm_chunk(chunk, gain, device)
                        try:
                            aplay_proc.stdin.write(amplified_chunk)
                            await aplay_proc.stdin.drain()
                        except (BrokenPipeError, ConnectionResetError, ValueError):
                            # stop_speaking() closed the pipe underneath us.
                            interrupted = True
                            break
            finally:
                _PLAYBACK.unregister(aplay_proc)
                try:
                    if aplay_proc.stdin:
                        aplay_proc.stdin.close()
                        await aplay_proc.stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
                    pass
                await aplay_proc.wait()

                if aplay_proc.returncode != 0 and not interrupted:
                    stderr = (await aplay_proc.stderr.read()) if aplay_proc.stderr else b""
                    logging.error(
                        "aplay returned %d: %s",
                        aplay_proc.returncode,
                        stderr.decode(errors="ignore")
                    )

            _raise_if_cancelled(generation)

        elif system == "Darwin":
            import tempfile

            chunks = []
            async for chunk in response.iter_bytes(chunk_size=4096):
                chunks.append(chunk)
            audio_data = b"".join(chunks)

            with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            try:
                if shutil.which("ffplay"):
                    proc = await asyncio.create_subprocess_exec(
                        "ffplay",
                        "-autoexit",
                        "-nodisp",
                        "-f", "s16le",
                        "-ar", "24000",
                        "-ac", "1",
                        temp_path,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    _PLAYBACK.register(proc)
                    try:
                        await proc.wait()
                    finally:
                        _PLAYBACK.unregister(proc)
                else:
                    wav_path = temp_path + ".wav"
                    proc = await asyncio.create_subprocess_exec(
                        "ffmpeg",
                        "-f", "s16le",
                        "-ar", "24000",
                        "-ac", "1",
                        "-i", temp_path,
                        "-y",
                        wav_path,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await proc.wait()
                    _raise_if_cancelled(generation)
                    proc = await asyncio.create_subprocess_exec("afplay", wav_path)
                    _PLAYBACK.register(proc)
                    try:
                        await proc.wait()
                    finally:
                        _PLAYBACK.unregister(proc)
                    os.unlink(wav_path)
                _raise_if_cancelled(generation)
            finally:
                os.unlink(temp_path)

        else:
            raise RuntimeError(f"OpenAI TTS not supported on OS={system}")

    logging.info("GO2_TTS: OpenAI async streaming complete")


def _is_piper_available() -> bool:
    """Check if Piper TTS is available."""
    return (
        _has("piper")
        and os.path.isfile(PIPER_MODEL)
        and os.path.isfile(PIPER_CONFIG)
    )


def _should_use_openai() -> bool:
    """Determine if OpenAI TTS should be used based on configuration."""
    if TTS_ENGINE == "openai":
        return True
    if TTS_ENGINE == "auto" and _is_openai_available():
        return True
    return False


def _should_runtime_fallback_to_offline_tts() -> bool:
    """
    Return whether runtime OpenAI failures should fall back to offline TTS.

    In `auto` mode we prefer OpenAI when it is healthy, but we do not want a
    timeout or transient network issue to silently drop speech. In explicit
    `openai` mode we preserve the failure so the caller can notice it.
    """
    return TTS_ENGINE == "auto"


# ---------------------------------------------------------------------
# Audio Cache for Pre-converted Phrases
# ---------------------------------------------------------------------

# Cache for pre-converted audio: phrase -> audio bytes
_audio_cache: Dict[str, bytes] = {}
_cache_lock = threading.Lock()


def prewarm_audio_cache(phrases: List[str]) -> int:
    """
    Pre-convert a list of phrases to audio and cache them for instant playback.

    Call this at startup with frequently-used phrases (e.g., acknowledgments)
    so they can be played instantly without conversion delay.

    Args:
        phrases: List of phrases to pre-convert

    Returns:
        Number of phrases successfully cached
    """
    if not _has("piper"):
        logging.warning("Piper not available, skipping audio cache prewarm")
        return 0

    if not os.path.isfile(PIPER_MODEL) or not os.path.isfile(PIPER_CONFIG):
        logging.warning("Piper model/config not found, skipping audio cache prewarm")
        return 0

    cached_count = 0
    logging.info("Pre-warming audio cache with %d phrases...", len(phrases))

    for phrase in phrases:
        clean_phrase = sanitize_tts_text(phrase)
        if not clean_phrase:
            continue

        # Skip if already cached
        with _cache_lock:
            if clean_phrase in _audio_cache:
                cached_count += 1
                continue

        try:
            audio_data = _convert_text_to_audio_piper(clean_phrase)
            with _cache_lock:
                _audio_cache[clean_phrase] = audio_data
            cached_count += 1
            logging.debug("Cached phrase: %s", clean_phrase[:30])
        except Exception:
            logging.exception("Failed to cache phrase: %s", phrase[:30])

    logging.info("Audio cache prewarm complete: %d/%d phrases cached",
                 cached_count, len(phrases))
    return cached_count


def get_cached_audio(text: str) -> Optional[bytes]:
    """
    Get pre-converted audio from cache if available.

    Args:
        text: Text to look up (will be sanitized before lookup)

    Returns:
        Cached audio bytes if found, None otherwise
    """
    clean_text = sanitize_tts_text(text)
    with _cache_lock:
        return _audio_cache.get(clean_text)


def say_cached(
    text: str,
    alsa_device: str | None = None,
    volume: float = 1.0,
) -> bool:
    """
    Play pre-cached audio if available, otherwise return False.

    This is the fastest way to play audio - no conversion needed.
    Use for frequently-used phrases like acknowledgments.

    Args:
        text: Text to speak (must have been pre-cached)
        alsa_device: ALSA device for Linux (default from env var)
        volume: Volume level 0.0-1.0 (default 1.0)

    Returns:
        True if cached audio was played, False if not in cache
    """
    audio_data = get_cached_audio(text)
    if audio_data is None:
        return False

    # Set volume and play
    volume_percent = int(volume * DEFAULT_VOLUME_PERCENT)
    _set_alsa_volume(volume_percent)

    generation = _PLAYBACK.generation
    with open(TTS_LOCK_FILE, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            _raise_if_cancelled(generation)
            _play_audio_bytes(audio_data, alsa_device)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return True


# ---------------------------------------------------------------------
# Text Sanitization
# ---------------------------------------------------------------------

def sanitize_tts_text(text: str) -> str:
    """
    Remove symbols and formatting that cause TTS to say unwanted words.

    For example:
    - "*" causes TTS to say "asterisk"
    - "#" causes TTS to say "hash" or "pound"
    - Markdown formatting like **bold** or _italic_ should be stripped

    Args:
        text: Raw text that may contain symbols/formatting

    Returns:
        Cleaned text suitable for TTS
    """
    if not text:
        return ""

    # Remove markdown bold/italic markers (**, *, __, _)
    # Handle **bold** and *italic* patterns
    result = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold** -> bold
    result = re.sub(r'\*([^*]+)\*', r'\1', result)    # *italic* -> italic
    result = re.sub(r'__([^_]+)__', r'\1', result)    # __bold__ -> bold
    result = re.sub(r'_([^_]+)_', r'\1', result)      # _italic_ -> italic

    # Remove standalone asterisks and underscores
    result = re.sub(r'\*+', '', result)
    result = re.sub(r'_+', ' ', result)

    # Remove markdown headers (# ## ### etc.)
    result = re.sub(r'^#+\s*', '', result, flags=re.MULTILINE)

    # Remove markdown bullet points (- or *)
    result = re.sub(r'^\s*[-*]\s+', '', result, flags=re.MULTILINE)

    # Remove markdown code blocks and inline code
    result = re.sub(r'```[^`]*```', '', result, flags=re.DOTALL)
    result = re.sub(r'`([^`]+)`', r'\1', result)

    # Remove markdown links [text](url) -> text
    result = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', result)

    # Remove other problematic symbols
    result = re.sub(r'[#@~^|\\<>{}[\]]', '', result)

    # Clean up multiple spaces and newlines
    result = re.sub(r'\s+', ' ', result)

    return result.strip()


def split_into_chunks(text: str, max_chunk_size: int = 50) -> List[str]:
    """
    Split text into small phrase-level chunks for fast TTS playback.

    Uses small chunks (~50 chars, ~8-10 words) so the first audio plays
    quickly. Splits on natural boundaries: sentence ends, commas, then spaces.

    Args:
        text: Text to split
        max_chunk_size: Maximum characters per chunk (default 50 for fast response)

    Returns:
        List of text chunks
    """
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    chunks = []
    current_chunk = ""

    for word in words:
        # Check if adding this word would exceed max size
        test_chunk = (current_chunk + " " + word).strip() if current_chunk else word

        if len(test_chunk) > max_chunk_size and current_chunk:
            # Save current chunk and start a new one
            chunks.append(current_chunk)
            current_chunk = word
        else:
            current_chunk = test_chunk

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _detect_usb_audio_device() -> str:
    """
    Detect the first USB Audio playback device by parsing 'aplay -l'.

    Returns:
        ALSA device string like "plughw:2,0", or "default" if not found.
    """
    try:
        result = subprocess.run(
            ["aplay", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        for line in result.stdout.splitlines():
            if "USB Audio" in line and line.startswith("card "):
                # e.g. "card 2: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]"
                card = line.split(":")[0].replace("card ", "").strip()
                device = line.split("device ")[1].split(":")[0].strip()
                detected = f"plughw:{card},{device}"
                logging.info("Auto-detected USB audio device: %s", detected)
                return detected
    except Exception as e:
        logging.warning("Failed to auto-detect USB audio device: %s", e)
    logging.warning("No USB Audio device found, falling back to 'default'")
    return "default"


def _resolve_alsa_device(device: str) -> str:
    """Resolve friendly output names to ALSA device strings."""
    normalized = device.strip().lower()
    if normalized in {"onboard", "internal"}:
        logging.info("Using onboard audio device: %s", ONBOARD_ALSA_DEVICE)
        return ONBOARD_ALSA_DEVICE
    if normalized in {"usb", "auto"}:
        return _detect_usb_audio_device()
    return device


# Resolve once at import time so we don't re-run aplay -l on every TTS call
_RESOLVED_ALSA_DEVICE = _resolve_alsa_device(DEFAULT_ALSA_DEVICE)


def _set_alsa_volume(volume_percent: int = DEFAULT_VOLUME_PERCENT) -> None:
    """
    Set ALSA mixer volume before playback to ensure consistent volume.

    This helps prevent volume drift that can occur on some systems where
    other processes may adjust mixer levels between playbacks. A duck that is
    already active wins, so starting a new utterance mid-barge-in cannot undo
    it by resetting the mixer to full volume.

    Args:
        volume_percent: Volume level 0-100 (default from GO2_TTS_VOLUME env var)
    """
    if _PLAYBACK.ducked:
        volume_percent = min(volume_percent, DUCK_VOLUME_PERCENT)
    _apply_mixer_volume(volume_percent)


def _apply_mixer_volume(volume_percent: int) -> None:
    """Push a volume level to the ALSA mixer, ignoring the duck state."""
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
                check=False,
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

    device = alsa_device or _RESOLVED_ALSA_DEVICE

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

    _run_playback(
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
    voice: str = "en-gb+f3",
    alsa_device: str | None = None,
) -> None:
    if not _has("espeak-ng"):
        raise RuntimeError("espeak-ng not installed")

    amp = max(0, min(200, int(volume * 200)))
    device = alsa_device or _RESOLVED_ALSA_DEVICE

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

    p1 = subprocess.Popen(espeak_cmd, stdout=subprocess.PIPE)  # nosec B603
    try:
        _run_playback(aplay_cmd, stdin=p1.stdout, check=True)
    finally:
        if p1.stdout:
            p1.stdout.close()
        p1.wait()


# ---------------------------------------------------------------------
# macOS: native 'say' command (preferred - truly blocking)
# ---------------------------------------------------------------------

MAC_VOICE = os.environ.get("GO2_MAC_VOICE", "Kate")


def _mac_say_via_subprocess(
    text: str,
    rate: int = 150,
) -> None:
    cmd = ["say"]
    if MAC_VOICE:
        cmd.extend(["-v", MAC_VOICE])
    cmd.extend(["-r", str(rate)])
    logging.info("GO2_TTS: macOS say command starting (rate=%d)", rate)
    _run_playback(cmd, input=text, text=True, check=True)
    logging.info("GO2_TTS: macOS say command completed")


# ---------------------------------------------------------------------
# Audio Conversion (for parallel processing)
# ---------------------------------------------------------------------

def _convert_text_to_audio_piper(text: str) -> bytes:
    """
    Convert text to raw audio bytes using Piper TTS.

    This function only converts - it does NOT play the audio.
    Used for parallel conversion while another chunk is playing.

    Args:
        text: Text to convert (should already be sanitized)

    Returns:
        Raw audio bytes (22050 Hz, 16-bit signed LE, mono)

    Raises:
        RuntimeError: If Piper is not available or conversion fails
    """
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
        "-m", PIPER_MODEL,
        "-c", PIPER_CONFIG,
        "--output-raw",
    ]

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

    return stdout


def _play_audio_bytes(audio_data: bytes, alsa_device: str | None = None) -> None:
    """
    Play raw audio bytes via aplay.

    Args:
        audio_data: Raw audio bytes (22050 Hz, 16-bit signed LE, mono)
        alsa_device: ALSA device to use (default from env var)
    """
    device = alsa_device or _RESOLVED_ALSA_DEVICE

    aplay_cmd = [
        "aplay",
        "-D", device,
        "-r", "22050",
        "-f", "S16_LE",
        "-c", "1",
        "-t", "raw",
    ]

    _run_playback(aplay_cmd, input=audio_data, check=True)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def _say_single_chunk(
    text: str,
    rate: int = 150,
    volume: float = 1.0,
    voice: str = "en-gb+f3",
    alsa_device: str | None = None,
) -> None:
    """
    Speak a single chunk of text (internal helper).

    This function handles the actual TTS for one chunk of text.
    It does NOT sanitize text - caller should sanitize before calling.
    """
    system = platform.system()

    if system == "Linux":
        try:
            _linux_say_via_piper(text, volume=volume, alsa_device=alsa_device)
            return
        except SpeechInterrupted:
            raise
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


def say_streaming(
    text: str,
    rate: int = 150,
    volume: float = 1.0,
    voice: str = "en-gb+f3",
    alsa_device: str | None = None,
    max_chunk_size: int = 50,
    on_chunk_start: Optional[Callable[[str, int, int], None]] = None,
) -> None:
    """
    Speak text using streaming TTS.

    If OpenAI is available and configured, uses true streaming where audio
    plays as it's generated. Otherwise falls back to Piper with parallel
    chunk conversion.

    Args:
        text: Text to speak (will be sanitized to remove symbols/formatting)
        rate: Speech rate (words per minute, default 150)
        volume: Volume level 0.0-1.0 (default 1.0)
        voice: Voice name for espeak fallback (default "en-gb+f3")
        alsa_device: ALSA device for Linux (default from env var)
        max_chunk_size: Maximum characters per chunk (default 50 for fast response)
        on_chunk_start: Callback(chunk_text, chunk_index, total_chunks) called
                        when each chunk starts playing
    """
    system = platform.system()

    # Sanitize text to remove symbols that cause TTS issues
    clean_text = sanitize_tts_text(text)
    if not clean_text:
        logging.warning("TTS: Empty text after sanitization, skipping")
        return

    logging.info("TTS streaming: text length=%d chars", len(clean_text))

    # Set ALSA volume once at the start
    volume_percent = int(volume * DEFAULT_VOLUME_PERCENT)
    _set_alsa_volume(volume_percent)

    generation = _PLAYBACK.generation
    with open(TTS_LOCK_FILE, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            _raise_if_cancelled(generation)
            # Use OpenAI streaming if available (true streaming, no chunking needed)
            if _should_use_openai():
                logging.info("TTS: Using OpenAI streaming")
                if on_chunk_start:
                    on_chunk_start(clean_text, 0, 1)
                try:
                    _openai_say_streaming(
                        clean_text,
                        volume=volume,
                        alsa_device=alsa_device,
                    )
                    return
                except SpeechInterrupted:
                    raise
                except Exception as exc:
                    if not _should_runtime_fallback_to_offline_tts():
                        raise
                    logging.warning(
                        "OpenAI TTS failed (%s), falling back to offline TTS",
                        exc,
                    )

            # Fall back to chunked Piper/espeak
            chunks = split_into_chunks(clean_text, max_chunk_size)
            if not chunks:
                return

            logging.info("TTS: Using Piper/espeak with %d chunks", len(chunks))

            if system == "Linux" and _is_piper_available():
                # Use parallel conversion for Piper TTS
                _say_streaming_piper(
                    chunks, alsa_device, on_chunk_start
                )
            else:
                # Fallback: sequential playback for other systems
                for i, chunk in enumerate(chunks):
                    _raise_if_cancelled(generation)
                    if on_chunk_start:
                        on_chunk_start(chunk, i, len(chunks))
                    _say_single_chunk(
                        chunk,
                        rate=rate,
                        volume=volume,
                        voice=voice,
                        alsa_device=alsa_device,
                    )
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _say_streaming_piper(
    chunks: List[str],
    alsa_device: str | None,
    on_chunk_start: Optional[Callable[[str, int, int], None]],
) -> None:
    """
    Stream TTS using Piper with parallel conversion.

    Converts chunk N+1 while playing chunk N for minimal latency.
    """
    total_chunks = len(chunks)
    audio_queue: queue.Queue[Optional[tuple]] = queue.Queue(maxsize=2)
    conversion_error: List[Exception] = []

    def convert_worker():
        """Background worker that converts chunks to audio."""
        try:
            for i, chunk in enumerate(chunks):
                logging.debug("Converting chunk %d/%d: %s...",
                              i + 1, total_chunks, chunk[:30])
                try:
                    audio_data = _convert_text_to_audio_piper(chunk)
                    audio_queue.put((i, chunk, audio_data))
                except Exception as e:
                    logging.exception("Failed to convert chunk %d", i)
                    conversion_error.append(e)
                    audio_queue.put(None)
                    return
            audio_queue.put(None)  # Signal end of conversion
        except Exception as e:
            logging.exception("Conversion worker error")
            conversion_error.append(e)
            audio_queue.put(None)

    # Start conversion in background thread
    converter_thread = threading.Thread(target=convert_worker, daemon=True)
    converter_thread.start()

    # Play audio as it becomes available
    generation = _PLAYBACK.generation
    interrupted = False
    while True:
        item = audio_queue.get()
        if item is None:
            break

        if _PLAYBACK.generation != generation:
            # Keep draining so the converter thread never blocks on a full
            # queue, but stop putting anything else on the speaker.
            interrupted = True
            continue

        chunk_idx, chunk_text, audio_data = item
        logging.debug("Playing chunk %d/%d", chunk_idx + 1, total_chunks)

        # Call the callback before playing (for UI updates)
        if on_chunk_start:
            try:
                on_chunk_start(chunk_text, chunk_idx, total_chunks)
            except Exception:
                logging.exception("on_chunk_start callback failed")

        # Play the audio
        try:
            _play_audio_bytes(audio_data, alsa_device)
        except SpeechInterrupted:
            interrupted = True
            continue
        except Exception:
            logging.exception("Failed to play chunk %d", chunk_idx)

    # Wait for converter to finish
    converter_thread.join(timeout=5.0)

    if interrupted:
        raise SpeechInterrupted("chunked playback cancelled by barge-in")

    # Re-raise any conversion errors
    if conversion_error:
        raise conversion_error[0]


def say(
    text: str,
    rate: int = 150,
    volume: float = 1.0,
    voice: str = "en-gb+f3",
    alsa_device: str | None = None,
    chunked: bool = True,
    max_chunk_size: int = 50,
    on_chunk_start: Optional[Callable[[str, int, int], None]] = None,
) -> None:
    """
    Speak text using TTS with automatic sanitization and optional chunking.

    When OpenAI is available and configured (GO2_TTS_ENGINE="openai" or "auto"),
    uses true streaming where audio plays as it's generated - no chunking needed.

    When using Piper (offline), chunked=True uses parallel conversion to minimize
    latency - converts chunk N+1 while playing chunk N.

    Args:
        text: Text to speak (will be sanitized to remove symbols/formatting)
        rate: Speech rate (words per minute, default 150)
        volume: Volume level 0.0-1.0 (default 1.0)
        voice: Voice name for espeak fallback (default "en-gb+f3")
        alsa_device: ALSA device for Linux (default from env var)
        chunked: If True, use streaming TTS with parallel conversion
        max_chunk_size: Maximum characters per chunk (default 50 for fast response)
        on_chunk_start: Callback(chunk_text, chunk_index, total_chunks) called
                        when each chunk starts playing (only when chunked=True)
    """
    if chunked:
        # Use streaming TTS with parallel conversion
        say_streaming(
            text=text,
            rate=rate,
            volume=volume,
            voice=voice,
            alsa_device=alsa_device,
            max_chunk_size=max_chunk_size,
            on_chunk_start=on_chunk_start,
        )
    else:
        # Speak entire text at once (no chunking)
        clean_text = sanitize_tts_text(text)
        if not clean_text:
            logging.warning("TTS: Empty text after sanitization, skipping")
            return

        # Captured before the lock: an utterance cancelled while it waited its
        # turn behind another process must be dropped, not spoken late.
        generation = _PLAYBACK.generation
        with open(TTS_LOCK_FILE, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                _raise_if_cancelled(generation)
                # Use OpenAI if available
                if _should_use_openai():
                    try:
                        _openai_say_streaming(
                            clean_text,
                            volume=volume,
                            alsa_device=alsa_device,
                        )
                    except SpeechInterrupted:
                        raise
                    except Exception as exc:
                        if not _should_runtime_fallback_to_offline_tts():
                            raise
                        logging.warning(
                            "OpenAI TTS failed (%s), falling back to offline TTS",
                            exc,
                        )
                        _say_single_chunk(
                            clean_text,
                            rate=rate,
                            volume=volume,
                            voice=voice,
                            alsa_device=alsa_device,
                        )
                else:
                    _say_single_chunk(
                        clean_text,
                        rate=rate,
                        volume=volume,
                        voice=voice,
                        alsa_device=alsa_device,
                    )
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


# ---------------------------------------------------------------------
# CodedTool wrapper
# ---------------------------------------------------------------------

async def say_async(
    text: str,
    volume: float = 1.0,
    alsa_device: str | None = None,
) -> None:
    """
    Async version of say() for use in async contexts.

    Currently only supports OpenAI TTS. Falls back to sync say() for other engines.

    Args:
        text: Text to speak (will be sanitized)
        volume: Volume level 0.0-1.0 (default 1.0)
        alsa_device: ALSA device for Linux (default from env var)
    """
    clean_text = sanitize_tts_text(text)
    if not clean_text:
        logging.warning("TTS: Empty text after sanitization, skipping")
        return

    if _should_use_openai():
        try:
            await _openai_say_streaming_async(
                clean_text,
                volume=volume,
                alsa_device=alsa_device,
            )
        except SpeechInterrupted:
            raise
        except Exception as exc:
            if not _should_runtime_fallback_to_offline_tts():
                raise
            logging.warning(
                "OpenAI async TTS failed (%s), falling back to offline TTS",
                exc,
            )
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: _say_single_chunk(
                    clean_text,
                    volume=volume,
                    alsa_device=alsa_device,
                ),
            )
    else:
        # Fall back to sync version in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: say(text, volume=volume, alsa_device=alsa_device),
        )


class Go2TTSTool(CodedTool):
    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return "Missing required 'text'"

        try:
            # Use async version if OpenAI is available for better performance
            if _should_use_openai():
                await say_async(
                    text=text,
                    volume=float(args.get("volume", 1.0)),
                    alsa_device=args.get("alsa_device"),
                )
            else:
                say(
                    text=text,
                    rate=int(args.get("rate", 150)),
                    volume=float(args.get("volume", 1.0)),
                    voice=args.get("voice", "en-gb+f3"),
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
    parser = argparse.ArgumentParser("TTS for Unitree Go2")
    parser.add_argument("text", nargs="*", help="Text to speak")
    parser.add_argument(
        "--engine",
        choices=["openai", "piper", "espeak", "auto"],
        default=None,
        help="TTS engine to use (overrides GO2_TTS_ENGINE env var)",
    )
    args = parser.parse_args()

    # Override engine if specified
    if args.engine:
        global TTS_ENGINE
        TTS_ENGINE = args.engine

    text = " ".join(args.text) if args.text else "Hello from Unitree!"
    say(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
