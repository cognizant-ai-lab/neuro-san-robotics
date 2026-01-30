"""
go2_tts.py — Offline TTS for Unitree Go2 EDU speaker (Linux/Jetson) and macOS.

Linux / Unitree Go2:
- PRIMARY: Piper TTS (neural, offline, British female voice - en_GB-cori-high)
- FALLBACK: espeak-ng -> ALSA via aplay (British female voice - en-gb+f3)

macOS:
- Native 'say' command (truly blocking, prevents audio overlap)
- Default voice: Kate (British female)
- Set GO2_MAC_VOICE env var to specify a different voice

Features:
- Text sanitization: Removes symbols/formatting that cause TTS to say "asterisk" etc.
- Chunked TTS: Splits long text into sentences and speaks chunks while converting next

Tested with Piper CLI requiring:
  piper -m MODEL -c CONFIG --output-raw | aplay
"""

import argparse
import fcntl
import logging
import os
import platform
import re
import shutil
import subprocess
from typing import Any, Dict, List

from neuro_san.interfaces.coded_tool import CodedTool


# ---------------------------------------------------------------------
# Configuration (override via env vars if needed)
# ---------------------------------------------------------------------

PIPER_MODEL = os.environ.get(
    "GO2_PIPER_MODEL",
    "/home/unitree/piper_models/en_GB-cori-high.onnx",
)

PIPER_CONFIG = os.environ.get(
    "GO2_PIPER_CONFIG",
    "/home/unitree/piper_models/en_GB-cori-high.onnx.json",
)

DEFAULT_ALSA_DEVICE = os.environ.get("GO2_TTS_DEVICE", "plughw:0,0")

# ALSA mixer control name for volume (common names: "Master", "PCM", "Speaker")
# Set via env var if the default doesn't work on your hardware
ALSA_MIXER_CONTROL = os.environ.get("GO2_ALSA_MIXER", "Master")

# Default volume percentage to set before each playback (0-100)
DEFAULT_VOLUME_PERCENT = int(os.environ.get("GO2_TTS_VOLUME", "100"))


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


def split_into_chunks(text: str, max_chunk_size: int = 200) -> List[str]:
    """
    Split text into chunks for progressive TTS playback.

    Splits on sentence boundaries (., !, ?) to create natural pauses.
    If a sentence is too long, it will be split on commas or spaces.

    Args:
        text: Text to split
        max_chunk_size: Maximum characters per chunk (default 200)

    Returns:
        List of text chunks
    """
    if not text:
        return []

    # First, split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence would exceed max size, save current chunk
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        # If the sentence itself is too long, split it further
        if len(sentence) > max_chunk_size:
            # Try splitting on commas first
            parts = re.split(r',\s*', sentence)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if current_chunk and len(current_chunk) + len(part) + 1 > max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                if len(part) > max_chunk_size:
                    # Last resort: split on spaces
                    words = part.split()
                    for word in words:
                        if current_chunk and len(current_chunk) + len(word) + 1 > max_chunk_size:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                        current_chunk = (current_chunk + " " + word).strip()
                else:
                    current_chunk = (current_chunk + " " + part).strip()
        else:
            current_chunk = (current_chunk + " " + sentence).strip()

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


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
    voice: str = "en-gb+f3",
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
    subprocess.run(cmd, input=text, text=True, check=True)
    logging.info("GO2_TTS: macOS say command completed")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

TTS_LOCK_FILE = "/tmp/go2_tts.lock"


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


def say(
    text: str,
    rate: int = 150,
    volume: float = 1.0,
    voice: str = "en-gb+f3",
    alsa_device: str | None = None,
    chunked: bool = True,
    max_chunk_size: int = 200,
) -> None:
    """
    Speak text using TTS with automatic sanitization and optional chunking.

    Args:
        text: Text to speak (will be sanitized to remove symbols/formatting)
        rate: Speech rate (words per minute, default 150)
        volume: Volume level 0.0-1.0 (default 1.0)
        voice: Voice name for espeak fallback (default "en-gb+f3")
        alsa_device: ALSA device for Linux (default from env var)
        chunked: If True, split long text into chunks for faster initial response
        max_chunk_size: Maximum characters per chunk when chunked=True
    """
    # Sanitize text to remove symbols that cause TTS issues
    clean_text = sanitize_tts_text(text)
    if not clean_text:
        logging.warning("TTS: Empty text after sanitization, skipping")
        return

    with open(TTS_LOCK_FILE, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if chunked:
                # Split into chunks and speak progressively
                chunks = split_into_chunks(clean_text, max_chunk_size)
                logging.info("TTS: Speaking %d chunks", len(chunks))
                for i, chunk in enumerate(chunks):
                    logging.debug("TTS: Speaking chunk %d/%d: %s...",
                                  i + 1, len(chunks), chunk[:50])
                    _say_single_chunk(
                        chunk,
                        rate=rate,
                        volume=volume,
                        voice=voice,
                        alsa_device=alsa_device,
                    )
            else:
                # Speak entire text at once
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
    parser = argparse.ArgumentParser("Offline TTS for Unitree Go2")
    parser.add_argument("text", nargs="*", help="Text to speak")
    args = parser.parse_args()

    text = " ".join(args.text) if args.text else "Hello from Unitree!"
    say(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
