"""
Offline speech-to-text, for when no hosted recogniser is reachable.

Text-to-speech already degrades when the network or the region lets it down:
it falls through to Piper and then espeak. Speech-to-text had no such path, so
a site whose Azure region carries no whisper and no gpt-4o-transcribe lost
voice input entirely. This closes the push-to-talk half of that.

It runs Whisper, the same model family as the hosted whisper-1 it stands in
for, through faster-whisper. That keeps the fallback boring: no second model
family to evaluate, and faster-whisper fetches its own weights on first use
and decodes webm/mp3/m4a itself, so there is no model installer and no ffmpeg
step to go wrong.

Environment:
  GO2_STT_ENGINE    "auto" (default), "openai" or "local".
                    auto   -- hosted first, falling back to local
                    openai -- hosted only, failures surface
                    local  -- skip the hosted call entirely, which is what a
                              site with no hosted recogniser wants: it avoids
                              a doomed request in front of every utterance
  GO2_STT_MODEL     whisper size: tiny, base (default), small, medium
  GO2_STT_DEVICE    "auto" (default), "cpu" or "cuda"
  GO2_STT_LANGUAGE  language hint, default "en"
"""

import logging
import os
import threading
from typing import Optional


_model = None
_model_lock = threading.Lock()


def engine() -> str:
    """Return the configured engine preference, normalised."""
    value = os.environ.get("GO2_STT_ENGINE", "auto").strip().lower()
    return value if value in {"auto", "openai", "local"} else "auto"


def model_size() -> str:
    """Which Whisper size to run."""
    return os.environ.get("GO2_STT_MODEL", "").strip() or "base"


def device() -> str:
    """
    Where to run it.

    Defaults to the CPU rather than the GPU. This robot's Orin already carries
    YOLO under TensorRT and deepface, and on a 16 GB Orin NX memory is the
    binding constraint, so a background recogniser stays off the GPU unless
    someone asks for it.
    """
    value = os.environ.get("GO2_STT_DEVICE", "auto").strip().lower()
    return "cpu" if value in {"auto", ""} else value


def is_model_cached() -> bool:
    """
    Whether the weights are already on disk.

    faster-whisper downloads on first use, and that download is hundreds of
    megabytes. Checking first is what keeps a robot that has never needed the
    fallback from stalling a request for minutes the one time a hosted call
    fails.
    """
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            f"Systran/faster-whisper-{model_size()}", local_files_only=True,
        )
        return True
    except Exception:
        return False


def available() -> bool:
    """
    Whether the automatic fallback may use a local transcription.

    Deliberately stricter than "is it installed": the weights must already be
    cached. An explicit GO2_STT_ENGINE="local" bypasses this and is allowed to
    download, because that is a decision someone made up front and expects to
    pay for once. The automatic path must never surprise a working robot with
    a long download in the middle of a request.
    """
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return is_model_cached()


def model():
    """
    Return the shared model, loading it on first use.

    Loading costs seconds and downloads weights the first time ever, so it
    happens once and is reused. The lock matters because Flask serves requests
    on threads: two people pressing the mic button together would otherwise
    each start a download.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel

                size, where = model_size(), device()
                logging.info("Loading local Whisper (%s, %s)", size, where)
                _model = WhisperModel(
                    size,
                    device=where,
                    # int8 on the CPU is the difference between usable and not;
                    # float16 is what the GPU path wants.
                    compute_type="int8" if where == "cpu" else "float16",
                )
                logging.info("Local Whisper ready")
    return _model


def transcribe(audio_path: str, language: Optional[str] = None) -> str:
    """
    Transcribe a recorded utterance offline.

    Blocking, and meant to be called from a request thread rather than an
    event loop. CTranslate2 releases the GIL while it decodes, so this does not
    stall the rest of the process the way a pure-Python loop would.
    """
    segments, _info = model().transcribe(
        audio_path,
        language=language or os.environ.get("GO2_STT_LANGUAGE", "en"),
    )
    return "".join(segment.text for segment in segments).strip()


def describe() -> str:
    """One-line summary for start-up logs."""
    if not available():
        return "unavailable (pip install faster-whisper)"
    return f"whisper-{model_size()} on {device()}"
