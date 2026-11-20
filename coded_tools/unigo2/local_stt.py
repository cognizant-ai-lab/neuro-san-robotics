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
import time
from typing import Optional


_model = None
_model_lock = threading.Lock()

# One decode at a time. faster-whisper's WhisperModel is not safe to call from
# two threads at once, and with GO2_STT_ENGINE=local both the mic button and
# ambient listening post to the same route, so they do collide in practice. The
# symptom is a request that never returns rather than an error.
_decode_lock = threading.Lock()


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
    event loop. CTranslate2 releases the GIL while it decodes, so waiting here
    does not stall the rest of the process.

    Serialised: the model cannot decode two clips at once, and callers that
    queue up simply wait their turn.
    """
    whisper = model()
    started = time.monotonic()
    with _decode_lock:
        waited = time.monotonic() - started
        if waited > 1.0:
            logging.info("Local transcription waited %.1fs for the model", waited)
        segments, _info = whisper.transcribe(
            audio_path,
            language=language or os.environ.get("GO2_STT_LANGUAGE", "en"),
        )
        # The generator is consumed inside the lock: that is where the decoding
        # actually happens, so releasing early would not serialise anything.
        text = "".join(segment.text for segment in segments).strip()
    logging.info("Local transcription took %.1fs (%d chars)",
                 time.monotonic() - started, len(text))
    return text


def warm_in_background() -> None:
    """
    Load the model now, off the request path.

    Otherwise the first person to press the mic button pays for the load, which
    on the robot's CPU is slow enough to look like a hang: the button sits on
    "transcribing" with nothing coming back. Started as a daemon so it cannot
    hold up shutdown, and errors are logged rather than raised because a warm-up
    failing should not stop the app from starting.
    """
    if engine() != "local" or not _importable():
        return

    def load():
        try:
            started = time.monotonic()
            model()
            logging.info("Local Whisper warmed in %.1fs", time.monotonic() - started)
        except Exception:
            logging.exception("Could not warm the local speech model")

    threading.Thread(target=load, name="whisper-warmup", daemon=True).start()


def ambient_mode(hosted_key_present: bool) -> str:
    """
    How always-on listening should run: "realtime", "local" or "unavailable".

    The browser asks this before it opens a microphone, so that a robot with no
    hosted recogniser goes straight to listening locally instead of negotiating
    a WebRTC session that cannot succeed.

    "local" here segments on silence rather than on a clock. An earlier version
    of ambient listening posted a recording every five seconds and was replaced
    precisely because that was slow and cut words in half; falling back to that
    would undo the change rather than stand in for it.
    """
    choice = engine()
    if choice == "local":
        # Chosen deliberately, so the weights may still be downloading.
        return "local" if _importable() else "unavailable"
    if choice == "openai":
        return "realtime" if hosted_key_present else "unavailable"
    if hosted_key_present:
        return "realtime"
    return "local" if available() else "unavailable"


def _importable() -> bool:
    """Whether faster-whisper is installed, regardless of cached weights."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def describe() -> str:
    """One-line summary for start-up logs."""
    if not available():
        return "unavailable (pip install faster-whisper)"
    return f"whisper-{model_size()} on {device()}"
