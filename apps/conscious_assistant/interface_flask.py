import atexit

import difflib
import logging
import os
import queue
import random
import re
import signal
import sys
import tempfile
import threading
import time

from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CODED_TOOLS_ROOT = REPO_ROOT / "coded_tools"
if str(CODED_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(CODED_TOOLS_ROOT))

logging.basicConfig(
    level=getattr(logging, os.environ.get("CONSCIOUS_LOG_LEVEL", "INFO").upper(), logging.INFO)
)

# pylint: disable=import-error
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask import send_file
from flask_socketio import SocketIO

from apps.conscious_assistant.agent_runtime import AgentRuntime
from scripts import setup_tls_certs as tls_certs
from apps.conscious_assistant.realtime_transcription import create_realtime_client_secret
from apps.conscious_assistant.robot_identity import robot_name
from apps.conscious_assistant.scene_observer import SceneObserver
from coded_tools.unigo2.agent_events import dispatch_agent_event
from coded_tools.unigo2.agent_events import queue_agent_event


# TLS certificate paths used by both Flask and the native runtime callback.
# Managed by apps.conscious_assistant.tls_certs, which is driven by
# TLS_CERT_DIR / ROBOT_HOST_IP from setmyenv.sh. Nothing is pinned in code so a
# fleet of robots -- some on DHCP, some on fixed addresses -- shares this file.
TLS_CERT = tls_certs.cert_path()
TLS_KEY = tls_certs.key_path()


def _ui_event_endpoint() -> str:
    """Return the local callback URL using the same transport as Flask."""
    scheme = "https" if TLS_CERT.exists() and TLS_KEY.exists() else "http"
    return f"{scheme}://127.0.0.1:5001/api/agent-output"

# Import TTS function used when the agent emits a say: block.
try:
    from coded_tools.unigo2.tts_go2 import SpeechInterrupted
    from coded_tools.unigo2.tts_go2 import duck_playback as tts_duck_playback
    from coded_tools.unigo2.tts_go2 import say as tts_say
    from coded_tools.unigo2.tts_go2 import stop_speaking as tts_stop_speaking
    TTS_AVAILABLE = True
except ImportError:
    logging.warning("TTS module not available - speech will be text-only")
    TTS_AVAILABLE = False
    tts_say = None
    tts_stop_speaking = None
    tts_duck_playback = None

    class SpeechInterrupted(RuntimeError):
        """Stand-in so the speech worker can catch barge-ins without TTS installed."""

# An explicit acknowledgement costs a full utterance of latency and queue
# pressure. With barge-in the robot is already interruptible, so this is off
# unless someone deliberately turns it back on.
ACKNOWLEDGE_USER_INPUT = os.environ.get(
    "CONSCIOUS_ACKNOWLEDGE_USER_INPUT",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}

ACKNOWLEDGMENT_PHRASES = [
    "Got it",
    "Okay",
    "On it",
    "Sure",
    "Right away",
    "Coming right up",
    "Let me check",
    "One moment",
    "Give me a second",
    "Let me think",
    "Hold on",
    "Beep boop beep",
]

os.environ.setdefault("AGENT_MANIFEST_FILE", str(REPO_ROOT / "registries" / "manifest.hocon"))
os.environ.setdefault("AGENT_TOOL_PATH", str(REPO_ROOT / "coded_tools"))
os.environ.setdefault("VISION_FACE_DB_PATH", str(REPO_ROOT / "face_database"))
# Flask owns this local callback; an inherited value can use the wrong scheme.
os.environ["CONSCIOUS_UI_EVENT_ENDPOINT"] = _ui_event_endpoint()
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"


@app.context_processor
def inject_template_identity() -> dict:
    """Give every template this robot's name and the current year.

    The lab branding stays hardcoded in the template: every robot lives in a
    Cognizant AI Lab, so only the robot's own name varies between units.
    """
    return {"robot_name": robot_name(), "year": datetime.now().year}


socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")
shutdown_event = threading.Event()

# Speech queue for TTS - allows non-blocking speech processing
speech_queue = queue.Queue()

# --- Barge-in state -------------------------------------------------------
#
# The mic stays open while the robot talks, so the robot hears itself. Two
# mechanisms keep that from turning into a feedback loop:
#
#   * a speech epoch, bumped by every barge-in, that makes queued utterances
#     and in-flight agent turns discard themselves once superseded, and
#   * a self-echo filter, which drops ambient transcripts that match what the
#     robot is saying right now.
#
# Voice-activity detection alone cannot tell the user apart from the robot, so
# a suspected barge-in ducks first and only cancels once a transcript confirms
# it was really a person.
_speech_state_lock = threading.RLock()
_speech_epoch = 0
_pending_turn_epoch = 0
_speech_active = False
_active_speech_text = ""
_active_speech_ended_at = 0.0
_last_barge_in_at = 0.0
_duck_release_timer = None

# How long after playback ends a transcript may still be the robot's own voice.
SELF_ECHO_TAIL_SECONDS = float(os.environ.get("CONSCIOUS_SELF_ECHO_TAIL_SECONDS", "1.5"))
# Fraction of a transcript's words that must appear in the spoken text for it
# to count as the robot hearing itself.
SELF_ECHO_OVERLAP = float(os.environ.get("CONSCIOUS_SELF_ECHO_OVERLAP", "0.6"))
# Consecutive words the robot never said that mark a transcript as a real
# person, whatever the overall overlap. Talking over the robot puts both voices
# in one transcript, and word overlap alone would read that as pure echo.
SELF_ECHO_NOVEL_WORDS = int(os.environ.get("CONSCIOUS_SELF_ECHO_NOVEL_WORDS", "2"))
# A duck with no transcript behind it is a false trigger; restore after this.
DUCK_RELEASE_SECONDS = float(os.environ.get("CONSCIOUS_DUCK_RELEASE_SECONDS", "2.5"))
# Words a transcript needs before it may cut the robot off, so a stray syllable
# or a one-word mis-transcription does not truncate an answer.
BARGE_IN_MIN_WORDS = int(os.environ.get("CONSCIOUS_BARGE_IN_MIN_WORDS", "2"))
# How long after a barge-in agent speech is still presumed to belong to the
# turn that was interrupted. Past this the agent has had time to start
# something new -- a navigation or observation announcement, say -- and
# silencing it would be wrong.
SUPERSEDED_TURN_WINDOW_SECONDS = float(
    os.environ.get("CONSCIOUS_SUPERSEDED_TURN_WINDOW_SECONDS", "12.0")
)
# Output that does not belong to a conversational turn and so cannot be
# superseded by one. A navigation outcome ("I arrived at the kitchen") is
# authoritative and must survive a barge-in that happened to land near it.
UNINTERRUPTIBLE_SPEECH_SOURCES = frozenset({"navigation"})

latest_observation = None
scene_image_path = SceneObserver(enabled=False).latest_image_path()
os.environ.setdefault("VISION_LATEST_IMAGE_PATH", str(scene_image_path))
os.environ.setdefault("VISION_LATEST_IMAGE_MAX_AGE_SECONDS", "0")
agent_runtime = AgentRuntime()


@app.before_request
def log_request_start():
    """Log incoming requests before route handlers can block."""
    logging.info("HTTP request started: %s %s", request.method, request.path)


@app.route("/api/health")
def health():
    """Lightweight readiness probe for browser/server connectivity checks."""
    return jsonify({"ok": True})


@app.route("/api/agent-output", methods=["POST"])
def receive_agent_output():
    """Receive explicit event-agent output from the local Neuro SAN process."""
    expected_token = os.environ.get("CONSCIOUS_UI_EVENT_TOKEN", "")
    if expected_token and request.headers.get("X-Conscious-Bridge-Token") != expected_token:
        return jsonify({"error": "unauthorized"}), 403

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "expected JSON object"}), 400

    thought = payload.get("thought", "")
    say = payload.get("say", "")
    heard = payload.get("heard", "")
    source = str(payload.get("source", ""))
    observation = payload.get("observation")
    if (
        not isinstance(thought, str)
        or not isinstance(say, str)
        or not isinstance(heard, str)
    ):
        return jsonify({"error": "thought, say, and heard must be strings"}), 400

    if heard.strip():
        socketio.emit("update_user_input", {"data": heard.strip()}, namespace="/chat")
    if thought.strip():
        socketio.emit("update_thoughts", {"data": thought.strip()}, namespace="/chat")
    if say.strip():
        # Thoughts stay in the UI log for context, but speech from a turn the
        # user already talked over must not reach the speaker.
        with _speech_state_lock:
            superseded = (
                source not in UNINTERRUPTIBLE_SPEECH_SOURCES
                and _speech_epoch > _pending_turn_epoch
                and time.monotonic() - _last_barge_in_at <= SUPERSEDED_TURN_WINDOW_SECONDS
            )
        if superseded:
            logging.info("Discarding speech from a superseded turn: %s", say.strip()[:60])
        else:
            enqueue_speech(say.strip(), emit_to_ui=True)
    if isinstance(observation, dict):
        global latest_observation  # pylint: disable=global-statement
        latest_observation = observation
        emit_observation_update(observation)

    return jsonify({"ok": True})


def emit_observation_update(observation=None, sid=None):
    """Send the latest observation image and caption data to clients."""
    payload = observation or latest_observation
    if not payload:
        return

    emit_kwargs = {"namespace": "/chat"}
    if sid is not None:
        emit_kwargs["to"] = sid
    socketio.emit("update_observation", payload, **emit_kwargs)



def sanitize_speech_text(text: str) -> str:
    """
    Remove tool-trace garbage from speech text.

    Sometimes the agent embeds function call traces like:
    'functions.say_out_loud ...: <text>'

    This function strips those out to get clean speech text.
    """
    if not text:
        return ""

    # Remove lines that look like function call traces
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        # Skip lines that look like function calls
        if re.match(r'^functions\.\w+', line.strip()):
            continue
        # Skip lines that look like tool invocations
        if re.match(r'^(CALL_TOOL|TOOL_CALL|call)\s*:', line.strip(), re.IGNORECASE):
            continue
        clean_lines.append(line)

    return '\n'.join(clean_lines).strip()


def speak_text_streaming(
    text: str,
    on_speech_complete=None,
) -> None:
    """
    Speak the given text using TTS without chunking.

    Speaks the full text at once for better prosody/tone.
    The callback is called AFTER speech completes to update the UI.

    Args:
        text: Text to speak
        on_speech_complete: Optional callback(text) called after speech finishes
    """
    if not TTS_AVAILABLE or tts_say is None:
        logging.info("TTS not available, skipping speech: %s", text[:50])
        return

    clean_text = sanitize_speech_text(text)
    if not clean_text:
        logging.info("No text to speak after sanitization")
        return

    try:
        logging.info("Speaking TTS utterance (%d chars)", len(clean_text))
        tts_say(clean_text, chunked=False)
        if on_speech_complete:
            on_speech_complete(clean_text)
    except SpeechInterrupted:
        # A barge-in cancelled this utterance; the worker decides what to do.
        raise
    except Exception:
        logging.exception("TTS failed for text: %s", clean_text[:50])


def speak_text(text: str) -> None:
    """
    Speak an agent-authored say: payload using TTS.

    Speaks the full text at once for better prosody.
    """
    speak_text_streaming(text, on_speech_complete=None)


def emit_speech_state(active: bool) -> None:
    """Notify the client that robot speech playback started or stopped."""
    event_name = "speech_started" if active else "speech_complete"
    try:
        socketio.emit(
            event_name,
            {"active": active},
            namespace="/chat",
        )
    except Exception:
        logging.exception("Failed to emit %s", event_name)


def _normalize_for_echo(text: str) -> str:
    """Reduce text to bare lowercase words so echoes compare cleanly."""
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def is_self_echo(transcript: str) -> bool:
    """
    Return whether a transcript is the robot hearing its own voice.

    Only transcripts captured while the robot was speaking (plus a short tail
    for room latency) are candidates, so ordinary conversation is never
    suppressed just because it repeats a word the robot happened to use.
    """
    with _speech_state_lock:
        spoken = _active_speech_text
        speaking = _speech_active
        ended_at = _active_speech_ended_at

    if not spoken:
        return False
    if not speaking and time.monotonic() - ended_at > SELF_ECHO_TAIL_SECONDS:
        return False

    heard_words = _normalize_for_echo(transcript).split()
    spoken_text = _normalize_for_echo(spoken)
    if not heard_words or not spoken_text:
        return False

    spoken_words = set(spoken_text.split())

    # A run of words the robot never said means a person spoke over it. This
    # has to win over the overlap test below: an interruption is captured
    # alongside the robot's own voice, so most of the transcript really is
    # echo, and only the novel run tells the two apart.
    longest_novel_run = 0
    novel_run = 0
    for word in heard_words:
        novel_run = 0 if word in spoken_words else novel_run + 1
        longest_novel_run = max(longest_novel_run, novel_run)
    if longest_novel_run >= SELF_ECHO_NOVEL_WORDS:
        return False

    overlap = sum(1 for word in heard_words if word in spoken_words) / len(heard_words)
    if overlap >= SELF_ECHO_OVERLAP:
        return True

    # A long transcript can drift from the spoken text word-for-word while
    # still clearly being the same sentence read back.
    ratio = difflib.SequenceMatcher(None, " ".join(heard_words), spoken_text).ratio()
    return ratio >= SELF_ECHO_OVERLAP


def _cancel_duck_release() -> None:
    """Drop any pending automatic un-duck."""
    global _duck_release_timer  # pylint: disable=global-statement
    with _speech_state_lock:
        timer = _duck_release_timer
        _duck_release_timer = None
    if timer is not None:
        timer.cancel()


def duck_speech(ducked: bool) -> None:
    """
    Lower or restore speech volume for a suspected barge-in.

    Ducking is instant and reversible, which is what makes it safe to trigger
    on raw voice activity: a cough or the robot's own voice costs a brief dip
    rather than a truncated sentence.
    """
    global _duck_release_timer  # pylint: disable=global-statement

    _cancel_duck_release()
    if TTS_AVAILABLE and tts_duck_playback is not None:
        try:
            tts_duck_playback(ducked)
        except Exception:
            logging.exception("Failed to %s speech", "duck" if ducked else "restore")

    if not ducked:
        return

    # Nothing confirmed the interruption, so schedule a restore rather than
    # leaving the robot permanently quiet.
    timer = threading.Timer(DUCK_RELEASE_SECONDS, _release_duck_after_timeout)
    timer.daemon = True
    with _speech_state_lock:
        _duck_release_timer = timer
    timer.start()


def _release_duck_after_timeout() -> None:
    """Undo a duck that no transcript ever confirmed."""
    logging.info("Barge-in was not confirmed by a transcript; restoring volume")
    duck_speech(False)


def cancel_speech(reason: str = "barge-in") -> None:
    """
    Stop the robot mid-sentence and discard everything it was about to say.

    Bumping the epoch is what makes this reliable: queued utterances, an
    utterance still being synthesized, and the agent turn that produced them
    all check the epoch and drop themselves once it moves.
    """
    global _speech_epoch, _last_barge_in_at  # pylint: disable=global-statement

    _cancel_duck_release()
    with _speech_state_lock:
        _speech_epoch += 1
        epoch = _speech_epoch
        _last_barge_in_at = time.monotonic()

    dropped = 0
    while True:
        try:
            job = speech_queue.get_nowait()
        except queue.Empty:
            break
        if job is None:
            # Preserve the shutdown sentinel; it is not ours to discard.
            speech_queue.put(None)
            speech_queue.task_done()
            break
        speech_queue.task_done()
        dropped += 1

    stopped = False
    if TTS_AVAILABLE and tts_stop_speaking is not None:
        try:
            stopped = tts_stop_speaking()
        except Exception:
            logging.exception("Failed to stop TTS playback")

    logging.info(
        "Speech cancelled (%s): epoch=%d dropped_queued=%d interrupted_audio=%s",
        reason,
        epoch,
        dropped,
        stopped,
    )


def _begin_utterance(text: str) -> None:
    """Record what is on the speaker so the self-echo filter can match it."""
    global _speech_active, _active_speech_text  # pylint: disable=global-statement
    with _speech_state_lock:
        _speech_active = True
        _active_speech_text = text


def _end_utterance() -> None:
    """Close the speaking window, keeping the text for the echo tail."""
    global _speech_active, _active_speech_ended_at  # pylint: disable=global-statement
    with _speech_state_lock:
        _speech_active = False
        _active_speech_ended_at = time.monotonic()


def speech_worker():
    """Background worker that processes the speech queue."""
    while True:
        got_item = False
        speech_active = False
        try:
            job = speech_queue.get(timeout=1.0)
            got_item = True
            if job is None:
                break
            if isinstance(job, dict):
                text = str(job.get("text", ""))
                emit_to_ui = bool(job.get("emit_to_ui", False))
                ui_text = str(job.get("ui_text", text))
                job_epoch = int(job.get("epoch", 0))
            else:
                text = str(job)
                emit_to_ui = False
                ui_text = text
                job_epoch = 0

            with _speech_state_lock:
                current_epoch = _speech_epoch
            if job_epoch < current_epoch:
                logging.info(
                    "Speech worker: dropping superseded utterance (epoch %d < %d)",
                    job_epoch,
                    current_epoch,
                )
                continue

            if text:
                speech_active = True
                _begin_utterance(text)
                emit_speech_state(True)

            logging.info("Speech worker: starting TTS job (%d chars)", len(text) if text else 0)
            if text:
                speak_text(text)
                logging.info("Speech worker: TTS completed")
                if emit_to_ui:
                    socketio.emit(
                        "update_speech",
                        {"data": ui_text},
                        namespace="/chat",
                    )
            else:
                logging.info("Speech worker: skipped empty TTS job")
        except queue.Empty:
            if shutdown_event.is_set():
                break
            continue
        except SpeechInterrupted:
            logging.info("Speech worker: utterance interrupted by barge-in")
        except Exception:
            logging.exception("Speech worker error")
        finally:
            if speech_active:
                _end_utterance()
                emit_speech_state(False)
            if got_item:
                speech_queue.task_done()
                logging.info("Speech worker: task_done() called")


def enqueue_speech(
    text: str,
    *,
    emit_to_ui: bool = False,
    ui_text: str | None = None,
) -> None:
    """Queue speech playback, optionally syncing the UI to speech start."""
    if shutdown_event.is_set():
        logging.debug("Skipping speech enqueue during shutdown")
        return

    with _speech_state_lock:
        epoch = _speech_epoch

    speech_queue.put(
        {
            "text": text,
            "emit_to_ui": emit_to_ui,
            "ui_text": text if ui_text is None else ui_text,
            "epoch": epoch,
        }
    )


# Start speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()
@socketio.on("connect", namespace="/chat")
def on_connect():
    """Send the retained observation without creating a second control loop."""
    logging.info("Socket client connected: %s", request.sid)
    emit_observation_update(sid=request.sid)


@app.route("/")
def index():
    """Return the html."""
    logging.info("Serving conscious assistant UI")
    return render_template("index.html")


@app.route("/api/observation/latest.jpg")
def latest_observation_image():
    """Return the latest retained observation image, if available."""
    if not scene_image_path.exists():
        return "", 404
    return send_file(scene_image_path, mimetype="image/jpeg", conditional=False, max_age=0)


@app.route("/api/transcribe", methods=["POST"])
def transcribe_audio():
    """
    Transcribe audio using OpenAI Whisper API.

    Expects a multipart/form-data POST with an 'audio' file.
    Returns JSON with 'text' field containing the transcription.
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return jsonify({
            "error": "OpenAI API key not configured. Set OPENAI_API_KEY env var."
        }), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty audio file"}), 400

    max_file_size = 25 * 1024 * 1024  # 25MB
    audio_file.seek(0, os.SEEK_END)
    file_size = audio_file.tell()
    audio_file.seek(0)

    if file_size > max_file_size:
        size_mb = file_size / 1024 / 1024
        return jsonify({"error": f"Audio file too large. Max 25MB, got {size_mb:.1f}MB"}), 413

    if file_size == 0:
        return jsonify({"error": "Audio file is empty"}), 400

    # Minimum file size check - very short recordings produce corrupted files
    min_audio_size = 1000  # 1KB minimum
    if file_size < min_audio_size:
        logging.warning("Audio file too small (%d bytes), likely a quick tap", file_size)
        return jsonify({"error": "Recording too short. Hold the mic button longer."}), 400

    temp_file = None
    try:
        suffix = ".webm"  # Default to webm
        if audio_file.filename.endswith(".wav"):
            suffix = ".wav"
        elif audio_file.filename.endswith(".mp3"):
            suffix = ".mp3"
        elif audio_file.filename.endswith(".m4a"):
            suffix = ".m4a"

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        audio_file.save(temp_file.name)
        temp_file.close()

        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)

            with open(temp_file.name, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="en"  # Optimize for English
                )

            return jsonify({"text": transcript.text})

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

    finally:
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                print(f"Failed to delete temp file: {e}")


@app.route("/api/realtime/transcription-token", methods=["POST"])
def realtime_transcription_token():
    """Mint a short-lived token for a browser transcription WebRTC session."""
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return jsonify({
            "error": "OpenAI API key not configured. Set OPENAI_API_KEY env var."
        }), 503

    model = os.environ.get(
        "CONSCIOUS_AMBIENT_TRANSCRIPTION_MODEL",
        "gpt-4o-transcribe",
    )
    try:
        status, content_type, response_body, request_id = create_realtime_client_secret(
            openai_api_key,
            model,
        )
    except OSError:
        logging.exception("Could not reach realtime transcription service")
        return jsonify({"error": "Realtime transcription service is unavailable"}), 502
    if status >= 400:
        logging.error(
            "Realtime transcription token creation failed (%d, request_id=%s): %s",
            status,
            request_id or "unavailable",
            response_body.decode("utf-8", errors="replace")[:1000],
        )
        return jsonify({"error": "Could not start realtime transcription"}), status

    logging.info(
        "Realtime transcription token created (request_id=%s)",
        request_id or "unavailable",
    )
    return response_body, status, {
        "Content-Type": content_type or "application/json",
        "Cache-Control": "no-store",
    }


@socketio.on("user_input", namespace="/chat")
def handle_user_input(json, *_):
    """
    Handles user input.

    :param json: A json object containing:
        - data: The user's input text
        - skip_echo: Optional boolean to skip echoing back to chat (used when
                     client has already displayed the text, e.g., from voice input)
    """
    user_input = str(json.get("data", "")).strip()
    if not user_input:
        return
    skip_echo = json.get("skip_echo", False)
    # Only emit update_user_input if client hasn't already displayed it
    if not skip_echo:
        socketio.emit("update_user_input", {"data": user_input}, namespace="/chat")

    # An explicit turn is unambiguous: it always supersedes whatever the robot
    # was saying and whatever it was still queued to say.
    cancel_speech(reason="user input")
    _mark_turn_dispatched()
    socketio.emit("processing_started", {"interactive": True}, namespace="/chat")
    if ACKNOWLEDGE_USER_INPUT:
        enqueue_speech(random.choice(ACKNOWLEDGMENT_PHRASES), emit_to_ui=True)

    def submit() -> None:
        accepted = dispatch_agent_event(user_input, source="user")
        if not accepted:
            logging.error("User event was not accepted by the Neuro SAN runtime")
        socketio.emit("processing_complete", {"interactive": True}, namespace="/chat")

    socketio.start_background_task(submit)


def _mark_turn_dispatched() -> None:
    """Record the epoch a turn was sent at, so a later barge-in can age it out."""
    global _pending_turn_epoch  # pylint: disable=global-statement
    with _speech_state_lock:
        _pending_turn_epoch = _speech_epoch


def _should_barge_in(transcript: str) -> bool:
    """Return whether a transcript is substantial enough to cut the robot off."""
    if len(transcript.split()) < BARGE_IN_MIN_WORDS:
        return False
    with _speech_state_lock:
        speaking = _speech_active
    return speaking or not speech_queue.empty()


@socketio.on("barge_in", namespace="/chat")
def handle_barge_in(json=None, *_):
    """
    Interrupt the robot, either provisionally or outright.

    Raw voice activity is ambiguous -- it fires the moment anyone starts
    talking, far sooner than a transcript arrives, but cannot tell a person
    from the robot's own voice. That case ducks and waits for the transcript
    to decide. A client that sets ``confirmed`` has an unambiguous signal (a
    held mic button), so it cancels immediately.
    """
    confirmed = bool((json or {}).get("confirmed", False))
    with _speech_state_lock:
        speaking = _speech_active

    if confirmed:
        cancel_speech(reason="push-to-talk")
        return

    if not speaking:
        return
    logging.info("Voice activity while speaking; ducking pending confirmation")
    duck_speech(True)


@socketio.on("ambient_transcript", namespace="/chat")
def handle_ambient_transcript(json, *_):
    """Queue an always-listening transcript without treating it as direct input.

    Ambient mode deliberately has no acknowledgement, processing indicator, or
    automatic speech.  The native agent receives every usable transcript and
    decides from its event instructions whether the robot was being addressed.

    This is also where a ducked barge-in is resolved: a transcript that matches
    what the robot is saying restores the volume, and anything else confirms a
    real interruption and cancels the utterance.
    """
    transcript = str((json or {}).get("data", "")).strip()
    if not transcript:
        return

    if is_self_echo(transcript):
        logging.info("Ignoring self-echo transcript: %s", transcript[:80])
        duck_speech(False)
        return

    if _should_barge_in(transcript):
        cancel_speech(reason="ambient speech")
    else:
        # Not enough to interrupt over, so undo any duck now rather than
        # leaving the robot quiet until the release timer fires.
        duck_speech(False)

    _mark_turn_dispatched()
    logging.info("Ambient transcript queued (%d chars): %s", len(transcript), transcript)
    socketio.emit(
        "ambient_transcript",
        {"data": transcript},
        namespace="/chat",
    )
    queue_agent_event(transcript, source="ambient")


cleaned_up = False


def cleanup(from_request=False):
    """Tear things down on exit."""
    global cleaned_up  # pylint: disable=global-statement
    if cleaned_up:
        return
    cleaned_up = True

    print("Bye!")
    shutdown_event.set()
    _cancel_duck_release()
    # Cut playback short so the worker is not stuck inside a long utterance
    # when we join it below.
    if TTS_AVAILABLE and tts_stop_speaking is not None:
        try:
            tts_stop_speaking()
        except Exception:
            logging.exception("Failed to stop TTS during shutdown")
    speech_queue.put(None)

    if threading.current_thread() is not speech_thread:
        speech_thread.join(timeout=3.0)

    agent_runtime.stop()

    if from_request:
        try:
            from flask import has_request_context
            if has_request_context():
                func = request.environ.get('werkzeug.server.shutdown')
                if func:
                    func()
                else:
                    app.logger.warning("Werkzeug shutdown function not available")
        except Exception as e:
            app.logger.warning("Server shutdown failed: %s", e)


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Shut down process."""
    cleanup(from_request=True)
    return "Capture ended"


@app.after_request
def add_header(response):
    """Add the header."""
    response.headers["Cache-Control"] = "no-store"
    return response


# Register the cleanup function
atexit.register(cleanup)


def handle_shutdown_signal(signum, _frame):
    """Make terminal interrupts leave the Werkzeug loop and run bounded cleanup."""
    logging.info("Received signal %s; shutting down", signum)
    raise KeyboardInterrupt

if __name__ == "__main__":
    import ssl

    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    try:
        agent_runtime.start()

        # Refresh the cert if this robot's address has moved since it was issued.
        # Idempotent, so it is a no-op on robots with a fixed address.
        regenerated, cert_reason = tls_certs.ensure_certs()
        logging.info(
            "TLS cert %s: %s",
            "regenerated" if regenerated else "reused",
            cert_reason,
        )

        ssl_ctx = None
        if TLS_CERT.exists() and TLS_KEY.exists():
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(TLS_CERT, TLS_KEY)
        else:
            logging.warning(
                "No TLS cert at %s; serving plain HTTP. Browser microphone "
                "access will fail from anything but localhost.",
                tls_certs.cert_dir(),
            )

        socketio.run(
            app,
            host="0.0.0.0",
            port=5001,
            debug=False,
            ssl_context=ssl_ctx,
            allow_unsafe_werkzeug=True,
            log_output=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        logging.info("Terminal interrupt received")
    finally:
        cleanup()
