import atexit

import logging
import os
import queue
import random
import re
import signal
import sys
import tempfile
import threading

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
from apps.conscious_assistant.scene_observer import SceneObserver
from coded_tools.unigo2.agent_events import dispatch_agent_event


# TLS certificate paths used by both Flask and the native runtime callback.
TLS_CERT = Path("/home/unitree/certs/cert.pem")
TLS_KEY = Path("/home/unitree/certs/key.pem")


def _ui_event_endpoint() -> str:
    """Return the local callback URL using the same transport as Flask."""
    scheme = "https" if TLS_CERT.exists() and TLS_KEY.exists() else "http"
    return f"{scheme}://127.0.0.1:5001/api/agent-output"

# Import TTS function used when the agent emits a say: block.
try:
    from coded_tools.unigo2.tts_go2 import say as tts_say
    TTS_AVAILABLE = True
except ImportError:
    logging.warning("TTS module not available - speech will be text-only")
    TTS_AVAILABLE = False
    tts_say = None

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
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")
shutdown_event = threading.Event()

# Speech queue for TTS - allows non-blocking speech processing
speech_queue = queue.Queue()
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
    observation = payload.get("observation")
    if not isinstance(thought, str) or not isinstance(say, str):
        return jsonify({"error": "thought and say must be strings"}), 400

    if thought.strip():
        socketio.emit("update_thoughts", {"data": thought.strip()}, namespace="/chat")
    if say.strip():
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
            else:
                text = str(job)
                emit_to_ui = False
                ui_text = text

            if text:
                speech_active = True
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
        except Exception:
            logging.exception("Speech worker error")
        finally:
            if speech_active:
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

    speech_queue.put(
        {
            "text": text,
            "emit_to_ui": emit_to_ui,
            "ui_text": text if ui_text is None else ui_text,
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
    socketio.emit("processing_started", {"interactive": True}, namespace="/chat")
    enqueue_speech(random.choice(ACKNOWLEDGMENT_PHRASES), emit_to_ui=True)

    def submit() -> None:
        accepted = dispatch_agent_event(user_input, source="user")
        if not accepted:
            logging.error("User event was not accepted by the Neuro SAN runtime")
        socketio.emit("processing_complete", {"interactive": True}, namespace="/chat")

    socketio.start_background_task(submit)


cleaned_up = False


def cleanup(from_request=False):
    """Tear things down on exit."""
    global cleaned_up  # pylint: disable=global-statement
    if cleaned_up:
        return
    cleaned_up = True

    print("Bye!")
    shutdown_event.set()
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

        ssl_ctx = None
        if TLS_CERT.exists() and TLS_KEY.exists():
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(TLS_CERT, TLS_KEY)

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
