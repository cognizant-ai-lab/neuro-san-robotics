import atexit

import logging
import os
import queue
import random
import re
import site
import sys
import tempfile
import threading
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


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse common boolean environment variable values."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _quiet_memory_tool_loggers() -> None:
    """Keep verbose memory maintenance from flooding the robot console."""
    if _env_flag("CONSCIOUS_VERBOSE_MEMORY_LOGS", default=False):
        return

    for logger_name in (
        "ListTopics",
        "RecallMemory",
        "CommitToMemory",
        "conscious_agent.reorganize_memory",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


_quiet_memory_tool_loggers()


def _should_preinitialize_robot_control() -> bool:
    return _env_flag(
        "CONSCIOUS_PREINIT_ROBOT",
        default=sys.platform.startswith("linux"),
    )


def _should_enable_scene_observer() -> bool:
    return _env_flag("CONSCIOUS_ENABLE_SCENE_OBSERVER", default=False)


def _should_enable_passive_agent_turns() -> bool:
    return _env_flag("CONSCIOUS_ENABLE_PASSIVE_AGENT_TURNS", default=False)


def _should_enable_vision_runtime_prime() -> bool:
    raw_value = os.environ.get("VISION_SKIP_EARLY_IMPORT")
    if raw_value is None:
        return sys.platform.startswith("linux") and _should_enable_scene_observer()
    return raw_value.strip().lower() not in {"1", "true", "yes", "on"}


def _candidate_libgomp_paths() -> list[Path]:
    candidates = []

    override_path = os.environ.get("VISION_LIBGOMP_PATH")
    if override_path:
        candidates.append(Path(override_path))

    for site_dir in site.getsitepackages():
        candidates.append(Path(site_dir) / "torch" / "lib" / "libgomp.so.1")

    common_system_paths = [
        "/usr/lib/aarch64-linux-gnu/libgomp.so.1",
        "/usr/lib/x86_64-linux-gnu/libgomp.so.1",
        "/lib/aarch64-linux-gnu/libgomp.so.1",
        "/lib/x86_64-linux-gnu/libgomp.so.1",
    ]
    candidates.extend(Path(path) for path in common_system_paths)

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        candidate_text = str(candidate)
        if candidate_text in seen:
            continue
        seen.add(candidate_text)
        unique_candidates.append(candidate)

    return unique_candidates


def _env_float(name: str, default: float) -> float:
    """Parse float environment variables with a safe fallback."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _prime_robot_control() -> None:
    """
    Initialize Unitree DDS/SportClient before loading heavy vision runtimes.

    On the robot, initializing CycloneDDS from the long-running Flask worker path
    can fail after Torch/DeepFace have loaded. Doing the robot client setup once
    on the main thread lets later Go2Macros instances reuse the cached client.
    """
    if not _should_preinitialize_robot_control():
        return

    try:
        from coded_tools.unigo2.go2_macros import Go2Macros as _Go2Macros

        robot = _Go2Macros()
        if getattr(robot, "available", False):
            print("[Go2] SportClient preinitialized for Flask runtime")
        else:
            print("[Go2] SportClient preinitialization did not complete")
    except Exception as exc:
        print(f"[Go2] SportClient preinitialization skipped: {exc}")


def _prime_vision_runtime_imports() -> None:
    """
    Prime YOLO dependencies before Flask imports on Linux.

    On some Jetson/ARM environments, importing torch/ultralytics later in the
    Flask startup path can fail with `libgomp.so.1: cannot allocate memory in
    static TLS block`, even though the same environment works in a simpler
    standalone process. Preloading libgomp and importing ultralytics early keeps
    the app closer to that standalone import order.
    """
    if not _should_enable_vision_runtime_prime():
        return

    try:
        import ctypes

        rtld_global = getattr(ctypes, "RTLD_GLOBAL", None)
        for candidate in _candidate_libgomp_paths():
            if not candidate.exists():
                continue
            try:
                if rtld_global is None:
                    ctypes.CDLL(str(candidate))
                else:
                    ctypes.CDLL(str(candidate), mode=rtld_global)
                print(f"[VisionCore] Preloaded libgomp: {candidate}")
                break
            except OSError:
                continue
    except Exception as exc:
        print(f"[VisionCore] libgomp preload skipped: {exc}")

    try:
        from ultralytics import YOLO as _EarlyYOLO  # noqa: F401
        print("[VisionCore] Early ultralytics import succeeded for Flask startup")
    except Exception as exc:
        print(f"[VisionCore] Early ultralytics import failed during Flask startup: {exc}")


_prime_robot_control()
_prime_vision_runtime_imports()

# pylint: disable=import-error
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask import send_file
from flask_socketio import SocketIO

from apps.conscious_assistant.agent_output import combine_speech_blocks
from apps.conscious_assistant.agent_output import parse_agent_output_blocks
from apps.conscious_assistant.conscious_assistant import conscious_thinker
from apps.conscious_assistant.conscious_assistant import set_up_conscious_assistant
from apps.conscious_assistant.scene_observer import SceneObserver
from apps.conscious_assistant.scene_observer import build_scene_input
from apps.conscious_assistant.scene_observer import observation_signature
from apps.conscious_assistant.conscious_assistant import tear_down_conscious_assistant


# SSL certificate paths
BASE_DIR = Path(__file__).resolve().parent
CERT = BASE_DIR / "certs" / "cert.pem"
KEY  = BASE_DIR / "certs" / "key.pem"

# Import TTS function for hardwired speech
try:
    from coded_tools.unigo2.tts_go2 import say as tts_say
    TTS_AVAILABLE = True
except ImportError:
    logging.warning("TTS module not available - speech will be text-only")
    TTS_AVAILABLE = False
    tts_say = None

# Import robot macros for motion during acknowledgment
try:
    from coded_tools.unigo2.go2_macros import Go2Macros
    ROBOT_AVAILABLE = True
except ImportError:
    logging.warning("Go2Macros not available - robot motions disabled")
    ROBOT_AVAILABLE = False
    Go2Macros = None

# Import deferred action executor for robot actions after speech
try:
    # Neuro-SAN loads CodedTools through AGENT_TOOL_PATH as unigo2.*.
    # Import the same module name here so the deferred-action queue is shared.
    from unigo2.robot_macros import clear_deferred_actions
    from unigo2.robot_macros import execute_deferred_actions
    DEFERRED_ACTIONS_AVAILABLE = True
except ImportError:
    try:
        from coded_tools.unigo2.robot_macros import clear_deferred_actions
        from coded_tools.unigo2.robot_macros import execute_deferred_actions
        DEFERRED_ACTIONS_AVAILABLE = True
    except ImportError:
        logging.warning("execute_deferred_actions not available - deferred robot actions disabled")
        DEFERRED_ACTIONS_AVAILABLE = False
        clear_deferred_actions = None
        execute_deferred_actions = None

THINKING_INTERVAL = _env_float("CONSCIOUS_THINKING_INTERVAL_SECONDS", 10.0)

# Robot motion configuration
ROBOT_MOTION_PROBABILITY = _env_float("CONSCIOUS_ROBOT_MOTION_PROBABILITY", 0.0)
ALLOWED_ROBOT_ACTIONS = [
    "sit_rise",
    "step_backward",
    "step_forward",
    "stretch",
    "content"
]

# Acknowledgment phrases to speak immediately when user input is received
ACKNOWLEDGMENT_PHRASES = [
    "Got it",
    "I'm on it...",
    "Let me check",
    "One moment...",
    "Sure thing",
    "Okay",
    "Understood",
    "Working on it...",
    "Let me see...",
    "Give me a second",
    "Right away",
    "On it",
    "You got it",
    "Absolutely...",
    "Let me think...",
    "Hold on...",
    "Just a sec...",
    "Coming right up...",
    "Perfect",
    "I hear you",
    "Hmm...",
    "Uh-huh",
    "Oh - okay",
    "Alright",
    "Thinking...",
    "Give me a sec...",
    "Um...",
    "Let me check with my agents...",
    "One minute please...",
    "I'm a bit hungry",
    "Haven't had my coffee yet today",
    "Just a moment please...",
    "Let me grab my thinking cap...",
    "My LLM is warming up...",
    "Loading neural pathways...",
    "Let me consult my artificial brain for a sec...",
    "I'm just a dog, but ok.",
    "Beep boop beep",
    "Bark",
    "Woof woof",
    "Bark bark",
]

os.environ.setdefault("AGENT_MANIFEST_FILE", str(REPO_ROOT / "registries" / "manifest.hocon"))
os.environ.setdefault("AGENT_TOOL_PATH", str(REPO_ROOT / "coded_tools"))
os.environ.setdefault("VISION_FACE_DB_PATH", str(REPO_ROOT / "face_database"))
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")
thread_started = False  # pylint: disable=invalid-name
thinking_task = None
shutdown_event = threading.Event()

user_input_queue = queue.Queue()

# Speech queue for TTS - allows non-blocking speech processing
speech_queue = queue.Queue()
navigation_status_lock = threading.Lock()
last_navigation_status_message = ""
last_navigation_status_at = 0.0
NAV_STATUS_REPEAT_SUPPRESS_SECONDS = _env_float("NAV_STATUS_REPEAT_SUPPRESS_SECONDS", 30.0)
scene_observer = SceneObserver(enabled=_should_enable_scene_observer())
os.environ.setdefault("VISION_LATEST_IMAGE_PATH", str(scene_observer.latest_image_path()))
os.environ.setdefault("VISION_LATEST_IMAGE_MAX_AGE_SECONDS", "0")


def emit_observation_update(observation=None, sid=None):
    """Send the latest observation image and caption data to clients."""
    payload = observation or scene_observer.latest_observation()
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


def normalize_agent_output(output) -> str:
    """Normalize agent responses so the Flask loop can parse them safely."""
    if output is None:
        return ""

    if isinstance(output, str):
        return output

    if isinstance(output, dict):
        candidate = output.get("last_chat_response") or output.get("data") or ""
        normalized = candidate if isinstance(candidate, str) else str(candidate)
        logging.warning(
            "Conscious thinker returned dict output; normalized to string (%d chars)",
            len(normalized),
        )
        return normalized

    if isinstance(output, (list, tuple)):
        parts = []
        for item in output:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                parts.append(text)
        normalized = "\n".join(parts)
        logging.warning(
            "Conscious thinker returned %s output; normalized to string (%d chars)",
            type(output).__name__,
            len(normalized),
        )
        return normalized

    normalized = str(output)
    logging.warning(
        "Conscious thinker returned unexpected %s output; normalized to string (%d chars)",
        type(output).__name__,
        len(normalized),
    )
    return normalized


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
    Speak the given text using TTS.

    This is the hardwired TTS function that gets called automatically
    whenever a 'say:' block is detected, ensuring speech always happens
    regardless of whether the agent's tool call worked.

    Speaks full text at once for better prosody.
    """
    speak_text_streaming(text, on_speech_complete=None)


def perform_random_robot_motion() -> None:
    """
    Perform 1 or 2 random robot motions from the allowed actions list.

    This function is called during user input acknowledgment to make the robot
    appear more engaged and responsive while the agent is processing.
    """
    if not ROBOT_AVAILABLE or Go2Macros is None:
        logging.info("Robot not available, skipping motion")
        return

    if ROBOT_MOTION_PROBABILITY <= 0:
        logging.info("Random acknowledgment robot motion disabled")
        return

    # Check probability - only perform motion when explicitly configured
    if random.random() > ROBOT_MOTION_PROBABILITY:
        logging.info("Skipping robot motion this time (probability check)")
        return

    try:
        go2 = Go2Macros()
        if not getattr(go2, "available", False):
            logging.info("Robot motion unavailable, skipping")
            return

        # Randomly select 1 action
        action = random.choice(ALLOWED_ROBOT_ACTIONS)

        logging.info("Performing robot motion: %s", action)

        if action == "content":
            go2.content()
        elif action == "sit_rise":
            go2.sit_rise()
        elif action == "step_backward":
            go2.step_backward()
        elif action == "step_forward":
            go2.step_forward()
        elif action == "stretch":
            go2.stretch()

        logging.info("Robot motion completed")

    except Exception as e:
        logging.exception("Robot motion failed")


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


def enqueue_navigation_status_update(message: str) -> None:
    """Speak terminal navigation updates emitted by NavCore's background loop."""
    global last_navigation_status_at, last_navigation_status_message  # pylint: disable=global-statement
    if not message:
        return

    now = datetime.now().timestamp()
    with navigation_status_lock:
        repeated = (
            message == last_navigation_status_message
            and now - last_navigation_status_at < NAV_STATUS_REPEAT_SUPPRESS_SECONDS
        )
        if repeated:
            logging.info("Suppressing repeated navigation status update: %s", message)
            return
        last_navigation_status_message = message
        last_navigation_status_at = now

    logging.info("Navigation status update: %s", message)
    enqueue_speech(message, emit_to_ui=True)


def register_navigation_status_callback() -> None:
    """Register the Flask speech bridge without eagerly constructing NavCore."""
    try:
        from coded_tools.unigo2.nav_core import NavCore

        NavCore.set_status_callback(enqueue_navigation_status_update)
        logging.info("Registered NavCore status callback for spoken navigation updates")
    except Exception:
        logging.exception("Failed to register NavCore status callback")


register_navigation_status_callback()


def discard_deferred_actions(reason: str) -> int:
    """Drop queued deferred robot actions when a turn should not execute them."""
    if not DEFERRED_ACTIONS_AVAILABLE or clear_deferred_actions is None:
        return 0

    cleared_count = clear_deferred_actions()
    if cleared_count:
        logging.info("Discarded %d deferred action(s): %s", cleared_count, reason)
    return cleared_count


def execute_deferred_actions_after_speech() -> None:
    """
    Run deferred robot actions after queued speech drains, without blocking the UI.

    The assistant should be ready for the next turn as soon as the text response
    is available, even if TTS playback or robot motions take longer.
    """
    if not DEFERRED_ACTIONS_AVAILABLE or execute_deferred_actions is None:
        return
    if shutdown_event.is_set():
        return

    try:
        speech_queue.join()
        results = execute_deferred_actions()
        if results:
            logging.info("Executed %d deferred robot actions", len(results))
    except Exception:
        logging.exception("Failed to execute deferred robot actions")


# Start speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

conscious_session, conscious_thread = set_up_conscious_assistant()


def conscious_thinking_process():
    """Main permanent agent-calling loop."""
    with app.app_context():  # Manually push the application context
        global conscious_thread  # pylint: disable=global-statement
        last_scene_signature = ()
        while not shutdown_event.is_set():
            is_interactive_turn = False
            processing_started = False
            thoughts = None
            try:
                timestamp = datetime.now().strftime("[%I:%M:%S%p]").lower()
                # Wait up to the configured interval for user input
                user_input = user_input_queue.get(timeout=THINKING_INTERVAL)
                logging.info("Received user input: %r", user_input)
                if user_input is None or user_input == "exit":
                    break
                is_interactive_turn = True
                thoughts = f"\n{timestamp} user: " + user_input
                socketio.emit("processing_started", {"interactive": True}, namespace="/chat")
                processing_started = True

                # Speak acknowledgment immediately to fill the gap
                acknowledgment = random.choice(ACKNOWLEDGMENT_PHRASES)
                logging.info("Speaking acknowledgment: %s", acknowledgment)
                enqueue_speech(acknowledgment, emit_to_ui=True)

                # Perform optional robot motion during the waiting time.
                # This happens while the speech is playing, filling the gap
                perform_random_robot_motion()

                logging.info("Acknowledgment queued, proceeding with agent")

            except queue.Empty:
                if shutdown_event.is_set():
                    break

                observation = scene_observer.observe()
                if observation is not None:
                    emit_observation_update(observation)

                scene_signature = observation_signature(observation)
                if not scene_signature:
                    last_scene_signature = ()
                    continue

                if scene_signature == last_scene_signature:
                    logging.debug(
                        "Scene observer saw unchanged entities; skipping agent turn: %s",
                        ", ".join(scene_signature),
                    )
                    continue

                # If a user speaks while we're observing the scene, let the next
                # loop iteration handle the user turn immediately instead.
                if not user_input_queue.empty():
                    logging.info("User input arrived during scene observation; prioritizing it")
                    continue

                if not _should_enable_passive_agent_turns():
                    last_scene_signature = scene_signature
                    logging.info(
                        "Scene observer detected updated entities without passive agent turn: %s",
                        ", ".join(scene_signature),
                    )
                    continue

                thoughts = build_scene_input(timestamp, list(scene_signature))
                if thoughts is None:
                    last_scene_signature = ()
                    continue

                last_scene_signature = scene_signature
                logging.info("Scene observer detected updated entities: %s", ", ".join(scene_signature))

            try:
                raw_output, conscious_thread = conscious_thinker(
                    conscious_session,
                    conscious_thread,
                    thoughts,
                )
                thoughts = normalize_agent_output(raw_output)

                if not thoughts:
                    if not is_interactive_turn:
                        discard_deferred_actions("passive scene turn returned no output")
                    logging.info("Conscious thinker returned no output")
                    continue

                # Separating thoughts and speeches
                thoughts_to_emit = []
                thought_blocks, speech_blocks = parse_agent_output_blocks(thoughts)

                for content in thought_blocks:
                    timestamp = datetime.now().strftime("[%I:%M:%S%p]").lower()
                    thoughts_to_emit.append(f"{timestamp} thought: {content}")

                # --- 2.  Emit the blocks ------------------------------------------------
                if thoughts_to_emit:
                    socketio.emit(
                        "update_thoughts",
                        {"data": "\n".join(thoughts_to_emit)},
                        namespace="/chat",
                    )

                if speech_blocks:
                    display_text, spoken_text = combine_speech_blocks(speech_blocks)
                    if spoken_text:
                        logging.info(
                            "Queueing %d speech block(s) as one utterance",
                            len(speech_blocks),
                        )
                        enqueue_speech(
                            spoken_text,
                            emit_to_ui=True,
                            ui_text=display_text,
                        )

                # Execute deferred robot actions after queued speech drains,
                # but do not block the interaction loop waiting for them.
                if DEFERRED_ACTIONS_AVAILABLE and execute_deferred_actions is not None:
                    if is_interactive_turn:
                        threading.Thread(
                            target=execute_deferred_actions_after_speech,
                            daemon=True,
                        ).start()
                    else:
                        discard_deferred_actions("passive scene turn")
            except Exception:
                logging.exception("Conscious thinking loop iteration failed")
            finally:
                if processing_started:
                    socketio.emit("processing_complete", {"interactive": True}, namespace="/chat")


@socketio.on("connect", namespace="/chat")
def on_connect():
    """Start background task on connect."""
    global thread_started, thinking_task  # pylint: disable=global-statement
    emit_observation_update(sid=request.sid)
    if not thread_started:
        thread_started = True
        # let socketio manage the green-thread
        thinking_task = socketio.start_background_task(conscious_thinking_process)


@app.route("/")
def index():
    """Return the html."""
    return render_template("index.html")


@app.route("/api/observation/latest.jpg")
def latest_observation_image():
    """Return the latest retained observation image, if available."""
    image_path = scene_observer.latest_image_path()
    if not image_path.exists():
        return "", 404
    return send_file(image_path, mimetype="image/jpeg", conditional=False, max_age=0)


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
    user_input = json["data"]
    skip_echo = json.get("skip_echo", False)
    user_input_queue.put(user_input)
    # Only emit update_user_input if client hasn't already displayed it
    if not skip_echo:
        socketio.emit("update_user_input", {"data": user_input}, namespace="/chat")


cleaned_up = False


def shutdown_nav_core_if_initialized():
    """Stop NavCore resources without importing/creating NavCore during app teardown."""
    for module_name in ("coded_tools.unigo2.nav_core", "unigo2.nav_core"):
        nav_module = sys.modules.get(module_name)
        nav_cls = getattr(nav_module, "NavCore", None) if nav_module else None
        nav_instance = getattr(nav_cls, "_instance", None) if nav_cls else None
        if nav_instance is None:
            continue

        try:
            nav_instance.shutdown()
        except Exception:
            logging.exception("Failed to shut down NavCore")
        finally:
            nav_cls._instance = None


def cleanup(from_request=False):
    """Tear things down on exit."""
    global cleaned_up  # pylint: disable=global-statement
    if cleaned_up:
        return
    cleaned_up = True

    print("Bye!")
    shutdown_event.set()
    user_input_queue.put(None)
    speech_queue.put(None)
    discard_deferred_actions("shutdown")

    try:
        if thinking_task is not None and hasattr(thinking_task, "join"):
            thinking_task.join(timeout=3.0)
    except RuntimeError:
        logging.debug("Skipping join on current thinking thread during shutdown")

    if threading.current_thread() is not speech_thread:
        speech_thread.join(timeout=3.0)

    shutdown_nav_core_if_initialized()
    scene_observer.cleanup()
    tear_down_conscious_assistant(conscious_session)

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

if __name__ == "__main__":
    import ssl

    CERT = "/home/unitree/certs/cert.pem"
    KEY = "/home/unitree/certs/key.pem"

    if scene_observer.available():
        logging.info("Pre-initializing scene observer on the main thread")
        if scene_observer.initialize():
            logging.info("Scene observer vision backend is ready")
        else:
            logging.warning("Scene observer vision backend did not initialize during startup")

    ssl_ctx = None
    if os.path.exists(CERT) and os.path.exists(KEY):
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(CERT, KEY)

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
