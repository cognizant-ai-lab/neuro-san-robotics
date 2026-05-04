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
import time
from datetime import datetime

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=getattr(logging, os.environ.get("CONSCIOUS_LOG_LEVEL", "INFO").upper(), logging.INFO)
)


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse common boolean environment variable values."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _should_preinitialize_robot_control() -> bool:
    return _env_flag(
        "CONSCIOUS_PREINIT_ROBOT",
        default=sys.platform.startswith("linux"),
    )


def _should_enable_scene_observer() -> bool:
    return _env_flag("CONSCIOUS_ENABLE_SCENE_OBSERVER", default=False)


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

from apps.conscious_assistant.conscious_assistant import conscious_thinker
from apps.conscious_assistant.conscious_assistant import set_up_conscious_assistant
from apps.conscious_assistant.scene_observer import SceneObserver
from apps.conscious_assistant.scene_observer import build_scene_input
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
    from coded_tools.unigo2.robot_macros import execute_deferred_actions
    DEFERRED_ACTIONS_AVAILABLE = True
except ImportError:
    logging.warning("execute_deferred_actions not available - deferred robot actions disabled")
    DEFERRED_ACTIONS_AVAILABLE = False
    execute_deferred_actions = None

THINKING_INTERVAL = _env_float("CONSCIOUS_THINKING_INTERVAL_SECONDS", 10.0)

# Robot motion configuration
ROBOT_MOTION_PROBABILITY = _env_float("CONSCIOUS_ROBOT_MOTION_PROBABILITY", 0.0)
ACK_WAIT_SECONDS = _env_float("CONSCIOUS_ACK_WAIT_SECONDS", 0.0)
TTS_TIMEOUT_SECONDS = _env_float("CONSCIOUS_TTS_TIMEOUT_SECONDS", 6.0)
AGENT_TIMEOUT_SECONDS = _env_float("CONSCIOUS_AGENT_TIMEOUT_SECONDS", 20.0)
AGENT_TIMEOUT_ENABLED = _env_flag("CONSCIOUS_ENABLE_AGENT_TIMEOUT", default=False)
IDLE_THINKING_ENABLED = _env_flag("CONSCIOUS_ENABLE_IDLE_THINKING", default=False)
SCENE_AGENT_INPUT_ENABLED = _env_flag("CONSCIOUS_ENABLE_SCENE_AGENT_INPUT", default=False)
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
_agent_executor = None

user_input_queue = queue.Queue()

# Speech queue for TTS - allows non-blocking speech processing
speech_queue = queue.Queue()
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

    if on_speech_complete:
        try:
            on_speech_complete(clean_text)
        except Exception:
            logging.exception("Failed to emit speech text to UI")

    def run_tts():
        try:
            logging.info("Speaking: %s", clean_text[:50])
            tts_say(clean_text, chunked=False)
        except Exception:
            logging.exception("TTS failed for text: %s", clean_text[:50])

    if TTS_TIMEOUT_SECONDS <= 0:
        run_tts()
        return

    tts_thread = threading.Thread(target=run_tts, daemon=True, name="tts-call")
    tts_thread.start()
    tts_thread.join(TTS_TIMEOUT_SECONDS)
    if tts_thread.is_alive():
        logging.warning(
            "TTS timed out after %.1fs; continuing so agent/actions do not block",
            TTS_TIMEOUT_SECONDS,
        )


def _wait_for_queue_drain(work_queue: queue.Queue, timeout_s: float) -> bool:
    if timeout_s <= 0:
        return False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if getattr(work_queue, "unfinished_tasks", 0) == 0:
            return True
        time.sleep(0.05)
    return getattr(work_queue, "unfinished_tasks", 0) == 0


def speak_text(text: str) -> None:
    """
    Speak the given text using TTS.

    This is the hardwired TTS function that gets called automatically
    whenever a 'say:' block is detected, ensuring speech always happens
    regardless of whether the agent's tool call worked.

    Speaks full text at once for better prosody.
    """
    speak_text_streaming(text, on_speech_complete=None)



def _direct_robot_action_for_text(user_text: str):
    """Map simple spoken commands to deterministic robot actions."""
    if not _env_flag("CONSCIOUS_DIRECT_ROBOT_COMMANDS", default=True):
        return None

    text = user_text.strip().lower()
    if not text:
        return None

    normalized = re.sub(r"[^a-z0-9]+", " ", text).strip()
    words = set(normalized.split())
    logging.info("Direct robot command parse: raw=%r normalized=%r words=%s", user_text, normalized, sorted(words))

    forward_words = {"forward", "forwards", "forth", "ahead"}
    backward_words = {"back", "backward", "backwards", "reverse"}
    move_words = {"step", "move", "walk", "go", "come", "run", "straight"}

    if words & {"stop", "halt", "freeze"}:
        return "stop_move", "Stopping now."
    if words & backward_words and (words & move_words or normalized in backward_words):
        return "step_backward", "Stepping backward now."
    if words & forward_words and (words & move_words or normalized in forward_words):
        return "step_forward", "Stepping forward now."
    if "step" in words:
        return "step_forward", "Stepping forward now."
    if words & {"dance", "dancing", "boogie"}:
        return "dance", "Dancing now."
    if words & {"shake", "shaking", "hello", "wave", "waving"}:
        return "shake", "Shaking now."
    if words & {"stretch", "stretching"}:
        return "stretch", "Stretching now."
    if "heart" in words:
        return "heart_pose", "Doing a heart pose now."
    if "sit" in words:
        return "sit", "Sitting now."
    if words & {"stand", "standing"}:
        return "balance_stand", "Standing now."

    return None


def execute_direct_robot_command(user_text: str) -> bool:
    """Execute obvious robot commands without waiting for the LLM agent."""
    match = _direct_robot_action_for_text(user_text)
    if match is None:
        return False

    action, speech = match
    logging.info("Direct robot command matched action=%s for input=%r", action, user_text)
    socketio.emit("update_speech", {"data": speech}, namespace="/chat")
    speech_queue.put(speech)

    if not ROBOT_AVAILABLE or Go2Macros is None:
        logging.warning("Robot not available for direct command %s", action)
        return True

    try:
        go2 = Go2Macros()
        if not getattr(go2, "available", False):
            logging.warning("Robot control unavailable for direct command %s", action)
            return True

        if action == "step_forward":
            go2.step_forward()
        elif action == "step_backward":
            go2.step_backward()
        elif action == "dance":
            go2.dance1()
        elif action == "shake":
            go2.shake()
        elif action == "stretch":
            go2.stretch()
        elif action == "heart_pose":
            go2.heart_pose()
        elif action == "sit":
            go2.sit()
        elif action == "balance_stand":
            go2.balance_stand()
        elif action == "stop_move":
            go2.stop_move()
        else:
            logging.warning("Unhandled direct robot action: %s", action)
            return True

        logging.info("Direct robot command completed: %s", action)
    except Exception:
        logging.exception("Direct robot command failed: %s", action)

    return True


def _call_conscious_thinker_with_timeout(thoughts, current_thread):
    """Run the LLM agent, optionally through a timeout wrapper for debugging."""
    if not AGENT_TIMEOUT_ENABLED:
        return conscious_thinker(
            conscious_session,
            current_thread,
            thoughts,
        )

    import concurrent.futures

    global _agent_executor  # pylint: disable=global-statement
    if _agent_executor is None:
        _agent_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="conscious-agent",
        )

    future = _agent_executor.submit(
        conscious_thinker,
        conscious_session,
        current_thread,
        thoughts,
    )
    try:
        return future.result(timeout=AGENT_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        future.cancel()
        logging.warning(
            "Conscious thinker timed out after %.1fs; releasing UI",
            AGENT_TIMEOUT_SECONDS,
        )
        return (
            "say: I got stuck thinking, but I am ready for another direct command.",
            current_thread,
        )


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

        if action == "look_left":
            go2.look_left()
        elif action == "look_right":
            go2.look_right()
        elif action == "sit":
            go2.sit()
        elif action == "rise_sit":
            go2.rise_sit()
        elif action == "step_backward":
            go2.step_backward()
        elif action == "step_forward":
            go2.step_forward()
        elif action == "stretch":
            go2.stretch()

        logging.info("Robot motion completed")

    except Exception as e:
        logging.exception("Robot motion failed")


def speech_worker():
    """Background worker that processes the speech queue."""
    while True:
        got_item = False
        try:
            text = speech_queue.get(timeout=1.0)
            got_item = True
            if text is None:
                break
            logging.info("Speech worker: starting TTS for text: %s...", text[:50] if text else "")
            speak_text(text)
            logging.info("Speech worker: TTS completed")
        except queue.Empty:
            continue
        except Exception:
            logging.exception("Speech worker error")
        finally:
            if got_item:
                speech_queue.task_done()
                logging.info("Speech worker: task_done() called")


# Start speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

conscious_session, conscious_thread = set_up_conscious_assistant()


def conscious_thinking_process():
    """Main permanent agent-calling loop."""
    with app.app_context():  # Manually push the application context
        global conscious_thread  # pylint: disable=global-statement
        thoughts = None  # Start with no initial thought - wait for user input
        while True:
            processing_started = False
            try:
                timestamp = datetime.now().strftime("[%I:%M:%S%p]").lower()
                # Wait up to the configured interval for user input
                user_input = user_input_queue.get(timeout=THINKING_INTERVAL)
                logging.info("Received user input: %r", user_input)
                if user_input == "exit":
                    break
                thoughts = f"\n{timestamp} user: " + user_input
                socketio.emit("processing_started", namespace="/chat")
                processing_started = True

                if execute_direct_robot_command(user_input):
                    thoughts = None
                    socketio.emit("processing_complete", namespace="/chat")
                    processing_started = False
                    continue

                # Speak acknowledgment immediately to fill the gap
                acknowledgment = random.choice(ACKNOWLEDGMENT_PHRASES)
                logging.info("Speaking acknowledgment: %s", acknowledgment)
                speech_queue.put(acknowledgment)
                # Emit to UI as well
                socketio.emit(
                    "update_speech",
                    {"data": acknowledgment},
                    namespace="/chat",
                )

                # Perform robot motion during the waiting time (50% chance)
                # This happens while the speech is playing, filling the gap
                perform_random_robot_motion()

                if ACK_WAIT_SECONDS > 0:
                    if _wait_for_queue_drain(speech_queue, ACK_WAIT_SECONDS):
                        logging.info("Acknowledgment speech complete, proceeding with agent")
                    else:
                        logging.warning(
                            "Acknowledgment speech still running after %.1fs; proceeding with agent",
                            ACK_WAIT_SECONDS,
                        )
                else:
                    logging.info("Proceeding with agent without waiting for acknowledgment TTS")

            except queue.Empty:
                observation = scene_observer.observe()
                if observation is not None:
                    emit_observation_update(observation)

                scene_input = None
                if SCENE_AGENT_INPUT_ENABLED and observation and observation.get("objects"):
                    scene_input = build_scene_input(timestamp, observation["objects"])
                    logging.info("Scene observer detected objects: %s", ", ".join(observation["objects"]))

                if scene_input is None:
                    if not IDLE_THINKING_ENABLED:
                        thoughts = None
                        continue
                    if thoughts is None:
                        continue

                if scene_input is None and thoughts is None:
                    continue

                thoughts = scene_input or (f"\n{timestamp} user: " + "[Silence]")
                # Emit processing_started for silence-triggered processing
                socketio.emit("processing_started", namespace="/chat")
                processing_started = True

            try:
                raw_output, conscious_thread = _call_conscious_thinker_with_timeout(
                    thoughts,
                    conscious_thread,
                )
                thoughts = normalize_agent_output(raw_output)
                print(thoughts)

                if not thoughts:
                    logging.info("Conscious thinker returned no output")
                    continue

                # Separating thoughts and speeches
                thoughts_to_emit = []
                speeches_to_emit = []

                # --- 1.  Slice the input into blocks ------------------------------------
                pattern = re.compile(
                    r"(?m)^(thought|say):[ \t]*(.*?)(?=^\s*(?:thought|say):|\Z)",
                    re.S,
                )

                for kind, raw in pattern.findall(thoughts):
                    content = raw.lstrip()
                    if not content:
                        continue

                    if kind == "thought":
                        timestamp = datetime.now().strftime("[%I:%M:%S%p]").lower()
                        thoughts_to_emit.append(f"{timestamp} thought: {content}")
                    else:
                        speeches_to_emit.append(content)

                # --- 2.  Emit the blocks ------------------------------------------------
                if thoughts_to_emit:
                    socketio.emit(
                        "update_thoughts",
                        {"data": "\n".join(thoughts_to_emit)},
                        namespace="/chat",
                    )

                if speeches_to_emit:
                    logging.info("Starting TTS for %d speech blocks", len(speeches_to_emit))

                    def emit_speech_to_ui(speech_text):
                        """Callback after speech completes to update UI."""
                        logging.debug("Emitting speech to UI: %s...", speech_text[:30])
                        socketio.emit(
                            "update_speech",
                            {"data": speech_text},
                            namespace="/chat",
                        )

                    for speech_text in speeches_to_emit:
                        speak_text_streaming(speech_text, on_speech_complete=emit_speech_to_ui)

                    logging.info("TTS complete")

                # Execute any deferred robot actions AFTER speech and UI update
                if DEFERRED_ACTIONS_AVAILABLE and execute_deferred_actions is not None:
                    try:
                        results = execute_deferred_actions()
                        if results:
                            logging.info("Executed %d deferred robot actions", len(results))
                    except Exception:
                        logging.exception("Failed to execute deferred robot actions")
            except Exception:
                logging.exception("Conscious thinking loop iteration failed")
            finally:
                if processing_started:
                    socketio.emit("processing_complete", namespace="/chat")


@socketio.on("connect", namespace="/chat")
def on_connect():
    """Start background task on connect."""
    global thread_started  # pylint: disable=global-statement
    emit_observation_update(sid=request.sid)
    if not thread_started:
        thread_started = True
        # let socketio manage the green-thread
        socketio.start_background_task(conscious_thinking_process)


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


def cleanup(from_request=False):
    """Tear things down on exit."""
    global cleaned_up  # pylint: disable=global-statement
    if cleaned_up:
        return
    cleaned_up = True

    print("Bye!")
    scene_observer.cleanup()
    if _agent_executor is not None:
        _agent_executor.shutdown(wait=False, cancel_futures=True)
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
