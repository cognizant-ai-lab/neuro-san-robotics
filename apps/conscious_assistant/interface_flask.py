import atexit

import logging
import os
import queue
import random
import re
import tempfile
import threading
import time
from datetime import datetime

from pathlib import Path

# pylint: disable=import-error
import schedule
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask_socketio import SocketIO

from apps.conscious_assistant.conscious_assistant import conscious_thinker
from apps.conscious_assistant.conscious_assistant import set_up_conscious_assistant
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

THINKING_INTERVAL = 30.0

# Robot motion configuration
ROBOT_MOTION_PROBABILITY = 0.5  # 50% chance of performing robot motion
ALLOWED_ROBOT_ACTIONS = [
    "look_left",
    "look_right",
    "sit",
    "rise_sit",
    "step_backward",
    "step_forward",
    "stretch"      
]

# Acknowledgment phrases to speak immediately when user input is received
ACKNOWLEDGMENT_PHRASES = [
    "Got it",
    "I'm on it",
    "Let me check",
    "One moment",
    "Sure thing",
    "Okay",
    "Understood",
    "Working on it",
    "Let me see",
    "Give me a second",
    "Right away",
    "On it",
    "You got it",
    "Absolutely",
    "Let me think",
    "Hold on",
    "Just a sec",
    "Coming right up",
    "Perfect",
    "I hear you",
    "Hmm",
    "Uh-huh",
    "Oh - okay",
    "Alright",
    "Thinking...",
    "Give me a sec",
    "Um",
    "Let me check with my agents...",
    "One minute please",
    "I'm a bit hungry",
    "Haven't had my coffee yet today",
    "Just a moment please",
    "Let me grab my thinking cap",
    "My LLM is warming up",
    "Loading neural pathways",
    "Consulting my artificial brain",
    "I'm just a dog, but ok.",
    "Beep boop beep",
    "Bark",
    "Woof woof",
    "Bark bark",
]

os.environ["AGENT_MANIFEST_FILE"] = "registries/manifest.hocon"
os.environ["AGENT_TOOL_PATH"] = "coded_tools"
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")
thread_started = False  # pylint: disable=invalid-name

user_input_queue = queue.Queue()

# Speech queue for TTS - allows non-blocking speech processing
speech_queue = queue.Queue()


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


def speak_text(text: str) -> None:
    """
    Speak the given text using TTS.

    This is the hardwired TTS function that gets called automatically
    whenever a 'say:' block is detected, ensuring speech always happens
    regardless of whether the agent's tool call worked.
    """
    if not TTS_AVAILABLE or tts_say is None:
        logging.info("TTS not available, skipping speech: %s", text[:50])
        return

    clean_text = sanitize_speech_text(text)
    if not clean_text:
        logging.info("No text to speak after sanitization")
        return

    try:
        logging.info("Speaking: %s", clean_text[:50])
        tts_say(clean_text)
    except Exception as e:
        logging.exception("TTS failed for text: %s", clean_text[:50])


def perform_random_robot_motion() -> None:
    """
    Perform 1 or 2 random robot motions from the allowed actions list.

    This function is called during user input acknowledgment to make the robot
    appear more engaged and responsive while the agent is processing.
    """
    if not ROBOT_AVAILABLE or Go2Macros is None:
        logging.info("Robot not available, skipping motion")
        return

    # Check probability - only perform motion 50% of the time (or as configured)
    if random.random() > ROBOT_MOTION_PROBABILITY:
        logging.info("Skipping robot motion this time (probability check)")
        return

    try:
        go2 = Go2Macros()

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
        except Exception as e:
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
            timestamp = datetime.now().strftime("[%I:%M:%S%p]").lower()
            try:
                # Wait up to 30 seconds for user input
                user_input = user_input_queue.get(timeout=THINKING_INTERVAL)
                if user_input == "exit":
                    break
                thoughts = f"\n{timestamp} user: " + user_input

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

                # Wait for acknowledgment to finish speaking
                speech_queue.join()
                logging.info("Acknowledgment speech complete, proceeding with agent")

            except queue.Empty:
                if thoughts is None:
                    continue
                thoughts = f"\n{timestamp} user: " + "[Silence]"
                # Emit processing_started for silence-triggered processing
                socketio.emit("processing_started", namespace="/chat")

            thoughts, conscious_thread = conscious_thinker(conscious_session, conscious_thread, thoughts)
            print(thoughts)

            # Separating thoughts and speeches
            thoughts_to_emit = []
            speeches_to_emit = []

            # --- 1.  Slice the input into blocks ----------------------------------------
            #     Each block begins with  "thought:"  or  "say:"  and continues until
            #     the next block or the end of the string.
            pattern = re.compile(
                r"(?m)^(thought|say):[ \t]*(.*?)(?=^\s*(?:thought|say):|\Z)", re.S  # look-ahead  # dot = newline
            )

            for kind, raw in pattern.findall(thoughts):
                content = raw.lstrip()  # drop the leading spaces/newline after the prefix
                if not content:
                    continue

                if kind == "thought":
                    timestamp = datetime.now().strftime("[%I:%M:%S%p]").lower()
                    thoughts_to_emit.append(f"{timestamp} thought: {content}")
                else:  # kind == "say"
                    speeches_to_emit.append(content)

            # --- 2.  Emit the blocks -----------------------------------------------------
            if thoughts_to_emit:
                socketio.emit(
                    "update_thoughts",
                    {"data": "\n".join(thoughts_to_emit)},
                    namespace="/chat",
                )

            if speeches_to_emit:
                # Hardwired TTS: queue each speech block for audio playback
                # NOTE: We intentionally speak FIRST, then render text in the UI.
                for speech_text in speeches_to_emit:
                    speech_queue.put(speech_text)

                # Wait for all speech to complete before showing the text
                # This makes the robot audio lead, then the UI catches up.
                logging.info("Waiting for TTS playback to complete before updating UI...")
                speech_queue.join()
                logging.info("TTS playback complete; now updating UI with speech text")

                # Now emit the full speech content to the UI
                socketio.emit(
                    "update_speech",
                    {"data": "\n".join(speeches_to_emit)},
                    namespace="/chat",
                )
            
            # Signal that processing is complete and user can send new input
            socketio.emit("processing_complete", namespace="/chat")


@socketio.on("connect", namespace="/chat")
def on_connect():
    """Start background task on connect."""
    global thread_started  # pylint: disable=global-statement
    if not thread_started:
        thread_started = True
        # let socketio manage the green-thread
        socketio.start_background_task(conscious_thinking_process)


@app.route("/")
def index():
    """Return the html."""
    return render_template("index.html")


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
            "error": "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        }), 503
    
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty audio file"}), 400
    
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
    audio_file.seek(0, os.SEEK_END)
    file_size = audio_file.tell()
    audio_file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({"error": f"Audio file too large. Maximum size is 25MB, got {file_size / 1024 / 1024:.1f}MB"}), 413
    
    if file_size == 0:
        return jsonify({"error": "Audio file is empty"}), 400
    
    # Minimum file size check - very short recordings produce corrupted files
    MIN_AUDIO_SIZE = 1000  # 1KB minimum
    if file_size < MIN_AUDIO_SIZE:
        logging.warning("Audio file too small (%d bytes), likely a quick tap", file_size)
        return jsonify({"error": "Recording too short. Please hold the mic button longer."}), 400
    
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
    global cleaned_up
    if cleaned_up:
        return
    cleaned_up = True
    
    print("Bye!")
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
            app.logger.warning(f"Server shutdown failed: {e}")


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


def run_scheduled_tasks():
    """Run the scheduled tasks."""
    while True:
        schedule.run_pending()
        time.sleep(1)


# Register the cleanup function
atexit.register(cleanup)

if __name__ == "__main__":
    import ssl

    CERT = "/home/unitree/certs/cert.pem"
    KEY = "/home/unitree/certs/key.pem"

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
