import atexit
import os
import queue
import re
import tempfile
import time
from datetime import datetime

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

THINKING_INTERVAL = 30.0

os.environ["AGENT_MANIFEST_FILE"] = "registries/manifest.hocon"
os.environ["AGENT_TOOL_PATH"] = "coded_tools"
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)
thread_started = False  # pylint: disable=invalid-name

user_input_queue = queue.Queue()

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
            except queue.Empty:
                if thoughts is None:
                    continue
                thoughts = f"\n{timestamp} user: " + "[Silence]"

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
                socketio.emit(
                    "update_speech",
                    {"data": "\n".join(speeches_to_emit)},
                    namespace="/chat",
                )


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

    :param json: A json object
    """
    user_input = json["data"]
    user_input_queue.put(user_input)
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
    socketio.run(app, debug=False, port=5001, allow_unsafe_werkzeug=True, log_output=True, use_reloader=False)
