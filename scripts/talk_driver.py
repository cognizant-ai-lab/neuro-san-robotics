"""Run scripted conversation scenarios against the conscious assistant and check the replies.

Turns go through the same Socket.IO events the web UI sends, so everything from
the Flask bridge through the LLM agent and its tools runs for real:

    user_input           typed text or push-to-talk (always addressed to the robot)
    ambient_transcript   always-listening transcript (the agent decides if it was addressed)

A scenario file (tests/scenarios/*.json) is {"name": ..., "turns": [...]}; each turn:

    {"say": "Hello", "via": "user_input",          # or "ambient"
     "expect": {"reply": true,                      # false = robot must stay quiet
                "speech_matches": "(?i)cail",       # regex over what it said
                "log_matches": "skipped 'dance'",   # regex over server output (needs --start-server)
                "max_reaction": 8}}                 # seconds from send to first reply (see below)

Usage:
    python scripts/talk_driver.py tests/scenarios/*.json                 # app already running
    python scripts/talk_driver.py --start-server tests/scenarios/*.json  # launch it in simulation mode

Reaction time is measured to the robot's first spoken word when the app reports
it (`speech_audio_started`), else to the end of speech (`update_speech`, emitted
after playback), e.g. in simulation without a speaker. Each result says which.

Exits 1 if any expectation fails.
"""

import argparse
import json
import os
import re
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Only speech counts: observation turns publish thoughts every few seconds and would pass any turn.
REPLY_EVENTS = ("update_speech",)
VIA_EVENTS = {"user_input": "user_input", "ambient": "ambient_transcript"}


def check_turn(expect, replies, log_text, reaction):
    """Return the list of failed expectations for one turn (empty = pass)."""
    failures = []
    if expect.get("reply", True):
        if reaction is None:
            failures.append("expected a reply, got none")
        elif "max_reaction" in expect and reaction > expect["max_reaction"]:
            failures.append(f"reaction {reaction:.1f}s > {expect['max_reaction']}s")
    elif replies:
        failures.append(f"expected silence, got {replies[0][1]!r}")
    spoken = "\n".join(text for event, text in replies if event == "update_speech")
    if "speech_matches" in expect and not re.search(expect["speech_matches"], spoken):
        failures.append(f"speech {spoken!r} does not match {expect['speech_matches']!r}")
    if "log_matches" in expect and not re.search(expect["log_matches"], log_text):
        failures.append(f"server output does not match {expect['log_matches']!r}")
    return failures


def turn_reaction(inbox, t0):
    """Seconds from sending to the reply, and what that measures; (None, None) without a reply."""
    if not inbox["replies"]:
        return None, None
    if inbox["audio_at"] is not None:
        return round(inbox["audio_at"] - t0, 6), "first word"
    return round(inbox["first_at"] - t0, 6), "end of speech"


def turn_satisfied(expect, replies, log_text):
    """True once a reply-expecting turn has everything it waits for."""
    return bool(replies) and not check_turn({**expect, "max_reaction": float("inf")}, replies, log_text, 0.0)


class ServerProcess:
    """Run interface_flask in simulation mode and keep its output for log_matches."""

    def __init__(self):
        env = {
            "NAV_SIMULATION_MODE": "1",
            "GO2_TTS_ENGINE": "espeak",  # offline; a missing binary just logs and skips audio
            **os.environ,
            "PYTHONUNBUFFERED": "1",  # child prints must reach the pipe immediately
        }
        self.lines = []
        self._lock = threading.Lock()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "apps.conscious_assistant.interface_flask"],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        verbose = bool(os.environ.get("TALK_DRIVER_VERBOSE"))
        for line in self.proc.stdout:
            if verbose:
                sys.stdout.write(f"  | {line}")
            with self._lock:
                self.lines.append(line)

    def mark(self):
        with self._lock:
            return len(self.lines)

    def output_since(self, mark):
        with self._lock:
            return "".join(self.lines[mark:])

    def wait_healthy(self, url, timeout):
        ctx = ssl._create_unverified_context()  # self-signed cert
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited during startup:\n" + self.output_since(0)[-4000:])
            try:
                with urllib.request.urlopen(f"{url}/api/health", context=ctx, timeout=2):
                    return
            except OSError:
                time.sleep(1)
        raise RuntimeError("server did not become healthy:\n" + self.output_since(0)[-4000:])

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def run_turn(sio, inbox, turn, server, args):
    expect = turn.get("expect", {})
    if args.max_reaction is not None:
        expect = {"max_reaction": args.max_reaction, **expect}
    wants_reply = expect.get("reply", True)
    # shortcut: a late reply from the previous turn can land here; --pause makes it unlikely.
    # Upgrade by tagging turns server-side if scenarios get flaky.
    with inbox["lock"]:
        inbox["replies"].clear()
        inbox["audio_at"] = None
    mark = server.mark() if server else 0
    t0 = time.monotonic()
    sio.emit(VIA_EVENTS[turn.get("via", "user_input")], {"data": turn["say"]}, namespace="/chat")

    # Silence can only be proven by waiting it out; a reply ends once its regexes are met.
    deadline = t0 + (args.timeout if wants_reply else args.silence)
    while time.monotonic() < deadline:
        with inbox["lock"]:
            replies = list(inbox["replies"])
        log_text = server.output_since(mark) if server else ""
        if wants_reply and turn_satisfied(expect, replies, log_text):
            break
        time.sleep(0.2)

    with inbox["lock"]:
        replies = list(inbox["replies"])
        reaction, measured = turn_reaction(inbox, t0)
    log_text = server.output_since(mark) if server else ""
    if "log_matches" in expect and not server:
        return ["log_matches needs --start-server"], reaction, measured
    return check_turn(expect, replies, log_text, reaction), reaction, measured


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="+", type=Path)
    ap.add_argument("--url", default="https://localhost:5001")
    ap.add_argument("--start-server", action="store_true", help="launch interface_flask in simulation mode")
    ap.add_argument("--timeout", type=float, default=60.0, help="max seconds to wait for an expected reply")
    ap.add_argument("--silence", type=float, default=15.0, help="seconds a reply:false turn must stay quiet")
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between turns")
    ap.add_argument("--max-reaction", type=float, help="reaction limit for turns that set none")
    args = ap.parse_args()

    import socketio  # pylint: disable=import-outside-toplevel

    server = ServerProcess() if args.start_server else None
    failed = 0
    try:
        if server:
            server.wait_healthy(args.url, timeout=180)

        inbox = {"lock": threading.Lock(), "replies": [], "first_at": 0.0, "audio_at": None}

        def on_audio(*_):
            with inbox["lock"]:
                if inbox["audio_at"] is None:
                    inbox["audio_at"] = time.monotonic()

        def on_reply(event):
            def handler(payload, *_):
                with inbox["lock"]:
                    if not inbox["replies"]:
                        inbox["first_at"] = time.monotonic()
                    inbox["replies"].append((event, str((payload or {}).get("data", ""))))
            return handler

        sio = socketio.Client(ssl_verify=False)  # robot uses a self-signed cert
        for event in REPLY_EVENTS:
            sio.on(event, on_reply(event), namespace="/chat")
        sio.on("speech_audio_started", on_audio, namespace="/chat")
        sio.connect(args.url, namespaces=["/chat"])
        try:
            for path in args.scenarios:
                scenario = json.loads(path.read_text())
                print(f"== {scenario.get('name', path.stem)}", flush=True)
                for turn in scenario["turns"]:
                    failures, reaction, measured = run_turn(sio, inbox, turn, server, args)
                    timing = f"{reaction:.1f}s {measured}" if reaction is not None else "silent"
                    print(f"  {'PASS' if not failures else 'FAIL'} [{timing}] {turn['say']!r}", flush=True)
                    for failure in failures:
                        print(f"      - {failure}", flush=True)
                    failed += bool(failures)
                    time.sleep(args.pause)
        finally:
            sio.disconnect()
    finally:
        if server:
            server.stop()

    print(f"{failed} failed turn(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
