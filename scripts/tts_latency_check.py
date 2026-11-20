#!/usr/bin/env python3
"""
Measure what a listener hears after the agent has finished replying: how long
until the first sound, whether playback ever stalls, and how long it all takes.

Runs the robot's real speech path (tts_go2.say, unchunked, exactly as the
Flask speech worker calls it) with `aplay` wrapped so every block of PCM
reaching the speaker is timestamped. The same command runs on the robot and
on a laptop; only the thing behind the wrapper differs:

  Linux robot   the real aplay, so what you hear is what is measured
  macOS         ffplay or sox `play` if one works, else silent pacing; the
                audio is also saved as WAV so you can `afplay` it afterwards

Usage:
    python scripts/tts_latency_check.py                      # built-in three paragraphs
    python scripts/tts_latency_check.py --file reply.txt
    pbpaste | python scripts/tts_latency_check.py --text -   # a real agent reply
    python scripts/tts_latency_check.py --engine openai      # the hosted path, for comparison
    python scripts/tts_latency_check.py --silent             # measure without sound

Piper on macOS: `pip install piper-tts` into the project venv (the Linux-only
marker in requirements.txt skips it), then fetch a voice with
scripts/install_piper_voice.py.

The "start" mark is taken immediately before say() is called, which is the
moment the speech worker has the agent's text in hand. Watch stderr on Linux
too: the real aplay prints its own "underrun!!!" lines if the pipe starves.
"""

import argparse
import logging
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Audio arriving later than this after the buffer ran dry counts as a stall.
# ALSA devices hold a few tens of milliseconds, so anything under that is
# absorbed and never heard.
GAP_TOLERANCE_S = 0.05

# For the progress bar only. Piper and the hosted voice both land near this;
# the bar is a hint of how far along the speech is, the report has the truth.
WORDS_PER_SECOND = 3.5
BAR_WIDTH = 30
BAR_FULL, BAR_EMPTY = "#", "."

DEFAULT_TEXT = (
    "Hello, I am CAIL-E, the robot dog at the Cognizant AI Lab in San Francisco. I can walk, look "
    "around, and talk with you about what I see. Right now I can see two people standing about two "
    "metres ahead of me, and a chair just to my left. "
    "If you would like, I can walk forward until I reach an obstacle and then stop. I use my depth "
    "camera and lidar together, so I will slow down as I get close to anything in my path. Just say "
    "stop at any time and I will hold still straight away. "
    "I also remember things you tell me. Earlier today someone asked me to remind them about the "
    "three o'clock demo, so I will mention it again when the time comes. Is there anything else you "
    "would like me to do?"
)


@dataclass
class Report:
    first_audio_s: Optional[float]
    speech_s: float
    gaps: List[Tuple[float, float]] = field(default_factory=list)  # (position in speech, length)


def analyse(start: float, arrivals: List[Tuple[float, int]], rate: int) -> Report:
    """
    Turn the wrapper's (arrival time, bytes) log into what the listener heard.

    Playback runs at `rate` from the first block onward. Each block extends
    how far the speaker can play; a block arriving after that point was
    reached means the speaker had nothing to play in between: a gap.
    """
    if not arrivals:
        return Report(first_audio_s=None, speech_s=0.0)

    first = arrivals[0][0]
    covered_until = first
    gaps: List[Tuple[float, float]] = []
    for arrived_at, nbytes in arrivals:
        late_by = arrived_at - covered_until
        if late_by > GAP_TOLERANCE_S:
            gaps.append((covered_until - first, late_by))
            covered_until = arrived_at  # playback resumes when data shows up
        covered_until += nbytes / 2 / rate
    speech_s = sum(n for _, n in arrivals) / 2 / rate
    return Report(first_audio_s=first - start, speech_s=speech_s, gaps=gaps)


def estimate_speech_s(text: str) -> float:
    """A rough length for the progress bar, from the word count alone."""
    return len(text.split()) / WORDS_PER_SECOND


def progress_line(now: float, start: float, arrivals: List[Tuple[float, int]], rate: int,
                  estimated_s: float) -> str:
    """
    One line describing the run so far, from the same trace the report uses.

    Before the first block: how long we have waited. After it: how much the
    listener has heard and whether the speaker is still being fed. The pipe
    into aplay fills at playback speed, so the trace cannot see how far ahead
    synthesis is; it can only see when it falls behind.
    """
    if not arrivals:
        return f"  waiting for first audio ... {now - start:.1f} s"
    report = analyse(start, arrivals, rate)
    first = arrivals[0][0]
    stalled = sum(length for _, length in report.gaps)
    played_s = min(max(now - first - stalled, 0.0), report.speech_s)
    filled = min(BAR_WIDTH, int(BAR_WIDTH * played_s / max(estimated_s, 0.1)))
    bar = BAR_FULL * filled + BAR_EMPTY * (BAR_WIDTH - filled)
    if now - first - stalled > report.speech_s + GAP_TOLERANCE_S:
        state = "STALLED, waiting for audio"
    else:
        state = "keeping up"
    gaps = f"   gaps {len(report.gaps)}" if report.gaps else ""
    return f"  [{bar}] played {played_s:.1f} s of ~{estimated_s:.0f} s   {state}{gaps}"


class Progress(threading.Thread):
    """Redraws progress_line on stderr every tenth of a second while say() runs."""

    def __init__(self, trace: Path, start: float, rate: int, estimated_s: float):
        super().__init__(daemon=True)
        self.trace, self.started_at, self.rate, self.estimated_s = trace, start, rate, estimated_s
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.wait(0.1):
            _sink, arrivals = read_trace(self.trace)
            line = progress_line(time.time(), self.started_at, arrivals, self.rate, self.estimated_s)
            sys.stderr.write(f"\r\x1b[K{line}")
            sys.stderr.flush()

    def finish(self) -> None:
        self.stop.set()
        self.join()
        sys.stderr.write("\r\x1b[K")
        sys.stderr.flush()


# The aplay stand-in. Written to a temp dir and put first on PATH, so the
# production code's `aplay` call lands here with its usual arguments.
APLAY_WRAPPER = r'''#!/usr/bin/env python3
import argparse, os, shutil, subprocess, sys, time, wave
p = argparse.ArgumentParser()
p.add_argument("-D"); p.add_argument("-r", type=int, default=22050); p.add_argument("-f")
p.add_argument("-c", type=int, default=1); p.add_argument("-t"); p.add_argument("-q", action="store_true")
a, _rest = p.parse_known_args()
trace_path, wav_path = os.environ["TTS_CHECK_TRACE"], os.environ.get("TTS_CHECK_WAV")
silent = os.environ.get("TTS_CHECK_SILENT") == "1"

def find_sink():
    me = os.path.dirname(os.path.abspath(__file__))
    path = os.pathsep.join(d for d in os.environ["PATH"].split(os.pathsep) if os.path.abspath(d) != me)
    real = shutil.which("aplay", path=path)
    if real:
        return [real] + sys.argv[1:]
    if shutil.which("ffplay") and subprocess.run(["ffplay", "-version"], capture_output=True).returncode == 0:
        return ["ffplay", "-f", "s16le", "-ar", str(a.r), "-ac", str(a.c), "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"]
    if shutil.which("play"):
        return ["play", "-q", "-t", "raw", "-r", str(a.r), "-e", "signed", "-b", "16", "-c", str(a.c), "-"]
    return None

sink_cmd = None if silent else find_sink()
sink = subprocess.Popen(sink_cmd, stdin=subprocess.PIPE) if sink_cmd else None
pcm, first, total = [], None, 0
with open(trace_path, "a") as trace:
    trace.write(f"sink {' '.join(sink_cmd) if sink_cmd else 'silent-paced'}\n")
    while True:
        buf = sys.stdin.buffer.read1(8192)
        if not buf:
            break
        now = time.time()
        if first is None:
            first = now
        total += len(buf)
        trace.write(f"{now:.6f} {len(buf)}\n"); trace.flush()  # survive being killed by a barge-in
        if sink:
            try:
                sink.stdin.write(buf); sink.stdin.flush()
            except BrokenPipeError:
                sys.exit(1)
        else:
            # Consume no faster than a speaker would, so synthesis must keep ahead.
            time.sleep(max(0.0, first + total / 2 / a.r - time.time()))
        if wav_path:
            pcm.append(buf)
if sink:
    sink.stdin.close(); sys.exit(sink.wait())
if wav_path and pcm:
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(a.c); w.setsampwidth(2); w.setframerate(a.r); w.writeframes(b"".join(pcm))
'''


def install_wrapper(directory: Path) -> None:
    wrapper = directory / "aplay"
    wrapper.write_text(APLAY_WRAPPER)
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ['PATH']}"


def read_trace(path: Path) -> Tuple[str, List[Tuple[float, int]]]:
    sink, arrivals = "unknown", []
    for line in path.read_text().splitlines():
        if line.startswith("sink "):
            sink = line[5:]
        else:
            stamp, nbytes = line.split()
            arrivals.append((float(stamp), int(nbytes)))
    return sink, arrivals


def load_text(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text().strip()
    if args.text == "-":
        return sys.stdin.read().strip()
    return args.text or DEFAULT_TEXT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--text", help="text to speak, or '-' to read it from stdin")
    parser.add_argument("--file", help="read the text from this file")
    parser.add_argument("--engine", default="piper",
                        help="GO2_TTS_ENGINE for this run (default piper; try openai or espeak)")
    parser.add_argument("--device", default=None,
                        help="ALSA device (default: what tts_go2 resolves, e.g. the USB speaker)")
    parser.add_argument("--silent", action="store_true",
                        help="pace the audio in real time without playing it")
    parser.add_argument("--keep-wav", default=None,
                        help="also save what reached the speaker as this WAV file")
    parser.add_argument("--verbose", action="store_true",
                        help="also show tts_go2's own log lines")
    args = parser.parse_args()

    os.environ["GO2_TTS_ENGINE"] = args.engine
    # A piper installed into this venv is on PATH only when the venv is
    # activated; the robot's service does not activate anything, so neither
    # does this. Put the interpreter's bin dir first and both cases work.
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR, stream=sys.stderr,
                        format="%(asctime)s.%(msecs)03d %(message)s", datefmt="%H:%M:%S")

    text = load_text(args)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        install_wrapper(tmpdir)
        trace = tmpdir / "trace.txt"
        trace.touch()
        os.environ["TTS_CHECK_TRACE"] = str(trace)
        os.environ["TTS_CHECK_SILENT"] = "1" if args.silent else "0"
        wav = args.keep_wav or (None if platform.system() == "Linux" else str(tmpdir / "heard.wav"))
        if wav:
            os.environ["TTS_CHECK_WAV"] = wav

        from coded_tools.unigo2 import tts_go2  # after env is set: it reads GO2_* at import

        if args.engine == "piper" and not tts_go2._is_piper_available():
            print("piper is not runnable here, so there is nothing to measure.", file=sys.stderr)
            print(f"  piper on PATH : {'yes' if shutil.which('piper') else 'NO  (uv pip install piper-tts)'}",
                  file=sys.stderr)
            print(f"  voice model   : {tts_go2.PIPER_MODEL} "
                  f"{'ok' if os.path.isfile(tts_go2.PIPER_MODEL) else 'MISSING (scripts/install_piper_voice.py)'}",
                  file=sys.stderr)
            return 2

        rate = tts_go2._piper_sample_rate() if args.engine != "openai" else tts_go2.OPENAI_PCM_RATE
        voice = Path(tts_go2.PIPER_MODEL).name.removesuffix(".onnx")
        print("TTS latency check")
        print(f"  engine {args.engine}   voice {voice} ({rate} Hz)   text {len(text)} chars, "
              f"{len(text.split())} words")

        start = time.time()
        progress = Progress(trace, start, rate, estimate_speech_s(text)) if sys.stderr.isatty() else None
        if progress:
            progress.start()
        try:
            tts_go2.say(text, chunked=False, alsa_device=args.device)
        except Exception as error:  # report, do not trace-dump
            print(f"  say() FAILED: {error}", file=sys.stderr)
        done = time.time()
        if progress:
            progress.finish()

        sink, arrivals = read_trace(trace)
        report = analyse(start, arrivals, rate)
        clock = lambda t: time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"

        print(f"  speaker fed by {sink}")
        print()
        print("TIMELINE")
        print(f"  {clock(start)}  start        agent reply in hand, say() called")
        if report.first_audio_s is None:
            print(f"  {clock(done)}  done         +{done - start:.2f} s, NO audio reached the speaker")
            return 1
        print(f"  {clock(arrivals[0][0])}  first audio  +{report.first_audio_s:.2f} s")
        for position, length in report.gaps:
            at = arrivals[0][0] + position
            print(f"  {clock(at)}  GAP          {length * 1000:.0f} ms of silence, "
                  f"{position:.1f} s into the speech")
        print(f"  {clock(done)}  done         +{done - start:.2f} s")
        print()
        print("RESULT")
        print(f"  time to first audio  {report.first_audio_s:6.2f} s   what the listener waits")
        print(f"  speech length        {report.speech_s:6.1f} s")
        print(f"  overhead             {done - start - report.speech_s:+6.2f} s   total minus speech")
        if report.gaps:
            worst = max(length for _, length in report.gaps)
            print(f"  playback gaps        {len(report.gaps):6d}     worst {worst * 1000:.0f} ms, audio stalled")
        else:
            print("  playback gaps          none     audio stayed ahead of the speaker")
        if wav and Path(wav).exists():
            keep = Path(args.keep_wav) if args.keep_wav else Path.cwd() / "tts_check_heard.wav"
            if not args.keep_wav:
                shutil.copy(wav, keep)
            print(f"  heard audio saved    {keep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
