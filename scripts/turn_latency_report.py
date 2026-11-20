#!/usr/bin/env python3
"""Summarise voice-turn latency from a conscious-assistant server log (#11, #13).

Reads the ``turn_timing`` lines the Flask app writes, one per spoken turn, from
a log captured with ``python -m apps.conscious_assistant.interface_flask 2>&1 |
tee run.log``. Prints, per mode, the median and p90 of every step, and the
share of the total that end-of-speech detection plus transcription take.

    python scripts/turn_latency_report.py run-ptt.log run-ambient.log
    python scripts/turn_latency_report.py --skip 2 run.log   # drop warm-up turns
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys

# Matches LOCAL_AMBIENT_SILENCE_MS in templates/index.html: ambient's t0 is the
# last loud frame, and the browser waits this long before it stops recording.
AMBIENT_HANGOVER_MS = 700

_LINE = re.compile(r"turn_timing (.*)$")


def parse(lines):
    """Return one dict per turn_timing line; *_ms values as floats."""
    turns = []
    for line in lines:
        match = _LINE.search(line)
        if not match:
            continue
        turn = dict(field.split("=", 1) for field in match.group(1).split())
        for key in list(turn):
            if key.endswith("_ms"):
                turn[key] = float(turn[key])
        turns.append(turn)
    return turns


def p90(values):
    """Nearest-rank 90th percentile."""
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def steps(turn):
    """Split a spoken turn into (row name, ms) pairs."""
    rows = []
    stt = turn["transcript_ms"]
    if turn["mode"] == "ambient":
        rows.append(("end-of-speech hangover", AMBIENT_HANGOVER_MS))
        rows.append(("upload + STT", stt - AMBIENT_HANGOVER_MS))
    else:
        rows.append(("upload + STT", stt))
    rows += [
        ("transcript -> dispatch", turn["dispatched_ms"] - stt),
        ("dispatch -> ui_output", turn["ui_output_ms"] - turn["dispatched_ms"]),
        ("ui_output -> first audio", turn["audio_ms"] - turn["ui_output_ms"]),
    ]
    if "speak_start_ms" in turn:  # logs from before #16 lack it
        rows += [
            ("  queue wait", turn["speak_start_ms"] - turn["ui_output_ms"]),
            ("  setup + 1st sentence", turn["audio_ms"] - turn["speak_start_ms"]),
        ]
    rows.append(("total", turn["audio_ms"]))
    return rows


def report(turns, mode):
    """Return the table for one mode as a string."""
    in_mode = [t for t in turns if t["mode"] == mode]
    spoken = [t for t in in_mode if t["reply"] == "spoken"]
    out = [f"== {mode}: {len(spoken)} spoken, {len(in_mode) - len(spoken)} no reply =="]
    if not spoken:
        return "\n".join(out)

    columns: dict[str, list[float]] = {}
    # Turns with the speak_start split first, so its rows sit above "total".
    for turn in sorted(spoken, key=lambda t: "speak_start_ms" not in t):
        for name, ms in steps(turn):
            columns.setdefault(name, []).append(ms)
    out.append(f"{'step':<26}{'median ms':>10}{'p90 ms':>10}")
    for name, values in columns.items():
        out.append(f"{name:<26}{statistics.median(values):>10.0f}{p90(values):>10.0f}")

    # transcript_ms already covers end-of-speech (+ hangover) and STT.
    eos_stt = statistics.median(t["transcript_ms"] for t in spoken)
    share = statistics.median(t["transcript_ms"] / t["audio_ms"] for t in spoken)
    out.append(f"end-of-speech + STT: median {eos_stt:.0f} ms, {share:.0%} of total (median share)")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", help="server log files")
    parser.add_argument("--skip", type=int, default=0,
                        help="drop the first N turns of each log (cold Piper/Whisper)")
    args = parser.parse_args(argv)

    turns = []
    for path in args.logs:
        with open(path, encoding="utf-8", errors="replace") as log:
            turns += parse(log)[args.skip:]
    if not turns:
        print("no turn_timing lines found", file=sys.stderr)
        return 1
    for mode in sorted({t["mode"] for t in turns}):
        print(report(turns, mode))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
