"""Per-turn timestamps for a spoken turn, logged as one line per turn (#13).

A voice turn's steps happen in different places -- the socket handler, the
event dispatcher, the ui_output callback and the speaker thread -- all in the
Flask process. They share a ``turn_id`` and report here; the turn is logged
once its first audio plays, or with ``reply=none`` when the next turn starts
without it ever having spoken.

The line is read by ``scripts/turn_latency_report.py``. Times are ms after t0,
the end of the user's speech, measured on this process's monotonic clock.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

logger = logging.getLogger(__name__)

# speak_start: the speech worker took the turn's reply off the queue (#16).
STEPS = ("transcript", "dispatched", "ui_output", "speak_start", "audio")

_LOCK = threading.Lock()
_OPEN: dict[str, dict] = {}


def start(mode: str, ended_ago_s: float) -> str:
    """Open a turn whose transcript just arrived; speech ended ``ended_ago_s`` ago."""
    now = time.monotonic()
    turn_id = uuid.uuid4().hex[:8]
    with _LOCK:
        stale = list(_OPEN)
        _OPEN[turn_id] = {
            "mode": mode,
            "t0": now - max(0.0, ended_ago_s),
            "transcript": now,
            "coalesced": 0,
        }
    # A new turn means the user stopped waiting for the earlier ones.
    for old_id in stale:
        _close(old_id, reply="none")
    return turn_id


def mark(turn_id: str | None, step: str, **details: object) -> None:
    """Record the first time ``turn_id`` reached ``step``; audio closes the turn.

    ``details`` (e.g. ``first_sentence_phonemes``) are logged as extra fields.
    """
    if not turn_id:
        return
    with _LOCK:
        turn = _OPEN.get(turn_id)
        if turn is None or step in turn:
            return
        turn[step] = time.monotonic()
        turn.setdefault("details", {}).update(details)
    if step == "audio":
        _close(turn_id, reply="spoken")


def merge(superseded_id: str | None, turn_id: str | None) -> None:
    """Fold a queued ambient turn into the newer one that replaced it."""
    if not superseded_id or not turn_id:
        return
    with _LOCK:
        old = _OPEN.pop(superseded_id, None)
        turn = _OPEN.get(turn_id)
        if old is not None and turn is not None:
            turn["coalesced"] += old["coalesced"] + 1


def flush() -> None:
    """Log every turn still open, e.g. at shutdown."""
    with _LOCK:
        open_ids = list(_OPEN)
    for turn_id in open_ids:
        _close(turn_id, reply="none")


def _close(turn_id: str, *, reply: str) -> None:
    with _LOCK:
        turn = _OPEN.pop(turn_id, None)
    if turn is None:
        return
    fields = [f"turn_id={turn_id}", f"mode={turn['mode']}"]
    for step in STEPS:
        if step in turn:
            fields.append(f"{step}_ms={(turn[step] - turn['t0']) * 1000:.0f}")
    fields += [f"{key}={value}" for key, value in turn.get("details", {}).items()]
    fields += [f"coalesced={turn['coalesced']}", f"reply={reply}"]
    logger.info("turn_timing %s", " ".join(fields))
