"""Small local adapters between the native Neuro SAN runtime and the web UI."""

from __future__ import annotations

import json
import logging
import os
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request
from urllib.request import urlopen


logger = logging.getLogger(__name__)
_EVENT_DISPATCHER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-events")
_AWARENESS_LOCK = threading.Lock()
_DEFAULT_AWARENESS_MAX_AGE_SECONDS = 6 * 60 * 60

# Sources whose queued events may be replaced by a newer one instead of
# queueing behind it. Room audio arrives faster than the agent can answer, so
# working through a backlog one utterance at a time leaves the robot replying
# to things the speaker has long since moved past.
_COALESCED_SOURCES = frozenset({"ambient"})
_PENDING_LOCK = threading.Lock()
_PENDING_BY_SOURCE: dict[str, str] = {}
# Called when an event cannot be delivered. The UI layer registers this so a
# robot that has lost its agent looks broken rather than merely uninterested:
# without it, every utterance is dropped behind a log line nobody is watching.
_DISPATCH_FAILURE_HOOK = None


def _navigation_awareness_path() -> Path:
    """Return the small cross-process state file used by the event bridge."""
    return Path(
        os.environ.get(
            "CONSCIOUS_NAVIGATION_AWARENESS_FILE",
            "/tmp/robot-navigation-awareness.json",
        )
    )


def remember_navigation_awareness(text: str) -> None:
    """Persist the newest authoritative navigation event for later user turns."""
    text = str(text).strip()
    if not text:
        return

    path = _navigation_awareness_path()
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    payload = {"text": text, "updated_at": time.time()}
    try:
        with _AWARENESS_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(temporary_path, path)
    except OSError:
        logger.warning("Could not retain navigation awareness", exc_info=True)


def route_navigation_status(text: str) -> None:
    """Retain routine nav state, but wake the agent for terminal outcomes only."""
    text = str(text).strip()
    if not text:
        return
    remember_navigation_awareness(text)
    terminal_prefixes = (
        "I arrived at ",
        "I am already at ",
        "I stopped before reaching ",
        "I did not move ",
        "I could not ",
    )
    if text.startswith(terminal_prefixes):
        # Terminal motion outcomes must never disappear into an internal-only
        # thought. Deliver the authoritative sentence to both UI and speech;
        # retain the agent event only as a fallback if the UI bridge is down.
        if not publish_ui_output(thought=text, say=text, source="navigation"):
            queue_agent_event(text, source="navigation")


def latest_navigation_awareness() -> str:
    """Return recent navigation awareness shared by the native and Flask processes."""
    path = _navigation_awareness_path()
    try:
        with _AWARENESS_LOCK:
            payload = json.loads(path.read_text(encoding="utf-8"))
        text = str(payload.get("text", "")).strip()
        updated_at = float(payload.get("updated_at", 0.0))
        max_age = float(
            os.environ.get(
                "CONSCIOUS_NAVIGATION_AWARENESS_MAX_AGE_SECONDS",
                _DEFAULT_AWARENESS_MAX_AGE_SECONDS,
            )
        )
        if not text or time.time() - updated_at > max(0.0, max_age):
            return ""
        return text
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return ""


def clear_navigation_awareness() -> None:
    """Discard awareness left by an earlier native runtime session."""
    path = _navigation_awareness_path()
    try:
        with _AWARENESS_LOCK:
            path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not clear stale navigation awareness", exc_info=True)


def _local_ssl_context(url: str) -> ssl.SSLContext | None:
    """Trust Flask's self-signed certificate only for a loopback callback."""
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.hostname in {"127.0.0.1", "::1", "localhost"}:
        return ssl._create_unverified_context()
    return None


def _post_json(url: str, payload: dict[str, Any], *, token: str = "", timeout: float = 5.0) -> None:
    """POST one local JSON event and fully consume its acknowledgement."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Conscious-Bridge-Token"] = token
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    ssl_context = _local_ssl_context(url)
    open_kwargs = {"timeout": timeout}
    if ssl_context is not None:
        open_kwargs["context"] = ssl_context
    with urlopen(request, **open_kwargs) as response:  # nosec B310 - endpoints are local runtime configuration
        # Event invocation sends a short acknowledgement before Neuro-SAN hands
        # the real work to EventWorkMonitor. Closing after one byte aborts that
        # handoff on some HTTP stacks.
        response.read()


def set_dispatch_failure_hook(hook) -> None:
    """Register a callback invoked with (source, exception) on a failed dispatch."""
    global _DISPATCH_FAILURE_HOOK  # pylint: disable=global-statement
    _DISPATCH_FAILURE_HOOK = hook


def _report_dispatch_failure(source: str, exc: Exception) -> None:
    """Escalate a dropped event past the log."""
    hook = _DISPATCH_FAILURE_HOOK
    if hook is None:
        return
    try:
        hook(source, exc)
    except Exception:  # pylint: disable=broad-except
        logger.exception("Dispatch failure hook raised")


def dispatch_agent_event(text: str, *, source: str) -> bool:
    """Wake the event-configured agent without waiting for its background work."""
    text = str(text).strip()
    if not text:
        return False

    if source == "navigation":
        remember_navigation_awareness(text)

    event_text = f"{source}: {text}"
    if source == "user":
        awareness = latest_navigation_awareness()
        if awareness:
            event_text += (
                "\nsystem: Current navigation awareness (authoritative): "
                f"{awareness}"
            )

    endpoint = os.environ.get(
        "CONSCIOUS_AGENT_EVENT_ENDPOINT",
        "http://127.0.0.1:8188/api/v1/conscious_agent/streaming_chat",
    )
    payload = {
        "user_message": {"type": "HUMAN", "text": event_text},
        "chat_filter": {"chat_filter_type": "MINIMAL"},
    }
    try:
        _post_json(endpoint, payload, timeout=5.0)
        return True
    except (OSError, URLError, ValueError) as exc:
        # This is not a warning. The utterance is gone and the robot will
        # appear to ignore whoever spoke, with nothing on screen to say why.
        logger.error(
            "Dropped %s event -- Neuro SAN unreachable at %s: %s",
            source,
            endpoint,
            exc,
        )
        _report_dispatch_failure(source, exc)
        return False


def _dispatch_latest(source: str) -> bool:
    """Dispatch the newest text queued for a coalescing source."""
    with _PENDING_LOCK:
        text = _PENDING_BY_SOURCE.pop(source, "")
    if not text:
        return False
    return dispatch_agent_event(text, source=source)


def queue_agent_event(text: str, *, source: str) -> None:
    """
    Queue an agent event without blocking a real-time producer.

    Ambient transcripts coalesce: while one is still waiting its turn, a newer
    one replaces it rather than queueing behind it, so the agent always answers
    the most recent thing it heard. Explicit sources keep strict FIFO -- a
    typed or spoken user turn is never superseded by a later one.
    """
    text = str(text).strip()
    if not text:
        return

    if source not in _COALESCED_SOURCES:
        _EVENT_DISPATCHER.submit(dispatch_agent_event, text, source=source)
        return

    with _PENDING_LOCK:
        superseded = _PENDING_BY_SOURCE.get(source)
        _PENDING_BY_SOURCE[source] = text

    if superseded is not None:
        # A task is already queued for this source and has not claimed its text
        # yet, so it will pick up the newer text when it runs.
        logger.info("Superseded a queued %s event: %s", source, superseded[:60])
        return

    _EVENT_DISPATCHER.submit(_dispatch_latest, source)


def publish_ui_output(
    *,
    thought: str = "",
    say: str = "",
    heard: str = "",
    source: str = "",
) -> bool:
    """Deliver agent-authored UI output to the local Flask presentation adapter.

    ``source`` marks who authored the output. Speech from a turn the user has
    already talked over is normally dropped, so an authoritative announcement
    that does not belong to a conversational turn -- a navigation outcome, say
    -- must identify itself to survive a barge-in.
    """
    thought = str(thought).strip()
    say = str(say).strip()
    heard = str(heard).strip()
    if not thought and not say:
        return False

    endpoint = os.environ.get("CONSCIOUS_UI_EVENT_ENDPOINT", "http://127.0.0.1:5001/api/agent-output")
    token = os.environ.get("CONSCIOUS_UI_EVENT_TOKEN", "")
    try:
        _post_json(
            endpoint,
            {"thought": thought, "say": say, "heard": heard, "source": source},
            token=token,
            timeout=10.0,
        )
        return True
    except (OSError, URLError, ValueError) as exc:
        logger.warning("Could not publish agent output to the UI: %s", exc)
        return False


def publish_observation(observation: dict[str, Any]) -> bool:
    """Deliver the newest scene metadata after the observer overwrote its JPEG."""
    endpoint = os.environ.get("CONSCIOUS_UI_EVENT_ENDPOINT", "http://127.0.0.1:5001/api/agent-output")
    token = os.environ.get("CONSCIOUS_UI_EVENT_TOKEN", "")
    try:
        _post_json(endpoint, {"observation": observation}, token=token, timeout=10.0)
        return True
    except (OSError, URLError, ValueError) as exc:
        logger.warning("Could not publish observation to the UI: %s", exc)
        return False
