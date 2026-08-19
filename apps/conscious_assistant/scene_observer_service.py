"""Periodically publish camera observations without involving an LLM agent."""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

from apps.conscious_assistant.scene_observer import SceneObserver
from coded_tools.unigo2.agent_events import publish_observation
from coded_tools.unigo2.agent_events import queue_agent_event


class SceneObserverService:
    """Capture scenes on a fixed interval and forward their metadata to the agent."""

    def __init__(
        self,
        *,
        observer: Optional[SceneObserver] = None,
        interval_seconds: float = 15.0,
    ):
        self._observer = observer or SceneObserver()
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def capture_once(self) -> bool:
        """Capture, publish, and forward one observation when available."""
        observation = self._observer.observe()
        if observation is None:
            return False

        publish_observation(observation)
        event = json.dumps(
            {
                "summary": observation.get("summary", "No detections"),
                "entities": observation.get("objects", []),
            },
            separators=(",", ":"),
        )
        queue_agent_event(event, source="observation")
        return True

    def start(self) -> None:
        """Start the observer once; the first capture follows one interval later."""
        if not self._observer.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="scene-observer",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            try:
                self.capture_once()
            except Exception:
                logging.exception("Periodic scene observation failed")

    def stop(self) -> None:
        """Stop future captures and release camera resources."""
        self._stop_event.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._observer.cleanup()
