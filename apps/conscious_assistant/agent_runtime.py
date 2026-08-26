"""Lifecycle helpers for the native Neuro SAN event service."""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 8188
MIN_NEURO_SAN_VERSION = (0, 6, 76)


def _require_native_event_support() -> None:
    """Reject Neuro-SAN versions that acknowledge events without continuing them."""
    try:
        installed = version("neuro-san")
        installed_parts = tuple(int(part) for part in installed.split(".")[:3])
    except PackageNotFoundError as exc:
        raise RuntimeError("neuro-san==0.6.76 is required for the conscious assistant") from exc
    except ValueError as exc:
        raise RuntimeError(
            "Could not verify the installed neuro-san version; "
            "install neuro-san==0.6.76"
        ) from exc

    if installed_parts < MIN_NEURO_SAN_VERSION:
        raise RuntimeError(
            f"neuro-san {installed} does not support native event continuation; "
            "install neuro-san==0.6.76"
        )


def _port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def _service_is_live(port: int, *, timeout: float = 1.0) -> bool:
    """
    Return whether the event service is reachable on this port.

    Deliberately one-sided: only a refused connection counts as dead. That is
    the signal from the failure this guards against -- Errno 111, nothing
    listening because a previous run had already let go of the port -- and it
    is the one thing a merely busy service cannot produce. Anything else, a
    slow answer or an unusual one, counts as alive, because wrongly declaring
    a working agent dead kills it mid-turn or blocks startup outright.

    Probes the root path rather than the event endpoint: a streaming endpoint
    may legitimately not answer a bare GET.
    """
    url = f"http://127.0.0.1:{port}/"
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout):  # nosec B310 - loopback
            return True
    except HTTPError:
        # It answered, which is all this needs to know.
        return True
    except ConnectionRefusedError:
        return False
    except URLError as exc:
        return not isinstance(exc.reason, ConnectionRefusedError)
    except OSError:
        return True


class AgentRuntime:
    """Start one local Neuro SAN service and leave Flask as a thin event bridge."""

    def __init__(self, *, port: int = DEFAULT_PORT):
        self.port = port
        self.process: Optional[subprocess.Popen] = None

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}/api/v1/conscious_agent/streaming_chat"

    def start(self) -> None:
        """Launch the native service unless an operator already launched it."""
        _require_native_event_support()
        if _service_is_live(self.port):
            # An operator already has one running; leave it alone.
            return

        from coded_tools.unigo2.agent_events import clear_navigation_awareness

        clear_navigation_awareness()

        environment = os.environ.copy()
        environment.setdefault("AGENT_MANIFEST_FILE", str(REPO_ROOT / "registries" / "manifest.hocon"))
        environment.setdefault("AGENT_TOOL_PATH", str(REPO_ROOT / "coded_tools"))
        environment.setdefault("AGENT_MANIFEST_UPDATE_PERIOD_SECONDS", "5")
        environment.setdefault("CONSCIOUS_UI_EVENT_ENDPOINT", "http://127.0.0.1:5001/api/agent-output")
        environment.setdefault("CONSCIOUS_UI_EVENT_TOKEN", secrets.token_urlsafe(32))
        environment["CONSCIOUS_AGENT_EVENT_ENDPOINT"] = self.endpoint
        os.environ.update(
            {
                "CONSCIOUS_UI_EVENT_ENDPOINT": environment["CONSCIOUS_UI_EVENT_ENDPOINT"],
                "CONSCIOUS_UI_EVENT_TOKEN": environment["CONSCIOUS_UI_EVENT_TOKEN"],
                "CONSCIOUS_AGENT_EVENT_ENDPOINT": self.endpoint,
            }
        )

        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "apps.conscious_assistant.native_server",
                "--http_port",
                str(self.port),
                "--http_server_instances",
                "1",
                "--manifest_update_period_seconds",
                environment["AGENT_MANIFEST_UPDATE_PERIOD_SECONDS"],
                "--mcp_enable",
                "false",
            ],
            cwd=REPO_ROOT,
            env=environment,
        )
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            # Waiting on the socket, not on an answer. This is our own child
            # coming up, so there is no stale process to be confused by, and a
            # stricter probe here only risks failing a launch that succeeded.
            if _port_is_open(self.port):
                return
            if self.process.poll() is not None:
                raise RuntimeError("Neuro SAN event service exited during startup")
            time.sleep(0.1)
        raise RuntimeError("Neuro SAN event service did not open its local port")

    def is_alive(self) -> bool:
        """Return whether a usable event service is reachable right now."""
        if self.process is not None and self.process.poll() is not None:
            return False
        return _service_is_live(self.port)

    def ensure_running(self) -> bool:
        """
        Bring the event service back if it has gone away.

        Returns whether a usable service is up afterwards. Flask has no value
        without one: every utterance it accepts would be dropped.
        """
        if self.is_alive():
            return True
        self.stop()
        self.process = None
        self.start()
        return self.is_alive()

    def stop(self) -> None:
        """Stop only the service process created by this Flask process."""
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2.0)
