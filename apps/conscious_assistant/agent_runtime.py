"""Lifecycle helpers for the native Neuro SAN event service."""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 8188


def _port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


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
        if _port_is_open(self.port):
            return

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
            if _port_is_open(self.port):
                return
            if self.process.poll() is not None:
                raise RuntimeError("Neuro SAN event service exited during startup")
            time.sleep(0.1)
        raise RuntimeError("Neuro SAN event service did not open its local port")

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
