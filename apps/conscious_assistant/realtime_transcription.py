"""Server-side credentials for browser WebRTC transcription sessions."""

import json
import time
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from apps.conscious_assistant.robot_identity import robot_home
from apps.conscious_assistant.robot_identity import robot_name


REALTIME_SESSION_URL = "https://api.openai.com/v1/realtime/client_secrets"
TRANSIENT_STATUSES = {502, 503, 504}


@dataclass(frozen=True)
class RealtimeSessionResponse:
    """
    Upstream reply to a browser transcription session request.

    ``payload`` is the raw upstream body. On success it carries the ephemeral
    credential the browser authenticates with, so it is forwarded straight to
    that browser and must never be logged or folded into an error message.
    Every other field is ordinary transport metadata and is safe to log.
    """

    status: int
    content_type: str
    payload: bytes
    request_id: str

    @property
    def failed(self) -> bool:
        """Whether the upstream call reported an error."""
        return self.status >= 400

    @property
    def transient(self) -> bool:
        """Whether a gateway hiccup makes a retry worth attempting."""
        return self.status in TRANSIENT_STATUSES


def transcription_prompt() -> str:
    """Bias the recogniser toward this robot's name so it is heard reliably."""
    return (
        f"Ambient speech in {robot_home()}. "
        f"{robot_name()} is the robot's name. "
        "Preserve names and technical terms accurately."
    )


def transcription_session_config(model: str) -> dict:
    """Return a transcription-only Realtime session tuned for room audio."""
    return {
        "type": "transcription",
        "audio": {
            "input": {
                "noise_reduction": {"type": "far_field"},
                "transcription": {
                    "model": model,
                    "language": "en",
                    "prompt": transcription_prompt(),
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.45,
                    "prefix_padding_ms": 400,
                    "silence_duration_ms": 700,
                },
            }
        },
    }


def _post_session_request(api_key: str, model: str) -> RealtimeSessionResponse:
    request_body = json.dumps({
        "session": transcription_session_config(model),
    }).encode("utf-8")
    upstream_request = Request(
        REALTIME_SESSION_URL,
        data=request_body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(upstream_request, timeout=20) as response:  # nosec B310
            return RealtimeSessionResponse(
                status=response.status,
                content_type=response.headers.get_content_type(),
                payload=response.read(),
                request_id=response.headers.get("x-request-id", ""),
            )
    except HTTPError as error:
        return RealtimeSessionResponse(
            status=error.code,
            content_type=error.headers.get_content_type(),
            payload=error.read(),
            request_id=error.headers.get("x-request-id", ""),
        )


def request_realtime_session(
    api_key: str,
    model: str,
    *,
    max_attempts: int = 2,
) -> RealtimeSessionResponse:
    """Open a short-lived browser session, retrying transient gateway failures."""
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        response = _post_session_request(api_key, model)
        if not response.transient or attempt == attempts - 1:
            return response
        time.sleep(0.5 * (attempt + 1))
    raise AssertionError("unreachable")
