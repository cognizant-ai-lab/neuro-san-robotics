"""Server-side credentials for browser WebRTC transcription sessions."""

import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from apps.conscious_assistant.robot_identity import robot_home
from apps.conscious_assistant.robot_identity import robot_name


REALTIME_CLIENT_SECRETS_URL = "https://api.openai.com/v1/realtime/client_secrets"
TRANSIENT_STATUSES = {502, 503, 504}


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


def _send_client_secret_request(api_key: str, model: str):
    request_body = json.dumps({
        "session": transcription_session_config(model),
    }).encode("utf-8")
    upstream_request = Request(
        REALTIME_CLIENT_SECRETS_URL,
        data=request_body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(upstream_request, timeout=20) as response:  # nosec B310
            return (
                response.status,
                response.headers.get_content_type(),
                response.read(),
                response.headers.get("x-request-id", ""),
            )
    except HTTPError as error:
        return (
            error.code,
            error.headers.get_content_type(),
            error.read(),
            error.headers.get("x-request-id", ""),
        )


def create_realtime_client_secret(
    api_key: str,
    model: str,
    *,
    max_attempts: int = 2,
):
    """Mint a short-lived browser token, retrying transient gateway failures."""
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        result = _send_client_secret_request(api_key, model)
        if result[0] not in TRANSIENT_STATUSES or attempt == attempts - 1:
            return result
        time.sleep(0.5 * (attempt + 1))
    raise AssertionError("unreachable")
