"""Server-side credentials for browser WebRTC transcription sessions.

Capture tuning is environment-driven because the right values depend on the
microphone, not the robot:

- CONSCIOUS_MIC_NOISE_REDUCTION: "far_field" (default) for a mic covering the
  room, "near_field" for one worn at the mouth, or "off" to send the audio
  through untouched. near_field on a mic that is not right at the speaker's
  mouth treats their voice as distant noise and suppresses it, which reads as
  the robot being hard of hearing however far the VAD threshold is lowered.
- CONSCIOUS_VAD_THRESHOLD: how loud speech must be to register (0.0-1.0).
  Raise it when robot noise keeps opening an utterance.
- CONSCIOUS_VAD_PREFIX_PADDING_MS: audio kept from before speech was detected.
- CONSCIOUS_VAD_SILENCE_MS: silence needed to close an utterance.
"""

import json
import os
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


def _env_value(name: str, default, cast):
    """Read one capture setting from the environment, ignoring junk."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


def noise_reduction_mode() -> str:
    """Return the capture profile matching this robot's microphone."""
    mode = (os.environ.get("CONSCIOUS_MIC_NOISE_REDUCTION") or "").strip().lower()
    return mode if mode in {"near_field", "far_field", "off"} else "far_field"


def transcription_session_config(model: str) -> dict:
    """Return a transcription-only Realtime session tuned for this robot's mic."""
    mode = noise_reduction_mode()
    audio_input = {
        # Omitted entirely for "off": there is no profile that means untouched,
        # and either profile shapes the audio before the VAD ever sees it.
        "transcription": {
            "model": model,
            "language": "en",
            "prompt": transcription_prompt(),
        },
        "turn_detection": {
            "type": "server_vad",
            "threshold": _env_value("CONSCIOUS_VAD_THRESHOLD", 0.45, float),
            "prefix_padding_ms": _env_value(
                "CONSCIOUS_VAD_PREFIX_PADDING_MS", 400, int
            ),
            "silence_duration_ms": _env_value("CONSCIOUS_VAD_SILENCE_MS", 700, int),
        },
    }
    if mode != "off":
        audio_input["noise_reduction"] = {"type": mode}

    return {
        "type": "transcription",
        "audio": {"input": audio_input},
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
