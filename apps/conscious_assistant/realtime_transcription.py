"""Server-side bridge for browser WebRTC transcription sessions."""

import json
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen


REALTIME_CALLS_URL = "https://api.openai.com/v1/realtime/calls"


def transcription_session_config(model: str) -> dict:
    """Return a transcription-only Realtime session tuned for room audio."""
    return {
        "type": "transcription",
        "audio": {
            "input": {
                "noise_reduction": {"type": "far_field"},
                "transcription": {
                    "model": model,
                    "languages": ["en"],
                    "prompt": (
                        "Ambient speech in the Cognizant AI Lab. CAIL-E is the robot's "
                        "name. Preserve names and technical terms accurately."
                    ),
                    "keywords": ["CAIL-E", "Cognizant", "Neuro-SAN", "Unitree"],
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


def create_realtime_call(offer_sdp: bytes, api_key: str, model: str):
    """Exchange a browser SDP offer for an OpenAI WebRTC SDP answer."""
    boundary = f"----conscious-assistant-{uuid.uuid4().hex}"
    session_json = json.dumps(transcription_session_config(model)).encode("utf-8")
    body = b"".join((
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="sdp"\r\n',
        b"Content-Type: application/sdp\r\n\r\n",
        offer_sdp,
        b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="session"\r\n',
        b"Content-Type: application/json\r\n\r\n",
        session_json,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ))
    upstream_request = Request(
        REALTIME_CALLS_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urlopen(upstream_request, timeout=20) as response:  # nosec B310
            return response.status, response.headers.get_content_type(), response.read()
    except HTTPError as error:
        return error.code, error.headers.get_content_type(), error.read()
