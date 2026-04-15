import logging
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict

try:
    import cv2
except ImportError:  # pragma: no cover - depends on runtime environment
    cv2 = None

from neuro_san.interfaces.coded_tool import CodedTool

from coded_tools.unigo2.vision_core import VisionCore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LATEST_IMAGE_PATH = (
    Path(tempfile.gettempdir()) / "neuro_san_conscious_assistant" / "latest_observation.jpg"
)
DEFAULT_FACE_DB_PATH = REPO_ROOT / "face_database"


def _normalize_person_name(raw_name: str) -> str:
    """Return a safe display name for the learned face."""
    person_name = re.sub(r"\s+", " ", str(raw_name or "")).strip()
    if not person_name:
        return ""
    if any(sep in person_name for sep in ("/", "\\", "\x00")):
        return ""
    return person_name


def _image_filename_stem(person_name: str) -> str:
    """Create a filesystem-friendly stem for the saved example image."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", person_name).strip("._")
    return stem or "person"


def _latest_image_path() -> Path:
    """Return the stable path for the retained observation image."""
    return Path(os.environ.get("VISION_LATEST_IMAGE_PATH", str(DEFAULT_LATEST_IMAGE_PATH)))


def _face_db_path() -> Path:
    """Return the face database path shared with the assistant."""
    return Path(os.environ.get("VISION_FACE_DB_PATH", str(DEFAULT_FACE_DB_PATH)))


def _max_image_age_seconds() -> float:
    """Return the maximum allowed age for a retained observation image."""
    raw_value = os.environ.get("VISION_LATEST_IMAGE_MAX_AGE_SECONDS", "0")
    try:
        return float(raw_value)
    except ValueError:
        return 0.0


class LearnFaceTool(CodedTool):
    """
    Save the latest retained observation image as a known face example.

    This tool is intended to be called when the user identifies a visible
    person by name and wants the robot to remember them for future recognition.
    """

    def invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> str:
        del sly_data  # Unused for this tool.

        if cv2 is None:
            return "Error: OpenCV is not available, so I cannot read the latest observation image."

        person_name = _normalize_person_name(
            args.get("person_name", "") or args.get("name", "")
        )
        if not person_name:
            return "Error: No valid person_name provided."

        image_path = _latest_image_path()
        if not image_path.exists():
            return (
                "Error: No latest observation image is available yet. "
                "Ask the person to stand in front of the robot and wait for a camera capture."
            )

        image_age_seconds = time.time() - image_path.stat().st_mtime
        max_age_seconds = _max_image_age_seconds()
        if max_age_seconds > 0 and image_age_seconds > max_age_seconds:
            return (
                f"Error: The latest observation image is too old ({int(image_age_seconds)} seconds). "
                "Ask the person to stand in front of the robot and wait for a fresh camera capture."
            )

        image = cv2.imread(str(image_path))
        if image is None or getattr(image, "size", 0) == 0:
            return f"Error: Could not read the latest observation image at {image_path}."

        face_db_path = _face_db_path()
        vision = VisionCore(
            face_db_path=str(face_db_path),
            initialize_yolo=False,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{_image_filename_stem(person_name)}_{timestamp}.jpg"

        if not vision.add_face_to_database(
            person_name=person_name,
            image=image,
            image_filename=image_filename,
        ):
            detail = getattr(vision, "last_face_db_error", "") or f"Failed to add '{person_name}' to the face database."
            return f"Error: {detail}"

        logging.info(
            "LearnFaceTool stored a new face example for %s from %s into %s",
            person_name,
            image_path,
            face_db_path,
        )
        return (
            f"Stored the latest observation image for '{person_name}' so I can try to recognize them later."
        )

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> str:
        """Delegate to the synchronous implementation."""
        return self.invoke(args, sly_data)
