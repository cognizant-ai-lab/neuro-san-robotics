import logging
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import cv2
except ImportError:  # pragma: no cover - exercised in runtime environments
    cv2 = None

try:
    from coded_tools.unigo2.vision_core import (
        VisionCore,
        detect_camera_snapshot,
        get_default_vision_core_settings,
        open_camera,
    )
except ImportError as exc:  # pragma: no cover - exercised in runtime environments
    VisionCore = None
    detect_camera_snapshot = None
    get_default_vision_core_settings = None
    open_camera = None
    _VISION_IMPORT_ERROR = exc
else:
    _VISION_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse common boolean environment variable values."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_repo_relative_path(path_text: str) -> str:
    """
    Resolve model/resource paths relative to the repo root when not absolute.

    The standalone vision script is typically run from the repo root, but the
    Flask app may be launched from elsewhere. Using an absolute path keeps both
    entrypoints aligned.
    """
    path = Path(path_text)
    if path.is_absolute():
        return str(path)

    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return str(repo_candidate)

    return str(path)


def summarize_observed_objects(objects: List[Dict[str, Any]]) -> List[str]:
    """
    Convert raw detection dictionaries into de-duplicated object names.

    Keeps the highest-confidence score per class and sorts classes by confidence
    so the most salient scene items are listed first.
    """
    ranked_objects: Dict[str, float] = {}
    for detected_object in objects:
        class_name = detected_object.get("class_name")
        if not class_name:
            continue
        confidence = float(detected_object.get("confidence", 0.0))
        ranked_objects[class_name] = max(confidence, ranked_objects.get(class_name, 0.0))

    return [
        class_name
        for class_name, _ in sorted(
            ranked_objects.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def summarize_observed_entities(results: Dict[str, Any]) -> List[str]:
    """
    Combine recognized face names with object labels for agent/UI consumption.

    When a known face is recognized, prefer the person's name over the generic
    `person` object label so downstream consumers can refer to the person.
    """
    object_names = summarize_observed_objects(results.get("objects", []))
    known_face_names = []
    for face in results.get("faces", []):
        face_name = str(face.get("name", "")).strip()
        if face_name and face_name != "Unknown" and face_name not in known_face_names:
            known_face_names.append(face_name)

    if known_face_names:
        object_names = [name for name in object_names if name != "person"]

    return known_face_names + object_names


def build_scene_input(timestamp: str, object_names: List[str]) -> Optional[str]:
    """Build the agent input payload for a silent observation interval."""
    if not object_names:
        return None

    lines = [f"{timestamp} user: [Silence]"]
    lines.extend(f"{timestamp} saw: {object_name}" for object_name in object_names)
    return "\n" + "\n".join(lines)


def observation_signature(observation: Optional[Dict[str, Any]]) -> tuple[str, ...]:
    """
    Normalize an observation into a stable signature for duplicate suppression.

    Repeated identical scene labels should not wake the top agent every interval,
    but the camera UI should still be free to refresh the latest image.
    """
    if not observation:
        return ()

    return tuple(
        str(object_name).strip()
        for object_name in observation.get("objects", [])
        if str(object_name).strip()
    )


class SceneObserver:
    """
    Captures a single scene observation on demand and keeps only the latest JPEG.

    The saved file is overwritten atomically in a temp directory so the web UI can
    show the latest scene without accumulating files on disk.
    """

    def __init__(
        self,
        *,
        camera_source: Optional[str] = None,
        public_image_url: str = "/api/observation/latest.jpg",
        enabled: Optional[bool] = None,
    ):
        self.enabled = (
            _env_flag("CONSCIOUS_ENABLE_SCENE_OBSERVER", default=sys.platform.startswith("linux"))
            if enabled is None
            else enabled
        )
        self.camera_source = (
            camera_source
            or os.environ.get("VISION_CAMERA_SOURCE")
            or os.environ.get("GO2_CAMERA_SOURCE")
        )
        self.public_image_url = public_image_url
        self.image_dir = Path(tempfile.gettempdir()) / "neuro_san_conscious_assistant"
        self.image_path = self.image_dir / "latest_observation.jpg"
        self._capture = None
        self._vision = None
        self._lock = threading.Lock()
        self._last_observation: Optional[Dict[str, Any]] = None
        self._warned_unavailable = False

    def available(self) -> bool:
        """Return whether the vision observer can run in this environment."""
        return (
            self.enabled
            and VisionCore is not None
            and detect_camera_snapshot is not None
            and get_default_vision_core_settings is not None
            and open_camera is not None
            and cv2 is not None
        )

    def _ensure_vision(self):
        if self._vision is not None:
            return self._vision

        if not self.available():
            if not self._warned_unavailable:
                logging.warning(
                    "Scene observer unavailable; install vision dependencies. Import error: %s",
                    _VISION_IMPORT_ERROR,
                )
                self._warned_unavailable = True
            return None

        yolo_model = _resolve_repo_relative_path(
            os.environ.get("VISION_YOLO_MODEL", "yolov8n.pt")
        )
        settings = get_default_vision_core_settings(
            yolo_model=yolo_model,
            face_model="Facenet",
            face_db_path=str(REPO_ROOT / "face_database"),
        )

        try:
            self._vision = VisionCore(**settings)
            if getattr(self._vision, "yolo", None) is None:
                logging.error(
                    "Scene observer initialized without a YOLO backend. "
                    "Resolved model path: %s",
                    yolo_model,
                )
        except Exception:  # pragma: no cover - hardware/runtime-dependent
            logging.exception("Failed to initialize SceneObserver vision backend")
            self._vision = None

        return self._vision

    def _ensure_camera(self) -> bool:
        if self._capture is not None and self._capture.isOpened():
            return True

        if open_camera is None:
            return False

        capture, camera_info = open_camera(
            camera_source=self.camera_source,
            verbose=False,
        )
        if capture is None:
            logging.warning(
                "Scene observer could not open a camera. Attempts: %s",
                camera_info.get("attempts", []),
            )
            return False

        self._capture = capture
        logging.info(
            "Scene observer connected to %s via %s",
            camera_info.get("description"),
            camera_info.get("backend"),
        )
        return True

    def initialize(self, *, connect_camera: bool = False) -> bool:
        """
        Eagerly initialize the observer so heavy imports can happen on the main thread.

        The native observer agent invokes captures periodically. On some deployments,
        importing the YOLO stack from that worker thread is less reliable than doing
        it once during startup.
        """
        if not self.enabled:
            return False

        with self._lock:
            vision = self._ensure_vision()
            if vision is None:
                return False
            if connect_camera and not self._ensure_camera():
                return False
            return True

    def _write_latest_image(self, annotated_frame) -> int:
        self.image_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.image_path.with_suffix(".tmp.jpg")
        cv2.imwrite(str(tmp_path), annotated_frame)
        os.replace(tmp_path, self.image_path)
        return int(time.time() * 1000)

    def latest_image_path(self) -> Path:
        """Expose the latest retained image path for Flask routes."""
        return self.image_path

    def latest_observation(self) -> Optional[Dict[str, Any]]:
        """Return the latest saved observation payload, if one exists."""
        with self._lock:
            if self._last_observation is None or not self.image_path.exists():
                return None
            return dict(self._last_observation)

    def observe(self) -> Optional[Dict[str, Any]]:
        """
        Capture one frame, detect objects, overwrite the latest JPEG, and return metadata.
        """
        if not self.enabled:
            return None

        with self._lock:
            vision = self._ensure_vision()
            if vision is None:
                return None

            if not self._ensure_camera():
                return None

            snapshot = detect_camera_snapshot(
                self._capture,
                vision,
                enable_face_recognition=True,
            )
            if snapshot is None:
                logging.warning("Scene observer snapshot capture returned no frame")
                self._release_capture()
                return None

            results = snapshot["results"]
            object_names = summarize_observed_entities(results)
            logging.info(
                "Scene observer snapshot summary: %s",
                results.get("summary", "No detections"),
            )
            updated_at_ms = self._write_latest_image(snapshot["annotated"])
            timestamp = datetime.now().strftime("[%I:%M:%S%p]").lower()

            self._last_observation = {
                "image_url": f"{self.public_image_url}?t={updated_at_ms}",
                "summary": results.get("summary", "No detections"),
                "objects": object_names,
                "faces": results.get("faces", []),
                "timestamp": timestamp,
            }
            return dict(self._last_observation)

    def _release_capture(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:  # pragma: no cover - hardware/runtime-dependent
                logging.exception("Failed to release SceneObserver camera")
            finally:
                self._capture = None

    def cleanup(self, timeout_s: float = 2.0) -> None:
        """Release resources and remove the retained image."""
        acquired = self._lock.acquire(timeout=timeout_s)
        if not acquired:
            logging.warning("Scene observer cleanup skipped because camera is busy")
            return

        try:
            self._release_capture()
            if self.image_path.exists():
                try:
                    self.image_path.unlink()
                except OSError:  # pragma: no cover - filesystem-dependent
                    logging.exception("Failed to delete latest observation image")
            self._last_observation = None
        finally:
            self._lock.release()
