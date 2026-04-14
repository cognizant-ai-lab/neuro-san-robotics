import logging
import os
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
    from coded_tools.unigo2.vision_core import VisionCore, open_camera
except ImportError as exc:  # pragma: no cover - exercised in runtime environments
    VisionCore = None
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


def _env_int(name: str, default: int) -> int:
    """Parse integer environment variables with a safe fallback."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Parse float environment variables with a safe fallback."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


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


def build_scene_input(timestamp: str, object_names: List[str]) -> Optional[str]:
    """Build the agent input payload for a silent observation interval."""
    if not object_names:
        return None

    lines = [f"{timestamp} user: [Silence]"]
    lines.extend(f"{timestamp} saw: {object_name}" for object_name in object_names)
    return "\n" + "\n".join(lines)


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
    ):
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
        return VisionCore is not None and open_camera is not None and cv2 is not None

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

        use_tensorrt = _env_flag("VISION_USE_JETSON_CONFIG", default=False)
        input_size = _env_int("VISION_OBSERVER_INPUT_SIZE", 640)
        confidence_threshold = _env_float(
            "VISION_OBSERVER_CONFIDENCE",
            0.65 if use_tensorrt else 0.55,
        )

        try:
            self._vision = VisionCore(
                yolo_model=os.environ.get("VISION_YOLO_MODEL", "yolov8n.pt"),
                face_model="Facenet",
                face_db_path=str(REPO_ROOT / "face_database"),
                use_tensorrt=use_tensorrt,
                confidence_threshold=confidence_threshold,
                iou_threshold=0.45,
                input_size=input_size,
                half_precision=use_tensorrt,
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

    def _read_frame(self):
        if not self._ensure_camera():
            return None

        for _ in range(2):
            ret, frame = self._capture.read()
            if ret and frame is not None and getattr(frame, "size", 0) > 0:
                return frame
            time.sleep(0.05)

        logging.warning("Scene observer failed to read a frame; reopening camera on next attempt")
        self._release_capture()
        return None

    def _annotate_frame(self, vision, frame, results: Dict[str, Any]):
        annotated = vision.visualize_detections(frame, results)
        cv2.putText(
            annotated,
            results.get("summary", "No detections"),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        return annotated

    def _detect_scene(self, vision, frame):
        """
        Run object detection on a resized copy of the frame and scale boxes back.

        This mirrors the `vision_core.py` demo structure more closely than the
        original full-resolution direct call and gives snapshot captures a more
        accurate, higher-resolution detection pass.
        """
        max_width = max(64, _env_int("VISION_OBSERVER_MAX_WIDTH", 640))
        orig_h, orig_w = frame.shape[:2]

        if orig_w > max_width:
            scale = max_width / float(orig_w)
            proc_w = max(1, int(orig_w * scale))
            proc_h = max(1, int(orig_h * scale))
            process_frame = cv2.resize(frame, (proc_w, proc_h))
        else:
            process_frame = frame
            proc_h, proc_w = frame.shape[:2]

        scale_x = orig_w / proc_w
        scale_y = orig_h / proc_h
        results = vision.detect_all(process_frame, detect_faces=False, verbose=False)

        for detected_object in results.get("objects", []):
            detected_object["bbox"] = [
                int(detected_object["bbox"][0] * scale_x),
                int(detected_object["bbox"][1] * scale_y),
                int(detected_object["bbox"][2] * scale_x),
                int(detected_object["bbox"][3] * scale_y),
            ]

        for face in results.get("faces", []):
            if face.get("bbox"):
                face["bbox"] = [
                    int(face["bbox"][0] * scale_x),
                    int(face["bbox"][1] * scale_y),
                    int(face["bbox"][2] * scale_x),
                    int(face["bbox"][3] * scale_y),
                ]

        return results

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
        with self._lock:
            vision = self._ensure_vision()
            if vision is None:
                return None

            frame = self._read_frame()
            if frame is None:
                return None

            results = self._detect_scene(vision, frame)
            object_names = summarize_observed_objects(results.get("objects", []))
            annotated = self._annotate_frame(vision, frame, results)
            updated_at_ms = self._write_latest_image(annotated)
            timestamp = datetime.now().strftime("[%I:%M:%S%p]").lower()

            self._last_observation = {
                "image_url": f"{self.public_image_url}?t={updated_at_ms}",
                "summary": results.get("summary", "No detections"),
                "objects": object_names,
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

    def cleanup(self) -> None:
        """Release resources and remove the retained image."""
        with self._lock:
            self._release_capture()
            if self.image_path.exists():
                try:
                    self.image_path.unlink()
                except OSError:  # pragma: no cover - filesystem-dependent
                    logging.exception("Failed to delete latest observation image")
            self._last_observation = None
