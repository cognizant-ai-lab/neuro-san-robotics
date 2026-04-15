import logging
import os
import tempfile
import threading
import time
from collections import deque
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


def _default_observer_vision_settings() -> Dict[str, Any]:
    """
    Mirror the standalone VisionCore defaults unless explicitly overridden.

    The standalone `vision_core.py` script already works on CAIL-E, so the Flask
    observer should start from the same runtime configuration instead of inventing
    its own thresholds and input size.
    """
    use_tensorrt = _env_flag("VISION_USE_JETSON_CONFIG", default=False)
    if use_tensorrt:
        return {
            "use_tensorrt": True,
            "half_precision": True,
            "input_size": _env_int("VISION_OBSERVER_INPUT_SIZE", 640),
            "confidence_threshold": _env_float("VISION_OBSERVER_CONFIDENCE", 0.65),
        }

    return {
        "use_tensorrt": False,
        "half_precision": False,
        "input_size": _env_int("VISION_OBSERVER_INPUT_SIZE", 256),
        "confidence_threshold": _env_float("VISION_OBSERVER_CONFIDENCE", 0.60),
    }


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

        settings = _default_observer_vision_settings()
        yolo_model = _resolve_repo_relative_path(
            os.environ.get("VISION_YOLO_MODEL", "yolov8n.pt")
        )

        try:
            self._vision = VisionCore(
                yolo_model=yolo_model,
                face_model="Facenet",
                face_db_path=str(REPO_ROOT / "face_database"),
                use_tensorrt=settings["use_tensorrt"],
                confidence_threshold=settings["confidence_threshold"],
                iou_threshold=0.45,
                input_size=settings["input_size"],
                half_precision=settings["half_precision"],
            )
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

    def _read_frame(self):
        if not self._ensure_camera():
            return None

        frame_reads = max(1, _env_int("VISION_OBSERVER_WARMUP_FRAMES", 5))
        last_frame = None

        for _ in range(frame_reads):
            ret, frame = self._capture.read()
            if ret and frame is not None and getattr(frame, "size", 0) > 0:
                last_frame = frame
            time.sleep(0.05)

        if last_frame is not None:
            return last_frame

        logging.warning("Scene observer failed to read a frame; reopening camera on next attempt")
        self._release_capture()
        return None

    def _read_detection_frames(self) -> List[Any]:
        """
        Capture a short burst of frames and return the latest good candidates.

        The standalone interactive demo benefits from repeated frames over a live
        stream. Capturing a burst here gives the observer a similar chance to
        avoid a blurred or poorly exposed single snapshot.
        """
        initial_frame = self._read_frame()
        if initial_frame is None:
            return []

        burst_size = max(1, _env_int("VISION_OBSERVER_BURST_FRAMES", 6))
        frames = deque([initial_frame], maxlen=burst_size)
        if self._capture is None:
            return list(frames)

        for _ in range(burst_size - 1):
            ret, frame = self._capture.read()
            if ret and frame is not None and getattr(frame, "size", 0) > 0:
                frames.append(frame)
            time.sleep(0.03)

        return list(frames)

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
        Run object detection on one or more resized copies of the frame.

        The standalone headless demo resizes to 320x240 before inference. We use
        that same snapshot path first, then retry with a larger 640x480 frame,
        and finally with the native frame if the smaller passes find nothing.
        """
        orig_h, orig_w = frame.shape[:2]
        primary_size = (
            max(64, _env_int("VISION_OBSERVER_PROCESS_WIDTH", 320)),
            max(48, _env_int("VISION_OBSERVER_PROCESS_HEIGHT", 240)),
        )
        fallback_size = (
            max(primary_size[0], _env_int("VISION_OBSERVER_FALLBACK_WIDTH", 640)),
            max(primary_size[1], _env_int("VISION_OBSERVER_FALLBACK_HEIGHT", 480)),
        )
        candidate_sizes = []
        for candidate_size in (primary_size, fallback_size, (orig_w, orig_h)):
            if candidate_size not in candidate_sizes:
                candidate_sizes.append(candidate_size)

        final_results = {"objects": [], "faces": [], "summary": "No detections"}

        for proc_w, proc_h in candidate_sizes:
            proc_w = max(1, min(proc_w, orig_w))
            proc_h = max(1, min(proc_h, orig_h))
            if proc_w == orig_w and proc_h == orig_h:
                process_frame = frame
            else:
                process_frame = cv2.resize(frame, (proc_w, proc_h))

            scale_x = orig_w / proc_w
            scale_y = orig_h / proc_h
            results = vision.detect_all(process_frame, detect_faces=False, verbose=False)
            logging.info(
                "Scene observer detection attempt at %sx%s found %d objects and %d faces",
                proc_w,
                proc_h,
                len(results.get("objects", [])),
                len(results.get("faces", [])),
            )

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

            final_results = results
            if results.get("objects") or results.get("faces"):
                break

        return final_results

    @staticmethod
    def _score_results(results: Dict[str, Any]) -> float:
        """Rank detection results so the best frame in a burst can be selected."""
        objects = results.get("objects", [])
        faces = results.get("faces", [])
        object_confidence = sum(float(obj.get("confidence", 0.0)) for obj in objects)
        face_confidence = sum(float(face.get("confidence", 0.0)) for face in faces)
        return (len(objects) * 100.0) + (len(faces) * 50.0) + object_confidence + face_confidence

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

            candidate_frames = self._read_detection_frames()
            if not candidate_frames:
                return None

            selected_frame = candidate_frames[-1]
            selected_results = {"objects": [], "faces": [], "summary": "No detections"}
            best_score = -1.0

            for frame in candidate_frames:
                results = self._detect_scene(vision, frame)
                score = self._score_results(results)
                if score > best_score:
                    best_score = score
                    selected_frame = frame
                    selected_results = results

            object_names = summarize_observed_objects(selected_results.get("objects", []))
            annotated = self._annotate_frame(vision, selected_frame, selected_results)
            updated_at_ms = self._write_latest_image(annotated)
            timestamp = datetime.now().strftime("[%I:%M:%S%p]").lower()

            self._last_observation = {
                "image_url": f"{self.public_image_url}?t={updated_at_ms}",
                "summary": selected_results.get("summary", "No detections"),
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
