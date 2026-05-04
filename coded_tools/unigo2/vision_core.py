"""
VisionCore - Object Detection and Face Recognition for Unitree Go2 EDU
Optimized for NVIDIA Jetson Orin

This module provides a unified vision system combining:
1. YOLO-based object detection (80 COCO classes)
2. DeepFace-based face recognition
3. TensorRT optimization for edge deployment
4. Face database management for known individuals

Performance optimizations:
- TensorRT acceleration (2-3x speedup on Jetson)
- Half-precision inference (FP16)
- Dynamic input resolution
- IoU-based NMS for better accuracy
"""

import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import json


CameraSource = Union[int, str]
_UNITREE_CHANNEL_STATE: Dict[str, Any] = {
    "initialized": False,
    "ifname": None,
}


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


def _is_jetson_platform() -> bool:
    """Detect whether we're running on NVIDIA Jetson hardware."""
    if _env_flag("VISION_FORCE_JETSON", default=False):
        return True

    if platform.machine().lower() != "aarch64":
        return False

    if Path("/etc/nv_tegra_release").exists():
        return True

    model_path = Path("/proc/device-tree/model")
    try:
        return "jetson" in model_path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False


def build_jetson_camera_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    display_width: Optional[int] = None,
    display_height: Optional[int] = None,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """Build a Jetson CSI camera pipeline that OpenCV can consume via GStreamer."""
    display_width = display_width or capture_width
    display_height = display_height or capture_height

    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true sync=false"
    )


def _camera_backend_label(backend: Optional[int]) -> str:
    """Return a human-readable backend name for logging."""
    if backend is None:
        return "default backend"
    if backend == getattr(cv2, "CAP_GSTREAMER", None):
        return "GStreamer"
    if backend == getattr(cv2, "CAP_V4L2", None):
        return "V4L2"
    return f"backend {backend}"


def _load_unitree_video_sdk():
    """Load the Unitree camera client from whichever package layout is installed."""
    import_errors = []

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.video.video_client import VideoClient
        return ChannelFactoryInitialize, VideoClient
    except Exception as exc:
        import_errors.append(exc)

    try:
        from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2_python.unitree_sdk2py.go2.video.video_client import VideoClient
        return ChannelFactoryInitialize, VideoClient
    except Exception as exc:
        import_errors.append(exc)

    error_messages = ", ".join(str(exc) for exc in import_errors if str(exc))
    raise ImportError(
        "Unitree camera SDK not available"
        + (f": {error_messages}" if error_messages else "")
    )


def _unitree_camera_interface(default_ifname: Optional[str] = None) -> Optional[str]:
    """Resolve the preferred network interface for Unitree SDK camera access."""
    return (
        default_ifname
        or os.environ.get("VISION_CAMERA_INTERFACE")
        or os.environ.get("GO2_CAMERA_INTERFACE")
        or os.environ.get("CYCLONEDDS_NETWORK_INTERFACE")
        or os.environ.get("IFNAME")
    )


def _unitree_camera_available() -> bool:
    """Return whether the Unitree front-camera SDK can be imported."""
    try:
        _load_unitree_video_sdk()
        return True
    except ImportError:
        return False


def _go2_channel_already_initialized() -> tuple[bool, Optional[str]]:
    """Detect DDS initialization done by Go2Macros in the same process."""
    try:
        from coded_tools.unigo2 import go2_macros
    except Exception:
        return False, None

    state = getattr(go2_macros, "_ROBOT_INIT_STATE", {})
    if not state.get("channel_initialized"):
        return False, None

    return True, getattr(go2_macros, "IFNAME", None)


class UnitreeVideoCapture:
    """Small VideoCapture-compatible wrapper around Unitree's Go2 VideoClient."""

    def __init__(self, ifname: Optional[str] = None, timeout: float = 3.0):
        self.ifname = _unitree_camera_interface(ifname)
        self.timeout = timeout
        self._opened = False
        self._last_frame: Optional[np.ndarray] = None
        self._initialize()

    def _initialize(self) -> None:
        channel_factory_initialize, video_client_cls = _load_unitree_video_sdk()

        channel_ifname = self.ifname
        if not _UNITREE_CHANNEL_STATE["initialized"]:
            go2_initialized, go2_ifname = _go2_channel_already_initialized()
            if go2_initialized:
                _UNITREE_CHANNEL_STATE["initialized"] = True
                _UNITREE_CHANNEL_STATE["ifname"] = go2_ifname or channel_ifname
            else:
                if channel_ifname:
                    channel_factory_initialize(0, channel_ifname)
                else:
                    channel_factory_initialize(0)
                _UNITREE_CHANNEL_STATE["initialized"] = True
                _UNITREE_CHANNEL_STATE["ifname"] = channel_ifname

        self._client = video_client_cls()
        self._client.SetTimeout(self.timeout)
        self._client.Init()
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            return False, None

        code, data = self._client.GetImageSample()
        if code != 0 or not data:
            return False, None

        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return False, None

        self._last_frame = frame
        return True, frame

    def release(self) -> None:
        self._opened = False

    def get(self, prop_id: int) -> float:
        if self._last_frame is None:
            return 0.0
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._last_frame.shape[1])
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._last_frame.shape[0])
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        # The Unitree camera service chooses its own output size.
        return False


def _opencv_gui_available() -> bool:
    """
    Return whether it is safe to use OpenCV HighGUI windows on this machine.

    On Linux, OpenCV window backends usually require an X11/Wayland session.
    CAIL-E is typically headless, so we default to non-GUI mode there.
    """
    if _env_flag("VISION_HEADLESS", default=False):
        return False

    if _env_flag("VISION_FORCE_GUI", default=False):
        return True

    system_name = platform.system().lower()
    if system_name == "linux":
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    return True


def _annotate_frame_with_summary(
    frame: np.ndarray,
    results: Optional[Dict[str, Any]],
    vision: "VisionCore",
) -> np.ndarray:
    """Create a saved preview image with detections and a summary caption."""
    annotated = frame.copy()
    if results is not None:
        annotated = vision.visualize_detections(frame, results)
        summary = results.get("summary", "No detections")
    else:
        summary = "No detections"

    cv2.putText(
        annotated,
        summary,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )
    return annotated


def _save_frame_snapshot(
    frame: np.ndarray,
    results: Optional[Dict[str, Any]],
    vision: "VisionCore",
    *,
    prefix: str = "detection",
) -> str:
    """Save an annotated frame and return the output filename."""
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    annotated = _annotate_frame_with_summary(frame, results, vision)
    cv2.imwrite(filename, annotated)
    return filename


def detect_camera_snapshot(
    cap,
    vision: "VisionCore",
    *,
    enable_face_recognition: bool = False,
    warmup_frames: Optional[int] = None,
    process_size: Tuple[int, int] = (320, 240),
) -> Optional[Dict[str, Any]]:
    """
    Capture one detection snapshot from an open camera using the headless demo path.

    This is the same warmup, resize, inference, and bbox-rescaling flow used by
    the standalone headless camera demo so other entrypoints can reuse it
    instead of maintaining their own detection pipeline.
    """
    if warmup_frames is None:
        warmup_frames = max(1, _env_int("VISION_HEADLESS_WARMUP_FRAMES", 5))

    last_frame = None
    for _ in range(max(1, warmup_frames)):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            last_frame = frame
        time.sleep(0.05)

    if last_frame is None:
        return None

    process_width = max(1, int(process_size[0]))
    process_height = max(1, int(process_size[1]))
    process_frame = cv2.resize(last_frame, (process_width, process_height))
    orig_h, orig_w = last_frame.shape[:2]
    proc_h, proc_w = process_frame.shape[:2]
    scale_x = orig_w / proc_w
    scale_y = orig_h / proc_h

    results = vision.detect_all(
        process_frame,
        detect_faces=False,
        verbose=False,
    )

    for obj in results['objects']:
        obj['bbox'] = [
            int(obj['bbox'][0] * scale_x),
            int(obj['bbox'][1] * scale_y),
            int(obj['bbox'][2] * scale_x),
            int(obj['bbox'][3] * scale_y)
        ]

    for face in results['faces']:
        if face['bbox']:
            face['bbox'] = [
                int(face['bbox'][0] * scale_x),
                int(face['bbox'][1] * scale_y),
            int(face['bbox'][2] * scale_x),
            int(face['bbox'][3] * scale_y)
        ]

    if enable_face_recognition:
        results['faces'] = vision.recognize_faces(last_frame)
        results['summary'] = vision._generate_summary(results)

    return {
        "frame": last_frame,
        "results": results,
        "annotated": _annotate_frame_with_summary(last_frame, results, vision),
    }


def _run_headless_camera_demo(
    cap,
    vision: "VisionCore",
    *,
    enable_face_recognition: bool = False,
) -> None:
    """
    Run a short non-GUI smoke test for headless robots.

    Captures a few frames, runs detection on the last good frame, saves a preview
    image to disk, and exits cleanly.
    """
    snapshot = detect_camera_snapshot(
        cap,
        vision,
        enable_face_recognition=enable_face_recognition,
    )

    if snapshot is None:
        print("[Headless] Failed to capture a frame from the camera.")
        return

    output_path = _save_frame_snapshot(
        snapshot["frame"],
        snapshot["results"],
        vision,
        prefix="headless_detection",
    )
    print("[Headless] No GUI session detected. Saved annotated snapshot instead of opening a window.")
    print(f"[Headless] Output: {output_path}")
    print(f"[Headless] Summary: {snapshot['results']['summary']}")
    print(f"[Headless] Objects: {len(snapshot['results']['objects'])}")
    print(f"[Headless] Faces: {len(snapshot['results']['faces'])}")


def _discover_v4l2_devices(limit: int = 6) -> List[str]:
    """Return available /dev/video* devices in numeric order."""
    devices = []
    for path in Path("/dev").glob("video*"):
        suffix = path.name.removeprefix("video")
        if suffix.isdigit():
            devices.append(path)

    devices.sort(key=lambda path: int(path.name.removeprefix("video")))
    return [str(path) for path in devices[: max(0, limit)]]


def _normalize_camera_source(camera_source: Optional[CameraSource]) -> Optional[Dict[str, Any]]:
    """Normalize a camera source override into a single candidate descriptor."""
    if camera_source is None:
        camera_source = (
            os.environ.get("VISION_CAMERA_SOURCE")
            or os.environ.get("GO2_CAMERA_SOURCE")
        )
        if not camera_source:
            return None

    if isinstance(camera_source, int):
        return {
            "kind": "opencv",
            "source": camera_source,
            "backend": None,
            "description": f"camera index {camera_source}",
        }

    raw_source = str(camera_source).strip()
    if not raw_source:
        return None

    if raw_source.lstrip("+-").isdigit():
        index = int(raw_source)
        return {
            "kind": "opencv",
            "source": index,
            "backend": None,
            "description": f"camera index {index}",
        }

    lowered = raw_source.lower()
    if lowered in {"unitree", "go2"} or lowered.startswith(("unitree:", "go2:")):
        _, _, interface_text = raw_source.partition(":")
        interface_name = interface_text.strip() or None
        description = "Unitree Go2 front camera"
        if interface_name:
            description += f" via {interface_name}"
        return {
            "kind": "unitree",
            "source": raw_source,
            "backend": None,
            "description": description,
            "ifname": interface_name,
        }

    if lowered.startswith(("jetson:", "csi:", "sensor:")):
        _, _, sensor_text = raw_source.partition(":")
        sensor_id = int(sensor_text.strip() or "0")
        return {
            "kind": "opencv",
            "source": build_jetson_camera_pipeline(sensor_id=sensor_id),
            "backend": getattr(cv2, "CAP_GSTREAMER", None),
            "description": f"Jetson CSI sensor {sensor_id}",
        }

    if lowered.startswith("gstreamer:"):
        pipeline = raw_source.split(":", 1)[1].strip()
        return {
            "kind": "opencv",
            "source": pipeline,
            "backend": getattr(cv2, "CAP_GSTREAMER", None),
            "description": "custom GStreamer pipeline",
        }

    if "!" in raw_source:
        return {
            "kind": "opencv",
            "source": raw_source,
            "backend": getattr(cv2, "CAP_GSTREAMER", None),
            "description": "inline GStreamer pipeline",
        }

    if raw_source.startswith("/dev/video"):
        return {
            "kind": "opencv",
            "source": raw_source,
            "backend": getattr(cv2, "CAP_V4L2", None),
            "description": raw_source,
        }

    return {
        "kind": "opencv",
        "source": raw_source,
        "backend": None,
        "description": raw_source,
    }


def get_camera_candidates(
    camera_source: Optional[CameraSource] = None,
    max_indices: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build camera candidates in a robot-friendly order.

    Priority:
    1. Explicit source override (`VISION_CAMERA_SOURCE`, `unitree:eth0`, `/dev/videoN`, index, or pipeline)
    2. Unitree Go2 front camera via SDK2 (on Jetson/robot installs)
    3. Jetson CSI sensors via GStreamer
    4. Present V4L2 devices under `/dev/video*`
    5. Plain OpenCV camera indices for laptop/desktop webcams
    """
    normalized = _normalize_camera_source(camera_source)
    if normalized is not None:
        return [normalized]

    if max_indices is None:
        max_indices = int(os.environ.get("VISION_CAMERA_SCAN_LIMIT", "6"))

    candidates: List[Dict[str, Any]] = []
    seen = set()

    def add_candidate(source: CameraSource, backend: Optional[int], description: str) -> None:
        key = (str(source), backend)
        if key in seen:
            return
        seen.add(key)
        candidates.append({
            "kind": "opencv",
            "source": source,
            "backend": backend,
            "description": description,
        })

    if _is_jetson_platform() and _unitree_camera_available():
        unitree_ifname = _unitree_camera_interface()
        unitree_description = "Unitree Go2 front camera"
        if unitree_ifname:
            unitree_description += f" via {unitree_ifname}"
        candidates.append({
            "kind": "unitree",
            "source": "unitree",
            "backend": None,
            "description": unitree_description,
            "ifname": unitree_ifname,
        })
        seen.add(("unitree", None))

    if _is_jetson_platform():
        for sensor_id in range(2):
            add_candidate(
                build_jetson_camera_pipeline(sensor_id=sensor_id),
                getattr(cv2, "CAP_GSTREAMER", None),
                f"Jetson CSI sensor {sensor_id}",
            )

    for device_path in _discover_v4l2_devices(limit=max_indices):
        add_candidate(device_path, getattr(cv2, "CAP_V4L2", None), device_path)

    for camera_index in range(max(0, max_indices)):
        add_candidate(camera_index, None, f"camera index {camera_index}")

    return candidates


def open_camera(
    camera_source: Optional[CameraSource] = None,
    *,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    warmup_reads: int = 3,
    verbose: bool = True,
) -> Tuple[Optional[cv2.VideoCapture], Dict[str, Any]]:
    """
    Open the first camera candidate that produces a real frame.

    This avoids the common Jetson/robot failure mode where `VideoCapture(0)`
    is a valid laptop assumption but not a valid camera source on the robot.
    """
    attempts: List[str] = []

    for candidate in get_camera_candidates(camera_source):
        kind = candidate.get("kind", "opencv")
        source = candidate["source"]
        backend = candidate["backend"]
        description = candidate["description"]
        backend_label = "Unitree SDK2" if kind == "unitree" else _camera_backend_label(backend)

        if verbose:
            print(f"[VisionCore] [Camera] Trying {description} via {backend_label}")

        try:
            if kind == "unitree":
                capture = UnitreeVideoCapture(ifname=candidate.get("ifname"))
            else:
                capture = cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)
        except Exception as exc:
            attempts.append(f"{description} via {backend_label}: exception while opening ({exc})")
            continue

        if not capture or not capture.isOpened():
            attempts.append(f"{description} via {backend_label}: could not open")
            if capture:
                capture.release()
            continue

        if frame_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        if frame_height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        frame_ok = False
        frame = None
        for _ in range(max(1, warmup_reads)):
            frame_ok, frame = capture.read()
            if frame_ok and frame is not None and getattr(frame, "size", 0) > 0:
                break
            time.sleep(0.05)

        if not frame_ok or frame is None or getattr(frame, "size", 0) == 0:
            attempts.append(f"{description} via {backend_label}: opened but produced no frame")
            capture.release()
            continue

        return capture, {
            "source": source,
            "backend": backend_label,
            "description": description,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "attempts": attempts,
        }

    return None, {
        "camera_source": camera_source,
        "attempts": attempts,
    }


def get_default_vision_core_settings(
    *,
    yolo_model: str = "yolov8n.pt",
    face_model: str = "Facenet",
    face_db_path: str = "./face_database",
) -> Dict[str, Any]:
    """
    Return the default VisionCore kwargs used by the standalone demo.

    Keeping this in one place lets the Flask observer and the standalone script
    share the same model, confidence, and input-size defaults.
    """
    use_jetson_config = _env_flag("VISION_USE_JETSON_CONFIG", default=False)
    if use_jetson_config:
        return {
            "yolo_model": yolo_model,
            "face_model": face_model,
            "face_db_path": face_db_path,
            "use_tensorrt": True,
            "confidence_threshold": 0.65,
            "iou_threshold": 0.45,
            "input_size": 640,
            "half_precision": True,
        }

    return {
        "yolo_model": yolo_model,
        "face_model": face_model,
        "face_db_path": face_db_path,
        "use_tensorrt": False,
        "confidence_threshold": 0.60,
        "iou_threshold": 0.45,
        "input_size": 256,
        "half_precision": False,
    }


class VisionCore:
    """
    Unified vision system for object detection and face recognition.

    Architecture:
    ┌─────────────────┐
    │  Input Image    │
    └────────┬────────┘
             │
    ┌────────▼────────────────────┐
    │  Object Detection (YOLO)    │  ← 80 COCO classes
    │  - Optimized with TensorRT  │
    │  - Configurable confidence  │
    │  - IoU-based filtering      │
    └────────┬────────────────────┘
             │
    ┌────────▼────────────────────┐
    │  Face Recognition (DeepFace)│  ← Identifies known faces
    │  - Lazy loading for speed   │
    │  - Database lookup          │
    │  - Similarity matching      │
    └────────┬────────────────────┘
             │
    ┌────────▼────────┐
    │  Results Dict   │
    │  - Objects list │
    │  - Faces list   │
    │  - Summary text │
    └─────────────────┘
    """

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        face_model: str = "Facenet",
        face_db_path: str = "./face_database",
        use_tensorrt: bool = False,
        confidence_threshold: float = 0.6,  # Increased from 0.5 for better accuracy
        iou_threshold: float = 0.45,        # IoU for Non-Maximum Suppression
        input_size: int = 640,              # Input resolution (lower = faster, higher = more accurate)
        half_precision: bool = False,       # FP16 mode (faster on Jetson, slight accuracy loss)
        initialize_yolo: bool = True
    ):
        """
        Initialize VisionCore system with optimized parameters.

        SPEED vs ACCURACY Trade-offs:
        ================================

        For Maximum Speed (Real-time on Jetson):
        - yolo_model: "yolov8n.pt" (nano - fastest)
        - use_tensorrt: True (2-3x speedup)
        - input_size: 416 or 480 (smaller = faster)
        - half_precision: True (FP16 - 2x faster)
        - confidence_threshold: 0.6+ (fewer false positives)

        For Maximum Accuracy:
        - yolo_model: "yolov8m.pt" or "yolov8l.pt"
        - use_tensorrt: True (still faster than PyTorch)
        - input_size: 640 or 1280 (larger = better detection)
        - half_precision: False (FP32 - more precise)
        - confidence_threshold: 0.5-0.7 (balance)

        Recommended for Go2 EDU Robot:
        - yolo_model: "yolov8n.pt" with TensorRT
        - input_size: 640 (good balance)
        - half_precision: True (Jetson Orin has good FP16 support)
        - confidence_threshold: 0.6 (reduces false positives like "cell phone")

        Args:
            yolo_model: YOLO model variant
                - yolov8n.pt: Nano (fastest, ~3MB)
                - yolov8s.pt: Small (balanced, ~11MB)
                - yolov8m.pt: Medium (accurate, ~25MB)

            face_model: DeepFace model backend
                - "Facenet": Fast, 128D embeddings (RECOMMENDED)
                - "Facenet512": More accurate, 512D embeddings
                - "VGG-Face": Classic, slower
                - "ArcFace": State-of-art accuracy, slower

            face_db_path: Directory storing known face images
                Structure: face_database/person_name/image.jpg

            use_tensorrt: Enable TensorRT optimization
                - Converts PyTorch model to optimized TensorRT engine
                - First run takes 3-5 min (one-time conversion)
                - Subsequent runs are 2-3x faster

            confidence_threshold: Min confidence for detections (0.0-1.0)
                - Higher = fewer detections, but more accurate
                - Lower = more detections, but more false positives
                - 0.6-0.7 is good for reducing misdetections

            iou_threshold: IoU threshold for NMS (Non-Maximum Suppression)
                - Removes overlapping bounding boxes
                - Higher = more boxes kept (may duplicate)
                - Lower = fewer boxes (may merge different objects)
                - 0.45 is YOLO default, works well

            input_size: Input image resolution (pixels)
                - YOLO processes square images (e.g., 640x640)
                - Options: 320, 416, 480, 640, 1280
                - Smaller = faster but misses small objects
                - Larger = slower but better for distant objects

            half_precision: Use FP16 instead of FP32
                - Reduces memory usage by 50%
                - 2x faster on Jetson Orin (Tensor Cores)
                - Minimal accuracy loss (<1%)
                - Only works with TensorRT on Jetson
        """
        # ============================================================
        # PERFORMANCE CONFIGURATION
        # Store all tuning parameters
        # ============================================================
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.half_precision = half_precision
        self.use_tensorrt = use_tensorrt
        self.initialize_yolo = initialize_yolo

        # ============================================================
        # FACE RECOGNITION CONFIGURATION
        # ============================================================
        self.face_db_path = Path(face_db_path)
        self.face_model = face_model

        # Create face database directory if it doesn't exist
        self.face_db_path.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # INITIALIZE OBJECT DETECTION (YOLO)
        # This loads the model and optionally converts to TensorRT
        # ============================================================
        self.backend = "Unknown"  # Will be set by _init_yolo
        self.yolo = None
        self.class_names = {}
        if self.initialize_yolo:
            self._init_yolo(yolo_model)
        else:
            self.backend = "Disabled"

        # ============================================================
        # INITIALIZE FACE RECOGNITION (LAZY LOADING)
        # DeepFace is only loaded when first needed to save memory
        # ============================================================
        self.deepface = None
        self._face_detection_enabled = False
        self.last_face_db_error: Optional[str] = None

        print(f"[VisionCore] ✓ Initialized successfully")
        print(f"  Backend: {self.backend}")
        print(f"  YOLO Model: {yolo_model}")
        print(f"  Face Model: {face_model}")
        print(f"  Confidence Threshold: {confidence_threshold}")
        print(f"  IoU Threshold: {iou_threshold}")
        print(f"  Input Size: {input_size}x{input_size}")
        print(f"  Half Precision (FP16): {half_precision}")
        print(f"  TensorRT: {use_tensorrt}")

    def _init_yolo(self, model_name: str):
        """
        Initialize YOLO object detection model.

        YOLO (You Only Look Once) Architecture:
        ========================================
        - Single-stage detector (faster than R-CNN family)
        - Divides image into grid, predicts bounding boxes per cell
        - Each box has: [x, y, w, h, confidence, class_probabilities]
        - Uses anchor boxes for different object sizes
        - Post-processing: Non-Maximum Suppression (NMS) to remove duplicates

        Backend Optimization Strategy:
        =============================
        - Jetson Orin: TensorRT (fastest, GPU optimized)
        - MacBook/CPU: ONNX Runtime (2-4x faster than PyTorch)
        - Fallback: PyTorch (slowest but universal)

        TensorRT Optimization (Jetson):
        ===============================
        - NVIDIA's inference optimization engine
        - Fuses layers, optimizes memory access
        - Uses Tensor Cores (FP16 operations)
        - Converts PyTorch → ONNX → TensorRT Engine (.engine file)
        - One-time conversion (3-5 min), then reused

        ONNX Runtime Optimization (CPU):
        ================================
        - Optimized inference engine for CPU
        - 2-4x faster than PyTorch on CPU
        - Quantization and graph optimizations
        - Converts PyTorch → ONNX (.onnx file)
        - One-time export (30 sec), then reused

        Args:
            model_name: Path to YOLO model file (.pt extension)
        """
        try:
            from ultralytics import YOLO

            # ============================================================
            # TENSORRT ACCELERATION PATH (Jetson Orin)
            # Best for GPU inference on Jetson
            # ============================================================
            if self.use_tensorrt:
                engine_path = model_name.replace('.pt', '.engine')

                if os.path.exists(engine_path):
                    # Optimized engine found - use it directly
                    print(f"[VisionCore] ⚡⚡⚡ Loading TensorRT engine: {engine_path}")
                    self.yolo = YOLO(engine_path)
                    self.backend = "TensorRT"
                else:
                    # No engine found - convert PyTorch model to TensorRT
                    print(f"[VisionCore] TensorRT engine not found. Creating one...")
                    print(f"[VisionCore] Loading PyTorch model: {model_name}")
                    self.yolo = YOLO(model_name)

                    print(f"[VisionCore] Exporting to TensorRT... (this takes 3-5 minutes, one-time only)")
                    print(f"[VisionCore] Parameters: input_size={self.input_size}, half={self.half_precision}")

                    # Export with optimization parameters
                    self.yolo.export(
                        format='engine',      # TensorRT format
                        device=0,             # GPU 0
                        half=self.half_precision,  # FP16 mode
                        imgsz=self.input_size      # Input resolution
                    )

                    # Load the newly created engine
                    self.yolo = YOLO(engine_path)
                    self.backend = "TensorRT"
                    print(f"[VisionCore] ✓ TensorRT engine ready: {engine_path}")

            # ============================================================
            # ONNX RUNTIME PATH (CPU optimization)
            # 2-4x faster than PyTorch on MacBook/Desktop CPUs
            # ============================================================
            else:
                # Try to use ONNX if available (much faster on CPU)
                onnx_path = model_name.replace('.pt', '.onnx')

                if os.path.exists(onnx_path):
                    # ONNX model found - use it for CPU speed
                    print(f"[VisionCore] ⚡⚡⚡ Loading ONNX model: {onnx_path}")
                    print(f"[VisionCore] ONNX Runtime: 2-4x faster than PyTorch on CPU")
                    self.yolo = YOLO(onnx_path, task='detect')
                    self.backend = "ONNX"  # Track which backend is used
                else:
                    # No ONNX found - use PyTorch (slower)
                    print(f"[VisionCore] Loading PyTorch model: {model_name}")
                    print(f"[VisionCore] ⚠️  PyTorch is SLOW on CPU (expect 3-6 FPS)")
                    print(f"[VisionCore] 💡 Run 'python export_yolo_onnx.py' for 2-4x speedup")
                    self.yolo = YOLO(model_name)
                    self.backend = "PyTorch"

            # ============================================================
            # LOAD CLASS NAMES (80 COCO classes)
            # Examples: person, bicycle, car, motorbike, aeroplane, bus, train,
            #           truck, boat, traffic light, fire hydrant, stop sign,
            #           parking meter, bench, bird, cat, dog, horse, sheep, cow,
            #           elephant, bear, zebra, giraffe, backpack, umbrella, handbag,
            #           tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
            #           baseball bat, baseball glove, skateboard, surfboard, tennis racket,
            #           bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
            #           sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake,
            #           chair, sofa, pottedplant, bed, diningtable, toilet, tvmonitor,
            #           laptop, mouse, remote, keyboard, cell phone, microwave, oven,
            #           toaster, sink, refrigerator, book, clock, vase, scissors,
            #           teddy bear, hair drier, toothbrush
            # ============================================================
            self.class_names = self.yolo.names
            print(f"[VisionCore] ✓ YOLO ready: {len(self.class_names)} classes available")

        except ImportError as exc:
            print(f"[VisionCore] ✗ ERROR importing ultralytics: {exc}")
            print("[VisionCore]   Fix: verify the active Python environment can import 'ultralytics'")
            self.yolo = None
        except Exception as e:
            print(f"[VisionCore] ✗ ERROR loading YOLO: {e}")
            self.yolo = None

    def _init_deepface(self):
        """
        Lazy initialization of DeepFace (only when face recognition is needed).

        Why Lazy Loading?
        =================
        - DeepFace loads large neural network models (~100-500MB)
        - If you only need object detection, no need to load face models
        - Saves memory and startup time
        - Face recognition is initialized on first use

        DeepFace Architecture:
        ======================
        - Uses pre-trained CNN models for face embeddings
        - Each face → 128D or 512D vector (depending on model)
        - Similarity measured by: Cosine similarity or Euclidean distance
        - Lower distance = more similar faces

        Face Recognition Pipeline:
        1. Detect face in image (using MTCNN, RetinaFace, or OpenCV)
        2. Align face (normalize rotation and scale)
        3. Extract embedding vector (run through CNN)
        4. Compare with known faces in database
        5. Return closest match if distance < threshold
        """
        if self.deepface is None:
            try:
                from deepface import DeepFace
                self.deepface = DeepFace
                self._face_detection_enabled = True
                print(f"[VisionCore] ✓ DeepFace initialized with {self.face_model} model")
            except ImportError:
                print("[VisionCore] ✗ ERROR: deepface not installed")
                print("[VisionCore]   Fix: pip install deepface tf-keras")
                self._face_detection_enabled = False
            except Exception as e:
                print(f"[VisionCore] ✗ ERROR initializing DeepFace: {e}")
                self._face_detection_enabled = False

    def _face_detector_backend(self) -> str:
        """Return the detector backend used for DeepFace face localization."""
        backend = os.environ.get("VISION_FACE_DETECTOR_BACKEND", "opencv").strip()
        return backend or "opencv"

    def _face_match_threshold(self) -> float:
        """Return the maximum embedding distance accepted as a known-person match."""
        override = os.environ.get("VISION_FACE_MATCH_THRESHOLD")
        if override is not None:
            try:
                return float(override)
            except ValueError:
                pass

        return {
            "Facenet": 0.40,
            "Facenet512": 0.30,
            "VGG-Face": 0.60,
            "ArcFace": 0.68,
        }.get(self.face_model, 0.40)

    def _extract_detected_faces(
        self,
        image: np.ndarray,
        *,
        require_face: bool,
    ) -> List[Dict[str, Any]]:
        """
        Detect real faces in an image and normalize DeepFace facial_area output.

        This keeps recognition from guessing a name when the detector cannot
        confirm that a face is actually present in the frame.
        """
        if not self._face_detection_enabled:
            self._init_deepface()

        if not self._face_detection_enabled:
            return []

        try:
            detected = self.deepface.extract_faces(
                img_path=image,
                detector_backend=self._face_detector_backend(),
                enforce_detection=require_face,
                align=True,
            )
        except Exception as exc:
            if require_face:
                return []
            print(f"[VisionCore] ✗ ERROR extracting faces: {exc}")
            return []

        faces: List[Dict[str, Any]] = []
        for face_data in detected or []:
            facial_area = face_data.get("facial_area") or {}
            try:
                x = int(float(facial_area["x"]))
                y = int(float(facial_area["y"]))
                w = int(float(facial_area["w"]))
                h = int(float(facial_area["h"]))
            except (KeyError, TypeError, ValueError):
                continue

            if w <= 0 or h <= 0:
                continue

            confidence = face_data.get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0

            faces.append(
                {
                    "bbox": [x, y, w, h],
                    "confidence": confidence,
                }
            )

        return faces

    def _crop_face_image(self, image: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Crop a detected face with a small margin to preserve context."""
        if image is None or getattr(image, "size", 0) == 0:
            return None

        x, y, w, h = bbox
        margin_ratio = max(0.0, _env_float("VISION_FACE_CROP_MARGIN", 0.20))
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)

        height, width = image.shape[:2]
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(width, x + w + margin_x)
        y2 = min(height, y + h + margin_y)

        if x2 <= x1 or y2 <= y1:
            return None

        return image[y1:y2, x1:x2].copy()

    def _invalidate_face_database_cache(self) -> None:
        """Remove cached DeepFace representations so new examples are picked up immediately."""
        for pattern in ("representations_*.pkl", "representations_*.pickle"):
            for cache_path in self.face_db_path.glob(pattern):
                try:
                    cache_path.unlink()
                except OSError:
                    print(f"[VisionCore] ⚠️  Could not remove face cache: {cache_path}")

    def detect_objects(
        self,
        image: np.ndarray,
        classes: Optional[List[str]] = None,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in an image using YOLO.

        YOLO Detection Process:
        =======================
        1. Resize image to input_size (e.g., 640x640)
        2. Normalize pixel values to [0, 1]
        3. Run through neural network (backbone + neck + head)
        4. Output: [N x (4+1+80)] where N = number of detections
           - 4 values: bounding box [x, y, w, h]
           - 1 value: objectness confidence
           - 80 values: class probabilities
        5. Apply NMS to remove duplicate detections
        6. Filter by confidence threshold

        Common False Positives and How to Fix:
        =======================================
        Problem: Everything detected as "cell phone"
        Causes:
        - Low confidence threshold (too many weak detections)
        - Object held at wrong angle (model trained on typical angles)
        - Poor lighting (model trained on well-lit images)
        - Small object (YOLO struggles with tiny objects)

        Solutions:
        - Increase confidence_threshold to 0.6-0.7 ✓
        - Add class filtering (only detect specific classes)
        - Use larger input_size (640 or 1280) for small objects
        - Improve lighting conditions
        - Use yolov8m or yolov8l for better accuracy

        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            classes: Optional list of class names to filter
                     e.g., ['person', 'dog', 'laptop']
                     If None, all 80 classes are detected
            verbose: Print detection details for debugging

        Returns:
            List of detection dictionaries, each containing:
            {
                'class_name': 'person',        # Human-readable class
                'confidence': 0.85,            # Detection confidence [0-1]
                'bbox': [x1, y1, x2, y2],     # Bounding box coordinates
                'class_id': 0                  # Numeric class ID (0-79)
            }
        """
        # ============================================================
        # VALIDATION: Check if YOLO is initialized
        # ============================================================
        if self.yolo is None:
            if verbose:
                print("[VisionCore] Object detection unavailable (YOLO not loaded)")
            return []

        detections = []

        try:
            # ============================================================
            # RUN YOLO INFERENCE
            # Parameters:
            # - conf: Minimum confidence threshold
            # - iou: IoU threshold for NMS (removes overlapping boxes)
            # - imgsz: Input size (image resized to this)
            # - verbose: Suppress YOLO output logs
            # ============================================================
            results = self.yolo(
                image,
                conf=self.confidence_threshold,  # Filter low-confidence detections
                iou=self.iou_threshold,          # Remove overlapping boxes
                imgsz=self.input_size,           # Input resolution
                verbose=False                     # Suppress logs
            )

            # ============================================================
            # PROCESS DETECTIONS
            # YOLO returns a list of Results objects (usually length 1)
            # Each Result contains detected boxes
            # ============================================================
            for result in results:
                boxes = result.boxes  # Boxes object with all detections

                for box in boxes:
                    # Extract confidence score
                    conf = float(box.conf[0])

                    # Double-check confidence (YOLO should already filter)
                    if conf < self.confidence_threshold:
                        continue

                    # Extract class ID and name
                    cls_id = int(box.cls[0])
                    class_name = self.class_names[cls_id]

                    # ============================================================
                    # CLASS FILTERING (Optional)
                    # If user specified specific classes, only keep those
                    # Example: classes=['person', 'dog'] → ignore cars, chairs, etc.
                    # ============================================================
                    if classes and class_name not in classes:
                        continue

                    # ============================================================
                    # EXTRACT BOUNDING BOX
                    # Format: [x1, y1, x2, y2] where (x1,y1) = top-left, (x2,y2) = bottom-right
                    # Convert from tensor to numpy array to regular ints
                    # ============================================================
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Add detection to results
                    detection = {
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class_id': cls_id
                    }
                    detections.append(detection)

                    if verbose:
                        print(f"  Detected: {class_name} ({conf:.2f}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

        except Exception as e:
            print(f"[VisionCore] ✗ ERROR in object detection: {e}")

        return detections

    def recognize_faces(
        self,
        image: np.ndarray,
        return_unknown: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Recognize faces in an image using DeepFace.

        Face Recognition Pipeline:
        ==========================
        1. Face Detection: Find faces in image (uses MTCNN or RetinaFace)
        2. Face Alignment: Normalize rotation and scale
        3. Embedding Extraction: Run through CNN → 128D or 512D vector
        4. Database Search: Compare embedding with known faces
        5. Similarity Matching: Find closest match using distance metric

        Distance Metrics:
        - Cosine Similarity: Measures angle between vectors (0-1, lower = more similar)
        - Euclidean Distance: L2 distance between vectors (0-∞, lower = more similar)

        Thresholds (distance < threshold = same person):
        - Facenet: 0.40
        - Facenet512: 0.30
        - VGG-Face: 0.60
        - ArcFace: 0.68

        Args:
            image: Input image as numpy array (BGR format)
            return_unknown: Include unknown faces in results
                           If False, only returns recognized people

        Returns:
            List of face recognition results:
            {
                'name': 'John',           # Person name or "Unknown"
                'confidence': 0.85,       # Recognition confidence [0-1]
                'bbox': [x, y, w, h],    # Face bounding box (if available)
                'distance': 0.25          # Embedding distance (lower = better match)
            }
        """
        # ============================================================
        # LAZY INITIALIZATION
        # Only load DeepFace when first needed
        # ============================================================
        if not self._face_detection_enabled:
            self._init_deepface()

        if not self._face_detection_enabled:
            return []

        faces = []

        try:
            # ============================================================
            # CHECK IF DATABASE HAS KNOWN FACES
            # Database structure: face_database/person_name/image.jpg
            # ============================================================
            has_known_faces = any(self.face_db_path.glob("*/*"))
            detected_faces = self._extract_detected_faces(image, require_face=True)

            if not detected_faces:
                return []

            if has_known_faces:
                # ============================================================
                # FACE RECOGNITION MODE (with known faces)
                # DeepFace.find() searches for faces in database
                # ============================================================
                match_threshold = max(0.0, self._face_match_threshold())
                for detected_face in detected_faces:
                    face_crop = self._crop_face_image(image, detected_face["bbox"])
                    if face_crop is None or getattr(face_crop, "size", 0) == 0:
                        continue

                    find_kwargs = {
                        "img_path": face_crop,
                        "db_path": str(self.face_db_path),
                        "model_name": self.face_model,
                        "enforce_detection": False,
                        "detector_backend": self._face_detector_backend(),
                        "silent": True,
                        "refresh_database": True,
                    }
                    try:
                        results = self.deepface.find(**find_kwargs)
                    except TypeError:
                        find_kwargs.pop("refresh_database", None)
                        results = self.deepface.find(**find_kwargs)

                    best_match = None
                    result_frames = results if isinstance(results, list) else [results]
                    for df in result_frames:
                        if df is None or len(df) == 0:
                            continue
                        candidate = df.iloc[0]
                        if best_match is None or float(candidate["distance"]) < float(best_match["distance"]):
                            best_match = candidate

                    if best_match is not None:
                        distance = float(best_match["distance"])
                        if distance <= match_threshold:
                            identity = str(best_match["identity"])
                            person_name = Path(identity).parent.name
                            confidence = max(
                                0.0,
                                min(1.0, 1.0 - (distance / max(match_threshold, 1e-6))),
                            )
                            faces.append(
                                {
                                    "name": person_name,
                                    "confidence": confidence,
                                    "bbox": detected_face["bbox"],
                                    "distance": distance,
                                }
                            )
                            continue

                    if return_unknown:
                        faces.append(
                            {
                                "name": "Unknown",
                                "confidence": 0.0,
                                "bbox": detected_face["bbox"],
                            }
                        )
            else:
                # ============================================================
                # FACE DETECTION MODE (no known faces)
                # Just detect faces without recognition
                # ============================================================
                if return_unknown:
                    for detected_face in detected_faces:
                        faces.append(
                            {
                                "name": "Unknown",
                                "confidence": detected_face["confidence"],
                                "bbox": detected_face["bbox"],
                            }
                        )

        except Exception as e:
            print(f"[VisionCore] ✗ ERROR in face recognition: {e}")

        return faces

    def detect_all(
        self,
        image: np.ndarray,
        detect_faces: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Perform both object detection and face recognition in one call.

        This is the main entry point for complete scene understanding.

        Workflow:
        =========
        1. Run YOLO object detection → Get all objects in scene
        2. Run DeepFace face recognition → Identify people
        3. Generate human-readable summary

        Use Cases:
        ==========
        - "What's in front of me?" → List all detected objects
        - "Who is this?" → Recognize face and recall past interactions
        - "Find a person sitting on a chair" → Combine object + person detection
        - Autonomous navigation → Detect obstacles (objects) and people

        Args:
            image: Input image as numpy array (BGR format)
            detect_faces: Whether to perform face recognition
                         Set to False if you only need object detection (faster)
            verbose: Print detailed detection info for debugging

        Returns:
            Dictionary with complete scene analysis:
            {
                'objects': [
                    {'class_name': 'person', 'confidence': 0.87, 'bbox': [...]},
                    {'class_name': 'chair', 'confidence': 0.92, 'bbox': [...]}
                ],
                'faces': [
                    {'name': 'John', 'confidence': 0.85, 'bbox': [...]}
                ],
                'summary': 'Objects: 1 person(s), 1 chair(s) | Recognized: John'
            }
        """
        results = {
            'objects': [],
            'faces': [],
            'summary': ''
        }

        # ============================================================
        # STEP 1: OBJECT DETECTION
        # Detect all objects in the scene using YOLO
        # ============================================================
        if verbose:
            print("[VisionCore] Running object detection...")

        results['objects'] = self.detect_objects(image, verbose=verbose)

        if verbose:
            print(f"[VisionCore] Found {len(results['objects'])} objects")

        # ============================================================
        # STEP 2: FACE RECOGNITION (Optional)
        # Identify people in the scene
        # ============================================================
        if detect_faces:
            if verbose:
                print("[VisionCore] Running face recognition...")

            results['faces'] = self.recognize_faces(image)

            if verbose:
                print(f"[VisionCore] Found {len(results['faces'])} faces")

        # ============================================================
        # STEP 3: GENERATE SUMMARY
        # Create human-readable description of the scene
        # ============================================================
        results['summary'] = self._generate_summary(results)

        return results

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of detections.

        This creates a concise text description that can be:
        - Displayed on screen
        - Spoken through robot speaker (TTS)
        - Logged for analytics
        - Passed to LLM agents for reasoning

        Examples:
        =========
        - "Objects: 1 person(s), 2 chair(s) | Recognized: John"
        - "Objects: 1 dog(s) | Unknown faces: 1"
        - "No detections"

        Args:
            results: Detection results from detect_all()

        Returns:
            Summary string
        """
        summary_parts = []

        # ============================================================
        # SUMMARIZE OBJECTS
        # Count instances of each class
        # ============================================================
        if results['objects']:
            obj_counts = {}
            for obj in results['objects']:
                name = obj['class_name']
                obj_counts[name] = obj_counts.get(name, 0) + 1

            # Format: "1 person(s), 2 chair(s), 1 laptop(s)"
            obj_str = ', '.join([f"{count} {name}(s)" for name, count in obj_counts.items()])
            summary_parts.append(f"Objects: {obj_str}")

        # ============================================================
        # SUMMARIZE FACES
        # List recognized people and count unknowns
        # ============================================================
        if results['faces']:
            known_faces = [f['name'] for f in results['faces'] if f['name'] != 'Unknown']
            unknown_count = sum(1 for f in results['faces'] if f['name'] == 'Unknown')

            if known_faces:
                summary_parts.append(f"Recognized: {', '.join(known_faces)}")
            if unknown_count > 0:
                summary_parts.append(f"Unknown faces: {unknown_count}")

        return ' | '.join(summary_parts) if summary_parts else "No detections"

    def add_face_to_database(
        self,
        person_name: str,
        image: np.ndarray,
        image_filename: Optional[str] = None
    ) -> bool:
        """
        Add a person's face to the recognition database.

        Database Structure:
        ===================
        face_database/
        ├── John/
        │   ├── john_20260122_101530.jpg
        │   └── john_20260122_103045.jpg
        ├── Sarah/
        │   └── sarah_20260122_120000.jpg
        └── Alice/
            └── alice_20260122_143000.jpg

        Multiple images per person improve recognition accuracy:
        - Different angles (front, side, 3/4 view)
        - Different lighting conditions
        - Different expressions

        Best Practices:
        ===============
        - Use clear, well-lit photos
        - Face should be centered and clearly visible
        - Multiple images per person (3-5 recommended)
        - Consistent naming convention

        Args:
            person_name: Name of the person (used as directory name)
            image: Face image as numpy array (BGR format)
                   Can be full frame (will auto-detect face) or cropped face
            image_filename: Optional custom filename
                          Auto-generated if None (person_YYYYMMDD_HHMMSS.jpg)

        Returns:
            True if successful, False otherwise
        """
        self.last_face_db_error = None

        try:
            if image is None or getattr(image, "size", 0) == 0:
                self.last_face_db_error = "Could not read a usable image for face learning."
                print(f"[VisionCore] ✗ ERROR adding face to database: {self.last_face_db_error}")
                return False

            detected_faces = self._extract_detected_faces(image, require_face=True)
            if not detected_faces:
                self.last_face_db_error = (
                    "No face was detected in the latest image. Ask one person to stand clearly in front of the robot."
                )
                print(f"[VisionCore] ✗ ERROR adding face to database: {self.last_face_db_error}")
                return False

            if len(detected_faces) > 1:
                self.last_face_db_error = (
                    "Multiple faces were detected in the latest image. Ask only one person to stand in view when learning a name."
                )
                print(f"[VisionCore] ✗ ERROR adding face to database: {self.last_face_db_error}")
                return False

            face_image = self._crop_face_image(image, detected_faces[0]["bbox"])
            if face_image is None or getattr(face_image, "size", 0) == 0:
                self.last_face_db_error = "Could not isolate the detected face for saving."
                print(f"[VisionCore] ✗ ERROR adding face to database: {self.last_face_db_error}")
                return False

            # ============================================================
            # CREATE PERSON DIRECTORY
            # Each person gets their own subdirectory
            # ============================================================
            person_dir = self.face_db_path / person_name
            person_dir.mkdir(exist_ok=True)

            # ============================================================
            # GENERATE FILENAME
            # Format: personname_YYYYMMDD_HHMMSS.jpg
            # ============================================================
            if image_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"{person_name}_{timestamp}.jpg"

            # ============================================================
            # SAVE IMAGE
            # OpenCV's imwrite handles BGR format automatically
            # ============================================================
            image_path = person_dir / image_filename
            cv2.imwrite(str(image_path), face_image)
            self._invalidate_face_database_cache()

            print(f"[VisionCore] ✓ Added face for '{person_name}': {image_path}")
            return True

        except Exception as e:
            self.last_face_db_error = str(e)
            print(f"[VisionCore] ✗ ERROR adding face to database: {e}")
            return False

    def list_known_faces(self) -> List[str]:
        """
        List all people in the face database.

        Returns:
            List of person names (directory names in face_database/)
        """
        try:
            return [d.name for d in self.face_db_path.iterdir() if d.is_dir()]
        except Exception as e:
            print(f"[VisionCore] ✗ ERROR listing known faces: {e}")
            return []

    def visualize_detections(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on image for visualization.

        Drawing Style:
        ==============
        - Objects: Green boxes with class name + confidence
        - Faces: Blue boxes with person name + confidence
        - Labels: Black text on colored background

        Args:
            image: Input image as numpy array (BGR format)
            results: Detection results from detect_all()
            show_confidence: Include confidence scores in labels

        Returns:
            Annotated image (copy of original, original is not modified)
        """
        # ============================================================
        # CREATE COPY TO AVOID MODIFYING ORIGINAL
        # ============================================================
        annotated = image.copy()

        # ============================================================
        # DRAW OBJECT BOUNDING BOXES (Green)
        # ============================================================
        for obj in results['objects']:
            x1, y1, x2, y2 = obj['bbox']

            # Create label with class name and confidence
            if show_confidence:
                label = f"{obj['class_name']} {obj['confidence']:.2f}"
            else:
                label = obj['class_name']

            # Draw rectangle (green color)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label background (filled rectangle)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1  # Filled rectangle
            )

            # Draw label text (black on green background)
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                1
            )

        # ============================================================
        # DRAW FACE BOUNDING BOXES (Blue)
        # ============================================================
        for face in results['faces']:
            if face['bbox']:
                x, y, w, h = face['bbox']

                # Create label with name and confidence
                if show_confidence:
                    label = f"{face['name']} {face['confidence']:.2f}"
                else:
                    label = face['name']

                # Draw rectangle (blue color)
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Draw label (white text on blue background)
                cv2.putText(
                    annotated,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

        return annotated


# ============================================================
# EXAMPLE USAGE AND TESTING
# Run this file directly to test VisionCore
# ============================================================
if __name__ == "__main__":
    """
    Example usage of VisionCore class with interactive webcam demo.

    This demonstrates:
    1. Initialization with optimized parameters
    2. Real-time object detection and face recognition
    3. Interactive face database management
    4. Performance monitoring
    """
    import sys

    print("=" * 60)
    print("VisionCore - Interactive Demo")
    print("=" * 60)

    # ============================================================
    # INITIALIZATION
    # Configure for your use case (Jetson vs Desktop)
    # ============================================================

    jetson_detected = _is_jetson_platform()
    use_jetson_config = _env_flag("VISION_USE_JETSON_CONFIG", default=False)

    if jetson_detected and not use_jetson_config:
        print("\n[Config] Jetson detected. Using safe CPU defaults.")
        print("[Config] Set VISION_USE_JETSON_CONFIG=1 to enable TensorRT-optimized inference.")

    vision_settings = get_default_vision_core_settings(
        yolo_model="yolov8n.pt",
        face_model="Facenet",
        face_db_path="./face_database",
    )

    if use_jetson_config:
        print("\n[Config] Using Jetson Orin optimized settings")
    else:
        # For desktop/laptop (no TensorRT):
        print("\n[Config] Using desktop/laptop CPU-optimized settings")
        print("[Config] Input size: 256x256 (very fast, good for nearby objects)")
        print("[Config] Will use ONNX if available (2-4x faster than PyTorch)")

    vision = VisionCore(**vision_settings)

    print("\n" + "=" * 60)
    print("Test 1: Object Detection from Webcam")
    print("=" * 60)

    # ============================================================
    # WEBCAM CAPTURE
    # Use default webcam resolution (driver-optimized)
    # ============================================================
    requested_camera_source = (
        os.environ.get("VISION_CAMERA_SOURCE")
        or os.environ.get("GO2_CAMERA_SOURCE")
    )
    cap, camera_info = open_camera(
        camera_source=requested_camera_source,
        verbose=True,
    )

    if cap is None:
        print("WARNING: Could not open any camera. Using test image instead.")
        if requested_camera_source:
            print(f"[Camera] Requested source override: {requested_camera_source}")
        for attempt in camera_info.get("attempts", []):
            print(f"  - {attempt}")
        print("[Camera] Hint: set VISION_CAMERA_SOURCE to an explicit source, e.g. 'unitree:eth0', 'jetson:0', '/dev/video2', or a full GStreamer pipeline.")

        # Create a test image with text
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            test_image,
            "No webcam available",
            (150, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        print("\nDetecting objects in test image...")
        results = vision.detect_all(test_image, detect_faces=True)
        print(f"Summary: {results['summary']}")
        print(f"Objects detected: {len(results['objects'])}")
        print(f"Faces detected: {len(results['faces'])}")

    else:
        actual_width = camera_info["width"]
        actual_height = camera_info["height"]
        print(
            f"[Camera] Using {camera_info['description']} via {camera_info['backend']} "
            f"at {actual_width}x{actual_height}"
        )
        gui_available = _opencv_gui_available()

        print("\nWebcam opened successfully!")

        if not gui_available:
            print("\n[Headless] No display detected; running a non-interactive smoke test.")
            _run_headless_camera_demo(
                cap,
                vision,
                enable_face_recognition=False,
            )
            cap.release()
        else:
            print("\n" + "!" * 60)
            print("IMPORTANT: Click on the OpenCV window to activate it!")
            print("Then press keys while the window is in focus:")
            print("  Q - Quit")
            print("  S - Save detection")
            print("  A - Add face to database")
            print("  V - Toggle verbose mode")
            print("  F - Toggle face recognition (turn OFF for speed)")
            print("!" * 60 + "\n")

            frame_count = 0
            current_frame = None
            current_results = None
            current_annotated = None
            verbose_mode = False

            # Performance monitoring
            fps_start_time = time.time()
            fps_frame_count = 0
            fps = 0.0

            # Detection timing
            last_detection_time = time.time()
            detection_interval = 0.0  # Will store actual detection time

            # ============================================================
            # PERFORMANCE OPTIMIZATION: Run face recognition less frequently
            # Face recognition is 10x slower than object detection
            # Only run it every N object detections
            # ============================================================
            face_recognition_interval = 10  # Run face recognition every 10 object detections
            detection_count = 0

            # Create named window
            window_name = 'VisionCore - Click here and press Q/S/A/V'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                current_frame = frame

                # ============================================================
                # PROCESS EVERY 3RD FRAME (balance speed and smoothness)
                # Adjust this number based on your hardware:
                # - MacBook/Desktop CPU with ONNX: every 2-3 frames (faster!)
                # - MacBook/Desktop CPU with PyTorch: every 5-10 frames (slow)
                # - Jetson with TensorRT: every 1 frame (can handle real-time)
                # This processes ~10 detections per second
                # ============================================================
                if frame_count % 3 == 0:
                    detection_start = time.time()

                    # ============================================================
                    # DOWNSCALE FRAME BEFORE PROCESSING (CPU optimization)
                    # This is MUCH better than changing webcam resolution
                    # Captures at native resolution, then downscales for processing
                    # Smaller = faster inference, but detection quality suffers
                    # ============================================================
                    process_frame = cv2.resize(frame, (320, 240))  # Aggressive downscale
                    # process_frame = cv2.resize(frame, (480, 360))  # Moderate downscale
                    # process_frame = frame  # Use full resolution (slowest)

                    # Calculate scale factors for bbox correction
                    orig_h, orig_w = frame.shape[:2]
                    proc_h, proc_w = process_frame.shape[:2]
                    scale_x = orig_w / proc_w
                    scale_y = orig_h / proc_h

                    # ============================================================
                    # SMART FACE RECOGNITION
                    # Only run face recognition every Nth detection to save time
                    # Object detection: ~30-50ms
                    # Face recognition: ~200-500ms (10x slower!)
                    # ============================================================
                    run_face_recognition = False
                    if face_recognition_interval > 0:
                        run_face_recognition = (detection_count % face_recognition_interval == 0)

                    if verbose_mode and run_face_recognition:
                        print(f"[Frame {frame_count}] Running face recognition...")

                    # Run detection on downscaled frame
                    current_results = vision.detect_all(
                        process_frame,
                        detect_faces=run_face_recognition,  # Only run faces periodically
                        verbose=verbose_mode
                    )

                    # ============================================================
                    # SCALE BOUNDING BOXES BACK TO ORIGINAL FRAME SIZE
                    # Detections are on downscaled frame, need to scale back up
                    # ============================================================
                    for obj in current_results['objects']:
                        obj['bbox'] = [
                            int(obj['bbox'][0] * scale_x),  # x1
                            int(obj['bbox'][1] * scale_y),  # y1
                            int(obj['bbox'][2] * scale_x),  # x2
                            int(obj['bbox'][3] * scale_y)   # y2
                        ]

                    for face in current_results['faces']:
                        if face['bbox']:
                            face['bbox'] = [
                                int(face['bbox'][0] * scale_x),  # x
                                int(face['bbox'][1] * scale_y),  # y
                                int(face['bbox'][2] * scale_x),  # w
                                int(face['bbox'][3] * scale_y)   # h
                            ]

                    detection_count += 1
                    detection_interval = time.time() - detection_start

                    # Visualize results on ORIGINAL frame (with scaled boxes)
                    current_annotated = vision.visualize_detections(frame, current_results)

                    # Display summary on image
                    cv2.putText(
                        current_annotated,
                        current_results['summary'],
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    # Calculate FPS
                    fps_frame_count += 1
                    if time.time() - fps_start_time >= 1.0:
                        fps = fps_frame_count / (time.time() - fps_start_time)
                        fps_start_time = time.time()
                        fps_frame_count = 0

                # ============================================================
                # ADD UI OVERLAY
                # ============================================================
                display_frame = current_annotated if current_annotated is not None else frame
                h, w = display_frame.shape[:2]

                # Semi-transparent black bar at bottom
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, h-80), (w, h), (0, 0, 0), -1)
                display_frame = cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0)

                # Instruction text
                cv2.putText(display_frame, "Q:Quit | S:Save | A:Add Face | V:Verbose | F:Toggle Faces",
                           (10, h-55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(display_frame, "Click this window first, then press keys!",
                           (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Performance stats
                face_status = f"ON (every {face_recognition_interval})" if face_recognition_interval > 0 else "OFF"
                backend_display = vision.backend if hasattr(vision, 'backend') else "Unknown"
                cv2.putText(display_frame, f"FPS: {fps:.1f} | {backend_display} | {detection_interval*1000:.0f}ms | Faces: {face_status}",
                           (10, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # Show frame
                cv2.imshow(window_name, display_frame)

                frame_count += 1

                # ============================================================
                # KEYBOARD INPUT HANDLING
                # ============================================================
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    print("\n[KEY DETECTED] Quitting...")
                    break

                elif key == ord('s') or key == ord('S'):
                    print("\n[KEY DETECTED] Save requested...")
                    if current_frame is not None and current_results is not None:
                        filename = _save_frame_snapshot(current_frame, current_results, vision)
                        print(f"✓ Saved detection to: {filename}")
                        print(f"  Summary: {current_results['summary']}")
                        print(f"  Objects: {len(current_results['objects'])}")
                        print(f"  Faces: {len(current_results['faces'])}")
                    else:
                        print("✗ No detection to save yet. Wait for processing...")

                elif key == ord('a') or key == ord('A'):
                    print("\n[KEY DETECTED] Add face requested...")
                    print("\n" + "="*40)
                    person_name = input("Enter person's name: ").strip()

                    if person_name:
                        # Extract face region if detected
                        if current_results and current_results['faces']:
                            face = current_results['faces'][0]
                            if face['bbox']:
                                x, y, w, h = face['bbox']
                                # Add padding around face
                                padding = 20
                                x = max(0, x - padding)
                                y = max(0, y - padding)
                                w = w + 2 * padding
                                h = h + 2 * padding

                                face_img = current_frame[y:y+h, x:x+w]
                                success = vision.add_face_to_database(person_name, face_img)
                            else:
                                success = vision.add_face_to_database(person_name, current_frame)
                        else:
                            print("No face detected. Using full frame...")
                            success = vision.add_face_to_database(person_name, current_frame)

                        if success:
                            print(f"✓ Successfully added '{person_name}' to database!")
                        else:
                            print("✗ Failed to add face to database")
                        print("="*40 + "\n")

                elif key == ord('v') or key == ord('V'):
                    verbose_mode = not verbose_mode
                    print(f"\n[KEY DETECTED] Verbose mode: {'ON' if verbose_mode else 'OFF'}")

                elif key == ord('f') or key == ord('F'):
                    # Toggle face recognition
                    if face_recognition_interval == 0:
                        face_recognition_interval = 10
                        print(f"\n[KEY DETECTED] Face recognition: ON (every {face_recognition_interval} frames)")
                    else:
                        face_recognition_interval = 0
                        print(f"\n[KEY DETECTED] Face recognition: OFF (⚡ maximum speed)")

            cap.release()
            cv2.destroyAllWindows()

    # ============================================================
    # FINAL STATISTICS
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 2: List Known Faces")
    print("=" * 60)

    known_faces = vision.list_known_faces()
    if known_faces:
        print(f"Known faces in database: {', '.join(known_faces)}")
    else:
        print("No faces in database yet.")

    print("\n" + "=" * 60)
    print("Test 3: Available Object Classes")
    print("=" * 60)

    if vision.yolo:
        print(f"Total classes: {len(vision.class_names)}")
        print("Sample classes:", ', '.join(list(vision.class_names.values())[:20]))

    print("\n" + "=" * 60)
    print("VisionCore Demo Complete!")
    print("=" * 60)
