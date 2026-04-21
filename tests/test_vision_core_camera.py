import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from coded_tools.unigo2 import vision_core


class _FakeCapture:
    def __init__(self, *, opened, frames=None, width=640, height=480):
        self._opened = opened
        self._frames = list(frames or [])
        self._width = width
        self._height = height
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if self._frames:
            return self._frames.pop(0)
        return False, None

    def release(self):
        self.released = True

    def get(self, prop_id):
        if prop_id == vision_core.cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        if prop_id == vision_core.cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        return 0

    def set(self, prop_id, value):
        if prop_id == vision_core.cv2.CAP_PROP_FRAME_WIDTH:
            self._width = value
        elif prop_id == vision_core.cv2.CAP_PROP_FRAME_HEIGHT:
            self._height = value
        return True


class _FakeVision:
    def __init__(self):
        self.frames = []
        self.face_frames = []

    def detect_all(self, frame, detect_faces=False, verbose=False):
        self.frames.append(frame.shape[:2])
        return {
            "objects": [
                {"class_name": "person", "confidence": 0.91, "bbox": [10, 5, 30, 25]},
            ],
            "faces": [],
            "summary": "Objects: 1 person(s)",
        }

    def recognize_faces(self, frame):
        self.face_frames.append(frame.shape[:2])
        return [
            {"name": "Alice", "confidence": 0.91, "bbox": [1, 2, 3, 4]},
        ]

    def _generate_summary(self, results):
        return (
            f"Objects: {len(results['objects'])} person(s) | "
            f"Recognized: {', '.join(face['name'] for face in results['faces'])}"
        )

    def visualize_detections(self, frame, results):
        return frame.copy()


class _FakeILoc:
    def __init__(self, row):
        self._row = row

    def __getitem__(self, idx):
        del idx
        return self._row


class _FakeDataFrame:
    def __init__(self, row):
        self.iloc = _FakeILoc(row)

    def __len__(self):
        return 1


class _FakeDeepFace:
    def __init__(self, result=None, extracted_faces=None, extract_raises=None):
        self._result = result
        self._extracted_faces = list(extracted_faces or [])
        self._extract_raises = extract_raises
        self.find_kwargs = None
        self.extract_kwargs = None
        self.find_calls = 0

    def find(self, **kwargs):
        self.find_kwargs = kwargs
        self.find_calls += 1
        return self._result

    def extract_faces(self, **kwargs):
        self.extract_kwargs = kwargs
        if self._extract_raises is not None:
            raise self._extract_raises
        return list(self._extracted_faces)


class VisionCoreCameraTests(unittest.TestCase):
    def test_vision_core_can_skip_yolo_initialization_for_face_db_updates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(vision_core.VisionCore, "_init_yolo") as init_yolo:
                vision = vision_core.VisionCore(
                    face_db_path=temp_dir,
                    initialize_yolo=False,
                )

        init_yolo.assert_not_called()
        self.assertEqual(vision.backend, "Disabled")
        self.assertIsNone(vision.yolo)

    def test_default_vision_core_settings_match_standalone_cpu_defaults(self):
        with patch.object(vision_core, "_env_flag", return_value=False):
            settings = vision_core.get_default_vision_core_settings()

        self.assertEqual(
            settings,
            {
                "yolo_model": "yolov8n.pt",
                "face_model": "Facenet",
                "face_db_path": "./face_database",
                "use_tensorrt": False,
                "confidence_threshold": 0.60,
                "iou_threshold": 0.45,
                "input_size": 256,
                "half_precision": False,
            },
        )

    def test_normalize_numeric_camera_source(self):
        candidate = vision_core._normalize_camera_source("3")

        self.assertEqual(candidate["source"], 3)
        self.assertEqual(candidate["description"], "camera index 3")
        self.assertIsNone(candidate["backend"])

    def test_normalize_jetson_camera_source(self):
        candidate = vision_core._normalize_camera_source("jetson:1")

        self.assertIn("sensor-id=1", candidate["source"])
        self.assertEqual(candidate["description"], "Jetson CSI sensor 1")
        self.assertEqual(candidate["backend"], getattr(vision_core.cv2, "CAP_GSTREAMER", None))
        self.assertEqual(candidate["kind"], "opencv")

    def test_normalize_unitree_camera_source(self):
        candidate = vision_core._normalize_camera_source("unitree:eth0")

        self.assertEqual(candidate["kind"], "unitree")
        self.assertEqual(candidate["ifname"], "eth0")
        self.assertIn("Unitree Go2 front camera", candidate["description"])

    @patch.object(vision_core, "_discover_v4l2_devices", return_value=["/dev/video2", "/dev/video4"])
    @patch.object(vision_core, "_unitree_camera_interface", return_value=None)
    @patch.object(vision_core, "_unitree_camera_available", return_value=True)
    @patch.object(vision_core, "_is_jetson_platform", return_value=True)
    def test_default_candidates_cover_jetson_v4l2_and_index_fallbacks(self, *_):
        candidates = vision_core.get_camera_candidates(max_indices=3)
        descriptions = [candidate["description"] for candidate in candidates]

        self.assertEqual(descriptions[0], "Unitree Go2 front camera")
        self.assertEqual(descriptions[1:3], ["Jetson CSI sensor 0", "Jetson CSI sensor 1"])
        self.assertIn("/dev/video2", descriptions)
        self.assertIn("/dev/video4", descriptions)
        self.assertIn("camera index 0", descriptions)
        self.assertIn("camera index 2", descriptions)

    def test_open_camera_falls_back_until_frame_is_available(self):
        first = _FakeCapture(opened=False)
        second = _FakeCapture(opened=True, frames=[(False, None)])
        third = _FakeCapture(
            opened=True,
            frames=[(True, np.ones((2, 2, 3), dtype=np.uint8))],
            width=800,
            height=600,
        )

        candidates = [
            {"source": 0, "backend": None, "description": "camera index 0"},
            {"source": 1, "backend": None, "description": "camera index 1"},
            {"source": 2, "backend": None, "description": "camera index 2"},
        ]

        with patch.object(vision_core, "get_camera_candidates", return_value=candidates):
            with patch.object(vision_core.cv2, "VideoCapture", side_effect=[first, second, third]):
                capture, info = vision_core.open_camera(verbose=False, warmup_reads=1)

        self.assertIs(capture, third)
        self.assertEqual(info["description"], "camera index 2")
        self.assertEqual(info["width"], 800)
        self.assertEqual(info["height"], 600)
        self.assertTrue(first.released)
        self.assertTrue(second.released)

    def test_open_camera_uses_unitree_capture_for_unitree_candidates(self):
        fake_capture = _FakeCapture(
            opened=True,
            frames=[(True, np.ones((4, 5, 3), dtype=np.uint8))],
            width=5,
            height=4,
        )
        candidates = [{
            "kind": "unitree",
            "source": "unitree",
            "backend": None,
            "description": "Unitree Go2 front camera via eth0",
            "ifname": "eth0",
        }]

        with patch.object(vision_core, "get_camera_candidates", return_value=candidates):
            with patch.object(vision_core, "UnitreeVideoCapture", return_value=fake_capture) as capture_cls:
                capture, info = vision_core.open_camera(verbose=False, warmup_reads=1)

        capture_cls.assert_called_once_with(ifname="eth0")
        self.assertIs(capture, fake_capture)
        self.assertEqual(info["description"], "Unitree Go2 front camera via eth0")
        self.assertEqual(info["backend"], "Unitree SDK2")

    def test_detect_camera_snapshot_reuses_headless_detection_flow(self):
        cap = _FakeCapture(
            opened=True,
            frames=[
                (True, np.zeros((240, 320, 3), dtype=np.uint8)),
                (True, np.ones((240, 320, 3), dtype=np.uint8)),
            ],
            width=320,
            height=240,
        )
        vision = _FakeVision()

        snapshot = vision_core.detect_camera_snapshot(
            cap,
            vision,
            warmup_frames=2,
            process_size=(160, 120),
        )

        self.assertEqual(vision.frames, [(120, 160)])
        self.assertEqual(snapshot["results"]["summary"], "Objects: 1 person(s)")
        self.assertEqual(snapshot["results"]["objects"][0]["bbox"], [20, 10, 60, 50])
        self.assertEqual(snapshot["annotated"].shape, (240, 320, 3))

    def test_detect_camera_snapshot_uses_larger_default_process_size_with_face_recognition(self):
        cap = _FakeCapture(
            opened=True,
            frames=[
                (True, np.zeros((1080, 1920, 3), dtype=np.uint8)),
                (True, np.ones((1080, 1920, 3), dtype=np.uint8)),
            ],
            width=1920,
            height=1080,
        )
        vision = _FakeVision()

        snapshot = vision_core.detect_camera_snapshot(
            cap,
            vision,
            enable_face_recognition=True,
            warmup_frames=2,
        )

        self.assertEqual(vision.frames, [(480, 640)])
        self.assertEqual(vision.face_frames, [(1080, 1920)])
        self.assertEqual(
            snapshot["results"]["summary"],
            "Objects: 1 person(s) | Recognized: Alice",
        )

    def test_recognize_faces_handles_single_deepface_dataframe_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_image = Path(temp_dir) / "Alice" / "alice.jpg"
            db_image.parent.mkdir(parents=True, exist_ok=True)
            db_image.write_bytes(b"fake")

            vision = vision_core.VisionCore(
                face_db_path=temp_dir,
                initialize_yolo=False,
            )
            vision._face_detection_enabled = True
            vision.deepface = _FakeDeepFace(
                result=_FakeDataFrame(
                    {
                        "identity": str(db_image),
                        "distance": 0.18,
                    }
                ),
                extracted_faces=[
                    {
                        "facial_area": {"x": 11, "y": 22, "w": 33, "h": 44},
                        "confidence": 0.99,
                    }
                ],
            )

            faces = vision.recognize_faces(np.zeros((96, 96, 3), dtype=np.uint8))

        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]["name"], "Alice")
        self.assertEqual(faces[0]["bbox"], [11, 22, 33, 44])
        self.assertTrue(vision.deepface.find_kwargs["refresh_database"])

    def test_recognize_faces_does_not_guess_when_no_face_is_detected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_image = Path(temp_dir) / "Alice" / "alice.jpg"
            db_image.parent.mkdir(parents=True, exist_ok=True)
            db_image.write_bytes(b"fake")

            vision = vision_core.VisionCore(
                face_db_path=temp_dir,
                initialize_yolo=False,
            )
            vision._face_detection_enabled = True
            vision.deepface = _FakeDeepFace(
                result=_FakeDataFrame(
                    {
                        "identity": str(db_image),
                        "distance": 0.12,
                    }
                ),
                extracted_faces=[],
            )

            faces = vision.recognize_faces(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual(faces, [])
        self.assertEqual(vision.deepface.find_calls, 0)

    def test_recognize_faces_rejects_weak_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_image = Path(temp_dir) / "Alice" / "alice.jpg"
            db_image.parent.mkdir(parents=True, exist_ok=True)
            db_image.write_bytes(b"fake")

            vision = vision_core.VisionCore(
                face_db_path=temp_dir,
                initialize_yolo=False,
            )
            vision._face_detection_enabled = True
            vision.deepface = _FakeDeepFace(
                result=_FakeDataFrame(
                    {
                        "identity": str(db_image),
                        "distance": 0.95,
                    }
                ),
                extracted_faces=[
                    {
                        "facial_area": {"x": 5, "y": 6, "w": 7, "h": 8},
                        "confidence": 0.98,
                    }
                ],
            )

            faces = vision.recognize_faces(np.zeros((16, 16, 3), dtype=np.uint8))

        self.assertEqual(
            faces,
            [
                {
                    "name": "Unknown",
                    "confidence": 0.0,
                    "bbox": [5, 6, 7, 8],
                }
            ],
        )

    def test_add_face_to_database_crops_single_detected_face_and_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "representations_facenet.pkl"
            cache_file.write_bytes(b"cache")

            vision = vision_core.VisionCore(
                face_db_path=temp_dir,
                initialize_yolo=False,
            )
            vision._face_detection_enabled = True
            vision.deepface = _FakeDeepFace(
                extracted_faces=[
                    {
                        "facial_area": {"x": 2, "y": 1, "w": 4, "h": 5},
                        "confidence": 0.99,
                    }
                ],
            )

            image = np.zeros((10, 10, 3), dtype=np.uint8)
            image[1:6, 2:6] = 255

            with patch.dict(os.environ, {"VISION_FACE_CROP_MARGIN": "0"}, clear=False):
                added = vision.add_face_to_database("Alice", image, "alice.jpg")

            saved_image_path = Path(temp_dir) / "Alice" / "alice.jpg"
            saved_image = cv2.imread(str(saved_image_path))

        self.assertTrue(added)
        self.assertFalse(cache_file.exists())
        self.assertIsNotNone(saved_image)
        self.assertEqual(saved_image.shape[:2], (5, 4))

    def test_add_face_to_database_rejects_multiple_faces(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vision = vision_core.VisionCore(
                face_db_path=temp_dir,
                initialize_yolo=False,
            )
            vision._face_detection_enabled = True
            vision.deepface = _FakeDeepFace(
                extracted_faces=[
                    {
                        "facial_area": {"x": 1, "y": 1, "w": 3, "h": 3},
                        "confidence": 0.95,
                    },
                    {
                        "facial_area": {"x": 5, "y": 5, "w": 3, "h": 3},
                        "confidence": 0.94,
                    },
                ],
            )

            added = vision.add_face_to_database(
                "Alice",
                np.zeros((12, 12, 3), dtype=np.uint8),
                "alice.jpg",
            )
            saved_images = list((Path(temp_dir) / "Alice").glob("*.jpg"))

        self.assertFalse(added)
        self.assertEqual(saved_images, [])
        self.assertIn("Multiple faces were detected", vision.last_face_db_error)


if __name__ == "__main__":
    unittest.main()
