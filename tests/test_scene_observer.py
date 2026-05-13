import unittest
from unittest.mock import patch

import numpy as np

from apps.conscious_assistant.scene_observer import SceneObserver
from apps.conscious_assistant.scene_observer import REPO_ROOT
from apps.conscious_assistant.scene_observer import _resolve_repo_relative_path
from apps.conscious_assistant.scene_observer import build_scene_input
from apps.conscious_assistant.scene_observer import observation_signature
from apps.conscious_assistant.scene_observer import summarize_observed_entities
from apps.conscious_assistant.scene_observer import summarize_observed_objects


class _VisionCtorResult:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.yolo = object()


class SceneObserverTests(unittest.TestCase):
    def test_resolve_repo_relative_path_uses_repo_copy_when_present(self):
        resolved = _resolve_repo_relative_path("yolov8n.pt")

        self.assertEqual(resolved, str(REPO_ROOT / "yolov8n.pt"))

    def test_summarize_observed_objects_deduplicates_and_sorts(self):
        objects = [
            {"class_name": "chair", "confidence": 0.33},
            {"class_name": "person", "confidence": 0.95},
            {"class_name": "chair", "confidence": 0.72},
        ]

        self.assertEqual(
            summarize_observed_objects(objects),
            ["person", "chair"],
        )

    def test_build_scene_input_uses_saw_lines(self):
        payload = build_scene_input("[04:20:00pm]", ["person", "chair"])

        self.assertEqual(
            payload,
            "\n[04:20:00pm] user: [Silence]\n[04:20:00pm] saw: person\n[04:20:00pm] saw: chair",
        )

    def test_observation_signature_returns_stable_object_tuple(self):
        signature = observation_signature(
            {
                "objects": ["Alice", "chair", "Alice", " "],
                "summary": "Objects: 1 chair(s) | Recognized: Alice",
            }
        )

        self.assertEqual(signature, ("Alice", "chair", "Alice"))

    def test_observation_signature_handles_missing_observation(self):
        self.assertEqual(observation_signature(None), ())

    def test_summarize_observed_entities_prefers_known_face_names_over_person(self):
        results = {
            "objects": [
                {"class_name": "person", "confidence": 0.95},
                {"class_name": "chair", "confidence": 0.60},
            ],
            "faces": [
                {"name": "Alice", "confidence": 0.88, "bbox": [1, 2, 3, 4]},
            ],
        }

        self.assertEqual(
            summarize_observed_entities(results),
            ["Alice", "chair"],
        )

    def test_ensure_vision_uses_shared_defaults_helper(self):
        observer = SceneObserver(enabled=True)
        captured_kwargs = {}

        def fake_defaults(**kwargs):
            captured_kwargs.update(kwargs)
            return {
                "yolo_model": kwargs["yolo_model"],
                "face_model": kwargs["face_model"],
                "face_db_path": kwargs["face_db_path"],
                "use_tensorrt": False,
                "confidence_threshold": 0.60,
                "iou_threshold": 0.45,
                "input_size": 256,
                "half_precision": False,
            }

        with (
            patch("apps.conscious_assistant.scene_observer.cv2", object()),
            patch("apps.conscious_assistant.scene_observer.detect_camera_snapshot", object()),
            patch("apps.conscious_assistant.scene_observer.open_camera", object()),
            patch("apps.conscious_assistant.scene_observer.get_default_vision_core_settings", side_effect=fake_defaults),
            patch("apps.conscious_assistant.scene_observer.VisionCore", side_effect=lambda **kwargs: _VisionCtorResult(**kwargs)) as ctor,
        ):
            vision = observer._ensure_vision()

        self.assertIsNotNone(vision)
        self.assertEqual(
            captured_kwargs,
            {
                "yolo_model": str(REPO_ROOT / "yolov8n.pt"),
                "face_model": "Facenet",
                "face_db_path": str(REPO_ROOT / "face_database"),
            },
        )
        ctor.assert_called_once()

    def test_observe_returns_latest_image_payload_from_shared_snapshot_helper(self):
        observer = SceneObserver(public_image_url="/api/observation/latest.jpg", enabled=True)
        frame = np.ones((16, 16, 3), dtype=np.uint8)
        vision = object()
        results = {
            "objects": [
                {"class_name": "chair", "confidence": 0.40, "bbox": [1, 1, 5, 5]},
                {"class_name": "person", "confidence": 0.91, "bbox": [0, 0, 10, 10]},
                {"class_name": "chair", "confidence": 0.70, "bbox": [2, 2, 6, 6]},
            ],
            "faces": [
                {"name": "Alice", "confidence": 0.82, "bbox": [0, 0, 4, 4]},
            ],
            "summary": "Objects: 1 person(s), 1 chair(s) | Recognized: Alice",
        }

        observer._capture = object()

        with patch.object(observer, "_ensure_vision", return_value=vision):
            with patch.object(observer, "_ensure_camera", return_value=True):
                with patch(
                    "apps.conscious_assistant.scene_observer.detect_camera_snapshot",
                    return_value={"frame": frame, "results": results, "annotated": frame},
                ) as snapshot_helper:
                    with patch.object(observer, "_write_latest_image", return_value=123456789):
                        observation = observer.observe()

        snapshot_helper.assert_called_once_with(
            observer._capture,
            vision,
            enable_face_recognition=True,
        )
        self.assertEqual(observation["objects"], ["Alice", "chair"])
        self.assertEqual(observation["summary"], "Objects: 1 person(s), 1 chair(s) | Recognized: Alice")
        self.assertEqual(observation["faces"], results["faces"])
        self.assertEqual(
            observation["image_url"],
            "/api/observation/latest.jpg?t=123456789",
        )
        self.assertTrue(observation["timestamp"].startswith("["))

    def test_observe_releases_capture_when_shared_snapshot_fails(self):
        observer = SceneObserver(enabled=True)

        class _FakeCapture:
            def __init__(self):
                self.released = False

            def release(self):
                self.released = True

        fake_capture = _FakeCapture()
        observer._capture = fake_capture

        with patch.object(observer, "_ensure_vision", return_value=object()):
            with patch.object(observer, "_ensure_camera", return_value=True):
                with patch("apps.conscious_assistant.scene_observer.detect_camera_snapshot", return_value=None):
                    with self.assertLogs(level="WARNING"):
                        observation = observer.observe()

        self.assertIsNone(observation)
        self.assertTrue(fake_capture.released)
        self.assertIsNone(observer._capture)

    def test_initialize_eagerly_loads_vision_backend(self):
        observer = SceneObserver(enabled=True)

        with patch.object(observer, "_ensure_vision", return_value=object()) as ensure_vision:
            with patch.object(observer, "_ensure_camera") as ensure_camera:
                initialized = observer.initialize()

        self.assertTrue(initialized)
        ensure_vision.assert_called_once()
        ensure_camera.assert_not_called()


if __name__ == "__main__":
    unittest.main()
