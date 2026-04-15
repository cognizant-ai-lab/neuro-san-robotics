import unittest
from unittest.mock import patch

import numpy as np

from apps.conscious_assistant.scene_observer import SceneObserver
from apps.conscious_assistant.scene_observer import REPO_ROOT
from apps.conscious_assistant.scene_observer import _resolve_repo_relative_path
from apps.conscious_assistant.scene_observer import build_scene_input
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

    def test_ensure_vision_uses_shared_defaults_helper(self):
        observer = SceneObserver()
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

        with patch("apps.conscious_assistant.scene_observer.get_default_vision_core_settings", side_effect=fake_defaults):
            with patch("apps.conscious_assistant.scene_observer.VisionCore", side_effect=lambda **kwargs: _VisionCtorResult(**kwargs)) as ctor:
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
        observer = SceneObserver(public_image_url="/api/observation/latest.jpg")
        frame = np.ones((16, 16, 3), dtype=np.uint8)
        results = {
            "objects": [
                {"class_name": "chair", "confidence": 0.40, "bbox": [1, 1, 5, 5]},
                {"class_name": "person", "confidence": 0.91, "bbox": [0, 0, 10, 10]},
                {"class_name": "chair", "confidence": 0.70, "bbox": [2, 2, 6, 6]},
            ],
            "faces": [],
            "summary": "Objects: 1 person(s), 1 chair(s)",
        }

        observer._capture = object()

        with patch.object(observer, "_ensure_vision", return_value=object()):
            with patch.object(observer, "_ensure_camera", return_value=True):
                with patch(
                    "apps.conscious_assistant.scene_observer.detect_camera_snapshot",
                    return_value={"frame": frame, "results": results, "annotated": frame},
                ) as snapshot_helper:
                    with patch.object(observer, "_write_latest_image", return_value=123456789):
                        observation = observer.observe()

        snapshot_helper.assert_called_once()
        self.assertEqual(observation["objects"], ["person", "chair"])
        self.assertEqual(observation["summary"], "Objects: 1 person(s), 1 chair(s)")
        self.assertEqual(
            observation["image_url"],
            "/api/observation/latest.jpg?t=123456789",
        )
        self.assertTrue(observation["timestamp"].startswith("["))

    def test_observe_releases_capture_when_shared_snapshot_fails(self):
        observer = SceneObserver()

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


if __name__ == "__main__":
    unittest.main()
