import unittest
from unittest.mock import patch

import numpy as np

from apps.conscious_assistant.scene_observer import SceneObserver
from apps.conscious_assistant.scene_observer import REPO_ROOT
from apps.conscious_assistant.scene_observer import _default_observer_vision_settings
from apps.conscious_assistant.scene_observer import _resolve_repo_relative_path
from apps.conscious_assistant.scene_observer import build_scene_input
from apps.conscious_assistant.scene_observer import summarize_observed_objects


class _FakeVision:
    def detect_all(self, frame, detect_faces=False, verbose=False):
        return {
            "objects": [
                {"class_name": "chair", "confidence": 0.40, "bbox": [1, 1, 5, 5]},
                {"class_name": "person", "confidence": 0.91, "bbox": [0, 0, 10, 10]},
                {"class_name": "chair", "confidence": 0.70, "bbox": [2, 2, 6, 6]},
            ],
            "faces": [],
            "summary": "Objects: 1 person(s), 1 chair(s)",
        }

    def visualize_detections(self, frame, results):
        return frame.copy()


class _FakeScaledVision:
    def detect_all(self, frame, detect_faces=False, verbose=False):
        return {
            "objects": [
                {"class_name": "person", "confidence": 0.90, "bbox": [10, 5, 30, 25]},
            ],
            "faces": [],
            "summary": "Objects: 1 person(s)",
        }

    def visualize_detections(self, frame, results):
        return frame.copy()


class _FallbackVision:
    def __init__(self):
        self.sizes = []

    def detect_all(self, frame, detect_faces=False, verbose=False):
        self.sizes.append((frame.shape[1], frame.shape[0]))
        if frame.shape[1] <= 320:
            return {
                "objects": [],
                "faces": [],
                "summary": "No detections",
            }
        return {
            "objects": [
                {"class_name": "person", "confidence": 0.88, "bbox": [16, 8, 32, 24]},
            ],
            "faces": [],
            "summary": "Objects: 1 person(s)",
        }

    def visualize_detections(self, frame, results):
        return frame.copy()


class _VisionCtorResult:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.yolo = object()


class SceneObserverTests(unittest.TestCase):
    def test_default_observer_vision_settings_match_standalone_cpu_defaults(self):
        with patch("apps.conscious_assistant.scene_observer._env_flag", return_value=False):
            settings = _default_observer_vision_settings()

        self.assertEqual(
            settings,
            {
                "use_tensorrt": False,
                "half_precision": False,
                "input_size": 256,
                "confidence_threshold": 0.60,
            },
        )

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

    def test_observe_returns_latest_image_payload(self):
        observer = SceneObserver(public_image_url="/api/observation/latest.jpg")
        frame = np.ones((16, 16, 3), dtype=np.uint8)

        with patch.object(observer, "_ensure_vision", return_value=_FakeVision()):
            with patch.object(observer, "_read_frame", return_value=frame):
                with patch.object(observer, "_write_latest_image", return_value=123456789):
                    observation = observer.observe()

        self.assertEqual(observation["objects"], ["person", "chair"])
        self.assertEqual(observation["summary"], "Objects: 1 person(s), 1 chair(s)")
        self.assertEqual(
            observation["image_url"],
            "/api/observation/latest.jpg?t=123456789",
        )
        self.assertTrue(observation["timestamp"].startswith("["))

    def test_detect_scene_scales_boxes_back_to_original_frame(self):
        observer = SceneObserver()
        frame = np.ones((100, 200, 3), dtype=np.uint8)

        with patch(
            "apps.conscious_assistant.scene_observer._env_int",
            side_effect=lambda name, default: 100 if name == "VISION_OBSERVER_PROCESS_WIDTH" else default,
        ):
            results = observer._detect_scene(_FakeScaledVision(), frame)

        self.assertEqual(results["objects"][0]["bbox"], [20, 5, 60, 25])

    def test_detect_scene_retries_with_fallback_width(self):
        observer = SceneObserver()
        frame = np.ones((240, 1280, 3), dtype=np.uint8)
        vision = _FallbackVision()

        def fake_env_int(name, default):
            if name == "VISION_OBSERVER_PROCESS_WIDTH":
                return 320
            if name == "VISION_OBSERVER_PROCESS_HEIGHT":
                return 240
            if name == "VISION_OBSERVER_FALLBACK_WIDTH":
                return 640
            if name == "VISION_OBSERVER_FALLBACK_HEIGHT":
                return 480
            return default

        with patch("apps.conscious_assistant.scene_observer._env_int", side_effect=fake_env_int):
            results = observer._detect_scene(vision, frame)

        self.assertEqual(vision.sizes, [(320, 240), (640, 240)])
        self.assertEqual(results["objects"][0]["bbox"], [32, 8, 64, 24])

    def test_read_frame_returns_last_good_warmup_frame(self):
        observer = SceneObserver()
        frames = [
            (True, np.zeros((4, 4, 3), dtype=np.uint8)),
            (True, np.ones((4, 4, 3), dtype=np.uint8)),
            (True, np.full((4, 4, 3), 2, dtype=np.uint8)),
        ]

        class _FakeCapture:
            def __init__(self, frame_pairs):
                self._frame_pairs = iter(frame_pairs)

            def isOpened(self):
                return True

            def read(self):
                return next(self._frame_pairs)

        observer._capture = _FakeCapture(frames)

        with patch.object(observer, "_ensure_camera", return_value=True):
            with patch("apps.conscious_assistant.scene_observer._env_int", side_effect=lambda name, default: 3 if name == "VISION_OBSERVER_WARMUP_FRAMES" else default):
                frame = observer._read_frame()

        self.assertEqual(int(frame[0, 0, 0]), 2)

    def test_ensure_vision_uses_repo_resolved_model_path(self):
        observer = SceneObserver()
        captured_kwargs = {}

        def fake_ctor(**kwargs):
            captured_kwargs.update(kwargs)
            return _VisionCtorResult(**kwargs)

        with patch("apps.conscious_assistant.scene_observer.VisionCore", side_effect=fake_ctor):
            vision = observer._ensure_vision()

        self.assertIsNotNone(vision)
        self.assertEqual(
            captured_kwargs["yolo_model"],
            str(REPO_ROOT / "yolov8n.pt"),
        )
        self.assertEqual(
            captured_kwargs["face_db_path"],
            str(REPO_ROOT / "face_database"),
        )

    def test_score_results_prefers_more_objects_then_confidence(self):
        low = {"objects": [{"confidence": 0.9}], "faces": [], "summary": ""}
        high = {
            "objects": [{"confidence": 0.4}, {"confidence": 0.3}],
            "faces": [],
            "summary": "",
        }

        self.assertGreater(
            SceneObserver._score_results(high),
            SceneObserver._score_results(low),
        )

    def test_observe_uses_best_result_from_frame_burst(self):
        observer = SceneObserver(public_image_url="/api/observation/latest.jpg")
        frames = [
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.ones((8, 8, 3), dtype=np.uint8),
            np.full((8, 8, 3), 2, dtype=np.uint8),
        ]
        results_by_marker = {
            0: {"objects": [], "faces": [], "summary": "No detections"},
            1: {
                "objects": [{"class_name": "person", "confidence": 0.82, "bbox": [0, 0, 2, 2]}],
                "faces": [],
                "summary": "Objects: 1 person(s)",
            },
            2: {"objects": [], "faces": [], "summary": "No detections"},
        }

        def fake_detect_scene(_vision, frame):
            marker = int(frame[0, 0, 0])
            return results_by_marker[marker]

        with patch.object(observer, "_ensure_vision", return_value=_FakeVision()):
            with patch.object(observer, "_read_detection_frames", return_value=frames):
                with patch.object(observer, "_detect_scene", side_effect=fake_detect_scene):
                    with patch.object(observer, "_annotate_frame", side_effect=lambda _vision, frame, _results: frame):
                        with patch.object(observer, "_write_latest_image", return_value=123456789):
                            observation = observer.observe()

        self.assertEqual(observation["objects"], ["person"])
        self.assertEqual(observation["summary"], "Objects: 1 person(s)")
        self.assertEqual(
            observation["image_url"],
            "/api/observation/latest.jpg?t=123456789",
        )


if __name__ == "__main__":
    unittest.main()
