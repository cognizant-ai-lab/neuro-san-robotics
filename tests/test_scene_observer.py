import unittest
from unittest.mock import patch

import numpy as np

from apps.conscious_assistant.scene_observer import SceneObserver
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


class SceneObserverTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
