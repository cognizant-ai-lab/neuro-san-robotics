import unittest
from unittest.mock import patch

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


class VisionCoreCameraTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
