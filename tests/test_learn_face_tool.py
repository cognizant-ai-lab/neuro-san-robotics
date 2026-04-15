import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from coded_tools.unigo2.learn_face import LearnFaceTool


class LearnFaceToolTests(unittest.TestCase):
    def test_invoke_requires_a_valid_person_name(self):
        tool = LearnFaceTool()

        result = tool.invoke({"person_name": "   "}, {})

        self.assertEqual(result, "Error: No valid person_name provided.")

    def test_invoke_requires_a_recent_observation_image(self):
        tool = LearnFaceTool()

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "latest_observation.jpg"
            with patch.dict(
                os.environ,
                {
                    "VISION_LATEST_IMAGE_PATH": str(image_path),
                    "VISION_FACE_DB_PATH": temp_dir,
                },
                clear=False,
            ):
                result = tool.invoke({"person_name": "Alice"}, {})

        self.assertIn("No latest observation image is available yet", result)

    def test_invoke_rejects_stale_observation_images(self):
        tool = LearnFaceTool()

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "latest_observation.jpg"
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            old_timestamp = time.time() - 120
            os.utime(image_path, (old_timestamp, old_timestamp))

            with patch.dict(
                os.environ,
                {
                    "VISION_LATEST_IMAGE_PATH": str(image_path),
                    "VISION_FACE_DB_PATH": temp_dir,
                    "VISION_LATEST_IMAGE_MAX_AGE_SECONDS": "30",
                },
                clear=False,
            ):
                result = tool.invoke({"person_name": "Alice"}, {})

        self.assertIn("latest observation image is too old", result)

    def test_invoke_stores_the_latest_observation_in_the_face_database(self):
        tool = LearnFaceTool()

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "latest_observation.jpg"
            face_db_path = Path(temp_dir) / "face_db"
            image = np.full((10, 10, 3), 200, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            with patch.dict(
                os.environ,
                {
                    "VISION_LATEST_IMAGE_PATH": str(image_path),
                    "VISION_FACE_DB_PATH": str(face_db_path),
                    "VISION_LATEST_IMAGE_MAX_AGE_SECONDS": "0",
                },
                clear=False,
            ):
                with patch("coded_tools.unigo2.learn_face.VisionCore") as vision_cls:
                    vision = vision_cls.return_value
                    vision.add_face_to_database.return_value = True

                    result = tool.invoke({"person_name": "Alice Example"}, {})

        self.assertIn("Stored the latest observation image for 'Alice Example'", result)
        vision_cls.assert_called_once_with(
            face_db_path=str(face_db_path),
            initialize_yolo=False,
        )
        vision.add_face_to_database.assert_called_once()


if __name__ == "__main__":
    unittest.main()
