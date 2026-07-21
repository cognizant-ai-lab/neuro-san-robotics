import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from coded_tools.unigo2.depth_processor import ObstacleGrid
from coded_tools.unigo2.nav_core import NavCore, NavGoal, RobotPose
from coded_tools.unigo2.structural_localization import (
    DepthMapLocalizer,
    StructuralMap,
    WallSegment,
)


def _grid_from_local_points(points, rows=100, cols=100, resolution=0.05):
    grid = np.zeros((rows, cols), dtype=np.float32)
    origin_row, origin_col = rows - 1, cols // 2
    for forward, left in points:
        row = origin_row - round(forward / resolution)
        col = origin_col - round(left / resolution)
        if 0 <= row < rows and 0 <= col < cols:
            grid[row, col] = 1.0
    return ObstacleGrid(
        grid=grid,
        resolution=resolution,
        origin_row=origin_row,
        origin_col=origin_col,
    )


class TestStructuralMap(unittest.TestCase):

    def test_loads_source_pixel_segments_in_map_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structure.json"
            path.write_text(
                '{"source_pixel_segments": [{"name": "wall", '
                '"start": [90, 80], "end": [70, 80]}]}',
                encoding="utf-8",
            )
            structural_map = StructuralMap.load_from_file(
                path,
                {
                    "source_floor_bbox_px": {"right": 100, "bottom": 100},
                    "scale_m_per_px": 0.1,
                },
            )

        self.assertEqual(len(structural_map.segments), 1)
        self.assertEqual(structural_map.segments[0].start, (2.0, 1.0))
        self.assertEqual(structural_map.segments[0].end, (2.0, 3.0))


class TestDepthMapLocalizer(unittest.TestCase):

    def test_corrects_local_pose_against_wall_corner(self):
        structural_map = StructuralMap(
            [
                WallSegment("front_wall", (2.0, -2.0), (2.0, 2.0)),
                WallSegment("left_wall", (0.4, 1.0), (3.0, 1.0)),
            ]
        )
        points = [(2.0, left) for left in np.linspace(-1.5, 1.5, 50)]
        points += [(forward, 1.0) for forward in np.linspace(0.5, 3.0, 45)]
        grid = _grid_from_local_points(points)

        correction = DepthMapLocalizer(structural_map).correct_pose(
            grid,
            pose_x=0.45,
            pose_y=-0.30,
            pose_yaw=math.radians(10.0),
        )

        self.assertIsNotNone(correction)
        self.assertAlmostEqual(correction.pose_x, 0.0, delta=0.16)
        self.assertAlmostEqual(correction.pose_y, 0.0, delta=0.16)
        self.assertAlmostEqual(correction.pose_yaw, 0.0, delta=math.radians(6.0))
        self.assertGreaterEqual(correction.support_segments, 2)

    def test_rejects_ambiguous_single_wall_observation(self):
        structural_map = StructuralMap(
            [WallSegment("front_wall", (2.0, -2.0), (2.0, 2.0))]
        )
        grid = _grid_from_local_points(
            [(2.0, left) for left in np.linspace(-1.5, 1.5, 50)]
        )

        correction = DepthMapLocalizer(structural_map).correct_pose(
            grid,
            pose_x=0.45,
            pose_y=-0.30,
            pose_yaw=math.radians(10.0),
        )

        self.assertIsNone(correction)


class TestNavCoreStructuralCorrection(unittest.TestCase):

    def test_accepted_match_reanchors_and_replans_semantic_route(self):
        structural_map = StructuralMap(
            [
                WallSegment("front_wall", (2.0, -2.0), (2.0, 2.0)),
                WallSegment("left_wall", (0.4, 1.0), (3.0, 1.0)),
            ]
        )
        points = [(2.0, left) for left in np.linspace(-1.5, 1.5, 50)]
        points += [(forward, 1.0) for forward in np.linspace(0.5, 3.0, 45)]

        nav = NavCore.__new__(NavCore)
        nav._depth_map_localizer = DepthMapLocalizer(structural_map)
        nav._last_depth_map_localization = 0.0
        nav.DEPTH_MAP_LOCALIZATION_INTERVAL_S = 0.0
        nav._odometry = MagicMock()
        nav._global_planner = MagicMock()
        nav._reset_progress_tracker = MagicMock()

        corrected = nav._correct_pose_from_depth_map(
            _grid_from_local_points(points),
            RobotPose(x=0.45, y=-0.30, yaw=math.radians(10.0)),
            NavGoal(goal_type="semantic", label="kitchen"),
        )

        self.assertAlmostEqual(corrected.x, 0.0, delta=0.16)
        self.assertAlmostEqual(corrected.y, 0.0, delta=0.16)
        nav._odometry.set_pose.assert_called_once()
        nav._global_planner.plan_path.assert_called_once_with(corrected, "kitchen")
        nav._reset_progress_tracker.assert_called_once()
