import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coded_tools.unigo2.metric_navigation import MetricOccupancyMap


class TestMetricOccupancyMap(unittest.TestCase):
    def test_path_routes_through_a_wall_opening(self):
        occupied = np.zeros((30, 30), dtype=bool)
        occupied[:, 15] = True
        occupied[13:18, 15] = False
        metric_map = MetricOccupancyMap(
            occupied,
            resolution_m=0.10,
            robot_clearance_m=0.10,
            preferred_clearance_m=0.30,
        )

        path = metric_map.plan_path((0.45, 0.45), (2.55, 2.55))

        self.assertIsNotNone(path)
        crossing = [point for point in path if 1.40 <= point[0] <= 1.60]
        self.assertTrue(crossing)
        self.assertTrue(any(1.25 <= point[1] <= 1.85 for point in crossing))

    def test_open_route_remains_direct_and_widely_spaced(self):
        metric_map = MetricOccupancyMap(
            np.zeros((40, 40), dtype=bool),
            resolution_m=0.10,
            robot_clearance_m=0.20,
        )

        path = metric_map.plan_path((0.25, 0.25), (3.25, 3.25))

        self.assertIsNotNone(path)
        route_length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        self.assertAlmostEqual(route_length, math.sqrt(18.0), delta=0.15)
        self.assertLess(len(path), 10)

    def test_clearance_inflation_closes_an_unsafe_narrow_gap(self):
        occupied = np.zeros((30, 30), dtype=bool)
        occupied[:, 15] = True
        occupied[14:17, 15] = False
        metric_map = MetricOccupancyMap(
            occupied,
            resolution_m=0.10,
            robot_clearance_m=0.20,
        )

        path = metric_map.plan_path((0.45, 1.55), (2.55, 1.55))

        self.assertIsNone(path)

    def test_dynamic_obstacle_changes_an_otherwise_direct_route(self):
        metric_map = MetricOccupancyMap(
            np.zeros((30, 40), dtype=bool),
            resolution_m=0.10,
            robot_clearance_m=0.10,
        )
        obstacles = np.asarray([[1.8, y] for y in np.linspace(1.2, 1.8, 10)])

        path = metric_map.plan_path(
            (0.25, 1.50),
            (3.55, 1.50),
            dynamic_obstacles_xy=obstacles,
            dynamic_clearance_m=0.25,
        )

        self.assertIsNotNone(path)
        self.assertTrue(any(abs(y - 1.50) > 0.35 for _x, y in path))

    def test_pose_match_corrects_cross_track_error_without_wall_slide(self):
        occupied = np.zeros((40, 40), dtype=bool)
        occupied[:, 15] = True
        metric_map = MetricOccupancyMap(
            occupied,
            resolution_m=0.10,
            robot_clearance_m=0.10,
        )
        points = np.column_stack(
            (
                np.full(80, 1.0),
                np.linspace(-1.0, 1.0, 80),
            )
        )

        correction = metric_map.match_pose(
            points,
            pose_x=0.66,
            pose_y=2.00,
            pose_yaw=0.0,
        )

        self.assertIsNotNone(correction)
        self.assertAlmostEqual(correction.x, 0.50, delta=0.09)
        self.assertAlmostEqual(correction.y, 2.00, delta=0.01)

    def test_saved_map_round_trips_without_image_dependencies(self):
        metric_map = MetricOccupancyMap(
            np.eye(10, dtype=bool),
            resolution_m=0.15,
            robot_clearance_m=0.30,
            preferred_clearance_m=0.80,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.npz"
            metric_map.save(path)
            loaded = MetricOccupancyMap.load(path)

        np.testing.assert_array_equal(loaded.occupied, metric_map.occupied)
        self.assertAlmostEqual(loaded.resolution_m, 0.15)
        self.assertAlmostEqual(loaded.robot_clearance_m, 0.30)


if __name__ == "__main__":
    unittest.main()

