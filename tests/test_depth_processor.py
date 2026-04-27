import math
import unittest

import numpy as np

from coded_tools.unigo2.depth_processor import (
    DepthProcessor,
    DepthProcessorConfig,
    ObstacleGrid,
    estimate_obstacle_distance_from_bbox,
)


def _make_config(**overrides) -> DepthProcessorConfig:
    defaults = dict(
        grid_rows=80,
        grid_cols=80,
        grid_resolution=0.05,
        ground_height=0.05,
        obstacle_max_height=0.60,
        camera_mount_height=0.30,
        robot_half_width=0.15,
        min_depth_m=0.1,
        max_depth_m=4.0,
        process_width=320,
        process_height=240,
        simulation_mode=True,
    )
    defaults.update(overrides)
    return DepthProcessorConfig(**defaults)


class TestDepthProcessorSimulation(unittest.TestCase):

    def test_simulation_mode_produces_grid(self):
        proc = DepthProcessor(config=_make_config(simulation_mode=True))
        grid = proc.get_obstacle_grid()

        self.assertIsNotNone(grid)
        self.assertIsInstance(grid, ObstacleGrid)
        self.assertEqual(grid.grid.shape, (80, 80))
        self.assertAlmostEqual(grid.resolution, 0.05)

    def test_simulation_grid_has_obstacles(self):
        """The synthetic depth has a wall at 2m — grid should have occupied cells."""
        proc = DepthProcessor(config=_make_config(simulation_mode=True))
        grid = proc.get_obstacle_grid()

        occupied = int(np.sum(grid.grid > 0))
        self.assertGreater(occupied, 0, "Simulation grid should contain obstacles")

    def test_grid_origin_at_bottom_center(self):
        proc = DepthProcessor(config=_make_config(simulation_mode=True))
        grid = proc.get_obstacle_grid()

        self.assertEqual(grid.origin_row, 79)
        self.assertEqual(grid.origin_col, 40)

    def test_nearest_obstacle_is_finite(self):
        proc = DepthProcessor(config=_make_config(simulation_mode=True))
        grid = proc.get_obstacle_grid()

        self.assertLess(grid.nearest_obstacle_m, float("inf"))
        self.assertGreater(grid.nearest_obstacle_m, 0.0)

    def test_backend_is_simulation(self):
        proc = DepthProcessor(config=_make_config(simulation_mode=True))
        self.assertEqual(proc.backend, "simulation")
        self.assertTrue(proc.is_available)

    def test_obstacle_summary_is_readable(self):
        proc = DepthProcessor(config=_make_config(simulation_mode=True))
        summary = proc.get_obstacle_summary()

        self.assertIsInstance(summary, str)
        self.assertIn("obstacle", summary.lower())


class TestDepthProcessing(unittest.TestCase):

    def test_empty_depth_produces_empty_grid(self):
        """All-zeros depth (out of range) should produce an empty grid."""
        proc = DepthProcessor(config=_make_config(simulation_mode=True))
        depth = np.zeros((240, 320), dtype=np.float32)
        grid = proc._process_depth_to_grid(depth)

        occupied = int(np.sum(grid.grid > 0))
        self.assertEqual(occupied, 0, "Zero-depth frame should produce no obstacles")

    def test_far_depth_produces_empty_grid(self):
        """Depth beyond max_depth_m should produce an empty grid."""
        proc = DepthProcessor(config=_make_config(simulation_mode=True))
        depth = np.full((240, 320), 10.0, dtype=np.float32)
        grid = proc._process_depth_to_grid(depth)

        occupied = int(np.sum(grid.grid > 0))
        self.assertEqual(occupied, 0, "Far-away depth should produce no obstacles")

    def test_close_obstacle_detected(self):
        """A wall at 1m ahead should appear in the grid."""
        proc = DepthProcessor(config=_make_config(simulation_mode=True))
        depth = np.full((240, 320), 1.0, dtype=np.float32)
        grid = proc._process_depth_to_grid(depth)

        occupied = int(np.sum(grid.grid > 0))
        self.assertGreater(occupied, 0, "1m wall should produce obstacles in grid")
        self.assertLess(grid.nearest_obstacle_m, 2.0)


class TestBboxDistanceEstimation(unittest.TestCase):

    def test_object_at_bottom_is_close(self):
        dist = estimate_obstacle_distance_from_bbox(
            bbox=[100, 400, 200, 470],
            image_height=480,
        )
        self.assertLess(dist, 2.0)

    def test_object_at_center_is_farther(self):
        dist = estimate_obstacle_distance_from_bbox(
            bbox=[100, 200, 200, 250],
            image_height=480,
        )
        self.assertGreater(dist, 1.0)

    def test_object_above_center_returns_max(self):
        dist = estimate_obstacle_distance_from_bbox(
            bbox=[100, 50, 200, 100],
            image_height=480,
        )
        self.assertAlmostEqual(dist, 10.0)

    def test_returns_bounded_values(self):
        dist = estimate_obstacle_distance_from_bbox(
            bbox=[0, 479, 100, 479],
            image_height=480,
        )
        self.assertGreaterEqual(dist, 0.1)
        self.assertLessEqual(dist, 10.0)


if __name__ == "__main__":
    unittest.main()
