import math
import os
import struct
import time
import unittest
from types import SimpleNamespace

import numpy as np

from coded_tools.unigo2.depth_processor import ObstacleGrid
from coded_tools.unigo2.lidar_processor import (
    LidarPerimeterConfig,
    LidarPerimeterService,
)
from coded_tools.unigo2.obstacle_grid_utils import ObstacleGridSpec, merge_obstacle_grids
from coded_tools.unigo2.obstacle_provider import (
    FusedObstacleProvider,
    create_default_obstacle_provider,
)


def _scan_with_cluster(index: int, distance_m: float, n: int = 360) -> np.ndarray:
    ranges = np.full(n, np.inf, dtype=np.float32)
    for offset in range(-3, 4):
        ranges[(index + offset) % n] = distance_m
    return ranges


def _empty_depth_grid() -> ObstacleGrid:
    return ObstacleGrid(
        grid=np.zeros((80, 80), dtype=np.float32),
        resolution=0.05,
        origin_row=79,
        origin_col=40,
        timestamp=time.time(),
        nearest_obstacle_m=float("inf"),
        nearest_obstacle_bearing=0.0,
        path_obstacle_m=float("inf"),
        path_obstacle_bearing=0.0,
        path_obstacle_points=0,
    )


class TestLidarPerimeterService(unittest.TestCase):
    def _service(self) -> LidarPerimeterService:
        return LidarPerimeterService(
            LidarPerimeterConfig(
                enabled=False,
                robot_half_width=0.15,
                path_obstacle_min_points=3,
                pointcloud_yaw_offset_rad=0.0,
            )
        )

    def test_range_cluster_ahead_becomes_path_obstacle(self):
        service = self._service()
        grid = service.grid_from_ranges(_scan_with_cluster(180, 0.50))

        self.assertIsNotNone(grid)
        self.assertEqual(grid.origin_row, 80)
        self.assertEqual(grid.origin_col, 80)
        self.assertAlmostEqual(grid.path_obstacle_m, 0.50, places=2)
        self.assertAlmostEqual(grid.path_obstacle_bearing, 0.0, places=2)

    def test_left_perimeter_obstacle_does_not_block_path_corridor(self):
        service = self._service()
        grid = service.grid_from_ranges(_scan_with_cluster(270, 0.40))

        self.assertIsNotNone(grid)
        self.assertAlmostEqual(grid.nearest_obstacle_m, 0.40, places=2)
        self.assertGreater(grid.nearest_obstacle_bearing, 1.0)
        self.assertEqual(grid.path_obstacle_m, float("inf"))

    def test_point_cloud_sample_is_projected(self):
        service = self._service()
        grid = service._grid_from_sample(
            SimpleNamespace(
                points=[
                    SimpleNamespace(x=0.6, y=0.0, z=0.2),
                    SimpleNamespace(x=0.6, y=0.03, z=0.2),
                    SimpleNamespace(x=0.6, y=-0.03, z=0.2),
                ]
            )
        )

        self.assertIsNotNone(grid)
        self.assertAlmostEqual(grid.path_obstacle_m, 0.60, places=2)

    def test_sensor_msgs_pointcloud2_sample_is_projected(self):
        service = self._service()
        fields = [
            SimpleNamespace(name="x", offset=0, datatype=7, count=1),
            SimpleNamespace(name="y", offset=4, datatype=7, count=1),
            SimpleNamespace(name="z", offset=8, datatype=7, count=1),
        ]
        points = [
            (0.55, 0.0, 0.15),
            (0.55, 0.03, 0.15),
            (0.55, -0.03, 0.15),
        ]
        data = b"".join(struct.pack("<ffff", x, y, z, 0.0) for x, y, z in points)

        grid = service._grid_from_sample(
            SimpleNamespace(
                width=len(points),
                height=1,
                fields=fields,
                is_bigendian=False,
                point_step=16,
                data=data,
            )
        )

        self.assertIsNotNone(grid)
        self.assertAlmostEqual(grid.path_obstacle_m, 0.55, places=2)

    def test_self_returns_inside_body_mask_do_not_block_path(self):
        service = self._service()
        grid = service.grid_from_points(
            [
                (0.30, 0.00, 0.15),
                (0.30, 0.03, 0.15),
                (0.30, -0.03, 0.15),
                (0.85, 0.00, 0.15),
                (0.85, 0.03, 0.15),
                (0.85, -0.03, 0.15),
            ]
        )

        self.assertIsNotNone(grid)
        self.assertAlmostEqual(grid.path_obstacle_m, 0.85, places=2)

    def test_go2_default_pointcloud_frame_rotates_negative_y_forward(self):
        service = LidarPerimeterService(
            LidarPerimeterConfig(
                enabled=False,
                robot_half_width=0.15,
                path_obstacle_min_points=3,
            )
        )
        yaw_offset = service._config.pointcloud_yaw_offset_rad

        def raw_point(forward_m: float, left_m: float) -> tuple[float, float, float]:
            return (
                forward_m * math.cos(yaw_offset) + left_m * math.sin(yaw_offset),
                -forward_m * math.sin(yaw_offset) + left_m * math.cos(yaw_offset),
                0.15,
            )

        grid = service.grid_from_points(
            [
                raw_point(0.90, 0.00),
                raw_point(0.90, 0.03),
                raw_point(0.90, -0.03),
            ]
        )

        self.assertIsNotNone(grid)
        self.assertAlmostEqual(grid.path_obstacle_m, 0.90, places=2)
        self.assertAlmostEqual(grid.path_obstacle_bearing, 0.0, places=2)


class TestObstacleGridFusion(unittest.TestCase):
    def test_merge_preserves_sensor_distance_metadata(self):
        service = LidarPerimeterService(
            LidarPerimeterConfig(
                enabled=False,
                robot_half_width=0.15,
                path_obstacle_min_points=3,
            )
        )
        lidar_grid = service.grid_from_ranges(_scan_with_cluster(180, 0.50))

        merged = merge_obstacle_grids(
            [_empty_depth_grid(), lidar_grid],
            ObstacleGridSpec(
                rows=160,
                cols=160,
                resolution=0.05,
                path_obstacle_min_points=3,
            ),
        )

        self.assertIsNotNone(merged)
        self.assertAlmostEqual(merged.path_obstacle_m, 0.50, places=2)
        self.assertGreater(int(np.sum(merged.grid > 0)), 0)

    def test_fused_provider_delegates_center_depth_to_depth_source(self):
        reading = SimpleNamespace(distance_m=1.2, coverage=0.8)
        depth = SimpleNamespace(
            _config=SimpleNamespace(
                grid_resolution=0.05,
                path_corridor_half_width=0.12,
                path_obstacle_min_points=6,
            ),
            backend="fake-depth",
            is_available=True,
            is_running=False,
            start=lambda: None,
            stop=lambda: None,
            get_obstacle_grid=_empty_depth_grid,
            get_single_frame_grid=_empty_depth_grid,
            get_center_depth_reading=lambda *args, **kwargs: reading,
        )
        provider = FusedObstacleProvider(depth=depth, perimeter_sources=[])

        self.assertIs(provider.get_center_depth_reading(), reading)
        self.assertEqual(provider.get_obstacle_grid().origin_row, 79)


class TestObstacleProviderDefaults(unittest.TestCase):
    def setUp(self):
        self._env_names = [
            "NAV_OBSTACLE_SOURCE",
            "NAV_SIMULATION_MODE",
            "NAV_USE_LIDAR",
        ]
        self._old_env = {name: os.environ.get(name) for name in self._env_names}
        for name in self._env_names:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_robot_default_is_depth_only(self):
        provider = create_default_obstacle_provider()

        self.assertIsNotNone(provider._depth)
        self.assertEqual(provider._perimeter_sources, [])

    def test_lidar_can_still_be_enabled_explicitly(self):
        os.environ["NAV_OBSTACLE_SOURCE"] = "lidar"

        provider = create_default_obstacle_provider()

        self.assertIsNone(provider._depth)
        self.assertEqual(len(provider._perimeter_sources), 1)

    def test_simulation_default_keeps_depth_provider(self):
        os.environ["NAV_SIMULATION_MODE"] = "1"

        provider = create_default_obstacle_provider()

        self.assertIsNotNone(provider._depth)
        self.assertEqual(provider._perimeter_sources, [])


if __name__ == "__main__":
    unittest.main()
