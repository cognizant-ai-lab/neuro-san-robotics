import math
import os
import threading
import time
import unittest
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np

from coded_tools.unigo2.depth_processor import (
    CenterDepthReading,
    DepthProcessor,
    DepthProcessorConfig,
    ObstacleGrid,
)
from coded_tools.unigo2.nav_core import (
    CAIL_LAB_MAP_FILE,
    GlobalPlanner,
    LocalPlanner,
    LocalObstacleMemory,
    MapEdge,
    MapNode,
    NavCore,
    NavGoal,
    NavState,
    OdometryProvider,
    RobotPose,
    SafetyMonitor,
    SdkSportModeOdometryProvider,
    TopologicalMap,
    VelocityCommand,
    _configured_map_file,
    _create_odometry_provider,
)
from coded_tools.unigo2.obstacle_confirmation import ObstacleConfirmationTracker
from coded_tools.unigo2.obstacle_grid_utils import is_transverse_wall


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _empty_grid(rows=80, cols=80, resolution=0.05) -> ObstacleGrid:
    return ObstacleGrid(
        grid=np.zeros((rows, cols), dtype=np.float32),
        resolution=resolution,
        origin_row=rows - 1,
        origin_col=cols // 2,
        timestamp=time.time(),
        nearest_obstacle_m=float("inf"),
        nearest_obstacle_bearing=0.0,
        path_obstacle_m=float("inf"),
        path_obstacle_bearing=0.0,
        path_obstacle_points=0,
    )


def _grid_with_wall_ahead(distance_m=1.0, rows=80, cols=80, resolution=0.05) -> ObstacleGrid:
    grid = np.zeros((rows, cols), dtype=np.float32)
    origin_row = rows - 1
    origin_col = cols // 2

    wall_row = origin_row - int(distance_m / resolution)
    if 0 <= wall_row < rows:
        grid[wall_row, :] = 1.0

    return ObstacleGrid(
        grid=grid,
        resolution=resolution,
        origin_row=origin_row,
        origin_col=origin_col,
        timestamp=time.time(),
        nearest_obstacle_m=distance_m,
        nearest_obstacle_bearing=0.0,
        path_obstacle_m=distance_m,
        path_obstacle_bearing=0.0,
        path_obstacle_points=100,
    )


def _grid_with_center_block_ahead(distance_m=0.25, rows=80, cols=80, resolution=0.05) -> ObstacleGrid:
    grid = np.zeros((rows, cols), dtype=np.float32)
    origin_row = rows - 1
    origin_col = cols // 2

    block_row = origin_row - int(distance_m / resolution)
    for row in range(block_row - 2, block_row + 3):
        for col in range(origin_col - 2, origin_col + 3):
            if 0 <= row < rows and 0 <= col < cols:
                grid[row, col] = 1.0

    return ObstacleGrid(
        grid=grid,
        resolution=resolution,
        origin_row=origin_row,
        origin_col=origin_col,
        timestamp=time.time(),
        nearest_obstacle_m=distance_m,
        nearest_obstacle_bearing=0.0,
        path_obstacle_m=distance_m,
        path_obstacle_bearing=0.0,
        path_obstacle_points=25,
    )


def _grid_with_wall_right(distance_m=0.5, rows=80, cols=80, resolution=0.05) -> ObstacleGrid:
    grid = np.zeros((rows, cols), dtype=np.float32)
    origin_row = rows - 1
    origin_col = cols // 2

    wall_col = origin_col - int(distance_m / resolution)
    if 0 <= wall_col < cols:
        grid[:, wall_col] = 1.0

    return ObstacleGrid(
        grid=grid,
        resolution=resolution,
        origin_row=origin_row,
        origin_col=origin_col,
        timestamp=time.time(),
        nearest_obstacle_m=distance_m,
        nearest_obstacle_bearing=-math.pi / 2,
        path_obstacle_m=float("inf"),
        path_obstacle_bearing=0.0,
        path_obstacle_points=0,
    )


def _grid_with_side_obstacle(distance_m=0.37, rows=80, cols=80, resolution=0.05) -> ObstacleGrid:
    grid = np.zeros((rows, cols), dtype=np.float32)
    origin_row = rows - 1
    origin_col = cols // 2

    side_col = origin_col + max(1, int(distance_m / resolution))
    if 0 <= side_col < cols:
        grid[origin_row, side_col] = 1.0

    return ObstacleGrid(
        grid=grid,
        resolution=resolution,
        origin_row=origin_row,
        origin_col=origin_col,
        timestamp=time.time(),
        nearest_obstacle_m=distance_m,
        nearest_obstacle_bearing=-math.pi / 2,
        path_obstacle_m=float("inf"),
        path_obstacle_bearing=0.0,
        path_obstacle_points=0,
    )


def _grid_with_obstacle_at_bearing(
    distance_m=0.30,
    bearing_rad=math.radians(-30),
    rows=80,
    cols=80,
    resolution=0.05,
) -> ObstacleGrid:
    grid = np.zeros((rows, cols), dtype=np.float32)
    origin_row = rows - 1
    origin_col = cols // 2

    forward_cells = int(round(distance_m * math.cos(bearing_rad) / resolution))
    lateral_cells = int(round(distance_m * math.sin(bearing_rad) / resolution))
    row = origin_row - forward_cells
    col = origin_col - lateral_cells
    if 0 <= row < rows and 0 <= col < cols:
        grid[row, col] = 1.0

    lateral_m = distance_m * math.sin(bearing_rad)
    path_distance = distance_m if abs(lateral_m) <= 0.12 else float("inf")
    path_bearing = bearing_rad if path_distance < float("inf") else 0.0
    path_points = 100 if path_distance < float("inf") else 0

    return ObstacleGrid(
        grid=grid,
        resolution=resolution,
        origin_row=origin_row,
        origin_col=origin_col,
        timestamp=time.time(),
        nearest_obstacle_m=distance_m,
        nearest_obstacle_bearing=bearing_rad,
        path_obstacle_m=path_distance,
        path_obstacle_bearing=path_bearing,
        path_obstacle_points=path_points,
    )


def _create_test_map() -> TopologicalMap:
    topo = TopologicalMap()
    topo.load_from_dict({
        "name": "test",
        "nodes": [
            {"name": "A", "x": 0.0, "y": 0.0},
            {"name": "B", "x": 3.0, "y": 0.0},
            {"name": "C", "x": 3.0, "y": 4.0},
        ],
        "edges": [
            {"from": "A", "to": "B", "distance": 3.0},
            {"from": "B", "to": "C", "distance": 4.0},
        ],
    })
    return topo


def _create_disconnected_map() -> TopologicalMap:
    topo = TopologicalMap()
    topo.load_from_dict({
        "name": "disconnected",
        "nodes": [
            {"name": "A", "x": 0.0, "y": 0.0},
            {"name": "B", "x": 3.0, "y": 0.0},
            {"name": "isolated", "x": 100.0, "y": 100.0},
        ],
        "edges": [
            {"from": "A", "to": "B", "distance": 3.0},
        ],
    })
    return topo


# ---------------------------------------------------------------------------
# ObstacleConfirmationTracker tests
# ---------------------------------------------------------------------------

class TestObstacleConfirmationTracker(unittest.TestCase):

    def test_confirms_after_required_readings_and_time(self):
        tracker = ObstacleConfirmationTracker(min_seconds=0.5, min_readings=3)

        confirmed, _ = tracker.update(0.4, 0.0, now=10.0)
        self.assertFalse(confirmed)
        confirmed, _ = tracker.update(0.4, 0.0, now=10.2)
        self.assertFalse(confirmed)
        confirmed, _ = tracker.update(0.4, 0.0, now=10.6)
        self.assertTrue(confirmed)

    def test_resets_when_reading_leaves_track_tolerance(self):
        tracker = ObstacleConfirmationTracker(
            min_seconds=0.0,
            min_readings=2,
            distance_tolerance_m=0.1,
            bearing_tolerance_rad=math.radians(5),
        )

        confirmed, _ = tracker.update(0.4, 0.0, now=10.0)
        self.assertFalse(confirmed)
        confirmed, started_new_track = tracker.update(0.7, 0.0, now=10.1)

        self.assertTrue(started_new_track)
        self.assertFalse(confirmed)
        self.assertEqual(tracker.count, 1)


# ---------------------------------------------------------------------------
# LocalPlanner tests
# ---------------------------------------------------------------------------

class TestLocalPlanner(unittest.TestCase):

    @staticmethod
    def _corridor_grid(
        slope: float = 0.10,
        center_offset: float = 0.08,
        include_right_wall: bool = True,
    ) -> ObstacleGrid:
        grid = _empty_grid()
        for forward in np.linspace(0.30, 1.80, 31):
            for lateral in (0.55, -0.55) if include_right_wall else (0.55,):
                lateral += slope * forward + center_offset
                row = grid.origin_row - round(forward / grid.resolution)
                col = grid.origin_col - round(lateral / grid.resolution)
                grid.grid[row, col] = 1.0
        return grid

    def test_parallel_corridor_walls_add_bounded_course_correction(self):
        planner = LocalPlanner(max_yaw_rate=0.08, avoidance_distance=0.30)
        cmd = VelocityCommand(vx=0.30, vy=0.0, vyaw=0.0)

        corrected = planner.apply_corridor_course_correction(
            cmd,
            self._corridor_grid(),
        )

        self.assertGreater(corrected.vyaw, 0.0)
        self.assertLessEqual(corrected.vyaw, 0.06)
        self.assertAlmostEqual(corrected.vx, cmd.vx)

    def test_one_sided_depth_geometry_continuously_corrects_course(self):
        planner = LocalPlanner(max_yaw_rate=0.08, avoidance_distance=0.30)
        cmd = VelocityCommand(vx=0.30, vy=0.0, vyaw=0.01)

        corrected = planner.apply_corridor_course_correction(
            cmd,
            self._corridor_grid(include_right_wall=False),
        )

        self.assertGreater(corrected.vyaw, cmd.vyaw)
        self.assertLessEqual(corrected.vyaw - cmd.vyaw, 0.06)

    def test_final_route_uses_single_wall_for_alignment_not_clearance_drift(self):
        planner = LocalPlanner(max_yaw_rate=0.08, avoidance_distance=0.30)
        cmd = VelocityCommand(vx=0.30, vy=0.0, vyaw=0.0)
        grid = self._corridor_grid(
            slope=0.0,
            center_offset=0.30,
            include_right_wall=False,
        )

        ordinary = planner.apply_corridor_course_correction(cmd, grid)
        final_route = planner.apply_corridor_course_correction(
            cmd,
            grid,
            align_only=True,
        )

        self.assertNotAlmostEqual(ordinary.vyaw, 0.0)
        self.assertAlmostEqual(final_route.vyaw, 0.0, places=3)

    def test_close_slanted_left_wall_steers_robot_right(self):
        planner = LocalPlanner(max_yaw_rate=0.08, avoidance_distance=0.75)
        grid = _empty_grid()
        for forward in np.linspace(0.25, 1.80, 32):
            lateral = 0.10 + 0.25 * forward
            row = grid.origin_row - round(forward / grid.resolution)
            col = grid.origin_col - round(lateral / grid.resolution)
            grid.grid[row, col] = 1.0

        corrected = planner.apply_corridor_course_correction(
            VelocityCommand(vx=0.30, vy=0.0, vyaw=0.0),
            grid,
        )

        self.assertLess(corrected.vyaw, 0.0)

    def test_converging_left_wall_overrides_route_command_toward_wall_early(self):
        planner = LocalPlanner(max_yaw_rate=0.08, avoidance_distance=0.75)
        grid = _empty_grid()
        for forward in np.linspace(0.25, 1.80, 32):
            # A left wall that draws inward with distance means the robot is
            # angled toward it.  Its current clearance is still recoverable.
            lateral = 0.65 - 0.12 * forward
            row = grid.origin_row - round(forward / grid.resolution)
            col = grid.origin_col - round(lateral / grid.resolution)
            grid.grid[row, col] = 1.0

        route_toward_wall = VelocityCommand(vx=0.28, vy=0.0, vyaw=0.08)
        corrected = planner.apply_corridor_course_correction(
            route_toward_wall,
            grid,
            correction_limit=0.02,
        )

        self.assertAlmostEqual(corrected.vx, route_toward_wall.vx)
        self.assertLess(corrected.vyaw, 0.0)
        self.assertGreaterEqual(
            corrected.vyaw,
            -planner.EARLY_WALL_MAX_ALIGNMENT_YAW_RPS,
        )

    def test_comfortably_distant_wall_does_not_reverse_mapped_route_steering(self):
        planner = LocalPlanner(max_yaw_rate=0.08, avoidance_distance=0.75)
        grid = _empty_grid()
        for forward in np.linspace(0.25, 1.80, 32):
            lateral = 0.95 - 0.05 * forward
            row = grid.origin_row - round(forward / grid.resolution)
            col = grid.origin_col - round(lateral / grid.resolution)
            grid.grid[row, col] = 1.0

        corrected = planner.apply_corridor_course_correction(
            VelocityCommand(vx=0.28, vy=0.0, vyaw=0.08),
            grid,
            correction_limit=0.02,
        )

        self.assertGreater(corrected.vyaw, 0.0)

    def test_final_route_alignment_does_not_use_side_clearance_override(self):
        planner = LocalPlanner(max_yaw_rate=0.08, avoidance_distance=0.75)
        grid = _empty_grid()
        for forward in np.linspace(0.25, 1.80, 32):
            lateral = 0.48 - 0.04 * forward
            row = grid.origin_row - round(forward / grid.resolution)
            col = grid.origin_col - round(lateral / grid.resolution)
            grid.grid[row, col] = 1.0

        corrected = planner.apply_corridor_course_correction(
            VelocityCommand(vx=0.18, vy=0.0, vyaw=0.08),
            grid,
            correction_limit=0.02,
            align_only=True,
        )

        self.assertGreater(corrected.vyaw, 0.0)

    def test_parallel_side_wall_supports_straight_mapped_route(self):
        planner = LocalPlanner(max_yaw_rate=0.08, avoidance_distance=0.75)
        grid = self._corridor_grid(
            slope=0.02,
            center_offset=0.0,
            include_right_wall=False,
        )

        self.assertTrue(
            planner.parallel_wall_supports_route(
                grid,
                route_heading=math.radians(2.0),
            )
        )

    def test_steering_reversal_requires_persistent_request(self):
        planner = LocalPlanner(max_yaw_rate=0.08)
        planner.STEERING_REVERSAL_CONFIRM_CYCLES = 3
        right = VelocityCommand(vx=0.2, vyaw=-0.08)
        left = VelocityCommand(vx=0.2, vyaw=0.08)

        self.assertEqual(
            planner.stabilize_translating_steering(right).vyaw,
            -0.08,
        )
        self.assertEqual(planner.stabilize_translating_steering(left).vyaw, 0.0)
        self.assertEqual(planner.stabilize_translating_steering(left).vyaw, 0.0)
        self.assertEqual(
            planner.stabilize_translating_steering(left).vyaw,
            0.08,
        )

    def test_pivot_direction_does_not_reverse_before_alignment(self):
        planner = LocalPlanner(pivot_yaw_rate=0.5)

        self.assertTrue(planner._should_pivot(math.radians(80.0), 2.0))
        self.assertGreater(planner._pivot_yaw_rate(math.radians(80.0)), 0.0)
        self.assertTrue(planner._should_pivot(math.radians(-40.0), 2.0))
        self.assertGreater(planner._pivot_yaw_rate(math.radians(-40.0)), 0.0)

        self.assertFalse(planner._should_pivot(math.radians(5.0), 2.0))
        self.assertLess(planner._pivot_yaw_rate(math.radians(-40.0)), 0.0)

    def test_half_turn_chooses_open_side_instead_of_normalized_angle_sign(self):
        planner = LocalPlanner(pivot_yaw_rate=0.5)
        grid = _empty_grid()
        for forward in np.linspace(0.25, 1.0, 20):
            lateral = -0.30
            row = grid.origin_row - round(forward / grid.resolution)
            col = grid.origin_col - round(lateral / grid.resolution)
            grid.grid[row, col] = 1.0

        cmd = planner.compute_velocity(
            grid,
            goal_direction=-math.pi,
            goal_distance=1.0,
        )

        self.assertAlmostEqual(cmd.vx, 0.0)
        self.assertGreater(cmd.vyaw, 0.0)

    def test_significant_route_recapture_ignores_wider_open_sector(self):
        planner = LocalPlanner(
            max_linear_speed=0.30,
            max_yaw_rate=0.08,
            avoidance_distance=0.75,
        )
        grid = self._corridor_grid(slope=0.0, center_offset=0.18)
        grid.path_obstacle_m = 2.50

        cmd = planner.compute_velocity(
            grid,
            goal_direction=math.radians(-12.0),
            goal_distance=1.0,
            pivot_heading=math.radians(-2.0),
            route_heading_authoritative=True,
        )

        self.assertGreater(cmd.vx, 0.0)
        self.assertAlmostEqual(cmd.vyaw, -planner.max_yaw_rate)

    def test_route_recapture_reverses_stale_steering_immediately(self):
        planner = LocalPlanner(max_yaw_rate=0.08)
        planner._steering_sign = 1

        corrected = planner.stabilize_translating_steering(
            VelocityCommand(vx=0.28, vy=0.0, vyaw=-0.08),
            allow_immediate_reversal=True,
        )

        self.assertAlmostEqual(corrected.vyaw, -0.08)
        self.assertEqual(planner._steering_sign, -1)

    def test_route_recapture_is_not_reversed_by_noncritical_wall_alignment(self):
        planner = LocalPlanner(max_yaw_rate=0.08, avoidance_distance=0.75)
        grid = _empty_grid()
        for forward in np.linspace(0.25, 1.80, 32):
            lateral = -0.78 + 0.12 * forward
            row = grid.origin_row - round(forward / grid.resolution)
            col = grid.origin_col - round(lateral / grid.resolution)
            grid.grid[row, col] = 1.0

        corrected = planner.apply_corridor_course_correction(
            VelocityCommand(vx=0.28, vy=0.0, vyaw=-0.08),
            grid,
            correction_limit=0.02,
            route_heading_authoritative=True,
        )

        self.assertLessEqual(corrected.vyaw, -0.06)

    def test_route_recapture_still_turns_away_from_wall_inside_target_clearance(self):
        planner = LocalPlanner(max_yaw_rate=0.08, avoidance_distance=0.75)
        grid = _empty_grid()
        for forward in np.linspace(0.25, 1.80, 32):
            lateral = -0.42
            row = grid.origin_row - round(forward / grid.resolution)
            col = grid.origin_col - round(lateral / grid.resolution)
            grid.grid[row, col] = 1.0

        corrected = planner.apply_corridor_course_correction(
            VelocityCommand(vx=0.28, vy=0.0, vyaw=-0.08),
            grid,
            correction_limit=0.02,
            route_heading_authoritative=True,
        )

        self.assertGreater(corrected.vyaw, 0.0)

    def test_open_space_centering_cannot_initiate_pivot_away_from_route(self):
        planner = LocalPlanner(max_yaw_rate=0.08, pivot_yaw_rate=0.5)

        cmd = planner._compute_direct_velocity(
            goal_direction=math.radians(45.0),
            goal_distance=1.0,
            path_nearest=3.0,
            pivot_heading=math.radians(-10.0),
        )

        self.assertGreater(cmd.vx, 0.0)
        self.assertAlmostEqual(cmd.vyaw, planner.max_yaw_rate)
        self.assertFalse(planner._pivoting)

    def test_mapped_segment_heading_drives_full_corner_pivot(self):
        planner = LocalPlanner(max_yaw_rate=0.08, pivot_yaw_rate=0.5)
        grid = _empty_grid()

        cmd = planner.compute_velocity(
            grid,
            goal_direction=math.radians(-42.0),
            goal_distance=1.0,
            pivot_heading=math.radians(-65.0),
        )

        self.assertAlmostEqual(cmd.vx, 0.0)
        self.assertAlmostEqual(cmd.vyaw, -0.5)

        continuing = planner.compute_velocity(
            grid,
            goal_direction=math.radians(-8.0),
            goal_distance=1.0,
            pivot_heading=math.radians(-8.0),
        )
        self.assertAlmostEqual(continuing.vx, 0.0)
        self.assertAlmostEqual(continuing.vyaw, -0.35)

    def test_pivot_remains_active_until_within_six_degrees(self):
        planner = LocalPlanner(pivot_yaw_rate=0.5)

        self.assertTrue(planner._should_pivot(math.radians(-65.0), 1.0))
        self.assertTrue(planner._should_pivot(math.radians(-8.0), 1.0))
        self.assertFalse(planner._should_pivot(math.radians(-5.0), 1.0))

    def test_route_heading_centers_before_reaching_close_right_wall(self):
        planner = LocalPlanner(max_yaw_rate=0.08, avoidance_distance=0.75)
        grid = self._corridor_grid(slope=0.0, center_offset=0.18)

        heading = planner._centered_route_heading(grid, goal_direction=0.0)

        self.assertGreater(heading, 0.0)

    def test_route_heading_keeps_direct_goal_when_robot_width_lane_is_open(self):
        planner = LocalPlanner(max_yaw_rate=0.08, avoidance_distance=0.75)
        grid = _empty_grid()
        for forward in np.linspace(0.30, 1.80, 31):
            row = grid.origin_row - round(forward / grid.resolution)
            col = grid.origin_col + round(0.70 / grid.resolution)
            grid.grid[row, col] = 1.0

        goal_heading = math.radians(8.0)
        heading = planner._centered_route_heading(grid, goal_heading)

        self.assertAlmostEqual(heading, goal_heading)

    def test_velocity_steers_toward_widest_open_route_before_avoidance_band(self):
        planner = LocalPlanner(
            max_linear_speed=0.30,
            max_yaw_rate=0.08,
            avoidance_distance=0.75,
        )
        grid = self._corridor_grid(slope=0.0, center_offset=0.18)
        grid.path_obstacle_m = float("inf")

        cmd = planner.compute_velocity(
            grid,
            goal_direction=0.0,
            goal_distance=3.0,
        )

        self.assertGreater(cmd.vx, 0.0)
        self.assertGreater(cmd.vyaw, 0.0)

    def test_metric_turn_is_not_overridden_by_open_space_centering(self):
        planner = LocalPlanner(max_yaw_rate=0.08, avoidance_distance=0.75)
        grid = self._corridor_grid(slope=0.0, center_offset=0.18)
        goal_heading = math.radians(30.0)

        heading = planner._centered_route_heading(grid, goal_heading)

        self.assertAlmostEqual(heading, goal_heading)


    def test_drives_toward_goal_in_clear_space(self):
        planner = LocalPlanner(max_linear_speed=0.3, safety_distance=0.4, avoidance_distance=0.8)
        grid = _empty_grid()
        cmd = planner.compute_velocity(grid, goal_direction=0.0, goal_distance=2.0)

        self.assertGreater(cmd.vx, 0.0, "Should drive forward in clear space")
        self.assertAlmostEqual(cmd.vyaw, 0.0, places=0,
                               msg="Should not turn when goal is straight ahead")

    def test_pivots_in_place_for_large_heading_error(self):
        planner = LocalPlanner(max_linear_speed=0.3, max_yaw_rate=0.5)
        grid = _empty_grid()

        cmd = planner.compute_velocity(
            grid,
            goal_direction=math.radians(90),
            goal_distance=2.0,
        )

        self.assertAlmostEqual(cmd.vx, 0.0)
        self.assertGreater(cmd.vyaw, 0.0)

    def test_moderate_transit_correction_keeps_moving(self):
        planner = LocalPlanner(max_linear_speed=0.3, max_yaw_rate=0.08)

        cmd = planner.compute_velocity(
            _empty_grid(),
            goal_direction=math.radians(24.0),
            goal_distance=0.65,
            slow_for_arrival=False,
        )

        self.assertGreater(cmd.vx, 0.0)
        self.assertGreaterEqual(cmd.vx, planner.MIN_TRANSIT_SPEED_MPS)
        self.assertGreater(cmd.vyaw, 0.0)

    def test_clear_final_approach_keeps_small_correction_and_effective_speed(self):
        planner = LocalPlanner(max_linear_speed=0.4, max_yaw_rate=0.08)
        steering = VelocityCommand(vx=0.098, vy=0.0, vyaw=0.08)

        cmd = planner.maintain_effective_final_approach(
            steering,
            _empty_grid(),
            goal_distance=0.98,
            arrival_tolerance=0.65,
        )

        self.assertAlmostEqual(cmd.vx, planner.MIN_EFFECTIVE_APPROACH_SPEED_MPS)
        self.assertAlmostEqual(cmd.vyaw, steering.vyaw)

    def test_pending_metric_arrival_keeps_effective_speed_inside_semantic_radius(self):
        planner = LocalPlanner(max_linear_speed=0.4, max_yaw_rate=0.08)
        steering = VelocityCommand(vx=0.063, vy=0.0, vyaw=0.02)

        cmd = planner.maintain_effective_final_approach(
            steering,
            _empty_grid(),
            goal_distance=0.63,
            arrival_tolerance=GlobalPlanner.FINAL_METRIC_LONGITUDINAL_TOLERANCE_M,
        )

        self.assertAlmostEqual(cmd.vx, planner.MIN_EFFECTIVE_APPROACH_SPEED_MPS)
        self.assertAlmostEqual(cmd.vyaw, steering.vyaw)

    def test_final_approach_does_not_override_tight_clearance(self):
        planner = LocalPlanner(max_linear_speed=0.4, max_yaw_rate=0.08)
        grid = replace(_empty_grid(), path_obstacle_m=0.50)
        steering = VelocityCommand(vx=0.098, vy=0.0, vyaw=0.08)

        cmd = planner.maintain_effective_final_approach(
            steering,
            grid,
            goal_distance=0.98,
            arrival_tolerance=0.65,
        )

        self.assertEqual(cmd, steering)

    def test_pivot_hysteresis_finishes_turn_without_threshold_oscillation(self):
        planner = LocalPlanner(max_linear_speed=0.3, max_yaw_rate=0.08)
        grid = _empty_grid()

        entering = planner.compute_velocity(
            grid,
            goal_direction=math.radians(40.0),
            goal_distance=1.0,
        )
        continuing = planner.compute_velocity(
            grid,
            goal_direction=math.radians(18.0),
            goal_distance=1.0,
        )
        finished = planner.compute_velocity(
            grid,
            goal_direction=math.radians(5.0),
            goal_distance=1.0,
        )

        self.assertAlmostEqual(entering.vx, 0.0)
        self.assertAlmostEqual(continuing.vx, 0.0)
        self.assertGreater(finished.vx, 0.0)

    def test_pivot_uses_configured_pivot_rate_independent_of_steering_limit(self):
        planner = LocalPlanner(
            max_linear_speed=0.3,
            max_yaw_rate=0.08,
            pivot_yaw_rate=0.50,
        )
        grid = _empty_grid()

        cmd = planner.compute_velocity(
            grid,
            goal_direction=math.radians(-90),
            goal_distance=2.0,
        )

        self.assertAlmostEqual(cmd.vx, 0.0)
        self.assertLess(cmd.vyaw, 0.0)
        self.assertAlmostEqual(abs(cmd.vyaw), 0.50)

    def test_clear_path_uses_direct_heading_without_vfh_wobble(self):
        planner = LocalPlanner(max_linear_speed=0.3, max_yaw_rate=0.08)
        grid = _grid_with_side_obstacle(distance_m=0.37)

        cmd = planner.compute_velocity(
            grid,
            goal_direction=math.radians(5.0),
            goal_distance=2.0,
        )

        self.assertGreater(cmd.vx, 0.0)
        self.assertGreater(cmd.vyaw, 0.0)
        self.assertLessEqual(abs(cmd.vyaw), 0.08)

    def test_stops_when_no_free_sectors(self):
        planner = LocalPlanner()
        grid = _empty_grid()
        grid.grid[:, :] = 1.0  # all occupied
        grid.nearest_obstacle_m = 0.2
        grid.path_obstacle_m = 0.2
        grid.path_obstacle_points = 100

        cmd = planner.compute_velocity(grid, goal_direction=0.0, goal_distance=2.0)
        self.assertAlmostEqual(cmd.vx, 0.0,
                               msg="Should stop when completely surrounded")

    def test_slows_near_obstacles(self):
        planner = LocalPlanner(max_linear_speed=0.3, safety_distance=0.4, avoidance_distance=0.8)
        grid = _grid_with_wall_ahead(distance_m=0.6)

        cmd = planner.compute_velocity(grid, goal_direction=0.0, goal_distance=2.0)
        self.assertLess(cmd.vx, 0.3, "Should slow down near obstacles")
        self.assertGreater(cmd.vx, 0.0, "Should still move (not stopped yet)")

    def test_zero_speed_at_safety_distance(self):
        planner = LocalPlanner(max_linear_speed=0.3, safety_distance=0.4, avoidance_distance=0.8)
        grid = _grid_with_wall_ahead(distance_m=0.3)

        cmd = planner.compute_velocity(grid, goal_direction=0.0, goal_distance=2.0)
        self.assertAlmostEqual(cmd.vx, 0.0,
                               msg="Should stop at safety distance")

    def test_steers_around_supported_obstacle_in_avoidance_band(self):
        planner = LocalPlanner(max_linear_speed=0.3, safety_distance=0.1, avoidance_distance=0.3)
        grid = _grid_with_center_block_ahead(distance_m=0.25)

        cmd = planner.compute_velocity(grid, goal_direction=0.0, goal_distance=2.0)

        self.assertGreater(cmd.vx, 0.0, "Should keep moving while circumnavigating")
        self.assertNotAlmostEqual(cmd.vyaw, 0.0, msg="Should steer around the block")

    def test_pivots_toward_free_space_when_obstacle_is_inside_safety_distance(self):
        planner = LocalPlanner(
            max_linear_speed=0.3,
            safety_distance=0.1,
            avoidance_distance=0.3,
        )
        grid = _grid_with_center_block_ahead(distance_m=0.08)

        cmd = planner.compute_velocity(grid, goal_direction=0.0, goal_distance=2.0)

        self.assertAlmostEqual(cmd.vx, 0.0)
        self.assertNotAlmostEqual(cmd.vyaw, 0.0)

    def test_transverse_wall_requires_broad_geometry_across_path(self):
        wall = _grid_with_wall_ahead(distance_m=0.25)
        block = _grid_with_center_block_ahead(distance_m=0.25)

        self.assertTrue(is_transverse_wall(wall, 0.25, 0.5))
        self.assertFalse(is_transverse_wall(block, 0.25, 0.5))

    def test_drives_when_only_side_obstacle_is_close(self):
        planner = LocalPlanner(max_linear_speed=0.3, safety_distance=0.4, avoidance_distance=0.8)
        grid = _grid_with_side_obstacle(distance_m=0.37)

        cmd = planner.compute_velocity(grid, goal_direction=0.0, goal_distance=2.0)

        self.assertGreater(cmd.vx, 0.0, "Side clutter should not block forward travel")

    def test_drives_when_diagonal_side_obstacle_is_close(self):
        planner = LocalPlanner(max_linear_speed=0.3, safety_distance=0.4, avoidance_distance=0.8)
        grid = _grid_with_obstacle_at_bearing(
            distance_m=0.30,
            bearing_rad=math.radians(-30.0),
        )

        cmd = planner.compute_velocity(grid, goal_direction=0.0, goal_distance=2.0)

        self.assertGreater(cmd.vx, 0.0, "A 30-degree side object should not block the route")

    def test_drives_when_nearest_anywhere_is_false_close_reading(self):
        planner = LocalPlanner(max_linear_speed=0.3, safety_distance=0.4, avoidance_distance=0.8)
        grid = _empty_grid()
        grid.nearest_obstacle_m = 0.19
        grid.nearest_obstacle_bearing = math.radians(35.0)
        grid.path_obstacle_m = float("inf")
        grid.path_obstacle_points = 0

        cmd = planner.compute_velocity(grid, goal_direction=0.0, goal_distance=2.0)

        self.assertGreater(
            cmd.vx,
            0.0,
            "A close reading outside the supported path corridor should not stop forward travel",
        )

    def test_slows_near_goal(self):
        planner = LocalPlanner(max_linear_speed=0.3)
        grid = _empty_grid()

        cmd = planner.compute_velocity(grid, goal_direction=0.0, goal_distance=0.3)
        self.assertLessEqual(cmd.vx, 0.1, "Should slow down near goal")

    def test_does_not_slow_for_close_intermediate_route_point(self):
        planner = LocalPlanner(max_linear_speed=0.3)
        grid = _empty_grid()

        cmd = planner.compute_velocity(
            grid,
            goal_direction=0.0,
            goal_distance=0.3,
            slow_for_arrival=False,
        )

        self.assertAlmostEqual(cmd.vx, 0.3)

    def test_close_intermediate_route_point_steers_through_without_pivoting(self):
        planner = LocalPlanner(
            max_linear_speed=0.3,
            max_yaw_rate=0.08,
            pivot_yaw_rate=0.5,
        )
        grid = _empty_grid()

        cmd = planner.compute_velocity(
            grid,
            goal_direction=math.radians(45.0),
            goal_distance=0.3,
            slow_for_arrival=False,
        )

        self.assertGreater(cmd.vx, 0.1)
        self.assertAlmostEqual(cmd.vyaw, 0.08)

    def test_avoidance_turns_away_from_obstacle(self):
        planner = LocalPlanner()
        grid = _grid_with_wall_right(distance_m=0.5)
        grid.nearest_obstacle_bearing = -0.5

        cmd = planner.compute_avoidance(grid)
        self.assertGreater(cmd.vyaw, 0.0,
                           msg="Should turn left (positive vyaw) to avoid obstacle on right")


# ---------------------------------------------------------------------------
# Local obstacle memory tests
# ---------------------------------------------------------------------------

class TestLocalObstacleMemory(unittest.TestCase):

    def test_retains_and_pose_aligns_recent_wall_geometry(self):
        memory = LocalObstacleMemory(ttl_s=0.8)
        wall = _grid_with_wall_ahead(distance_m=1.0)
        memory.update(wall, RobotPose(x=0.0, y=0.0, yaw=0.0), now=10.0)

        retained = memory.update(
            _empty_grid(),
            RobotPose(x=0.10, y=0.0, yaw=0.0),
            now=10.1,
        )

        occupied = np.argwhere(retained.grid > 0)
        forward = (retained.origin_row - occupied[:, 0]) * retained.resolution
        self.assertTrue(np.any(np.isclose(forward, 0.90, atol=retained.resolution)))
        self.assertEqual(retained.path_obstacle_m, float("inf"))

    def test_expires_old_wall_geometry(self):
        memory = LocalObstacleMemory(ttl_s=0.5)
        memory.update(
            _grid_with_wall_ahead(distance_m=1.0),
            RobotPose(),
            now=10.0,
        )

        retained = memory.update(_empty_grid(), RobotPose(), now=10.6)

        self.assertFalse(np.any(retained.grid > 0))


# ---------------------------------------------------------------------------
# SafetyMonitor tests
# ---------------------------------------------------------------------------

class TestSafetyMonitor(unittest.TestCase):

    def test_estop_at_close_distance(self):
        safety = SafetyMonitor(safety_distance=0.4)
        cmd = VelocityCommand(vx=0.3, vy=0.0, vyaw=0.0)

        filtered, event = safety.filter_command(cmd, nearest_obstacle_m=0.2)
        self.assertAlmostEqual(filtered.vx, 0.0)
        self.assertAlmostEqual(filtered.vy, 0.0)
        self.assertIsNotNone(event)
        self.assertIn("e_stop", event)

    def test_allows_in_place_pivot_near_obstacle(self):
        safety = SafetyMonitor(safety_distance=0.4, pivot_hard_stop_distance=0.2)
        cmd = VelocityCommand(vx=0.0, vy=0.0, vyaw=0.5)

        filtered, event = safety.filter_command(cmd, nearest_obstacle_m=0.25)

        self.assertIsNone(event)
        self.assertAlmostEqual(filtered.vx, 0.0)
        self.assertAlmostEqual(filtered.vyaw, 0.5)

    def test_allows_close_pivot_explicitly_directed_away_from_geometry(self):
        safety = SafetyMonitor(
            safety_distance=0.4,
            pivot_hard_stop_distance=0.4,
            pivot_emergency_stop_distance=0.10,
        )

        filtered, event = safety.filter_command(
            VelocityCommand(vyaw=0.5),
            nearest_obstacle_m=0.18,
            nearest_obstacle_bearing=math.radians(-20.0),
            pivot_escape_allowed=True,
        )

        self.assertIsNone(event)
        self.assertAlmostEqual(filtered.vyaw, 0.5)

    def test_extremely_close_geometry_still_blocks_pivot_escape(self):
        safety = SafetyMonitor(
            safety_distance=0.4,
            pivot_hard_stop_distance=0.4,
            pivot_emergency_stop_distance=0.10,
        )

        filtered, event = safety.filter_command(
            VelocityCommand(vyaw=0.5),
            nearest_obstacle_m=0.08,
            pivot_escape_allowed=True,
        )

        self.assertEqual(event, "e_stop:obstacle_too_close")
        self.assertAlmostEqual(filtered.vyaw, 0.0)

    def test_allows_translation_past_close_side_obstacle(self):
        safety = SafetyMonitor(safety_distance=0.4, pivot_hard_stop_distance=0.2)
        cmd = VelocityCommand(vx=0.3, vy=0.0, vyaw=0.0)

        filtered, event = safety.filter_command(
            cmd,
            nearest_obstacle_m=0.37,
            nearest_obstacle_bearing=-math.pi / 2,
        )

        self.assertIsNone(event)
        self.assertAlmostEqual(filtered.vx, 0.3)

    def test_allows_translation_past_close_diagonal_side_obstacle(self):
        safety = SafetyMonitor(safety_distance=0.4, pivot_hard_stop_distance=0.2)
        cmd = VelocityCommand(vx=0.3, vy=0.0, vyaw=0.0)

        filtered, event = safety.filter_command(
            cmd,
            nearest_obstacle_m=0.30,
            nearest_obstacle_bearing=math.radians(-30.0),
        )

        self.assertIsNone(event)
        self.assertAlmostEqual(filtered.vx, 0.3)

    def test_no_event_when_clear(self):
        safety = SafetyMonitor(safety_distance=0.4, avoidance_distance=0.8)
        cmd = VelocityCommand(vx=0.3, vy=0.0, vyaw=0.0)

        filtered, event = safety.filter_command(cmd, nearest_obstacle_m=2.0)
        self.assertAlmostEqual(filtered.vx, 0.3)
        self.assertIsNone(event)

    def test_speed_ramp_in_avoidance_zone(self):
        safety = SafetyMonitor(safety_distance=0.4, avoidance_distance=0.8)
        cmd = VelocityCommand(vx=0.3, vy=0.0, vyaw=0.0)

        filtered, event = safety.filter_command(cmd, nearest_obstacle_m=0.6)
        self.assertGreater(filtered.vx, 0.0, "Should still move in avoidance zone")
        self.assertLess(filtered.vx, 0.3, "Should be slower than full speed")
        self.assertIsNone(event)

    def test_cliff_detection_stops(self):
        safety = SafetyMonitor()
        cmd = VelocityCommand(vx=0.3, vy=0.0, vyaw=0.0)

        filtered, event = safety.filter_command(
            cmd, nearest_obstacle_m=5.0, ground_plane_valid=False
        )
        self.assertAlmostEqual(filtered.vx, 0.0)
        self.assertIn("ground_plane", event)

    def test_stuck_detection(self):
        safety = SafetyMonitor(stuck_timeout=10.0)
        cmd = VelocityCommand(vx=0.3, vy=0.0, vyaw=0.0)

        filtered, event = safety.filter_command(
            cmd, nearest_obstacle_m=5.0, seconds_since_progress=15.0
        )
        self.assertAlmostEqual(filtered.vx, 0.0)
        self.assertIn("stuck", event)


# ---------------------------------------------------------------------------
# GlobalPlanner tests
# ---------------------------------------------------------------------------

class TestGlobalPlanner(unittest.TestCase):

    def test_metric_guidance_follows_segment_with_small_early_correction(self):
        planner = GlobalPlanner(_create_test_map())
        planner.install_metric_path_points(
            RobotPose(0.0, 0.0, 0.0),
            "C",
            [(0.0, 0.0), (2.0, 0.0), (2.0, -2.0), (4.0, 0.0)],
        )

        heading = planner.current_segment_guidance(RobotPose(0.5, 0.20, 0.0))
        upcoming = planner.upcoming_turn()

        self.assertAlmostEqual(heading, math.radians(-7.6), places=2)
        self.assertIsNotNone(upcoming)
        self.assertAlmostEqual(upcoming[2], math.radians(-90.0))
        planner.advance_current_waypoint()
        self.assertAlmostEqual(
            planner.current_turn_change(),
            math.radians(-90.0),
        )

    def test_rejects_detour_that_is_both_a_u_turn_and_much_longer(self):
        pose = RobotPose(0.0, 0.0, 0.0)
        points = [(0.0, 0.0), (-1.0, 0.0), (-12.0, 0.0)]

        self.assertFalse(
            GlobalPlanner.metric_detour_is_reasonable(
                pose,
                points,
                reference_length=6.0,
                reference_bearing=0.0,
            )
        )

    def test_allows_short_local_detour_even_when_it_starts_behind(self):
        pose = RobotPose(0.0, 0.0, 0.0)
        points = [(0.0, 0.0), (-0.3, 0.0), (5.5, 0.0)]

        self.assertTrue(
            GlobalPlanner.metric_detour_is_reasonable(
                pose,
                points,
                reference_length=6.0,
                reference_bearing=0.0,
            )
        )

    def test_rejected_detour_does_not_replace_active_route(self):
        topo = _create_test_map()
        topo.metric_map = MagicMock()
        topo.metric_map.plan_path.return_value = [
            (0.0, 0.0),
            (-1.0, 0.0),
            (-12.0, 0.0),
        ]
        planner = GlobalPlanner(topo)
        planner.install_metric_path_points(
            RobotPose(0.0, 0.0, 0.0),
            "C",
            [(0.0, 0.0), (1.0, 0.0), (6.0, 0.0)],
        )

        path = planner.replan_path_around_obstacles(
            RobotPose(0.0, 0.0, 0.0),
            "C",
            np.asarray([[0.5, 0.0]]),
        )

        self.assertIsNone(path)
        self.assertAlmostEqual(planner.get_current_waypoint().x, 1.0)

    def test_metric_transit_point_advances_within_route_following_tolerance(self):
        topo = _create_test_map()
        planner = GlobalPlanner(topo)
        planner.install_metric_path_points(
            RobotPose(0.0, 0.0, 0.0),
            "C",
            [(0.0, 0.0), (0.8, 0.0), (3.0, 0.0)],
        )

        waypoint = planner.get_next_waypoint(RobotPose(0.40, 0.0, 0.0))

        self.assertIsNotNone(waypoint)
        self.assertEqual(waypoint.name, "C")

    def test_metric_map_routes_around_static_wall(self):
        topo = _create_test_map()
        occupied = np.zeros((30, 50), dtype=bool)
        occupied[:22, 20] = True
        from coded_tools.unigo2.metric_navigation import MetricOccupancyMap
        topo.metric_map = MetricOccupancyMap(
            occupied,
            resolution_m=0.10,
            robot_clearance_m=0.10,
        )
        topo.nodes["B"].x = 4.0
        topo.nodes["B"].y = 0.5
        planner = GlobalPlanner(topo)

        path = planner.plan_path(RobotPose(0.5, 0.5, 0.0), "B")

        self.assertIsNotNone(path)
        self.assertTrue(all("metric_transit" in node.tags for node in path[:-1]))
        self.assertEqual(path[-1].name, "B")
        self.assertTrue(any(node.y > 2.2 for node in path))

    def test_projects_remote_correction_onto_active_segment(self):
        planner = GlobalPlanner(_create_test_map())
        planner.plan_path(RobotPose(0.0, 0.0, 0.0), "C")

        correction = planner.project_onto_current_segment(
            RobotPose(1.4, 0.8, 0.5)
        )

        self.assertIsNotNone(correction)
        pose, start, target = correction
        self.assertEqual((start.name, target.name), ("A", "B"))
        self.assertAlmostEqual(pose.x, 1.4)
        self.assertAlmostEqual(pose.y, 0.0)
        self.assertAlmostEqual(pose.yaw, 0.0)
        self.assertEqual(planner._waypoint_index, 1)

    def test_dijkstra_finds_shortest_path(self):
        topo = _create_test_map()
        planner = GlobalPlanner(topo)

        path = planner.plan_path(RobotPose(0, 0, 0), "C")
        self.assertIsNotNone(path)
        self.assertEqual(path[0].name, "A")
        self.assertEqual(path[-1].name, "C")
        self.assertEqual(len(path), 3)  # A -> B -> C

    def test_direct_neighbor(self):
        topo = _create_test_map()
        planner = GlobalPlanner(topo)

        path = planner.plan_path(RobotPose(0, 0, 0), "B")
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 2)  # A -> B

    def test_unreachable_returns_none(self):
        topo = _create_disconnected_map()
        planner = GlobalPlanner(topo)

        path = planner.plan_path(RobotPose(0, 0, 0), "isolated")
        self.assertIsNone(path)

    def test_unknown_destination_returns_none(self):
        topo = _create_test_map()
        planner = GlobalPlanner(topo)

        path = planner.plan_path(RobotPose(0, 0, 0), "nonexistent")
        self.assertIsNone(path)

    def test_already_at_goal(self):
        topo = _create_test_map()
        planner = GlobalPlanner(topo)

        path = planner.plan_path(RobotPose(0, 0, 0), "A")
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0].name, "A")

    def test_single_node_path_can_still_provide_waypoint_when_offset(self):
        topo = _create_test_map()
        planner = GlobalPlanner(topo)

        path = planner.plan_path(RobotPose(0.5, 0.0, 0.0), "A")
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 1)

        waypoint = planner.get_next_waypoint(RobotPose(0.5, 0.0, 0.0), tolerance_m=0.3)
        self.assertIsNotNone(waypoint)
        self.assertEqual(waypoint.name, "A")

    def test_get_next_waypoint_advances(self):
        topo = _create_test_map()
        planner = GlobalPlanner(topo)
        planner.plan_path(RobotPose(0, 0, 0), "C")
        advances = []

        wp1 = planner.get_next_waypoint(RobotPose(0, 0, 0), tolerance_m=0.3)
        self.assertIsNotNone(wp1)
        self.assertEqual(wp1.name, "B")

        wp2 = planner.get_next_waypoint(
            RobotPose(3.0, 0.0, 0),
            tolerance_m=0.3,
            on_advance=lambda reached, upcoming: advances.append(
                (reached.name, upcoming.name)
            ),
        )
        self.assertIsNotNone(wp2)
        self.assertEqual(wp2.name, "C")
        self.assertEqual(advances, [("B", "C")])

    def test_get_next_waypoint_returns_none_at_goal(self):
        topo = _create_test_map()
        planner = GlobalPlanner(topo)
        planner.plan_path(RobotPose(0, 0, 0), "B")

        wp = planner.get_next_waypoint(RobotPose(3.0, 0.0, 0), tolerance_m=0.3)
        self.assertIsNone(wp, "Should return None when at goal")

    def test_get_next_waypoint_uses_map_arrival_tolerance(self):
        topo = _create_test_map()
        topo.nodes["B"].arrival_tolerance_m = 0.65
        planner = GlobalPlanner(topo)
        planner.plan_path(RobotPose(0, 0, 0), "B")

        wp = planner.get_next_waypoint(RobotPose(2.45, 0.0, 0), tolerance_m=0.15)

        self.assertIsNone(wp)

    def test_metric_final_does_not_arrive_at_outer_edge_of_radius(self):
        topo = _create_test_map()
        topo.nodes["B"].arrival_tolerance_m = 0.65
        planner = GlobalPlanner(topo)
        planner.install_metric_path_points(
            RobotPose(0.0, 0.0, 0.0),
            "B",
            [(0.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        )
        planner.get_next_waypoint(RobotPose(2.0, 0.0, 0.0), tolerance_m=0.65)

        waypoint = planner.get_next_waypoint(
            RobotPose(2.40, 0.0, 0.0),
            tolerance_m=0.65,
        )

        self.assertIsNotNone(waypoint)
        self.assertEqual(waypoint.name, "B")

    def test_metric_final_arrives_after_crossing_aligned_entrance_gate(self):
        topo = _create_test_map()
        topo.nodes["B"].arrival_tolerance_m = 0.65
        planner = GlobalPlanner(topo)
        planner.install_metric_path_points(
            RobotPose(0.0, 0.0, 0.0),
            "B",
            [(0.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        )
        planner.get_next_waypoint(RobotPose(2.0, 0.0, 0.0), tolerance_m=0.65)

        waypoint = planner.get_next_waypoint(
            RobotPose(2.75, 0.10, math.radians(5.0)),
            tolerance_m=0.65,
            final_arrival_sensor_confirmed=True,
        )

        self.assertIsNone(waypoint)

    def test_metric_final_requires_independent_sensor_confirmation(self):
        topo = _create_test_map()
        topo.nodes["B"].arrival_tolerance_m = 0.65
        planner = GlobalPlanner(topo)
        planner.install_metric_path_points(
            RobotPose(0.0, 0.0, 0.0),
            "B",
            [(0.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        )
        planner.get_next_waypoint(RobotPose(2.0, 0.0, 0.0), tolerance_m=0.65)

        deferred = planner.get_next_waypoint(
            RobotPose(2.75, 0.0, 0.0),
            tolerance_m=0.65,
            final_arrival_sensor_confirmed=False,
        )

        self.assertIsNotNone(deferred)
        self.assertEqual(deferred.name, "B")

    def test_metric_final_accepts_strict_route_position_when_map_match_is_weak(self):
        topo = _create_test_map()
        topo.nodes["B"].arrival_tolerance_m = 0.65
        planner = GlobalPlanner(topo)
        planner.install_metric_path_points(
            RobotPose(0.0, 0.0, 0.0),
            "B",
            [(0.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        )
        planner.get_next_waypoint(RobotPose(2.0, 0.0, 0.0), tolerance_m=0.65)

        arrived = planner.get_next_waypoint(
            RobotPose(2.90, 0.0, math.radians(90.0)),
            tolerance_m=0.65,
            final_arrival_sensor_confirmed=False,
        )

        self.assertIsNone(arrived)

    def test_metric_final_accepts_sensor_confirmed_pose_inside_arrival_radius(self):
        topo = _create_test_map()
        topo.nodes["B"].arrival_tolerance_m = 0.65
        planner = GlobalPlanner(topo)
        planner.install_metric_path_points(
            RobotPose(0.0, 0.0, 0.0),
            "B",
            [(0.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        )
        planner.get_next_waypoint(RobotPose(2.0, 0.0, 0.0), tolerance_m=0.65)

        arrived = planner.get_next_waypoint(
            RobotPose(2.40, 0.0, 0.0),
            tolerance_m=0.65,
            final_arrival_sensor_confirmed=True,
        )

        self.assertIsNone(arrived)

    def test_metric_final_rejects_wrong_heading_or_lateral_approach(self):
        topo = _create_test_map()
        topo.nodes["B"].arrival_tolerance_m = 0.65
        planner = GlobalPlanner(topo)
        planner.install_metric_path_points(
            RobotPose(0.0, 0.0, 0.0),
            "B",
            [(0.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        )
        planner.get_next_waypoint(RobotPose(2.0, 0.0, 0.0), tolerance_m=0.65)

        wrong_heading = planner.get_next_waypoint(
            RobotPose(2.80, 0.0, math.radians(90.0)),
            tolerance_m=0.65,
        )
        wide = planner.get_next_waypoint(
            RobotPose(2.80, 0.50, 0.0),
            tolerance_m=0.65,
        )

        self.assertIsNotNone(wrong_heading)
        self.assertIsNotNone(wide)
        self.assertEqual(wrong_heading.name, "B")
        self.assertEqual(wide.name, "B")

    def test_get_next_waypoint_accepts_passed_intermediate_waypoint(self):
        topo = _create_test_map()
        topo.nodes["B"].pass_through_tolerance_m = 0.75
        planner = GlobalPlanner(topo)
        planner.plan_path(RobotPose(0, 0, 0), "C")

        wp = planner.get_next_waypoint(RobotPose(3.2, 0.4, 0), tolerance_m=0.15)

        self.assertIsNotNone(wp)
        self.assertEqual(wp.name, "C")

    def test_get_next_waypoint_does_not_accept_wide_pass(self):
        topo = _create_test_map()
        topo.nodes["B"].pass_through_tolerance_m = 0.75
        planner = GlobalPlanner(topo)
        planner.plan_path(RobotPose(0, 0, 0), "C")

        wp = planner.get_next_waypoint(RobotPose(3.2, 1.0, 0), tolerance_m=0.15)

        self.assertIsNotNone(wp)
        self.assertEqual(wp.name, "B")

    def test_straight_metric_waypoint_accepts_corridor_width_pass(self):
        planner = GlobalPlanner(_create_test_map())
        planner.install_metric_path_points(
            RobotPose(0.0, 0.0, 0.0),
            "C",
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (4.0, 0.0)],
        )

        waypoint = planner.get_next_waypoint(
            RobotPose(1.10, 0.62, 0.0),
            tolerance_m=0.45,
        )

        self.assertIsNotNone(waypoint)
        self.assertEqual(waypoint.name, "__metric_002__")

    def test_metric_corner_keeps_tight_pass_tolerance(self):
        planner = GlobalPlanner(_create_test_map())
        planner.install_metric_path_points(
            RobotPose(0.0, 0.0, 0.0),
            "C",
            [(0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (4.0, 0.0)],
        )

        waypoint = planner.get_next_waypoint(
            RobotPose(1.10, 0.62, 0.0),
            tolerance_m=0.45,
        )

        self.assertIsNotNone(waypoint)
        self.assertEqual(waypoint.name, "__metric_001__")

    def test_metric_corner_does_not_turn_from_logged_radial_tolerance_pose(self):
        planner = GlobalPlanner(_create_test_map())
        planner.install_metric_path_points(
            RobotPose(6.384, 12.656, math.radians(90.0)),
            "C",
            [
                (6.384, 12.656),
                (6.350, 13.450),
                (5.650, 13.470),
                (4.0, 13.50),
            ],
        )

        premature = planner.get_next_waypoint(
            RobotPose(6.54, 13.05, math.radians(96.0)),
            tolerance_m=0.45,
        )
        at_corner = planner.get_next_waypoint(
            RobotPose(6.50, 13.40, math.radians(96.0)),
            tolerance_m=0.45,
        )

        self.assertIsNotNone(premature)
        self.assertEqual(premature.name, "__metric_001__")
        self.assertIsNotNone(at_corner)
        self.assertEqual(at_corner.name, "__metric_002__")

    def test_metric_corner_advances_after_entering_outgoing_corridor(self):
        planner = GlobalPlanner(_create_test_map())
        planner.install_metric_path_points(
            RobotPose(6.31, 12.82, math.radians(95.0)),
            "C",
            [
                (6.31, 12.82),
                (6.25, 13.55),
                (5.57, 13.55),
                (4.89, 13.55),
                (4.0, 13.55),
            ],
        )

        waypoint = planner.get_next_waypoint(
            RobotPose(5.49, 13.60, math.radians(93.0)),
            tolerance_m=0.45,
        )

        self.assertIsNotNone(waypoint)
        self.assertEqual(waypoint.name, "__metric_003__")
        self.assertAlmostEqual(
            abs(planner.current_segment_heading()),
            math.pi,
        )

    def test_metric_corner_rejects_straight_overshoot_outside_outgoing_corridor(self):
        planner = GlobalPlanner(_create_test_map())
        planner.install_metric_path_points(
            RobotPose(6.31, 12.82, math.radians(95.0)),
            "C",
            [
                (6.31, 12.82),
                (6.25, 13.55),
                (5.57, 13.55),
                (4.0, 13.55),
            ],
        )

        waypoint = planner.get_next_waypoint(
            RobotPose(6.80, 14.20, math.radians(95.0)),
            tolerance_m=0.45,
        )

        self.assertIsNotNone(waypoint)
        self.assertEqual(waypoint.name, "__metric_001__")

    def test_metric_corner_rejects_pose_on_wrong_side_of_outgoing_segment(self):
        planner = GlobalPlanner(_create_test_map())
        planner.install_metric_path_points(
            RobotPose(6.31, 12.82, math.radians(95.0)),
            "C",
            [
                (6.31, 12.82),
                (6.25, 13.55),
                (5.57, 13.55),
                (4.0, 13.55),
            ],
        )

        waypoint = planner.get_next_waypoint(
            RobotPose(6.80, 13.60, math.radians(95.0)),
            tolerance_m=0.45,
        )

        self.assertIsNotNone(waypoint)
        self.assertEqual(waypoint.name, "__metric_001__")

    def test_current_segment_reports_signed_cross_track_error(self):
        planner = GlobalPlanner(_create_test_map())
        planner.install_metric_path_points(
            RobotPose(0.0, 0.0, 0.0),
            "C",
            [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)],
        )

        self.assertAlmostEqual(
            planner.current_segment_cross_track_error(
                RobotPose(0.5, 0.40, 0.0)
            ),
            0.40,
        )

    def test_replan_does_not_reinstate_completed_waypoint(self):
        topo = _create_test_map()
        planner = GlobalPlanner(topo)
        planner.plan_path(RobotPose(0, 0, 0), "C")
        planner.advance_current_waypoint()
        self.assertEqual(planner.get_current_waypoint().name, "C")

        planner.replan_path_preserving_progress(RobotPose(0, 0, 0), "C")

        self.assertEqual(planner.get_current_waypoint().name, "C")


# ---------------------------------------------------------------------------
# TopologicalMap tests
# ---------------------------------------------------------------------------

class TestTopologicalMap(unittest.TestCase):

    def test_load_from_dict(self):
        topo = _create_test_map()
        self.assertTrue(topo.is_loaded)
        self.assertEqual(len(topo.nodes), 3)
        self.assertEqual(len(topo.edges), 2)

    def test_default_map_loads_required_metric_occupancy(self):
        topo = TopologicalMap()

        loaded = topo.load_from_file(str(CAIL_LAB_MAP_FILE))

        self.assertTrue(loaded)
        self.assertIsNotNone(topo.metric_map)
        self.assertGreater(topo.metric_map.occupied.size, 100_000)

    def test_charging_pose_is_not_enclosed_by_its_map_annotation(self):
        topo = TopologicalMap()
        self.assertTrue(topo.load_from_file(str(CAIL_LAB_MAP_FILE)))
        charging = topo.get_node("charging_station")
        immersive = topo.get_node("immersive_room")

        start_cell = topo.metric_map.world_to_cell(charging.x, charging.y)
        self.assertGreater(float(topo.metric_map.distance_m[start_cell]), 0.75)
        path = topo.metric_map.plan_path(
            (charging.x, charging.y),
            (immersive.x, immersive.y),
        )

        self.assertIsNotNone(path)
        first_bearing = math.atan2(
            path[1][1] - charging.y,
            path[1][0] - charging.x,
        )
        heading = math.radians(charging.heading_degrees)
        self.assertLess(
            abs(math.atan2(math.sin(first_bearing - heading), math.cos(first_bearing - heading))),
            math.radians(45.0),
        )

    def test_default_map_blocks_false_marker_4_to_f_core_shortcut(self):
        topo = TopologicalMap()
        self.assertTrue(topo.load_from_file(str(CAIL_LAB_MAP_FILE)))
        marker_4 = topo.get_node("entrance")
        immersive = topo.get_node("immersive_room")

        path = topo.metric_map.plan_path(
            (marker_4.x, marker_4.y),
            (immersive.x, immersive.y),
        )

        self.assertIsNotNone(path)
        self.assertFalse(
            any(
                9.9 <= x <= 16.9 and 10.0 <= y <= 30.2
                for x, y in path
            )
        )

    def test_default_map_routes_clear_of_fixed_furniture(self):
        topo = TopologicalMap()
        self.assertTrue(topo.load_from_file(str(CAIL_LAB_MAP_FILE)))
        charging = topo.get_node("charging_station")
        immersive = topo.get_node("immersive_room")
        furniture = {
            "conference table beyond B": (15.5, 17.9, 1.3, 5.5),
            "kitchen table beside marker 0": (5.20, 8.40, 1.30, 4.70),
            "south kitchen table": (2.60, 5.60, 0.00, 3.10),
            "desk cluster near marker 9": (1.30, 5.70, 3.30, 7.35),
            "desk cluster near marker 7": (1.20, 5.70, 8.30, 12.70),
            "desk cluster near marker 2": (1.20, 5.70, 14.40, 18.00),
            "desk cluster near marker 5": (0.60, 5.70, 23.70, 28.30),
            "desk cluster near marker M": (0.50, 5.70, 28.50, 33.70),
            "gathering-area tables": (3.00, 6.40, 38.60, 42.50),
            "immersive flower desk": (21.00, 25.90, 7.60, 11.80),
            "immersive lounge tables": (19.00, 26.30, 0.80, 6.60),
            "ping-pong table": (17.7, 20.6, 6.8, 8.7),
            "immersive-room half-circle desk": (18.8, 25.5, 13.2, 17.5),
        }

        path = topo.metric_map.plan_path(
            (charging.x, charging.y),
            (immersive.x, immersive.y),
        )

        self.assertIsNotNone(path)
        for name, (x0, x1, y0, y1) in furniture.items():
            center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            row, col = topo.metric_map.world_to_cell(*center)
            self.assertTrue(topo.metric_map.occupied[row, col], name)
            margin = 0.25
            self.assertFalse(
                any(
                    x0 - margin <= x <= x1 + margin
                    and y0 - margin <= y <= y1 + margin
                    for x, y in path
                ),
                name,
            )

    def test_loads_map_declared_arrival_landmark(self):
        topo = TopologicalMap()
        topo.load_from_dict({
            "nodes": [{
                "name": "turn",
                "x": 1.0,
                "y": 0.0,
                "arrival_landmarks": [{
                    "type": "wall",
                    "approach_from": ["start"],
                    "max_pose_error_m": 0.8,
                }],
            }],
        })

        landmark = topo.nodes["turn"].arrival_landmarks[0]
        self.assertEqual(landmark["type"], "wall")
        self.assertEqual(landmark["approach_from"], ["start"])
        self.assertAlmostEqual(landmark["max_pose_error_m"], 0.8)

    def test_loads_map_declared_arrival_regions(self):
        topo = TopologicalMap()
        topo.load_from_dict({
            "nodes": [{
                "name": "kitchen_entrance",
                "x": 1.0,
                "y": 0.0,
                "arrival_tolerance_m": 0.65,
                "pass_through_tolerance_m": 0.75,
            }],
        })

        node = topo.nodes["kitchen_entrance"]
        self.assertAlmostEqual(node.arrival_tolerance_m, 0.65)
        self.assertAlmostEqual(node.pass_through_tolerance_m, 0.75)

    def test_find_nearest_node(self):
        topo = _create_test_map()
        node = topo.find_nearest_node(2.8, 0.1)
        self.assertEqual(node.name, "B")

    def test_list_destinations(self):
        topo = _create_test_map()
        names = topo.list_destinations()
        self.assertEqual(names, ["A", "B", "C"])

    def test_get_node(self):
        topo = _create_test_map()
        self.assertIsNotNone(topo.get_node("A"))
        self.assertIsNone(topo.get_node("Z"))

    def test_get_node_accepts_natural_language_aliases(self):
        topo = TopologicalMap()
        topo.load_from_dict({
            "name": "suite21",
            "nodes": [
                {
                    "name": "ai_hall_of_fame",
                    "x": 0,
                    "y": 0,
                    "description": "AI hall of fame, red marker 3",
                },
                {
                    "name": "shrushtis_desk",
                    "x": 1,
                    "y": 0,
                    "description": "Shrushti's desk, red marker 2",
                    "aliases": [
                        "Shush desk",
                        "Shush this desk",
                        "Shushdi Fest",
                        "Srushti desk",
                        "Srishti desk",
                    ],
                },
                {
                    "name": "charging_station",
                    "x": -1,
                    "y": 0,
                    "description": "Charging station, red marker 1",
                },
            ],
            "edges": [],
        })

        self.assertEqual(topo.get_node("AI Hall of Fame").name, "ai_hall_of_fame")
        self.assertEqual(topo.get_node("Shrushti's desk").name, "shrushtis_desk")
        self.assertEqual(topo.get_node("Shushti's desk").name, "shrushtis_desk")
        self.assertEqual(topo.get_node("Srushti's desk").name, "shrushtis_desk")
        self.assertEqual(topo.get_node("Srishti's desk").name, "shrushtis_desk")
        self.assertEqual(topo.get_node("Xuxi's desk").name, "shrushtis_desk")
        self.assertEqual(topo.get_node("Shushti's death").name, "shrushtis_desk")
        self.assertEqual(topo.get_node("shush desk").name, "shrushtis_desk")
        self.assertEqual(topo.get_node("Shush this desk").name, "shrushtis_desk")
        self.assertEqual(topo.get_node("Shushdi Fest").name, "shrushtis_desk")
        self.assertEqual(topo.get_node("base").name, "charging_station")

    def test_load_from_file(self):
        import tempfile
        import json

        data = {
            "name": "temp",
            "nodes": [{"name": "X", "x": 0, "y": 0}],
            "edges": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            topo = TopologicalMap()
            result = topo.load_from_file(path)
            self.assertTrue(result)
            self.assertEqual(len(topo.nodes), 1)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# OdometryProvider tests
# ---------------------------------------------------------------------------

class TestOdometryProvider(unittest.TestCase):

    @patch("coded_tools.unigo2.nav_core.SdkSportModeOdometryProvider")
    def test_sdk_odometry_factory_defaults_to_eth0_interface(self, mock_sdk_provider):
        provider = MagicMock()
        provider.subscriber_error = None
        mock_sdk_provider.return_value = provider

        with patch.dict(os.environ, {}, clear=True):
            self.assertIs(_create_odometry_provider(), provider)

        mock_sdk_provider.assert_called_once_with(
            topic="rt/sportmodestate",
            network_interface="eth0",
        )

    def test_initial_pose_is_zero(self):
        odom = OdometryProvider()
        pose = odom.get_pose()
        self.assertAlmostEqual(pose.x, 0.0)
        self.assertAlmostEqual(pose.y, 0.0)
        self.assertAlmostEqual(pose.yaw, 0.0)

    def test_dead_reckoning_forward(self):
        odom = OdometryProvider()
        cmd = VelocityCommand(vx=1.0, vy=0.0, vyaw=0.0)
        odom.update_from_velocity(cmd, dt=1.0)

        pose = odom.get_pose()
        self.assertAlmostEqual(pose.x, 1.0, places=2)
        self.assertAlmostEqual(pose.y, 0.0, places=2)

    def test_dead_reckoning_rotation(self):
        odom = OdometryProvider()
        cmd = VelocityCommand(vx=0.0, vy=0.0, vyaw=math.pi / 2)
        odom.update_from_velocity(cmd, dt=1.0)

        pose = odom.get_pose()
        self.assertAlmostEqual(pose.yaw, math.pi / 2, places=2)

    def test_dead_reckoning_lateral_motion(self):
        odom = OdometryProvider()
        odom.update_from_velocity(VelocityCommand(vy=0.2), dt=1.0)

        pose = odom.get_pose()
        self.assertAlmostEqual(pose.x, 0.0, places=2)
        self.assertAlmostEqual(pose.y, 0.2, places=2)

    def test_reset(self):
        odom = OdometryProvider()
        odom.update_from_velocity(VelocityCommand(vx=1.0), dt=1.0)
        odom.reset()
        pose = odom.get_pose()
        self.assertAlmostEqual(pose.x, 0.0)

    def test_sdk_odometry_aligns_measured_yaw_to_map_anchor(self):
        odom = SdkSportModeOdometryProvider(start_subscriber=False)
        odom.set_pose(5.62, 15.70, math.radians(0.0))

        odom._handle_sample(SimpleNamespace(
            position=[10.0, 20.0, 0.0],
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, math.radians(30.0)]),
        ))
        odom._handle_sample(SimpleNamespace(
            position=[10.0, 20.0, 0.0],
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, math.radians(-60.0)]),
        ))

        pose = odom.get_pose()
        self.assertAlmostEqual(pose.x, 5.62, places=2)
        self.assertAlmostEqual(pose.y, 15.70, places=2)
        self.assertAlmostEqual(pose.yaw, math.radians(-90.0), places=2)

    def test_static_fresh_sdk_sample_does_not_overwrite_fallback_motion(self):
        odom = SdkSportModeOdometryProvider(start_subscriber=False)
        odom.set_pose(0.0, 0.0, 0.0)
        sample = SimpleNamespace(
            position=[0.0, 0.0, 0.0],
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
        )
        odom._handle_sample(sample)

        odom.update_from_velocity(VelocityCommand(vx=1.0), dt=1.0)
        odom._handle_sample(sample)

        pose = odom.get_pose()
        self.assertAlmostEqual(pose.x, 1.0, places=2)
        self.assertFalse(odom._sdk_translation_confirmed)

    def test_yaw_only_sdk_motion_keeps_command_integrated_translation(self):
        odom = SdkSportModeOdometryProvider(start_subscriber=False)
        odom.set_pose(0.0, 0.0, 0.0)
        odom._handle_sample(SimpleNamespace(
            position=[0.0, 0.0, 0.0],
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
        ))
        odom._handle_sample(SimpleNamespace(
            position=[0.0, 0.0, 0.0],
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.1]),
        ))

        odom.update_from_velocity(VelocityCommand(vx=1.0), dt=1.0)

        pose = odom.get_pose()
        self.assertFalse(odom._sdk_translation_confirmed)
        self.assertTrue(odom._sdk_yaw_confirmed)
        self.assertAlmostEqual(math.hypot(pose.x, pose.y), 1.0, places=2)
        self.assertAlmostEqual(pose.yaw, 0.1, places=2)

    def test_sdk_translation_is_enabled_by_default(self):
        odom = SdkSportModeOdometryProvider(start_subscriber=False)
        odom.set_pose(5.0, 10.0, 0.0)
        odom._handle_sample(SimpleNamespace(
            position=[0.0, 0.0, 0.0],
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
        ))
        odom._handle_sample(SimpleNamespace(
            position=[1.0, 0.0, 0.0],
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
        ))

        pose = odom.get_pose()
        self.assertTrue(odom._sdk_translation_confirmed)
        self.assertAlmostEqual(pose.x, 6.0, places=2)
        self.assertAlmostEqual(pose.y, 10.0, places=2)

    def test_metric_correction_preserves_confirmed_sdk_odometry(self):
        odom = SdkSportModeOdometryProvider(start_subscriber=False)
        odom.set_pose(5.0, 10.0, 0.0)
        odom._handle_sample(SimpleNamespace(
            position=[0.0, 0.0, 0.0],
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
        ))
        odom._handle_sample(SimpleNamespace(
            position=[1.0, 0.0, 0.0],
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.1]),
        ))

        odom.apply_pose_correction(6.1, 10.1, 0.08)

        self.assertTrue(odom._sdk_translation_confirmed)
        self.assertTrue(odom._sdk_yaw_confirmed)
        self.assertTrue(odom.has_confirmed_translation())
        pose = odom.get_pose()
        self.assertAlmostEqual(pose.x, 6.1, places=2)
        self.assertAlmostEqual(pose.y, 10.1, places=2)

    def test_sdk_translation_can_be_disabled_explicitly(self):
        with patch.dict(os.environ, {"NAV_USE_SDK_TRANSLATION_ODOMETRY": "0"}):
            odom = SdkSportModeOdometryProvider(start_subscriber=False)
        odom.set_pose(5.0, 10.0, 0.0)
        odom._handle_sample(SimpleNamespace(
            position=[0.0, 0.0, 0.0],
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
        ))
        odom._handle_sample(SimpleNamespace(
            position=[1.0, 0.0, 0.0],
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
        ))

        pose = odom.get_pose()
        self.assertFalse(odom._sdk_translation_confirmed)
        self.assertAlmostEqual(pose.x, 5.0, places=2)
        self.assertAlmostEqual(pose.y, 10.0, places=2)

    def test_wireless_controller_activity_has_release_grace(self):
        odom = SdkSportModeOdometryProvider(start_subscriber=False)
        odom._handle_remote_sample(
            SimpleNamespace(lx=0.0, ly=0.2, rx=0.0, ry=0.0, keys=0)
        )
        self.assertTrue(odom.is_manual_control_active())

        with odom._lock:
            odom._manual_control_until = time.monotonic() - 0.01
        self.assertFalse(odom.is_manual_control_active())

    def test_sdk_odometry_falls_back_when_sample_is_stale(self):
        odom = SdkSportModeOdometryProvider(start_subscriber=False)
        odom.set_pose(0.0, 0.0, 0.0)
        odom._handle_sample(SimpleNamespace(
            position=[0.0, 0.0, 0.0],
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
        ))

        with odom._lock:
            x, y, yaw, _timestamp = odom._latest_sdk_pose
            odom._latest_sdk_pose = (x, y, yaw, time.monotonic() - 10.0)

        odom.update_from_velocity(VelocityCommand(vx=1.0), dt=1.0)
        pose = odom.get_pose()
        self.assertAlmostEqual(pose.x, 1.0, places=2)


# ---------------------------------------------------------------------------
# NavCore integration (lightweight, no hardware)
# ---------------------------------------------------------------------------

class TestNavCoreStatus(unittest.TestCase):

    def test_metric_arrival_requires_repeated_scan_to_map_consistency(self):
        core = NavCore.__new__(NavCore)
        metric_map = MagicMock()
        metric_map.pose_consistency.return_value = (0.05, 0.80)
        core._topo_map = SimpleNamespace(metric_map=metric_map)
        core._arrival_map_consistency_readings = 0
        core.ARRIVAL_MAP_MAX_SCORE_M = 0.16
        core.ARRIVAL_MAP_MIN_MATCHED_FRACTION = 0.35
        core.ARRIVAL_MAP_CONFIRM_READINGS = 2
        grid = _grid_with_wall_right(0.8)
        pose = RobotPose(6.75, 5.75, math.radians(-90.0))

        first = core._metric_arrival_sensor_is_confirmed(grid, pose)
        second = core._metric_arrival_sensor_is_confirmed(grid, pose)

        self.assertFalse(first)
        self.assertTrue(second)
        self.assertEqual(metric_map.pose_consistency.call_count, 2)

        metric_map.pose_consistency.return_value = (0.40, 0.05)
        self.assertFalse(core._metric_arrival_sensor_is_confirmed(grid, pose))
        self.assertEqual(core._arrival_map_consistency_readings, 0)

    def test_parallel_wall_reanchors_bad_initial_route_heading_after_confirmation(self):
        core = NavCore.__new__(NavCore)
        core._topo_map = _create_test_map()
        core._global_planner = GlobalPlanner(core._topo_map)
        core._global_planner.install_metric_path_points(
            RobotPose(0.0, 0.0, math.radians(26.0)),
            "C",
            [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)],
        )
        core._local_planner = LocalPlanner()
        core._odometry = MagicMock()
        core._route_wall_heading_candidate = None
        core._route_wall_heading_readings = 0
        grid = replace(
            _grid_with_wall_right(0.8),
            path_obstacle_m=float("inf"),
            path_obstacle_bearing=0.0,
        )
        pose = RobotPose(0.0, 0.0, math.radians(26.0))

        for _ in range(core.ROUTE_WALL_HEADING_CONFIRM_READINGS):
            pose = core._maybe_align_heading_to_route_wall(grid, pose)

        self.assertAlmostEqual(pose.yaw, 0.0, places=2)
        core._odometry.apply_pose_correction.assert_called_once()

    def test_parallel_wall_does_not_reanchor_on_final_destination_segment(self):
        core = NavCore.__new__(NavCore)
        core._topo_map = _create_test_map()
        core._global_planner = GlobalPlanner(core._topo_map)
        core._global_planner.install_metric_path_points(
            RobotPose(0.0, 0.0, math.radians(14.0)),
            "C",
            [(0.0, 0.0), (4.0, 0.0)],
        )
        core._local_planner = LocalPlanner()
        core._odometry = MagicMock()
        core._route_wall_heading_candidate = None
        core._route_wall_heading_readings = 0
        grid = replace(
            _grid_with_wall_right(0.8),
            path_obstacle_m=float("inf"),
            path_obstacle_bearing=0.0,
        )
        pose = RobotPose(3.0, 0.0, math.radians(14.0))

        for _ in range(core.ROUTE_WALL_HEADING_CONFIRM_READINGS):
            pose = core._maybe_align_heading_to_route_wall(grid, pose)

        self.assertAlmostEqual(pose.yaw, math.radians(14.0), places=2)
        core._odometry.apply_pose_correction.assert_not_called()

    def test_parallel_wall_does_not_reanchor_late_metric_transit_segment(self):
        core = NavCore.__new__(NavCore)
        core._topo_map = _create_test_map()
        core._global_planner = GlobalPlanner(core._topo_map)
        core._global_planner.install_metric_path_points(
            RobotPose(0.0, 0.0, math.radians(20.0)),
            "C",
            [(float(index), 0.0) for index in range(7)],
        )
        for _ in range(3):
            core._global_planner.advance_current_waypoint()
        self.assertEqual(
            core._global_planner.get_current_waypoint().name,
            "__metric_004__",
        )
        core._local_planner = LocalPlanner()
        core._odometry = MagicMock()
        core._route_wall_heading_candidate = None
        core._route_wall_heading_readings = 0
        grid = replace(
            _grid_with_wall_right(0.8),
            path_obstacle_m=float("inf"),
            path_obstacle_bearing=0.0,
        )
        pose = RobotPose(4.0, 0.0, math.radians(20.0))

        for _ in range(core.ROUTE_WALL_HEADING_CONFIRM_READINGS):
            pose = core._maybe_align_heading_to_route_wall(grid, pose)

        self.assertAlmostEqual(pose.yaw, math.radians(20.0), places=2)
        core._odometry.apply_pose_correction.assert_not_called()

    def test_expected_transverse_wall_advances_metric_corner_early(self):
        core = NavCore.__new__(NavCore)
        core._topo_map = _create_test_map()
        core._global_planner = GlobalPlanner(core._topo_map)
        core._global_planner.install_metric_path_points(
            RobotPose(0.0, 0.0, 0.0),
            "C",
            [(0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (1.0, -2.0)],
        )
        core._local_planner = LocalPlanner()
        core._odometry = MagicMock()
        core._notify_status_change = MagicMock()
        core._reset_progress_tracker = MagicMock()

        corrected, accepted = core._accept_expected_metric_corner(
            RobotPose(0.4, 0.0, 0.0),
            _grid_with_wall_ahead(1.0),
        )

        self.assertTrue(accepted)
        self.assertAlmostEqual(corrected.x, 1.0)
        self.assertEqual(
            core._global_planner.get_current_waypoint().name,
            "__metric_002__",
        )
        core._odometry.apply_pose_correction.assert_called_once()

    def test_mapped_clear_segment_keeps_moving_past_safe_side_reading(self):
        core = NavCore.__new__(NavCore)
        core._topo_map = _create_test_map()
        core._global_planner = GlobalPlanner(core._topo_map)
        core._global_planner.install_metric_path_points(
            RobotPose(0.0, 0.0, 0.0),
            "C",
            [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)],
        )
        core._local_planner = LocalPlanner()
        grid = replace(
            _empty_grid(),
            path_obstacle_m=0.55,
            path_obstacle_bearing=math.radians(-45.0),
            path_obstacle_points=20,
        )

        self.assertTrue(
            core._parallel_wall_projection_is_clear(
                grid,
                RobotPose(0.5, 0.0, 0.0),
            )
        )

        close_grid = replace(grid, path_obstacle_m=0.35)
        self.assertFalse(
            core._parallel_wall_projection_is_clear(
                close_grid,
                RobotPose(0.5, 0.0, 0.0),
            )
        )

    def test_metric_transit_does_not_emit_agent_status_events(self):
        core = NavCore.__new__(NavCore)
        core._topo_map = _create_test_map()
        core._local_planner = MagicMock()
        core._notify_status_change = MagicMock()
        transit = MapNode(
            name="__metric_001__",
            x=1.0,
            y=1.0,
            tags=["metric_transit"],
        )

        core._global_planner = MagicMock()
        core._global_planner.current_turn_change.return_value = 0.0
        core._notify_waypoint_advance(transit, core._topo_map.nodes["B"])

        core._local_planner.reset_navigation_state.assert_not_called()
        core._notify_status_change.assert_not_called()

    def test_persistent_obstacle_schedules_background_metric_route(self):
        core = NavCore.__new__(NavCore)
        metric_map = MagicMock()
        metric_map.robot_points_to_world.return_value = np.asarray([[1.0, 0.0]])
        safe_goal = MapNode(name="kitchen", x=3.8, y=1.2)
        core._topo_map = SimpleNamespace(
            metric_map=metric_map,
            get_node=MagicMock(return_value=safe_goal),
        )
        core._global_planner = MagicMock()
        core._global_planner.get_current_waypoint.return_value = MapNode(
            name="__metric_001__", x=2.0, y=0.0
        )
        core._global_planner.remaining_metric_route.return_value = (4.0, 0.0)
        core._local_planner = MagicMock()
        core._reset_progress_tracker = MagicMock()
        core._metric_replan_executor = MagicMock()
        pending = Future()
        core._metric_replan_executor.submit.return_value = pending
        core._metric_replan_future = None
        core._metric_replan_context = None
        core._path_obstacle_active = True
        core._path_obstacle_active_since = time.monotonic() - 3.0
        core._path_obstacle_clear_since = None
        core._last_metric_replan_time = 0.0
        core._last_translation_progress_time = time.monotonic() - 3.0
        core._last_motion_command = VelocityCommand(vx=0.2)
        goal = NavGoal(goal_type="semantic", x=4.0, y=1.0, label="Kitchen")
        pose = RobotPose(0.0, 0.0, 0.0)

        replanned = core._maybe_replan_blocked_metric_route(
            goal,
            pose,
            _grid_with_wall_ahead(0.6),
        )

        self.assertTrue(replanned)
        self.assertIs(core._metric_replan_future, pending)
        self.assertEqual((goal.x, goal.y), (4.0, 1.0))
        core._metric_replan_executor.submit.assert_called_once()
        core._local_planner.reset_navigation_state.assert_not_called()

    def test_metric_replan_is_not_scheduled_during_route_pivot(self):
        core = NavCore.__new__(NavCore)
        metric_map = MagicMock()
        core._topo_map = SimpleNamespace(metric_map=metric_map)
        core._global_planner = MagicMock()
        core._global_planner.get_current_waypoint.return_value = MapNode(
            name="__metric_001__", x=0.0, y=2.0
        )
        core._metric_replan_future = None
        core._path_obstacle_active = True
        core._path_obstacle_active_since = time.monotonic() - 3.0
        core._last_metric_replan_time = 0.0
        core._last_translation_progress_time = time.monotonic() - 3.0
        core._last_motion_command = VelocityCommand(vyaw=0.5)

        scheduled = core._maybe_replan_blocked_metric_route(
            NavGoal(goal_type="semantic", x=4.0, y=1.0, label="Kitchen"),
            RobotPose(0.0, 0.0, 0.0),
            _grid_with_wall_ahead(0.6),
        )

        self.assertFalse(scheduled)
        metric_map.robot_points_to_world.assert_not_called()

    def test_remote_control_release_preserves_pose_and_plans_fresh_route(self):
        core = NavCore.__new__(NavCore)
        core._manual_override_active = False
        core._state_lock = threading.Lock()
        core._topo_map = _create_test_map()
        core._global_planner = GlobalPlanner(core._topo_map)
        core._global_planner.plan_path(RobotPose(0.0, 0.0, 0.0), "C")
        measured_pose = RobotPose(x=1.5, y=0.7, yaw=0.2)
        core._odometry = MagicMock()
        core._odometry.is_manual_control_active.side_effect = [True, False]
        core._odometry.get_pose.return_value = measured_pose
        core._notify_status_change = MagicMock()
        core._reset_progress_tracker = MagicMock()
        core._local_planner = MagicMock()
        goal = NavGoal(x=3.0, y=4.0, goal_type="semantic", label="C")

        self.assertTrue(core._handle_manual_override(goal))
        self.assertFalse(core._handle_manual_override(goal))

        core._odometry.set_pose.assert_not_called()
        core._reset_progress_tracker.assert_called_once_with()
        self.assertIn(
            "preserved the corrected position and heading",
            core._notify_status_change.call_args.args[0],
        )
        self.assertEqual(core._global_planner._current_path[-1].name, "C")

    def test_remote_control_release_does_not_abort_without_route_edge(self):
        core = NavCore.__new__(NavCore)
        core._manual_override_active = True
        core._state_lock = threading.Lock()
        core._topo_map = _create_test_map()
        core._global_planner = GlobalPlanner(core._topo_map)
        core._odometry = MagicMock()
        core._odometry.is_manual_control_active.return_value = False
        core._odometry.get_pose.return_value = RobotPose(1.0, 1.0, 0.4)
        core._notify_status_change = MagicMock()
        core._reset_progress_tracker = MagicMock()
        core._local_planner = MagicMock()
        goal = NavGoal(x=3.0, y=1.0, goal_type="semantic", label="B")

        self.assertFalse(core._handle_manual_override(goal))

        core._odometry.set_pose.assert_not_called()
        self.assertIn(
            "continuing toward B",
            core._notify_status_change.call_args.args[0],
        )

    def test_remote_course_correction_realigns_badly_drifted_map_heading(self):
        core = NavCore.__new__(NavCore)
        core._manual_override_active = False
        core._manual_override_start_pose = None
        core._state_lock = threading.Lock()
        core._global_planner = MagicMock()
        first_target = MapNode(name="__metric_001__", x=1.0, y=-4.0)
        core._global_planner.plan_path.return_value = [
            MapNode(name="start", x=0.0, y=0.0),
            first_target,
            MapNode(name="kitchen", x=2.0, y=-5.0),
        ]
        start_pose = RobotPose(0.0, 0.0, math.radians(-95.0))
        released_pose = RobotPose(0.0, 0.0, math.radians(-6.0))
        core._odometry = MagicMock()
        core._odometry.is_manual_control_active.side_effect = [True, False]
        core._odometry.get_pose.side_effect = [start_pose, released_pose]
        core._notify_status_change = MagicMock()
        core._reset_progress_tracker = MagicMock()
        core._local_planner = MagicMock()
        goal = NavGoal(goal_type="semantic", label="kitchen")

        self.assertTrue(core._handle_manual_override(goal))
        self.assertFalse(core._handle_manual_override(goal))

        expected_yaw = math.atan2(-4.0, 1.0)
        corrected = core._odometry.apply_pose_correction.call_args.args
        self.assertAlmostEqual(corrected[2], expected_yaw)
        self.assertIsNone(core._manual_override_start_pose)

    def test_robot_navigation_defaults_are_in_code(self):
        self.assertAlmostEqual(NavCore.MAX_LINEAR_SPEED, 0.40)
        self.assertAlmostEqual(NavCore.MAX_YAW_RATE, 0.08)
        self.assertAlmostEqual(NavCore.PIVOT_YAW_RATE, 0.50)
        self.assertAlmostEqual(NavCore.SAFETY_DISTANCE_M, 0.20)
        self.assertAlmostEqual(NavCore.AVOIDANCE_DISTANCE_M, 0.75)
        self.assertAlmostEqual(NavCore.PIVOT_HARD_STOP_DISTANCE_M, 0.40)
        self.assertAlmostEqual(NavCore.PIVOT_EMERGENCY_STOP_DISTANCE_M, 0.10)
        self.assertAlmostEqual(NavCore.CLOSE_OBSTACLE_CONFIRM_S, 0.7)
        self.assertEqual(NavCore.CLOSE_OBSTACLE_CONFIRM_READINGS, 6)
        self.assertAlmostEqual(NavCore.PATH_OBSTACLE_CONFIRM_S, 0.3)
        self.assertEqual(NavCore.PATH_OBSTACLE_CONFIRM_READINGS, 3)
        self.assertAlmostEqual(NavCore.PATH_OBSTACLE_CENTER_DEPTH_MARGIN_M, 0.15)
        self.assertAlmostEqual(NavCore.GOAL_TOLERANCE_M, 0.15)
        self.assertAlmostEqual(NavCore.SEMANTIC_ARRIVAL_TOLERANCE_M, 0.65)
        self.assertAlmostEqual(
            NavCore.LOCOMOTION_MIN_VERIFICATION_COMMAND_MPS,
            0.15,
        )
        self.assertAlmostEqual(
            GlobalPlanner.CORNER_METRIC_LONGITUDINAL_TOLERANCE_M,
            0.08,
        )
        self.assertAlmostEqual(
            GlobalPlanner.FINAL_METRIC_STRICT_PROXIMITY_M,
            0.15,
        )
        self.assertAlmostEqual(NavCore.OBSTACLE_GRID_MAX_AGE_S, 0.50)
        self.assertAlmostEqual(NavCore.OBSTACLE_GRID_LOSS_GRACE_S, 3.0)
        self.assertAlmostEqual(NavCore.METRIC_ROUTE_RECAPTURE_CROSS_TRACK_M, 0.35)
        self.assertAlmostEqual(NavCore.METRIC_CORNER_REGRESSION_DISTANCE_M, 0.25)
        self.assertAlmostEqual(NavCore.METRIC_CORNER_REGRESSION_CONFIRM_S, 0.30)
        self.assertAlmostEqual(NavCore.STALL_ESCAPE_SPEED_MPS, 0.15)
        self.assertAlmostEqual(NavCore.STALL_ESCAPE_DISTANCE_M, 0.15)
        self.assertAlmostEqual(NavCore.STALL_ESCAPE_CLEARANCE_M, 0.40)
        self.assertAlmostEqual(NavCore.STALL_ESCAPE_ROBOT_HALF_LENGTH_M, 0.35)
        self.assertAlmostEqual(NavCore.STALL_BACKUP_SPEED_MPS, 0.12)
        self.assertAlmostEqual(NavCore.STALL_BACKUP_DISTANCE_M, 0.15)
        self.assertAlmostEqual(NavCore.STALL_SCAN_YAW_RATE_RPS, 0.35)
        self.assertAlmostEqual(
            NavCore.STALL_SCAN_MIN_ANGLE_RAD,
            math.radians(20.0),
        )
        self.assertAlmostEqual(
            NavCore.STALL_SCAN_MAX_ANGLE_RAD,
            math.radians(50.0),
        )

    def test_terminal_semantic_failure_preserves_destination_for_resume(self):
        nav = NavCore.__new__(NavCore)
        nav._state = NavState.NAVIGATING
        nav._goal = None
        nav._suspended_goal = None
        nav._last_stop_reason = None
        nav._state_lock = threading.Lock()
        nav._go2 = MagicMock(available=True)
        nav._ensure_go2 = MagicMock()
        nav._clear_planner_and_obstacle_state = MagicMock()
        nav._notify_status_change = MagicMock()
        nav._stop_depth_when_idle = MagicMock()
        goal = NavGoal(
            goal_type="semantic",
            x=2.85,
            y=13.57,
            label="charging_station",
        )

        nav._abort_active_navigation(
            goal,
            "E-STOP: path obstacle at 0.18m",
            "I stopped before reaching the charging station.",
            state=NavState.E_STOP,
        )

        self.assertEqual(nav.state, NavState.E_STOP)
        self.assertIsNone(nav._goal)
        self.assertEqual(nav._suspended_goal.label, "charging_station")

    def test_resume_replans_suspended_route_from_latest_measured_pose(self):
        nav = NavCore.__new__(NavCore)
        nav._state = NavState.E_STOP
        nav._goal = None
        nav._suspended_goal = NavGoal(
            goal_type="semantic",
            x=2.85,
            y=13.57,
            label="charging_station",
        )
        nav._last_stop_reason = "E-STOP"
        nav._state_lock = threading.Lock()
        nav._odometry = MagicMock()
        nav._odometry.get_pose.return_value = RobotPose(
            7.10,
            6.20,
            math.radians(90.0),
        )
        nav.navigate_to = MagicMock(return_value=True)
        nav._notify_status_change = MagicMock()

        resumed = nav.resume()

        self.assertTrue(resumed)
        nav.navigate_to.assert_called_once_with("charging_station")
        self.assertIsNone(nav._suspended_goal)
        nav._notify_status_change.assert_called_once()

    def test_stale_obstacle_grid_is_rejected(self):
        core = NavCore.__new__(NavCore)
        grid = _empty_grid()
        grid.timestamp = time.time() - 1.0

        self.assertIsNone(core._fresh_obstacle_grid(grid))

    def test_a_robot_with_no_configured_map_gets_no_map(self):
        # Each site sets NAV_MAP_FILE in its own setmyenv.sh. A robot that has
        # not been told where it lives must say so rather than load some other
        # office and offer destinations that do not exist here.
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_configured_map_file(), "")

    def test_configured_map_file_is_used(self):
        with patch.dict(os.environ, {"NAV_MAP_FILE": "/tmp/bengaluru.json"}, clear=True):
            self.assertEqual(_configured_map_file(), "/tmp/bengaluru.json")

    def test_simulation_mode_disables_the_map(self):
        environment = {"NAV_SIMULATION_MODE": "1"}
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(_configured_map_file(), "")

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_initial_location_missing_from_the_map_is_reported(self, mock_go2):
        import json
        import tempfile

        mock_go2.return_value = MagicMock()
        map_data = {
            "name": "somewhere else",
            "nodes": [{"name": "reception", "x": 0.0, "y": 0.0}],
            "edges": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump(map_data, handle)
            map_path = handle.name

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        os.environ["NAV_MAP_FILE"] = map_path
        os.environ["NAV_INITIAL_LOCATION"] = "charging_station"
        try:
            with self.assertLogs("coded_tools.unigo2.nav_core", level="WARNING") as logs:
                nav = NavCore.get_instance()
            self.assertTrue(
                any("charging_station" in line for line in logs.output),
                f"expected a warning naming the missing node, got {logs.output}",
            )
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)
            os.environ.pop("NAV_MAP_FILE", None)
            os.environ.pop("NAV_INITIAL_LOCATION", None)
            os.unlink(map_path)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_get_status_summary(self, mock_go2):
        mock_go2.return_value = MagicMock()

        # Reset singleton for test isolation
        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            summary = nav.get_status_summary()
            self.assertIn("idle", summary.lower())
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_list_destinations_no_map(self, mock_go2):
        mock_go2.return_value = MagicMock()

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            result = nav.list_destinations()
            self.assertIn("No map loaded", result)
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    def test_navigation_odometry_is_decoupled_from_forward_timing(self):
        self.assertLess(
            NavCore.ODOMETRY_LINEAR_SPEED_RATIO,
            NavCore.FORWARD_ACTUAL_SPEED_RATIO,
        )

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_set_location_anchors_pose_to_map_node(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._topo_map.load_from_dict({
                "name": "suite21",
                "nodes": [
                    {
                        "name": "charging_station",
                        "x": 1.3,
                        "y": 15.7,
                        "description": "Charging station, red marker 1",
                    },
                    {
                        "name": "shrushtis_desk",
                        "x": 5.62,
                        "y": 15.7,
                        "description": "Shrushti's desk, red marker 2",
                    },
                ],
                "edges": [
                    {
                        "from": "charging_station",
                        "to": "shrushtis_desk",
                        "distance": 4.32,
                    },
                ],
            })

            result = nav.set_location("base", heading_rad=0.0)

            self.assertTrue(result)
            pose = nav._odometry.get_pose()
            self.assertAlmostEqual(pose.x, 1.3)
            self.assertAlmostEqual(pose.y, 15.7)
            path = nav._global_planner.plan_path(pose, "Shrushti's desk")
            self.assertEqual(
                [node.name for node in path],
                ["charging_station", "shrushtis_desk"],
            )
            self.assertEqual(nav.state, NavState.IDLE)
            self.assertIn(
                "Current mapped location: Charging station",
                nav.get_status_summary(),
            )
            fake_go2.stop_move.assert_called()
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_map_load_anchors_initial_pose_to_charging_station(self, mock_go2):
        mock_go2.return_value = MagicMock()

        import json
        import tempfile

        map_data = {
            "name": "suite21",
            "nodes": [
                {
                    "name": "charging_station",
                    "x": 1.3,
                    "y": 15.7,
                    "heading_degrees": 28.3,
                    "description": "Charging station, red marker 1",
                },
                {
                    "name": "shrushtis_desk",
                    "x": 5.62,
                    "y": 15.7,
                    "description": "Shrushti's desk, red marker 2",
                },
            ],
            "edges": [
                {
                    "from": "charging_station",
                    "to": "shrushtis_desk",
                    "distance": 4.32,
                },
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(map_data, f)
            map_path = f.name

        NavCore._instance = None
        old_heading = os.environ.pop("NAV_INITIAL_HEADING_DEGREES", None)
        os.environ["NAV_SIMULATION_MODE"] = "1"
        os.environ["NAV_MAP_FILE"] = map_path
        try:
            nav = NavCore.get_instance()
            pose = nav._odometry.get_pose()
            self.assertAlmostEqual(pose.x, 1.3)
            self.assertAlmostEqual(pose.y, 15.7)
            self.assertAlmostEqual(pose.yaw, math.radians(28.3))
            path = nav._global_planner.plan_path(pose, "Shrushti's desk")
            self.assertEqual(
                [node.name for node in path],
                ["charging_station", "shrushtis_desk"],
            )
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)
            os.environ.pop("NAV_MAP_FILE", None)
            if old_heading is not None:
                os.environ["NAV_INITIAL_HEADING_DEGREES"] = old_heading
            os.unlink(map_path)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_scales_dead_reckoning_for_calibrated_motion(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            nav._depth_processor = fake_depth

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0)
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            commanded_vx = fake_go2.move.call_args.kwargs["vx"]
            pose = nav._odometry.get_pose()
            expected_x = commanded_vx * nav.ODOMETRY_LINEAR_SPEED_RATIO / nav.NAV_LOOP_HZ
            self.assertAlmostEqual(pose.x, expected_x, places=4)
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_status_reports_estop_obstacle_reason(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        original_confirm_s = NavCore.CLOSE_OBSTACLE_CONFIRM_S
        NavCore.CLOSE_OBSTACLE_CONFIRM_S = 0.0
        events = []

        def record_event(message):
            events.append(message)

        NavCore.set_status_callback(record_event)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _grid_with_wall_ahead(distance_m=0.08)
            nav._depth_processor = fake_depth

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0)
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            for _ in range(NavCore.CLOSE_OBSTACLE_CONFIRM_READINGS):
                nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.E_STOP)
            self.assertIn("E-STOP: path obstacle at 0.08m", nav.get_status_summary())
            self.assertEqual(
                events,
                [
                    "I stopped before reaching the destination because my depth sensor "
                    "reported something in my path at 0.08 meters."
                ],
            )
            fake_go2.stop_move.assert_called()
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore.CLOSE_OBSTACLE_CONFIRM_S = original_confirm_s
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_semantic_close_obstacle_attempts_recovery_before_estop(self, mock_go2):
        fake_go2 = MagicMock(available=True)
        mock_go2.return_value = fake_go2
        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        original_confirm_s = NavCore.CLOSE_OBSTACLE_CONFIRM_S
        NavCore.CLOSE_OBSTACLE_CONFIRM_S = 0.0
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            blocked = _grid_with_wall_ahead(distance_m=0.08)
            nav._depth_processor = MagicMock()
            nav._depth_processor.get_obstacle_grid.return_value = blocked
            nav._filter_transient_path_obstacle = lambda grid: grid
            nav._global_planner = MagicMock()
            nav._global_planner.get_next_waypoint.return_value = MapNode(
                name="__metric_001__",
                x=2.0,
                y=0.0,
                tags=["metric_transit"],
            )
            nav._global_planner.current_segment.return_value = None
            nav._local_planner = MagicMock()
            nav._local_planner.compute_velocity.return_value = VelocityCommand(vx=0.2)
            nav._local_planner.apply_corridor_course_correction.side_effect = (
                lambda cmd, _grid, **_kwargs: cmd
            )
            nav._recover_from_stall = MagicMock(return_value=True)
            goal = NavGoal(
                goal_type="semantic",
                x=2.0,
                y=0.0,
                label="Immersive Room",
            )
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            for _ in range(NavCore.CLOSE_OBSTACLE_CONFIRM_READINGS):
                nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            self.assertIs(nav._goal, goal)
            nav._recover_from_stall.assert_called_once()
            fake_go2.stop_move.assert_called()
            nav.shutdown()
        finally:
            NavCore.CLOSE_OBSTACLE_CONFIRM_S = original_confirm_s
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_defers_transient_close_obstacle_before_estop(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.side_effect = [
                _grid_with_wall_ahead(distance_m=0.08),
                _empty_grid(),
            ]
            nav._depth_processor = fake_depth

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Shrushti's desk")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            fake_go2.stop_move.assert_called_once()
            fake_go2.move.assert_not_called()
            self.assertEqual(events, [])

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            fake_go2.move.assert_called()
            self.assertGreater(fake_go2.move.call_args.kwargs["vx"], 0.0)
            self.assertEqual(events, [])
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_ignores_transient_avoidance_path_obstacle(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.side_effect = [
                _grid_with_wall_ahead(distance_m=0.25),
                _empty_grid(),
            ]
            nav._depth_processor = fake_depth

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            fake_go2.move.assert_called()
            self.assertGreater(fake_go2.move.call_args.kwargs["vx"], 0.0)
            self.assertLess(
                fake_go2.move.call_args.kwargs["vx"],
                nav.MAX_LINEAR_SPEED,
            )
            self.assertEqual(events, [])

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            self.assertEqual(events, [])
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_requires_persistent_avoidance_path_obstacle_before_crawl(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        original_confirm_s = NavCore.PATH_OBSTACLE_CONFIRM_S
        original_confirm_readings = NavCore.PATH_OBSTACLE_CONFIRM_READINGS
        original_clear_confirm_s = NavCore.PATH_OBSTACLE_CLEAR_CONFIRM_S
        NavCore.PATH_OBSTACLE_CONFIRM_S = 0.0
        NavCore.PATH_OBSTACLE_CONFIRM_READINGS = 3
        NavCore.PATH_OBSTACLE_CLEAR_CONFIRM_S = 0.0
        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _grid_with_wall_ahead(distance_m=0.25)
            nav._depth_processor = fake_depth
            nav._local_planner = MagicMock()
            nav._local_planner.compute_velocity.return_value = VelocityCommand(
                vx=0.10,
                vy=0.0,
                vyaw=0.0,
            )

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)
            first_grid = nav._local_planner.compute_velocity.call_args.args[0]
            self.assertEqual(first_grid.path_obstacle_m, float("inf"))

            nav._nav_cycle(NavState.NAVIGATING, goal)
            second_grid = nav._local_planner.compute_velocity.call_args.args[0]
            self.assertEqual(second_grid.path_obstacle_m, float("inf"))

            nav._nav_cycle(NavState.NAVIGATING, goal)
            third_grid = nav._local_planner.compute_velocity.call_args.args[0]
            self.assertAlmostEqual(third_grid.path_obstacle_m, 0.25)
            self.assertEqual(len(events), 1)
            self.assertIn("encountered an obstacle", events[0])

            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            nav._nav_cycle(NavState.NAVIGATING, goal)
            nav._nav_cycle(NavState.NAVIGATING, goal)
            self.assertEqual(len(events), 2)
            self.assertIn("path is clear again", events[1])

            self.assertEqual(nav.state, NavState.NAVIGATING)
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore.PATH_OBSTACLE_CONFIRM_S = original_confirm_s
            NavCore.PATH_OBSTACLE_CONFIRM_READINGS = original_confirm_readings
            NavCore.PATH_OBSTACLE_CLEAR_CONFIRM_S = original_clear_confirm_s
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_rejects_uncorroborated_avoidance_path_obstacle(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        original_confirm_s = NavCore.PATH_OBSTACLE_CONFIRM_S
        original_confirm_readings = NavCore.PATH_OBSTACLE_CONFIRM_READINGS
        NavCore.PATH_OBSTACLE_CONFIRM_S = 0.0
        NavCore.PATH_OBSTACLE_CONFIRM_READINGS = 1
        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _grid_with_wall_ahead(distance_m=0.25)
            fake_depth.get_center_depth_reading.return_value = CenterDepthReading(
                distance_m=2.0,
                coverage=0.5,
            )
            nav._depth_processor = fake_depth

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            fake_go2.move.assert_called()
            self.assertGreater(fake_go2.move.call_args.kwargs["vx"], 0.0)
            self.assertEqual(events, [])
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore.PATH_OBSTACLE_CONFIRM_S = original_confirm_s
            NavCore.PATH_OBSTACLE_CONFIRM_READINGS = original_confirm_readings
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_ignores_repeated_false_17cm_path_metadata(self, mock_go2):
        fake_go2 = MagicMock(available=True)
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            false_close = replace(
                _empty_grid(),
                nearest_obstacle_m=0.70,
                nearest_obstacle_bearing=math.radians(-45.0),
                path_obstacle_m=0.17,
                path_obstacle_bearing=math.radians(-15.0),
                path_obstacle_points=6,
            )
            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = false_close
            fake_depth.get_center_depth_reading.return_value = CenterDepthReading(
                distance_m=1.50,
                coverage=0.5,
            )
            nav._depth_processor = fake_depth

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            for _ in range(10):
                nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            self.assertIs(nav._goal, goal)
            self.assertGreaterEqual(fake_go2.move.call_count, 1)
            fake_go2.stop_move.assert_not_called()
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_moves_past_close_side_obstacle(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _grid_with_side_obstacle(distance_m=0.37)
            nav._depth_processor = fake_depth

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Shrushti's desk")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            fake_go2.move.assert_called()
            self.assertGreater(fake_go2.move.call_args.kwargs["vx"], 0.0)
            fake_go2.stop_move.assert_not_called()
            self.assertEqual(events, [])
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_ignores_close_nearest_when_path_corridor_is_clear(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            grid = _empty_grid()
            grid.nearest_obstacle_m = 0.19
            grid.nearest_obstacle_bearing = math.radians(35.0)
            grid.path_obstacle_m = float("inf")
            grid.path_obstacle_points = 0

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = grid
            nav._depth_processor = fake_depth

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Shrushti's desk")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            fake_go2.move.assert_called()
            self.assertGreater(fake_go2.move.call_args.kwargs["vx"], 0.0)
            fake_go2.stop_move.assert_not_called()
            self.assertEqual(events, [])
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_reports_robot_control_unavailable(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = False
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            nav._depth_processor = fake_depth

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Shrushti's desk")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.E_STOP)
            self.assertIn("Robot control unavailable", nav.get_status_summary())
            self.assertEqual(
                events,
                [
                    "I did not move toward Shrushti's desk because robot motor "
                    "control is unavailable."
                ],
            )
            fake_go2.move.assert_not_called()
            fake_depth.stop.assert_called()
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_waits_for_transient_obstacle_grid_dropout(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = None
            nav._depth_processor = fake_depth
            nav._local_planner = MagicMock()
            nav._local_planner.compute_velocity.return_value = VelocityCommand(vx=0.2)
            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            self.assertIs(nav._goal, goal)
            fake_go2.stop_move.assert_called()
            self.assertIn("stopped safely", events[0])

            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            fake_go2.move.assert_called()
            self.assertIn("obstacle view recovered", events[1])
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_aborts_after_obstacle_grid_grace_period(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = None
            nav._depth_processor = fake_depth
            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)
            nav._obstacle_grid_unavailable_since = (
                time.monotonic() - nav.OBSTACLE_GRID_LOSS_GRACE_S - 0.1
            )
            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.E_STOP)
            self.assertIsNone(nav._goal)
            self.assertIn("did not recover", events[-1])
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_holds_pivot_when_leg_sweep_is_near_obstacle(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _grid_with_wall_ahead(distance_m=0.25)
            nav._depth_processor = fake_depth

            goal = NavGoal(goal_type="relative", x=0.0, y=2.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            fake_go2.move.assert_not_called()
            fake_go2.stop_move.assert_called_once()
            self.assertEqual(events, [])
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_redirects_pivot_away_from_close_corner(self, mock_go2):
        fake_go2 = MagicMock(available=True)
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            close_right = _grid_with_obstacle_at_bearing(
                distance_m=0.18,
                bearing_rad=math.radians(-20.0),
            )
            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = close_right
            nav._depth_processor = fake_depth
            nav._robust_pivot_clearance = MagicMock(
                return_value=(0.18, math.radians(-20.0))
            )
            goal = NavGoal(goal_type="relative", x=-2.0, y=0.0)
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            move = fake_go2.move.call_args.kwargs
            self.assertAlmostEqual(move["vx"], 0.0)
            self.assertGreater(move["vyaw"], 0.0)
        finally:
            NavCore._instance.shutdown()
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_mapped_wall_landmark_corrects_pose_and_pivots_to_next_waypoint(self, mock_go2):
        fake_go2 = MagicMock(available=True)
        mock_go2.return_value = fake_go2

        original_confirm_s = NavCore.PATH_OBSTACLE_CONFIRM_S
        original_confirm_readings = NavCore.PATH_OBSTACLE_CONFIRM_READINGS
        NavCore.PATH_OBSTACLE_CONFIRM_S = 0.0
        NavCore.PATH_OBSTACLE_CONFIRM_READINGS = 1
        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            nav._topo_map.load_from_dict({
                "name": "landmark route",
                "nodes": [
                    {"name": "start", "x": 0.0, "y": 0.0},
                    {
                        "name": "turn",
                        "x": 3.0,
                        "y": 0.0,
                        "arrival_landmarks": [{
                            "type": "wall",
                            "approach_from": ["start"],
                            "max_pose_error_m": 1.0,
                            "max_lateral_error_m": 0.5,
                            "max_detection_distance_m": 0.3,
                            "min_span_m": 0.5,
                        }],
                    },
                    {"name": "goal", "x": 3.0, "y": 4.0},
                ],
                "edges": [
                    {"from": "start", "to": "turn", "distance": 3.0},
                    {"from": "turn", "to": "goal", "distance": 4.0},
                ],
            })
            path = nav._global_planner.plan_path(RobotPose(0.0, 0.0, 0.0), "goal")
            nav._odometry.set_pose(2.3, 0.2, 0.0)

            wall = _grid_with_wall_ahead(distance_m=0.25)
            fake_depth = MagicMock(supports_center_depth=True)
            fake_depth.get_obstacle_grid.return_value = wall
            fake_depth.get_center_depth_reading.return_value = CenterDepthReading(
                distance_m=0.25,
                coverage=0.5,
            )
            nav._depth_processor = fake_depth

            goal = NavGoal(
                goal_type="semantic",
                x=path[-1].x,
                y=path[-1].y,
                label="goal",
            )
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            pose = nav._odometry.get_pose()
            self.assertAlmostEqual(pose.x, 3.0, places=2)
            self.assertAlmostEqual(pose.y, 0.2, places=2)
            fake_go2.move.assert_not_called()
            fake_go2.stop_move.assert_called_once()
            self.assertTrue(any("reached turn" in event for event in events))
            self.assertEqual(nav._global_planner.current_segment()[1].name, "goal")
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore.PATH_OBSTACLE_CONFIRM_S = original_confirm_s
            NavCore.PATH_OBSTACLE_CONFIRM_READINGS = original_confirm_readings
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_mapped_wall_outside_correction_bound_remains_an_obstacle(self, mock_go2):
        fake_go2 = MagicMock(available=True)
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            nav._topo_map.load_from_dict({
                "nodes": [
                    {"name": "start", "x": 0.0, "y": 0.0},
                    {
                        "name": "turn",
                        "x": 3.0,
                        "y": 0.0,
                        "arrival_landmarks": [{
                            "type": "wall",
                            "approach_from": ["start"],
                            "max_pose_error_m": 1.0,
                        }],
                    },
                    {"name": "goal", "x": 3.0, "y": 4.0},
                ],
                "edges": [
                    {"from": "start", "to": "turn", "distance": 3.0},
                    {"from": "turn", "to": "goal", "distance": 4.0},
                ],
            })
            pose = RobotPose(1.5, 0.0, 0.0)
            nav._odometry.set_pose(pose.x, pose.y, pose.yaw)
            nav._global_planner.plan_path(RobotPose(0.0, 0.0, 0.0), "goal")
            waypoint = nav._global_planner.get_next_waypoint(pose)

            corrected, upcoming, accepted = nav._accept_expected_landmark(
                pose,
                _grid_with_wall_ahead(distance_m=0.25),
                waypoint,
            )

            self.assertFalse(accepted)
            self.assertEqual(upcoming.name, "turn")
            self.assertEqual(corrected, pose)
            self.assertEqual(nav._global_planner.current_segment()[1].name, "turn")
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_pivots_for_right_angle_mapped_waypoint(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            nav._local_planner = LocalPlanner(
                max_linear_speed=0.40,
                max_yaw_rate=0.08,
                pivot_yaw_rate=0.50,
                safety_distance=0.10,
                avoidance_distance=0.30,
            )
            nav._topo_map.load_from_dict({
                "name": "suite21",
                "nodes": [
                    {"name": "shrushtis_desk", "x": 5.62, "y": 15.70},
                    {"name": "wellness_room", "x": 5.62, "y": 11.59},
                    {"name": "kitchen", "x": 5.62, "y": 4.11},
                ],
                "edges": [
                    {"from": "shrushtis_desk", "to": "wellness_room", "distance": 4.11},
                    {"from": "wellness_room", "to": "kitchen", "distance": 7.48},
                ],
            })
            nav._odometry.set_pose(5.62, 15.70, 0.0)
            path = nav._global_planner.plan_path(nav._odometry.get_pose(), "kitchen")

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            nav._depth_processor = fake_depth

            goal = NavGoal(goal_type="semantic", x=path[-1].x, y=path[-1].y, label="kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            fake_go2.move.assert_called_once()
            self.assertAlmostEqual(fake_go2.move.call_args.kwargs["vx"], 0.0)
            self.assertLess(fake_go2.move.call_args.kwargs["vyaw"], 0.0)
            self.assertAlmostEqual(abs(fake_go2.move.call_args.kwargs["vyaw"]), 0.50)
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_counts_yaw_as_progress_during_planned_pivot(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            nav._depth_processor = fake_depth
            nav._odometry.set_pose(0.0, 0.0, math.radians(10.0))
            escape_pose = RobotPose(0.0, 0.15, math.radians(10.0))
            nav._execute_stall_escape = MagicMock(
                return_value=(escape_pose, "left")
            )
            nav._execute_stall_turn_scan = MagicMock(
                return_value=(escape_pose, "right")
            )

            goal = NavGoal(goal_type="relative", x=0.0, y=2.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._last_progress_pose = RobotPose(0.0, 0.0, 0.0)
                nav._last_progress_time = time.monotonic() - nav.STUCK_TIMEOUT_S - 1.0

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            fake_go2.move.assert_called()
            self.assertAlmostEqual(fake_go2.move.call_args.kwargs["vx"], 0.0)
            self.assertGreater(fake_go2.move.call_args.kwargs["vyaw"], 0.0)
            self.assertEqual(events, [])
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_measured_pivot_progress_does_not_trigger_translation_stall(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            nav._depth_processor = fake_depth
            nav._odometry.set_pose(0.0, 0.0, math.radians(10.0))
            escape_pose = RobotPose(0.0, 0.15, math.radians(10.0))
            nav._execute_stall_escape = MagicMock(
                return_value=(escape_pose, "left")
            )
            nav._execute_stall_turn_scan = MagicMock(
                return_value=(escape_pose, "right")
            )

            goal = NavGoal(goal_type="relative", x=0.0, y=2.0, label="Kitchen")
            stale_time = time.monotonic() - nav.STUCK_TIMEOUT_S - 1.0
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._last_progress_pose = RobotPose(0.0, 0.0, 0.0)
                nav._last_progress_time = stale_time
                nav._last_translation_progress_pose = nav._odometry.get_pose()
                nav._last_translation_progress_time = stale_time

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            self.assertEqual(events, [])
            fake_go2.move.assert_called()
            self.assertAlmostEqual(fake_go2.move.call_args.kwargs["vx"], 0.0)
            self.assertGreater(fake_go2.move.call_args.kwargs["vyaw"], 0.0)
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    def test_path_obstacle_clear_event_requires_stable_clearance(self):
        nav = NavCore.__new__(NavCore)
        nav._path_obstacle_active = False
        nav._path_obstacle_clear_since = None
        nav._on_status_change = MagicMock()
        goal = NavGoal(label="Kitchen")

        with patch(
            "coded_tools.unigo2.nav_core.time.monotonic",
            side_effect=[0.0, 0.1, 0.2, 0.3, 1.2, 1.4],
        ):
            nav._update_path_obstacle_event(0.5, goal)
            nav._update_path_obstacle_event(float("inf"), goal)
            nav._update_path_obstacle_event(0.5, goal)
            nav._update_path_obstacle_event(float("inf"), goal)
            nav._update_path_obstacle_event(float("inf"), goal)
            nav._update_path_obstacle_event(float("inf"), goal)

        self.assertEqual(nav._on_status_change.call_count, 2)
        self.assertIn(
            "encountered an obstacle",
            nav._on_status_change.call_args_list[0].args[0],
        )
        self.assertIn(
            "path is clear again",
            nav._on_status_change.call_args_list[1].args[0],
        )

    def test_stall_escape_moves_away_from_right_obstacle(self):
        nav = NavCore.__new__(NavCore)
        nav.STALL_ESCAPE_CLEARANCE_M = 0.40
        grid = _grid_with_obstacle_at_bearing(
            distance_m=0.50,
            bearing_rad=math.radians(-30.0),
        )
        grid.path_obstacle_m = 0.50
        grid.path_obstacle_bearing = math.radians(-30.0)

        self.assertEqual(nav._choose_stall_escape_direction(grid), 1.0)

    def test_stall_escape_randomizes_when_both_sides_are_clear(self):
        nav = NavCore.__new__(NavCore)
        nav.STALL_ESCAPE_CLEARANCE_M = 0.40
        grid = _empty_grid()

        with patch(
            "coded_tools.unigo2.nav_core.random.choice",
            return_value=-1.0,
        ) as choose:
            direction = nav._choose_stall_escape_direction(grid)

        self.assertEqual(direction, -1.0)
        choose.assert_called_once()

    def test_front_wall_does_not_falsely_block_lateral_escape(self):
        nav = NavCore.__new__(NavCore)
        nav.STALL_ESCAPE_CLEARANCE_M = 0.40
        nav.STALL_ESCAPE_ROBOT_HALF_LENGTH_M = 0.35
        grid = _grid_with_wall_ahead(distance_m=0.50)

        with patch(
            "coded_tools.unigo2.nav_core.random.choice",
            return_value=1.0,
        ):
            direction = nav._choose_stall_escape_direction(grid)

        self.assertEqual(direction, 1.0)

    def test_stall_escape_refuses_when_neither_side_is_safe(self):
        nav = NavCore.__new__(NavCore)
        nav.STALL_ESCAPE_CLEARANCE_M = 0.40
        nav.STALL_ESCAPE_ROBOT_HALF_LENGTH_M = 0.35
        grid = _empty_grid()
        row = grid.origin_row - 5
        grid.grid[row, grid.origin_col - 4] = 1.0
        grid.grid[row, grid.origin_col + 4] = 1.0

        self.assertIsNone(nav._choose_stall_escape_direction(grid))

    def test_stall_escape_backs_up_and_reroutes_when_sides_stay_blocked(self):
        nav = NavCore.__new__(NavCore)
        goal = NavGoal(goal_type="semantic", label="Kitchen")
        grid = _empty_grid()
        backed_pose = RobotPose(-0.15, 0.0, 0.0)
        nav._choose_stall_escape_direction = MagicMock(side_effect=[None, None])
        nav._execute_stall_backup = MagicMock(return_value=backed_pose)
        nav._depth_processor = MagicMock()
        nav._depth_processor.get_obstacle_grid.return_value = grid

        result = nav._execute_stall_escape(goal, grid)

        self.assertEqual(result, (backed_pose, "backward"))
        nav._execute_stall_backup.assert_called_once_with(goal, grid)
        self.assertEqual(nav._choose_stall_escape_direction.call_count, 2)

    def test_stall_escape_rechecks_sides_after_backing_up(self):
        nav = NavCore.__new__(NavCore)
        goal = NavGoal(goal_type="semantic", label="Kitchen")
        grid = _empty_grid()
        backed_pose = RobotPose(-0.15, 0.0, 0.0)
        lateral_pose = RobotPose(-0.15, 0.15, 0.0)
        nav._choose_stall_escape_direction = MagicMock(side_effect=[None, 1.0])
        nav._execute_stall_backup = MagicMock(return_value=backed_pose)
        nav._execute_lateral_stall_escape = MagicMock(
            return_value=(lateral_pose, "left")
        )
        nav._depth_processor = MagicMock()
        nav._depth_processor.get_obstacle_grid.return_value = grid

        result = nav._execute_stall_escape(goal, grid)

        self.assertEqual(result, (lateral_pose, "backward then left"))
        nav._execute_lateral_stall_escape.assert_called_once_with(goal, 1.0)

    def test_stall_backup_commands_short_reverse_motion(self):
        nav = NavCore.__new__(NavCore)
        nav.STALL_BACKUP_SPEED_MPS = 0.12
        nav.STALL_BACKUP_DISTANCE_M = 0.15
        nav.STALL_BACKUP_CLEARANCE_DROP_M = 0.08
        nav.STALL_ESCAPE_COMMAND_PERIOD_S = 0.10
        nav._go2 = MagicMock(available=True)
        nav._ensure_go2 = MagicMock()
        nav._odometry = OdometryProvider()
        grid = _grid_with_wall_ahead(distance_m=0.50)
        nav._depth_processor = MagicMock()
        nav._depth_processor.get_obstacle_grid.return_value = grid
        goal = NavGoal(goal_type="semantic", label="Kitchen")

        with (
            patch("coded_tools.unigo2.nav_core.random.uniform", return_value=1.0),
            patch(
                "coded_tools.unigo2.nav_core.time.monotonic",
                side_effect=[0.0, 0.1, 2.0],
            ),
            patch("coded_tools.unigo2.nav_core.time.sleep"),
        ):
            pose = nav._execute_stall_backup(goal, grid)

        nav._go2.move.assert_called_once_with(vx=-0.12, vy=0.0, vyaw=0.0)
        nav._go2.stop_move.assert_called_once()
        self.assertLess(pose.x, 0.0)

    def test_stall_turn_scan_alternates_after_failed_recovery(self):
        nav = NavCore.__new__(NavCore)
        nav._last_stall_scan_direction = None
        grid = _grid_with_obstacle_at_bearing(
            distance_m=0.50,
            bearing_rad=math.radians(30.0),
        )

        first = nav._choose_stall_scan_direction(grid)
        nav._last_stall_scan_direction = first
        second = nav._choose_stall_scan_direction(grid)

        self.assertEqual(first, -1.0)
        self.assertEqual(second, 1.0)

    def test_stall_turn_scan_rotates_to_inspect_opening(self):
        nav = NavCore.__new__(NavCore)
        nav.STALL_SCAN_YAW_RATE_RPS = 0.35
        nav.STALL_SCAN_MIN_ANGLE_RAD = 0.03
        nav.STALL_SCAN_MAX_ANGLE_RAD = 0.03
        nav.STALL_ESCAPE_COMMAND_PERIOD_S = 0.10
        nav.AVOIDANCE_DISTANCE_M = 0.75
        nav._last_stall_scan_direction = None
        nav._go2 = MagicMock(available=True)
        nav._ensure_go2 = MagicMock()
        nav._odometry = OdometryProvider()
        nav._choose_stall_scan_direction = MagicMock(return_value=1.0)
        nav._depth_processor = MagicMock()
        nav._depth_processor.get_obstacle_grid.return_value = _empty_grid()
        goal = NavGoal(goal_type="semantic", label="Kitchen")

        with patch("coded_tools.unigo2.nav_core.time.sleep"):
            pose, direction = nav._execute_stall_turn_scan(
                goal,
                _empty_grid(),
            )

        nav._go2.move.assert_called_once_with(vx=0.0, vy=0.0, vyaw=0.35)
        nav._go2.stop_move.assert_called_once()
        self.assertEqual(direction, "left")
        self.assertGreater(pose.yaw, 0.0)

    def test_failed_stall_escape_pauses_and_retries_instead_of_aborting(self):
        nav = NavCore.__new__(NavCore)
        nav.MAX_STUCK_RECOVERY_ATTEMPTS = 2
        nav._stuck_recovery_attempts = 0
        nav._state = NavState.NAVIGATING
        nav._state_lock = threading.Lock()
        nav._execute_stall_escape = MagicMock(return_value=None)
        nav._execute_stall_turn_scan = MagicMock(return_value=None)
        nav._depth_processor = MagicMock()
        nav._depth_processor.get_obstacle_grid.return_value = blocked_grid = _empty_grid()
        nav._reset_progress_tracker = MagicMock()
        nav._notify_status_change = MagicMock()
        goal = NavGoal(goal_type="semantic", label="Kitchen")

        blocked_grid.path_obstacle_m = 0.50
        recovered = nav._recover_from_stall(goal, RobotPose(), blocked_grid)

        self.assertTrue(recovered)
        self.assertEqual(nav._stuck_recovery_attempts, 1)
        nav._reset_progress_tracker.assert_called_once_with(
            reset_recovery_attempts=False
        )
        self.assertIn(
            "pause and try again",
            nav._notify_status_change.call_args.args[0],
        )

    def test_failed_translation_escape_turns_and_reroutes(self):
        nav = NavCore.__new__(NavCore)
        nav.MAX_STUCK_RECOVERY_ATTEMPTS = 2
        nav._stuck_recovery_attempts = 0
        nav._state = NavState.NAVIGATING
        nav._state_lock = threading.Lock()
        nav._last_stop_reason = None
        nav._path_obstacle_active = True
        nav._path_obstacle_active_since = 1.0
        nav._path_obstacle_clear_since = None
        nav._execute_stall_escape = MagicMock(return_value=None)
        turned_pose = RobotPose(1.0, 2.0, math.radians(-35.0))
        nav._execute_stall_turn_scan = MagicMock(
            return_value=(turned_pose, "right")
        )
        blocked_grid = _empty_grid()
        blocked_grid.path_obstacle_m = 0.20
        nav._depth_processor = MagicMock()
        nav._depth_processor.get_obstacle_grid.return_value = blocked_grid
        nav._fresh_obstacle_grid = MagicMock(return_value=blocked_grid)
        nav._global_planner = MagicMock()
        nav._global_planner.get_current_waypoint.return_value = MapNode(
            name="goal", x=3.0, y=2.0
        )
        nav._go2 = MagicMock(available=True)
        nav._ensure_go2 = MagicMock()
        nav._local_planner = MagicMock()
        nav._reset_progress_tracker = MagicMock()
        nav._notify_status_change = MagicMock()
        goal = NavGoal(goal_type="relative", x=3.0, y=2.0, label="Kitchen")

        recovered = nav._recover_from_stall(goal, RobotPose(), blocked_grid)

        self.assertTrue(recovered)
        self.assertEqual(nav._stuck_recovery_attempts, 1)
        nav._execute_stall_turn_scan.assert_called_once_with(goal, blocked_grid)
        self.assertIn(
            "turned right, rerouted",
            nav._notify_status_change.call_args.args[0],
        )

    def test_clear_stall_near_transit_point_advances_without_escape_or_replan(self):
        nav = NavCore.__new__(NavCore)
        nav.MAX_STUCK_RECOVERY_ATTEMPTS = 2
        nav.CLEAR_STALL_TRANSIT_SKIP_M = 0.65
        nav.AVOIDANCE_DISTANCE_M = 0.75
        nav._stuck_recovery_attempts = 0
        nav._clear_motion_recovery_attempts = 0
        nav._state = NavState.NAVIGATING
        nav._state_lock = threading.Lock()
        nav._go2 = MagicMock(available=True)
        nav._go2.recover_locomotion.return_value = True
        nav._ensure_go2 = MagicMock()
        nav._local_planner = MagicMock()
        nav._reset_progress_tracker = MagicMock()
        nav._notify_status_change = MagicMock()
        nav._notify_waypoint_advance = MagicMock()
        nav._execute_stall_escape = MagicMock()
        transit = MapNode(
            name="__metric_003__",
            x=0.55,
            y=0.0,
            tags=["metric_transit"],
        )
        upcoming = MapNode(
            name="__metric_004__",
            x=1.35,
            y=0.0,
            tags=["metric_transit"],
        )
        nav._global_planner = MagicMock()
        nav._global_planner.get_current_waypoint.return_value = transit
        nav._global_planner.advance_current_waypoint.return_value = (
            transit,
            upcoming,
        )

        recovered = nav._recover_from_stall(
            NavGoal(goal_type="semantic", label="immersive_room"),
            RobotPose(),
            _empty_grid(),
        )

        self.assertTrue(recovered)
        nav._execute_stall_escape.assert_not_called()
        nav._global_planner.advance_current_waypoint.assert_called_once()
        nav._local_planner.reset_navigation_state.assert_called_once()
        self.assertEqual(nav._stuck_recovery_attempts, 0)
        self.assertEqual(nav._clear_motion_recovery_attempts, 1)
        nav._go2.recover_locomotion.assert_called_once()

    def test_rotation_does_not_reset_stall_recovery_attempts(self):
        nav = NavCore.__new__(NavCore)
        nav._last_progress_pose = RobotPose(0.0, 0.0, 0.0)
        nav._last_translation_progress_pose = RobotPose(0.0, 0.0, 0.0)
        nav._last_progress_time = 0.0
        nav._last_translation_progress_time = 0.0
        nav._stuck_recovery_attempts = 1

        nav._update_progress(RobotPose(0.0, 0.0, math.radians(10.0)))
        self.assertEqual(nav._stuck_recovery_attempts, 1)

        nav._update_progress(RobotPose(0.2, 0.0, math.radians(10.0)))
        self.assertEqual(nav._stuck_recovery_attempts, 0)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_stuck_replans_and_keeps_goal(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            blocked_grid = _empty_grid()
            blocked_grid.path_obstacle_m = 0.50
            fake_depth.get_obstacle_grid.return_value = blocked_grid
            nav._depth_processor = fake_depth
            nav._filter_transient_path_obstacle = lambda grid: grid
            nav._local_planner = MagicMock()
            nav._local_planner.compute_velocity.return_value = VelocityCommand(vx=0.2)
            nav._local_planner.apply_corridor_course_correction.side_effect = (
                lambda cmd, _grid, **_kwargs: cmd
            )
            waypoint = MapNode(name="kitchen", x=2.0, y=0.0)
            nav._global_planner = MagicMock()
            nav._global_planner.get_next_waypoint.return_value = waypoint
            nav._global_planner.current_segment.return_value = None
            nav._global_planner.replan_path_preserving_progress.return_value = [
                waypoint
            ]
            escape_pose = RobotPose(0.0, 0.15, 0.0)
            scan_pose = RobotPose(0.0, 0.15, math.radians(-30.0))
            nav._execute_stall_escape = MagicMock(
                return_value=(escape_pose, "left")
            )
            nav._execute_stall_turn_scan = MagicMock(
                return_value=(scan_pose, "right")
            )

            goal = NavGoal(goal_type="semantic", x=2.0, y=0.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._last_progress_pose = nav._odometry.get_pose()
                nav._last_progress_time = time.monotonic() - nav.STUCK_TIMEOUT_S - 1.0
                nav._last_translation_progress_pose = nav._odometry.get_pose()
                nav._last_translation_progress_time = (
                    time.monotonic() - nav.STUCK_TIMEOUT_S - 1.0
                )

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            self.assertIs(nav._goal, goal)
            self.assertEqual(
                events,
                [
                    "I encountered an obstacle in my path at 0.50 meters while "
                    "heading to Kitchen. My local planner is navigating around it.",
                    "I stalled while heading to Kitchen. I stepped left, turned "
                    "right, rerouted, and am continuing."
                ],
            )
            fake_go2.stop_move.assert_called()
            fake_go2.move.assert_not_called()
            fake_depth.stop.assert_not_called()
            nav._global_planner.replan_path_preserving_progress.assert_called_once_with(
                scan_pose,
                "Kitchen",
            )
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_stuck_aborts_after_recovery_limit(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            blocked_grid = _empty_grid()
            blocked_grid.path_obstacle_m = 0.50
            fake_depth.get_obstacle_grid.return_value = blocked_grid
            nav._depth_processor = fake_depth
            nav._filter_transient_path_obstacle = lambda grid: grid
            nav._local_planner = MagicMock()
            nav._local_planner.compute_velocity.return_value = VelocityCommand(vx=0.2)

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._stuck_recovery_attempts = nav.MAX_STUCK_RECOVERY_ATTEMPTS
                nav._last_progress_pose = nav._odometry.get_pose()
                nav._last_progress_time = time.monotonic() - nav.STUCK_TIMEOUT_S - 1.0
                nav._last_translation_progress_pose = nav._odometry.get_pose()
                nav._last_translation_progress_time = (
                    time.monotonic() - nav.STUCK_TIMEOUT_S - 1.0
                )

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.STUCK)
            self.assertIsNone(nav._goal)
            self.assertEqual(
                events,
                [
                    "I encountered an obstacle in my path at 0.50 meters while "
                    "heading to Kitchen. My local planner is navigating around it.",
                    "I stopped before reaching Kitchen because I was not making progress.",
                ],
            )
            fake_depth.stop.assert_called()
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_clear_path_motion_failure_never_uses_obstacle_recovery(self, mock_go2):
        fake_go2 = MagicMock(available=True)
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            nav._depth_processor = fake_depth
            nav._local_planner = MagicMock()
            nav._local_planner.compute_velocity.return_value = VelocityCommand(vx=0.2)
            nav._execute_stall_escape = MagicMock()

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            stale_time = time.monotonic() - nav.STUCK_TIMEOUT_S - 1.0
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._clear_motion_recovery_attempts = (
                    nav.MAX_CLEAR_MOTION_RECOVERY_ATTEMPTS
                )
                nav._last_progress_time = stale_time
                nav._last_translation_progress_time = stale_time

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.STUCK)
            nav._execute_stall_escape.assert_not_called()
            self.assertEqual(
                events,
                [
                    "I stopped before reaching Kitchen because the path was clear, "
                    "but locomotion recovery could not be verified. locomotion "
                    "recovery completed, but subsequent commands still produced no "
                    "measured translation"
                ],
            )
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_clear_path_motion_watchdog_recovers_before_general_stall_timeout(
        self,
        mock_go2,
    ):
        fake_go2 = MagicMock(available=True)
        fake_go2.recover_locomotion.return_value = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            nav._depth_processor = fake_depth
            nav._local_planner = MagicMock()
            # The logged final approach used 0.18m/s. It is meaningful motion
            # and must verify locomotion rather than falling into obstacle
            # escape/replanning merely because it is below the old 0.20 limit.
            nav._local_planner.compute_velocity.return_value = VelocityCommand(
                vx=0.18,
            )
            nav._execute_stall_escape = MagicMock()

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._last_translation_progress_time = (
                    time.monotonic() - nav.CLEAR_MOTION_ACK_TIMEOUT_S - 0.1
                )

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            fake_go2.recover_locomotion.assert_called_once()
            nav._execute_stall_escape.assert_not_called()
            self.assertEqual(nav._clear_motion_recovery_attempts, 1)

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertGreaterEqual(
                fake_go2.move.call_args.kwargs["vx"],
                nav.LOCOMOTION_VERIFICATION_SPEED_MPS,
            )
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    def test_second_clear_motion_recovery_reinitializes_client_and_preserves_route(self):
        nav = NavCore.__new__(NavCore)
        nav.MAX_CLEAR_MOTION_RECOVERY_ATTEMPTS = 2
        nav.AVOIDANCE_DISTANCE_M = 0.75
        nav.CLEAR_STALL_TRANSIT_SKIP_M = 0.65
        nav._clear_motion_recovery_attempts = 1
        nav._stuck_recovery_attempts = 0
        nav._locomotion_recovery_verification_pending = (
            "soft locomotion-mode reset"
        )
        nav._locomotion_recovery_error = None
        nav._state = NavState.NAVIGATING
        nav._state_lock = threading.Lock()
        nav._last_stop_reason = None
        nav._last_motion_command = VelocityCommand(vx=0.2)
        nav._go2 = MagicMock(available=True)
        nav._go2.reinitialize_locomotion.return_value = True
        nav._ensure_go2 = MagicMock()
        nav._local_planner = MagicMock()
        nav._reset_progress_tracker = MagicMock()
        nav._notify_status_change = MagicMock()
        destination = MapNode(name="kitchen", x=2.0, y=0.0)
        nav._global_planner = MagicMock()
        nav._global_planner.get_current_waypoint.return_value = destination
        pose = RobotPose(1.0, 0.2, 0.3)

        recovered = nav._recover_from_stall(
            NavGoal(goal_type="semantic", x=2.0, y=0.0, label="Kitchen"),
            pose,
            _empty_grid(),
        )

        self.assertTrue(recovered)
        nav._go2.reinitialize_locomotion.assert_called_once()
        nav._go2.recover_locomotion.assert_not_called()
        nav._global_planner.clear.assert_not_called()
        nav._global_planner.advance_current_waypoint.assert_not_called()
        self.assertIs(nav._global_planner.get_current_waypoint(), destination)
        self.assertEqual(nav._clear_motion_recovery_attempts, 2)
        self.assertEqual(
            nav._locomotion_recovery_verification_pending,
            "SportClient reinitialization",
        )

    def test_obstacle_recovery_never_reinitializes_locomotion_on_filtered_clear_grid(self):
        nav = NavCore.__new__(NavCore)
        nav.AVOIDANCE_DISTANCE_M = 0.75
        nav.MAX_STUCK_RECOVERY_ATTEMPTS = 2
        nav._stuck_recovery_attempts = 0
        nav._clear_motion_recovery_attempts = 0
        nav._state = NavState.NAVIGATING
        nav._state_lock = threading.Lock()
        nav._go2 = MagicMock(available=True)
        nav._ensure_go2 = MagicMock()
        nav._execute_stall_escape = MagicMock(return_value=None)
        nav._execute_stall_turn_scan = MagicMock(
            return_value=(RobotPose(0.0, 0.0, 0.2), "left")
        )
        nav._global_planner = MagicMock()
        nav._global_planner.get_current_waypoint.return_value = MapNode(
            name="__metric_001__", x=1.0, y=0.0, tags=["metric_transit"]
        )
        nav._global_planner.replan_path_preserving_progress.return_value = [
            MapNode(name="Kitchen", x=2.0, y=0.0)
        ]
        nav._topo_map = MagicMock(metric_map=None)
        nav._depth_processor = MagicMock()
        nav._fresh_obstacle_grid = MagicMock(return_value=_empty_grid())
        nav._local_planner = MagicMock()
        nav._reset_progress_tracker = MagicMock()
        nav._notify_status_change = MagicMock()

        recovered = nav._recover_from_stall(
            NavGoal(goal_type="semantic", label="Kitchen"),
            RobotPose(),
            _empty_grid(),
            allow_locomotion_recovery=False,
        )

        self.assertTrue(recovered)
        nav._go2.recover_locomotion.assert_not_called()
        nav._go2.reinitialize_locomotion.assert_not_called()
        nav._execute_stall_escape.assert_called_once()
        nav._execute_stall_turn_scan.assert_called_once()

    def test_sparse_center_depth_does_not_corroborate_close_grid_artifact(self):
        nav = NavCore.__new__(NavCore)
        nav.AVOIDANCE_DISTANCE_M = 0.75
        nav.PATH_OBSTACLE_CENTER_DEPTH_MARGIN_M = 0.15
        nav.PATH_OBSTACLE_MIN_CENTER_COVERAGE = 0.06
        nav._read_center_depth = MagicMock(
            return_value=(
                CenterDepthReading(distance_m=0.19, coverage=0.025),
                True,
            )
        )

        self.assertFalse(nav._path_obstacle_matches_center_depth(0.19))

        nav._read_center_depth.return_value = (
            CenterDepthReading(distance_m=0.19, coverage=0.20),
            True,
        )
        self.assertTrue(nav._path_obstacle_matches_center_depth(0.19))

    def test_metric_route_regression_replans_before_running_farther_away(self):
        nav = NavCore.__new__(NavCore)
        nav.METRIC_ROUTE_REGRESSION_DISTANCE_M = 0.65
        nav.METRIC_ROUTE_REGRESSION_CONFIRM_S = 0.0
        nav._route_progress_waypoint_name = None
        nav._route_progress_best_distance = float("inf")
        nav._route_regression_since = None
        nav._go2 = MagicMock(available=True)
        nav._ensure_go2 = MagicMock()
        nav._global_planner = MagicMock()
        nav._global_planner.plan_path.return_value = [
            MapNode(name="start", x=0.0, y=0.0, tags=["metric_transit"]),
            MapNode(name="Kitchen", x=2.0, y=0.0),
        ]
        nav._local_planner = MagicMock()
        nav._reset_progress_tracker = MagicMock()
        waypoint = MapNode(
            name="__metric_008__",
            x=1.0,
            y=0.0,
            tags=["metric_transit"],
        )
        goal = NavGoal(goal_type="semantic", x=2.0, y=0.0, label="Kitchen")

        self.assertFalse(
            nav._recover_regressing_metric_route(
                goal, RobotPose(), waypoint, 0.50
            )
        )
        self.assertFalse(
            nav._recover_regressing_metric_route(
                goal, RobotPose(), waypoint, 1.20
            )
        )
        self.assertTrue(
            nav._recover_regressing_metric_route(
                goal, RobotPose(), waypoint, 1.20
            )
        )

        nav._go2.stop_move.assert_called_once()
        nav._global_planner.plan_path.assert_called_once()
        nav._local_planner.reset_navigation_state.assert_called_once()
        nav._reset_progress_tracker.assert_called_once()

    def test_metric_corner_regression_replans_after_small_overshoot(self):
        nav = NavCore.__new__(NavCore)
        nav.METRIC_ROUTE_REGRESSION_DISTANCE_M = 0.65
        nav.METRIC_ROUTE_REGRESSION_CONFIRM_S = 1.0
        nav.METRIC_CORNER_REGRESSION_DISTANCE_M = 0.25
        nav.METRIC_CORNER_REGRESSION_CONFIRM_S = 0.0
        nav.EXPECTED_CORNER_MIN_TURN_RAD = math.radians(45.0)
        nav._route_progress_waypoint_name = None
        nav._route_progress_best_distance = float("inf")
        nav._route_regression_since = None
        nav._go2 = MagicMock(available=True)
        nav._ensure_go2 = MagicMock()
        nav._global_planner = MagicMock()
        nav._global_planner.active_waypoint_is_corner.return_value = True
        nav._global_planner.plan_path.return_value = [
            MapNode(name="start", x=0.0, y=0.0, tags=["metric_transit"]),
            MapNode(name="Charging station", x=2.0, y=0.0),
        ]
        nav._local_planner = MagicMock()
        nav._reset_progress_tracker = MagicMock()
        waypoint = MapNode(
            name="__metric_010__",
            x=1.0,
            y=0.0,
            tags=["metric_transit"],
        )
        goal = NavGoal(
            goal_type="semantic",
            x=2.0,
            y=0.0,
            label="Charging station",
        )

        self.assertFalse(
            nav._recover_regressing_metric_route(
                goal, RobotPose(), waypoint, 0.75
            )
        )
        self.assertFalse(
            nav._recover_regressing_metric_route(
                goal, RobotPose(), waypoint, 1.01
            )
        )
        self.assertTrue(
            nav._recover_regressing_metric_route(
                goal, RobotPose(), waypoint, 1.01
            )
        )

        nav._go2.stop_move.assert_called_once()
        nav._global_planner.plan_path.assert_called_once()

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_raw_obstacle_cannot_trigger_clear_path_locomotion_reinit(self, mock_go2):
        fake_go2 = MagicMock(available=True)
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            raw_blocked = _grid_with_wall_ahead(distance_m=0.50)
            nav._depth_processor = MagicMock()
            nav._depth_processor.get_obstacle_grid.return_value = raw_blocked
            nav._filter_transient_path_obstacle = MagicMock(
                return_value=_empty_grid()
            )
            nav._local_planner = MagicMock()
            nav._local_planner.compute_velocity.return_value = VelocityCommand(vx=0.20)
            nav._recover_from_stall = MagicMock(return_value=True)
            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            stale = time.monotonic() - nav.STUCK_TIMEOUT_S - 0.1
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._last_progress_time = stale
                nav._last_translation_progress_time = stale

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertFalse(
                nav._recover_from_stall.call_args.kwargs[
                    "allow_locomotion_recovery"
                ]
            )
            fake_go2.recover_locomotion.assert_not_called()
            fake_go2.reinitialize_locomotion.assert_not_called()
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    def test_pivot_clearance_ignores_single_raw_minimum_and_uses_supported_grid(self):
        nav = NavCore.__new__(NavCore)
        nav.PIVOT_CLEARANCE_PERCENTILE = 10.0
        nav.PIVOT_GRID_INFLATION_M = 0.15
        grid = _grid_with_wall_ahead(distance_m=0.50)
        grid.nearest_obstacle_m = 0.19
        grid.nearest_obstacle_bearing = math.radians(55.0)

        clearance, bearing = nav._robust_pivot_clearance(grid)

        self.assertGreater(clearance, 0.40)
        self.assertLess(abs(bearing), math.radians(20.0))

    def test_close_pivot_is_redirected_away_from_observed_corner(self):
        nav = NavCore.__new__(NavCore)
        nav.PIVOT_HARD_STOP_DISTANCE_M = 0.40
        nav.PIVOT_EMERGENCY_STOP_DISTANCE_M = 0.10
        nav._local_planner = LocalPlanner(pivot_yaw_rate=0.5)
        nav._pivot_clearance_escape_active = False
        grid = _grid_with_obstacle_at_bearing(
            distance_m=0.18,
            bearing_rad=math.radians(-20.0),
        )

        redirected, allowed = nav._prepare_close_pivot(
            VelocityCommand(vyaw=-0.5),
            grid,
            clearance_m=0.18,
            obstacle_bearing=math.radians(-20.0),
        )

        self.assertTrue(allowed)
        self.assertGreater(redirected.vyaw, 0.0)
        self.assertEqual(nav._local_planner._pivot_direction, 1)

    def test_measured_translation_verifies_recovery_and_resets_watchdog(self):
        nav = NavCore.__new__(NavCore)
        nav._last_progress_pose = RobotPose()
        nav._last_translation_progress_pose = RobotPose()
        nav._last_progress_time = 0.0
        nav._last_translation_progress_time = 0.0
        nav._stuck_recovery_attempts = 1
        nav._clear_motion_recovery_attempts = 2
        nav._locomotion_recovery_verification_pending = (
            "SportClient reinitialization"
        )
        nav._locomotion_recovery_error = None
        nav._last_stall_scan_direction = None
        nav._notify_status_change = MagicMock()

        nav._update_progress(RobotPose(0.12, 0.0, 0.0))

        self.assertEqual(nav._clear_motion_recovery_attempts, 0)
        self.assertIsNone(nav._locomotion_recovery_verification_pending)
        nav._notify_status_change.assert_called_once_with(
            "Locomotion recovery was verified by measured movement, and I am "
            "continuing the existing route."
        )

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_stuck_abort_keeps_current_pose(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            nav._odometry.set_pose(5.0, 15.7, 0.0)
            goal = NavGoal(goal_type="semantic", x=5.62, y=15.7, label="Shrushti's desk")

            nav._abort_active_navigation(
                goal,
                "Stuck: no progress toward the goal",
                "navigation_status: code=stuck_no_progress",
            )

            pose = nav._odometry.get_pose()
            self.assertAlmostEqual(pose.x, 5.0)
            self.assertAlmostEqual(pose.y, 15.7)
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_reports_no_safe_motion_command(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            nav._depth_processor = fake_depth
            nav._local_planner = MagicMock()
            nav._local_planner.compute_velocity.return_value = VelocityCommand()

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Shrushti's desk")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.STUCK)
            self.assertIn("Blocked: no safe motion command", nav.get_status_summary())
            self.assertEqual(
                events,
                [
                    "I did not move toward Shrushti's desk because my local planner "
                    "could not find a safe motion command."
                ],
            )
            fake_go2.move.assert_not_called()
            fake_go2.stop_move.assert_called()
            fake_depth.stop.assert_called()
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_does_not_abort_for_slowdown_band_obstacle(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        original_confirm_s = NavCore.PATH_OBSTACLE_CONFIRM_S
        original_confirm_readings = NavCore.PATH_OBSTACLE_CONFIRM_READINGS
        NavCore.PATH_OBSTACLE_CONFIRM_S = 0.0
        NavCore.PATH_OBSTACLE_CONFIRM_READINGS = 1
        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []

        def record_event(message):
            events.append(message)

        NavCore.set_status_callback(record_event)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _grid_with_wall_ahead(distance_m=0.25)
            nav._depth_processor = fake_depth
            nav._local_planner = MagicMock()
            nav._local_planner.compute_velocity.return_value = VelocityCommand(
                vx=0.10,
                vy=0.0,
                vyaw=0.0,
            )

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            self.assertEqual(len(events), 1)
            self.assertIn("encountered an obstacle", events[0])
            fake_go2.move.assert_called_once()
            fake_go2.stop_move.assert_not_called()
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore.PATH_OBSTACLE_CONFIRM_S = original_confirm_s
            NavCore.PATH_OBSTACLE_CONFIRM_READINGS = original_confirm_readings
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_center_depth_slows_forward_command_when_grid_misses_wall(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            fake_depth.get_center_depth_reading.return_value = CenterDepthReading(
                distance_m=0.475,
                coverage=0.5,
            )
            nav._depth_processor = fake_depth
            nav._local_planner = MagicMock()
            nav._local_planner.compute_velocity.return_value = VelocityCommand(vx=0.40)

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            fake_go2.move.assert_called_once()
            _, kwargs = fake_go2.move.call_args
            self.assertAlmostEqual(kwargs["vx"], 0.20, places=2)
            self.assertAlmostEqual(kwargs["vyaw"], 0.0, places=2)
            fake_go2.stop_move.assert_not_called()
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    def test_parallel_side_wall_uses_center_depth_for_forward_safety(self):
        nav = NavCore.__new__(NavCore)
        nav.AVOIDANCE_DISTANCE_M = 0.75
        nav._read_center_depth = MagicMock(
            return_value=(CenterDepthReading(distance_m=0.60, coverage=0.5), True)
        )
        nav._reset_center_only_close_confirmation = MagicMock()

        distance, bearing = nav._forward_clearance_for_safety(
            VelocityCommand(vx=0.20),
            path_dist=0.27,
            path_bearing=math.radians(18.0),
            parallel_wall_clear=True,
        )

        self.assertAlmostEqual(distance, 0.60)
        self.assertAlmostEqual(bearing, 0.0)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_center_depth_estops_forward_command_when_grid_misses_close_wall(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        original_confirm_s = NavCore.CLOSE_OBSTACLE_CONFIRM_S
        original_confirm_readings = NavCore.CLOSE_OBSTACLE_CONFIRM_READINGS
        original_center_confirm_s = NavCore.CENTER_ONLY_CLOSE_CONFIRM_S
        original_center_confirm_readings = NavCore.CENTER_ONLY_CLOSE_CONFIRM_READINGS
        NavCore.CLOSE_OBSTACLE_CONFIRM_S = 0.0
        NavCore.CLOSE_OBSTACLE_CONFIRM_READINGS = 1
        NavCore.CENTER_ONLY_CLOSE_CONFIRM_S = 0.0
        NavCore.CENTER_ONLY_CLOSE_CONFIRM_READINGS = 3
        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            fake_depth.get_center_depth_reading.return_value = CenterDepthReading(
                distance_m=0.08,
                coverage=0.5,
            )
            nav._depth_processor = fake_depth
            nav._local_planner = MagicMock()
            nav._local_planner.compute_velocity.return_value = VelocityCommand(vx=0.20)

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            for _ in range(2):
                nav._nav_cycle(NavState.NAVIGATING, goal)
                self.assertEqual(nav.state, NavState.NAVIGATING)

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.E_STOP)
            self.assertIsNone(nav._goal)
            self.assertEqual(
                events,
                [
                    "I stopped before reaching Kitchen because my depth sensor "
                    "reported something in my path at 0.08 meters."
                ],
            )
            fake_go2.move.assert_not_called()
            fake_go2.stop_move.assert_called()
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore.CLOSE_OBSTACLE_CONFIRM_S = original_confirm_s
            NavCore.CLOSE_OBSTACLE_CONFIRM_READINGS = original_confirm_readings
            NavCore.CENTER_ONLY_CLOSE_CONFIRM_S = original_center_confirm_s
            NavCore.CENTER_ONLY_CLOSE_CONFIRM_READINGS = original_center_confirm_readings
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_ignores_isolated_center_depth_close_reading(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            fake_depth.get_center_depth_reading.side_effect = [
                CenterDepthReading(distance_m=0.08, coverage=0.5),
                CenterDepthReading(distance_m=1.0, coverage=0.5),
            ]
            nav._depth_processor = fake_depth
            nav._local_planner = MagicMock()
            nav._local_planner.compute_velocity.return_value = VelocityCommand(vx=0.20)

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)
            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            fake_go2.move.assert_called_once()
            fake_go2.stop_move.assert_called_once()
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_center_depth_does_not_block_pivot(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            fake_depth.get_center_depth_reading.return_value = CenterDepthReading(
                distance_m=0.10,
                coverage=0.5,
            )
            nav._depth_processor = fake_depth
            nav._local_planner = MagicMock()
            nav._local_planner.compute_velocity.return_value = VelocityCommand(vyaw=0.50)

            goal = NavGoal(goal_type="relative", x=0.0, y=2.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.NAVIGATING)
            fake_go2.move.assert_called_once()
            _, kwargs = fake_go2.move.call_args
            self.assertAlmostEqual(kwargs["vx"], 0.0, places=2)
            self.assertAlmostEqual(kwargs["vyaw"], 0.50, places=2)
            fake_go2.stop_move.assert_not_called()
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_navigate_to_waits_for_first_depth_grid(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        original_timeout = NavCore.DEPTH_READY_TIMEOUT_S
        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            nav._depth_processor.stop()

            nav._topo_map.load_from_dict({
                "name": "suite21",
                "nodes": [
                    {"name": "charging_station", "x": 1.3, "y": 15.7},
                    {
                        "name": "shrushtis_desk",
                        "x": 5.62,
                        "y": 15.7,
                        "description": "Shrushti's desk, red marker 2",
                    },
                ],
                "edges": [
                    {"from": "charging_station", "to": "shrushtis_desk", "distance": 4.32},
                ],
            })
            nav._odometry.set_pose(1.3, 15.7, 0.0)

            calls = {"count": 0}

            def get_obstacle_grid():
                calls["count"] += 1
                if calls["count"] == 1:
                    return None
                return _empty_grid()

            fake_depth = MagicMock()
            fake_depth.is_available = True
            fake_depth.get_obstacle_grid.side_effect = get_obstacle_grid
            nav._depth_processor = fake_depth
            NavCore.DEPTH_READY_TIMEOUT_S = 0.2

            result = nav.navigate_to("Shrushti's desk")

            self.assertTrue(result)
            fake_depth.start.assert_called()
            self.assertGreaterEqual(calls["count"], 2)
            self.assertEqual(nav.state, NavState.NAVIGATING)
            nav.shutdown()
        finally:
            NavCore.DEPTH_READY_TIMEOUT_S = original_timeout
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_navigate_to_keeps_initial_global_route_independent_of_live_scan(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            nav._topo_map.load_from_dict({
                "name": "suite21",
                "nodes": [
                    {"name": "charging_station", "x": 0.0, "y": 0.0},
                    {"name": "immersive_room", "x": 4.0, "y": 0.0},
                ],
                "edges": [
                    {"from": "charging_station", "to": "immersive_room", "distance": 4.0},
                ],
            })
            metric_map = MagicMock()
            nav._topo_map.metric_map = metric_map
            nav._odometry.set_pose(0.0, 0.0, 0.0)

            fake_depth = MagicMock()
            fake_depth.is_available = True
            fake_depth.get_obstacle_grid.return_value = _grid_with_wall_ahead(0.6)
            nav._depth_processor = fake_depth
            planned_path = [
                MapNode(name="__metric_start__", x=0.0, y=0.0, tags=["metric_transit"]),
                MapNode(name="immersive_room", x=4.0, y=0.0),
            ]
            nav._global_planner = MagicMock()
            nav._global_planner.plan_path.return_value = planned_path
            nav._ensure_running = MagicMock()

            self.assertTrue(nav.navigate_to("immersive room"))

            nav._global_planner.plan_path.assert_called_once()
            plan_call = nav._global_planner.plan_path.call_args
            self.assertEqual(plan_call.args[1], "immersive room")
            self.assertIsNone(plan_call.kwargs["dynamic_obstacles_xy"])
            metric_map.robot_points_to_world.assert_not_called()
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_navigate_to_reports_depth_grid_unavailable_before_moving(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        original_timeout = NavCore.DEPTH_READY_TIMEOUT_S
        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            nav._depth_processor.stop()

            nav._topo_map.load_from_dict({
                "name": "suite21",
                "nodes": [
                    {"name": "charging_station", "x": 1.3, "y": 15.7},
                    {
                        "name": "shrushtis_desk",
                        "x": 5.62,
                        "y": 15.7,
                        "description": "Shrushti's desk, red marker 2",
                    },
                ],
                "edges": [
                    {"from": "charging_station", "to": "shrushtis_desk", "distance": 4.32},
                ],
            })
            nav._odometry.set_pose(1.3, 15.7, 0.0)

            fake_depth = MagicMock()
            fake_depth.is_available = True
            fake_depth.get_obstacle_grid.return_value = None
            nav._depth_processor = fake_depth
            NavCore.DEPTH_READY_TIMEOUT_S = 0.0

            result = nav.navigate_to("Shrushti's desk")

            self.assertFalse(result)
            self.assertEqual(nav.state, NavState.E_STOP)
            self.assertIn("E-STOP: obstacle grid unavailable", nav.get_status_summary())
            self.assertEqual(
                events,
                [
                    "I did not move toward Shrushti's desk because my obstacle grid "
                    "was not available."
                ],
            )
            fake_depth.start.assert_called()
            fake_depth.stop.assert_called()
            fake_go2.stop_move.assert_called()
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore.DEPTH_READY_TIMEOUT_S = original_timeout
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_navigate_to_reports_unknown_destination_without_waiting_for_agent(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.is_available = True
            fake_depth.get_obstacle_grid.return_value = None
            nav._depth_processor = fake_depth

            nav._topo_map.load_from_dict({
                "name": "suite21",
                "nodes": [
                    {"name": "charging_station", "x": 1.3, "y": 15.7},
                    {
                        "name": "shrushtis_desk",
                        "x": 5.62,
                        "y": 15.7,
                        "description": "Shrushti's desk, red marker 2",
                    },
                ],
                "edges": [
                    {"from": "charging_station", "to": "shrushtis_desk", "distance": 4.32},
                ],
            })
            nav._odometry.set_pose(1.3, 15.7, 0.0)

            result = nav.navigate_to("somewhere imaginary")

            self.assertFalse(result)
            self.assertEqual(nav.state, NavState.IDLE)
            self.assertIn("Unknown destination: somewhere imaginary", nav.get_status_summary())
            self.assertEqual(len(events), 1)
            self.assertIn("I did not move because I do not recognize somewhere imaginary", events[0])
            fake_depth.start.assert_not_called()
            fake_go2.move.assert_not_called()
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_navigate_to_reports_already_at_destination_without_moving(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []
        NavCore.set_status_callback(events.append)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.is_available = True
            fake_depth.get_obstacle_grid.return_value = None
            nav._depth_processor = fake_depth

            nav._topo_map.load_from_dict({
                "name": "suite21",
                "nodes": [
                    {
                        "name": "charging_station",
                        "x": 1.3,
                        "y": 15.7,
                        "description": "Charging station, red marker 0",
                    },
                    {
                        "name": "shrushtis_desk",
                        "x": 5.62,
                        "y": 15.7,
                        "description": "Shrushti's desk, red marker 2",
                    },
                ],
                "edges": [
                    {"from": "charging_station", "to": "shrushtis_desk", "distance": 4.32},
                ],
            })
            nav._odometry.set_pose(1.9, 15.7, 0.0)

            result = nav.navigate_to("charging station")

            self.assertTrue(result)
            self.assertEqual(nav.state, NavState.IDLE)
            self.assertIn("Already at Charging station", nav.get_status_summary())
            self.assertIn(
                "Current mapped location: Charging station",
                nav.get_status_summary(),
            )
            self.assertEqual(events, ["I am already at Charging station."])
            fake_depth.start.assert_not_called()
            fake_go2.move.assert_not_called()
            fake_go2.stop_move.assert_called()
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_releases_depth_after_goal(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _empty_grid()
            nav._depth_processor = fake_depth
            nav._topo_map.load_from_dict({
                "name": "suite21",
                "nodes": [{
                    "name": "kitchen_entrance",
                    "x": 1.0,
                    "y": 1.0,
                    "description": "Kitchen entrance",
                    "aliases": ["kitchen"],
                }],
                "edges": [],
            })
            nav._odometry.set_pose(1.0, 1.0, 0.0)

            goal = NavGoal(goal_type="semantic", x=1.0, y=1.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.IDLE)
            self.assertIn(
                "Current mapped location: Kitchen entrance",
                nav.get_status_summary(),
            )
            fake_go2.stop_move.assert_called()
            fake_depth.stop.assert_called()
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_guarded_forward_moves_until_center_depth_threshold(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            fake_depth = MagicMock()
            fake_depth.is_available = True
            fake_depth.get_center_depth_reading.side_effect = [
                CenterDepthReading(distance_m=2.0, coverage=0.5),
                CenterDepthReading(distance_m=2.0, coverage=0.5),
                CenterDepthReading(distance_m=0.7, coverage=0.5),
            ]
            nav._depth_processor = fake_depth

            result = nav.move_forward_guarded(
                stop_distance_m=0.75,
                speed=0.45,
                max_seconds=1.0,
                command_period_s=0.0,
            )

            self.assertIn("Stopped forward movement", result)
            self.assertGreaterEqual(fake_go2.move.call_count, 1)
            fake_go2.stop_move.assert_called()
            self.assertEqual(nav.state, NavState.IDLE)
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_guarded_forward_uses_obstacle_grid_without_center_depth(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            fake_sensor = MagicMock()
            fake_sensor.supports_center_depth = False
            fake_sensor.get_obstacle_grid.side_effect = [
                _empty_grid(),
                _empty_grid(),
                _grid_with_wall_ahead(distance_m=0.70),
            ]
            nav._depth_processor = fake_sensor

            result = nav.move_forward_guarded(
                stop_distance_m=0.75,
                speed=0.45,
                max_seconds=1.0,
                command_period_s=0.0,
            )

            self.assertIn("Stopped forward movement", result)
            self.assertIn("0.70m", result)
            self.assertGreaterEqual(fake_go2.move.call_count, 1)
            fake_go2.stop_move.assert_called()
            self.assertEqual(nav.state, NavState.IDLE)
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_guarded_forward_refuses_without_center_depth(self, mock_go2):
        mock_go2.return_value = MagicMock()

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        try:
            nav = NavCore.get_instance()
            fake_depth = MagicMock()
            fake_depth.is_available = True
            fake_depth.get_center_depth_reading.return_value = None
            nav._depth_processor = fake_depth

            result = nav.move_forward_guarded(max_seconds=1.0, command_period_s=0.0)

            self.assertIn("Cannot move forward", result)
            mock_go2.assert_not_called()
            nav.shutdown()
        finally:
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)


# ---------------------------------------------------------------------------
# DepthProcessor lifecycle tests
# ---------------------------------------------------------------------------

class TestDepthProcessorLifecycle(unittest.TestCase):

    def test_stop_ignores_unstarted_capture_thread(self):
        processor = DepthProcessor(DepthProcessorConfig(simulation_mode=True))
        processor._running = True
        processor._thread = threading.Thread(target=lambda: None)

        processor.stop()

        self.assertFalse(processor.is_running)
        self.assertIsNone(processor._thread)


if __name__ == "__main__":
    unittest.main()
