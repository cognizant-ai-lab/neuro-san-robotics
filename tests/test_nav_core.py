import math
import os
import threading
import time
import unittest
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
    DEFAULT_MAP_FILE,
    GlobalPlanner,
    LocalPlanner,
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
        self.assertLessEqual(corrected.vyaw, 0.04)
        self.assertAlmostEqual(corrected.vx, cmd.vx)

    def test_one_sided_depth_geometry_does_not_change_course(self):
        planner = LocalPlanner(max_yaw_rate=0.08, avoidance_distance=0.30)
        cmd = VelocityCommand(vx=0.30, vy=0.0, vyaw=0.01)

        corrected = planner.apply_corridor_course_correction(
            cmd,
            self._corridor_grid(include_right_wall=False),
        )

        self.assertEqual(corrected, cmd)

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

    def test_avoidance_turns_away_from_obstacle(self):
        planner = LocalPlanner()
        grid = _grid_with_wall_right(distance_m=0.5)
        grid.nearest_obstacle_bearing = -0.5

        cmd = planner.compute_avoidance(grid)
        self.assertGreater(cmd.vyaw, 0.0,
                           msg="Should turn left (positive vyaw) to avoid obstacle on right")


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


# ---------------------------------------------------------------------------
# TopologicalMap tests
# ---------------------------------------------------------------------------

class TestTopologicalMap(unittest.TestCase):

    def test_load_from_dict(self):
        topo = _create_test_map()
        self.assertTrue(topo.is_loaded)
        self.assertEqual(len(topo.nodes), 3)
        self.assertEqual(len(topo.edges), 2)

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

    def test_remote_control_release_accepts_current_route_segment(self):
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
        goal = NavGoal(x=3.0, y=4.0, goal_type="semantic", label="C")

        self.assertTrue(core._handle_manual_override(goal))
        self.assertFalse(core._handle_manual_override(goal))

        core._odometry.set_pose.assert_called_once_with(1.5, 0.0, 0.0)
        core._reset_progress_tracker.assert_called_once_with()
        self.assertIn(
            "route from A to B",
            core._notify_status_change.call_args.args[0],
        )

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
        goal = NavGoal(x=3.0, y=1.0, goal_type="semantic", label="B")

        self.assertFalse(core._handle_manual_override(goal))

        core._odometry.set_pose.assert_called_once_with(1.0, 1.0, 0.0)
        self.assertIn(
            "continuing toward B",
            core._notify_status_change.call_args.args[0],
        )

    def test_robot_navigation_defaults_are_in_code(self):
        self.assertAlmostEqual(NavCore.MAX_LINEAR_SPEED, 0.40)
        self.assertAlmostEqual(NavCore.MAX_YAW_RATE, 0.08)
        self.assertAlmostEqual(NavCore.PIVOT_YAW_RATE, 0.50)
        self.assertAlmostEqual(NavCore.SAFETY_DISTANCE_M, 0.10)
        self.assertAlmostEqual(NavCore.AVOIDANCE_DISTANCE_M, 0.30)
        self.assertAlmostEqual(NavCore.PIVOT_HARD_STOP_DISTANCE_M, 0.00)
        self.assertAlmostEqual(NavCore.CLOSE_OBSTACLE_CONFIRM_S, 0.7)
        self.assertEqual(NavCore.CLOSE_OBSTACLE_CONFIRM_READINGS, 6)
        self.assertAlmostEqual(NavCore.PATH_OBSTACLE_CONFIRM_S, 0.3)
        self.assertEqual(NavCore.PATH_OBSTACLE_CONFIRM_READINGS, 3)
        self.assertAlmostEqual(NavCore.PATH_OBSTACLE_CENTER_DEPTH_MARGIN_M, 0.15)
        self.assertAlmostEqual(NavCore.GOAL_TOLERANCE_M, 0.15)
        self.assertAlmostEqual(NavCore.OBSTACLE_GRID_MAX_AGE_S, 0.50)

    def test_stale_obstacle_grid_is_rejected(self):
        core = NavCore.__new__(NavCore)
        grid = _empty_grid()
        grid.timestamp = time.time() - 1.0

        self.assertIsNone(core._fresh_obstacle_grid(grid))

    def test_robot_map_default_is_in_code(self):
        old_map = os.environ.pop("NAV_MAP_FILE", None)
        old_sim = os.environ.pop("NAV_SIMULATION_MODE", None)
        try:
            self.assertEqual(_configured_map_file(), str(DEFAULT_MAP_FILE))
            os.environ["NAV_SIMULATION_MODE"] = "1"
            self.assertEqual(_configured_map_file(), "")
        finally:
            if old_map is not None:
                os.environ["NAV_MAP_FILE"] = old_map
            if old_sim is not None:
                os.environ["NAV_SIMULATION_MODE"] = old_sim

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
        NavCore.PATH_OBSTACLE_CONFIRM_S = 0.0
        NavCore.PATH_OBSTACLE_CONFIRM_READINGS = 3
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
            self.assertEqual(len(events), 2)
            self.assertIn("path is clear again", events[1])

            self.assertEqual(nav.state, NavState.NAVIGATING)
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore.PATH_OBSTACLE_CONFIRM_S = original_confirm_s
            NavCore.PATH_OBSTACLE_CONFIRM_READINGS = original_confirm_readings
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
    def test_nav_cycle_allows_pivot_before_translation_near_obstacle(self, mock_go2):
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
            fake_go2.move.assert_called()
            self.assertAlmostEqual(fake_go2.move.call_args.kwargs["vx"], 0.0)
            self.assertGreater(fake_go2.move.call_args.kwargs["vyaw"], 0.0)
            fake_go2.stop_move.assert_not_called()
            self.assertEqual(events, [])
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
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
    def test_nav_cycle_stuck_is_terminal_and_clears_goal(self, mock_go2):
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
            nav._local_planner.compute_velocity.return_value = VelocityCommand(vx=0.2)

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._last_progress_pose = nav._odometry.get_pose()
                nav._last_progress_time = time.monotonic() - nav.STUCK_TIMEOUT_S - 1.0

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.STUCK)
            self.assertIsNone(nav._goal)
            self.assertEqual(
                events,
                ["I stopped before reaching Kitchen because I was not making progress."],
            )
            fake_go2.stop_move.assert_called()
            fake_go2.move.assert_not_called()
            fake_depth.stop.assert_called()

            nav._nav_cycle(NavState.STUCK, goal)
            self.assertEqual(
                events,
                ["I stopped before reaching Kitchen because I was not making progress."],
            )
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
            NavCore._instance = None
            os.environ.pop("NAV_SIMULATION_MODE", None)

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
                distance_m=0.20,
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

    @patch("coded_tools.unigo2.nav_core._get_go2_macros")
    def test_nav_cycle_center_depth_estops_forward_command_when_grid_misses_close_wall(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        original_confirm_s = NavCore.CLOSE_OBSTACLE_CONFIRM_S
        original_confirm_readings = NavCore.CLOSE_OBSTACLE_CONFIRM_READINGS
        NavCore.CLOSE_OBSTACLE_CONFIRM_S = 0.0
        NavCore.CLOSE_OBSTACLE_CONFIRM_READINGS = 1
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
            nav._odometry.set_pose(1.3, 15.7, 0.0)

            result = nav.navigate_to("charging station")

            self.assertTrue(result)
            self.assertEqual(nav.state, NavState.IDLE)
            self.assertIn("Already at Charging station", nav.get_status_summary())
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
            nav._odometry.set_pose(1.0, 1.0, 0.0)

            goal = NavGoal(goal_type="semantic", x=1.0, y=1.0, label="Kitchen")
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.IDLE)
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
