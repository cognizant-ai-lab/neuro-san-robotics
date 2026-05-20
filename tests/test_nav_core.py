import math
import os
import time
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from coded_tools.unigo2.depth_processor import CenterDepthReading, ObstacleGrid
from coded_tools.unigo2.nav_core import (
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
    TopologicalMap,
    VelocityCommand,
)


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
# LocalPlanner tests
# ---------------------------------------------------------------------------

class TestLocalPlanner(unittest.TestCase):

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

    def test_stops_when_no_free_sectors(self):
        planner = LocalPlanner()
        grid = _empty_grid()
        grid.grid[:, :] = 1.0  # all occupied
        grid.nearest_obstacle_m = 0.2

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

    def test_get_next_waypoint_advances(self):
        topo = _create_test_map()
        planner = GlobalPlanner(topo)
        planner.plan_path(RobotPose(0, 0, 0), "C")

        wp1 = planner.get_next_waypoint(RobotPose(0, 0, 0), tolerance_m=0.3)
        self.assertIsNotNone(wp1)
        self.assertEqual(wp1.name, "B")

        wp2 = planner.get_next_waypoint(RobotPose(3.0, 0.0, 0), tolerance_m=0.3)
        self.assertIsNotNone(wp2)
        self.assertEqual(wp2.name, "C")

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


# ---------------------------------------------------------------------------
# NavCore integration (lightweight, no hardware)
# ---------------------------------------------------------------------------

class TestNavCoreStatus(unittest.TestCase):

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
        os.environ["NAV_SIMULATION_MODE"] = "1"
        os.environ["NAV_MAP_FILE"] = map_path
        try:
            nav = NavCore.get_instance()
            pose = nav._odometry.get_pose()
            self.assertAlmostEqual(pose.x, 1.3)
            self.assertAlmostEqual(pose.y, 15.7)
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
        events = []

        def record_event(message):
            events.append(message)

        NavCore.set_status_callback(record_event)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _grid_with_wall_ahead(distance_m=0.23)
            nav._depth_processor = fake_depth

            goal = NavGoal(goal_type="relative", x=2.0, y=0.0)
            with nav._state_lock:
                nav._state = NavState.NAVIGATING
                nav._goal = goal
                nav._reset_progress_tracker()

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.E_STOP)
            self.assertIn("E-STOP: obstacle at 0.23m", nav.get_status_summary())
            self.assertEqual(
                events,
                [
                    "I stopped before reaching the destination because my depth sensor "
                    "reported something at 0.23 meters."
                ],
            )
            fake_go2.stop_move.assert_called()
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
    def test_nav_cycle_reports_sustained_obstacle_limited_crawl(self, mock_go2):
        fake_go2 = MagicMock()
        fake_go2.available = True
        mock_go2.return_value = fake_go2

        NavCore._instance = None
        os.environ["NAV_SIMULATION_MODE"] = "1"
        events = []

        def record_event(message):
            events.append(message)

        NavCore.set_status_callback(record_event)
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            nav.OBSTACLE_LIMITED_TIMEOUT_S = 0.1
            nav.OBSTACLE_LIMITED_SPEED_MPS = 0.12

            fake_depth = MagicMock()
            fake_depth.get_obstacle_grid.return_value = _grid_with_wall_ahead(distance_m=0.55)
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
                nav._obstacle_limited_since = time.monotonic() - 1.0

            nav._nav_cycle(NavState.NAVIGATING, goal)

            self.assertEqual(nav.state, NavState.STUCK)
            self.assertIn("Blocked: nearby obstacle or wall at 0.55m", nav.get_status_summary())
            self.assertEqual(
                events,
                [
                    "I stopped before reaching Kitchen because my depth sensor kept "
                    "seeing something nearby at 0.55 meters and I was only able to crawl."
                ],
            )
            fake_go2.stop_move.assert_called()
            nav.shutdown()
        finally:
            NavCore.set_status_callback(None)
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
                    {"name": "shrushtis_desk", "x": 5.62, "y": 15.7},
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
        try:
            nav = NavCore.get_instance()
            nav._go2 = fake_go2
            nav._depth_processor.stop()

            nav._topo_map.load_from_dict({
                "name": "suite21",
                "nodes": [
                    {"name": "charging_station", "x": 1.3, "y": 15.7},
                    {"name": "shrushtis_desk", "x": 5.62, "y": 15.7},
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
            self.assertIn("E-STOP: depth grid unavailable", nav.get_status_summary())
            fake_go2.stop_move.assert_called()
            nav.shutdown()
        finally:
            NavCore.DEPTH_READY_TIMEOUT_S = original_timeout
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


if __name__ == "__main__":
    unittest.main()
