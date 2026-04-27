
# Copyright (C) 2023-2025 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# neuro-san SDK Software in commercial settings.
#
# END COPYRIGHT

"""
NavCore - Navigation Engine for Unitree Go2 EDU

Provides autonomous navigation for the CAIL-E robot dog:
- Local reactive obstacle avoidance (VFH+ algorithm)
- Global path planning on topological maps (Dijkstra)
- Safety monitoring with emergency stop
- Odometry tracking via Unitree SDK2 DDS
- Background navigation loop at configurable frequency

Architecture:
  NavCore (singleton) runs a background thread at ~10 Hz.
  Each cycle: read sensors -> plan -> safety filter -> execute via Go2Macros.

See docs/nav_core_design.md for full architecture documentation.
"""

import heapq
import json
import math
import os
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from coded_tools.unigo2.depth_processor import (
    DepthProcessor,
    DepthProcessorConfig,
    ObstacleGrid,
    _env_flag,
    _env_float,
    _env_int,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import of Go2Macros to avoid import-time SDK initialization
# ---------------------------------------------------------------------------

_go2_macros_cls = None


def _get_go2_macros():
    """Lazily import and instantiate Go2Macros to avoid triggering DDS init at import time."""
    global _go2_macros_cls
    if _go2_macros_cls is None:
        from coded_tools.unigo2.go2_macros import Go2Macros
        _go2_macros_cls = Go2Macros
    return _go2_macros_cls()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class NavState(Enum):
    """Navigation state machine states."""
    IDLE = "idle"
    NAVIGATING = "navigating"
    AVOIDING = "avoiding"
    STUCK = "stuck"
    E_STOP = "e_stop"


@dataclass
class RobotPose:
    """Robot position and heading in the world frame (meters, radians)."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    timestamp: float = 0.0


@dataclass
class VelocityCommand:
    """Velocity command for Go2Macros.move(). Units: m/s and rad/s."""
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0


@dataclass
class NavGoal:
    """Navigation goal with type, target position, and optional label.

    goal_type: 'relative' (distance/angle), 'pose' (x/y), or 'semantic' (map node name).
    """
    goal_type: str = "relative"    # "relative", "pose", "semantic"
    x: float = 0.0
    y: float = 0.0
    yaw: Optional[float] = None
    label: str = ""
    timeout_s: float = 60.0


@dataclass
class MapNode:
    """A named location in the topological map with position and metadata."""
    name: str
    x: float
    y: float
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class MapEdge:
    """A bidirectional connection between two MapNodes with traversal cost."""
    from_node: str
    to_node: str
    distance: float
    traversable: bool = True
    description: str = ""


# ---------------------------------------------------------------------------
# TopologicalMap
# ---------------------------------------------------------------------------

class TopologicalMap:
    """Graph-based semantic map loaded from JSON."""

    def __init__(self):
        """Initialize an empty topological map."""
        self.name: str = ""
        self.nodes: Dict[str, MapNode] = {}
        self.edges: List[MapEdge] = []
        self._adjacency: Dict[str, List[Tuple[str, float]]] = {}

    def load_from_file(self, path: str) -> bool:
        """Load map from a JSON file. Returns True if at least one node was loaded."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._parse(data)
        except Exception as exc:
            logger.error("TopologicalMap: failed to load %s: %s", path, exc)
            return False

    def load_from_dict(self, data: Dict[str, Any]) -> bool:
        """Load map from a dictionary (same schema as the JSON file)."""
        return self._parse(data)

    def _parse(self, data: Dict[str, Any]) -> bool:
        """Parse map data, building nodes, edges, and adjacency list."""
        self.name = data.get("name", "")
        self.nodes.clear()
        self.edges.clear()
        self._adjacency.clear()

        for node_data in data.get("nodes", []):
            name = node_data["name"]
            self.nodes[name] = MapNode(
                name=name,
                x=float(node_data.get("x", 0)),
                y=float(node_data.get("y", 0)),
                description=node_data.get("description", ""),
                tags=node_data.get("tags", []),
            )
            self._adjacency.setdefault(name, [])

        for edge_data in data.get("edges", []):
            edge = MapEdge(
                from_node=edge_data["from"],
                to_node=edge_data["to"],
                distance=float(edge_data.get("distance", 1.0)),
                traversable=edge_data.get("traversable", True),
                description=edge_data.get("description", ""),
            )
            self.edges.append(edge)
            if edge.traversable:
                self._adjacency.setdefault(edge.from_node, []).append(
                    (edge.to_node, edge.distance)
                )
                self._adjacency.setdefault(edge.to_node, []).append(
                    (edge.from_node, edge.distance)
                )

        logger.info(
            "TopologicalMap: loaded '%s' with %d nodes, %d edges",
            self.name, len(self.nodes), len(self.edges),
        )
        return len(self.nodes) > 0

    def save_to_file(self, path: str):
        """Serialize the map to a JSON file, creating parent directories if needed."""
        data = {
            "name": self.name,
            "version": "1.0",
            "nodes": [
                {
                    "name": n.name, "x": n.x, "y": n.y,
                    "description": n.description, "tags": n.tags,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "from": e.from_node, "to": e.to_node,
                    "distance": e.distance,
                    "traversable": e.traversable,
                    "description": e.description,
                }
                for e in self.edges
            ],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_node(self, name: str) -> Optional[MapNode]:
        """Look up a node by name. Returns None if not found."""
        return self.nodes.get(name)

    def find_nearest_node(self, x: float, y: float) -> Optional[MapNode]:
        """Find the map node closest to the given (x, y) position."""
        best = None
        best_dist = float("inf")
        for node in self.nodes.values():
            d = math.hypot(node.x - x, node.y - y)
            if d < best_dist:
                best_dist = d
                best = node
        return best

    def list_destinations(self) -> List[str]:
        """Return sorted list of all node names."""
        return sorted(self.nodes.keys())

    @property
    def is_loaded(self) -> bool:
        """True if the map has at least one node."""
        return len(self.nodes) > 0


# ---------------------------------------------------------------------------
# GlobalPlanner (Dijkstra)
# ---------------------------------------------------------------------------

class GlobalPlanner:
    """Dijkstra path planning on the topological map."""

    def __init__(self, topo_map: TopologicalMap):
        """Initialize the global planner with a topological map reference."""
        self._map = topo_map
        self._current_path: List[MapNode] = []
        self._waypoint_index: int = 0

    def plan_path(self, current_pose: RobotPose, goal_label: str) -> Optional[List[MapNode]]:
        """Plan a path from the nearest node to current_pose to the goal node.

        Returns:
            List of MapNodes from start to goal, or None if unreachable/unknown.
        """
        goal_node = self._map.get_node(goal_label)
        if goal_node is None:
            logger.warning("GlobalPlanner: unknown destination '%s'", goal_label)
            return None

        start_node = self._map.find_nearest_node(current_pose.x, current_pose.y)
        if start_node is None:
            return None

        if start_node.name == goal_node.name:
            return [goal_node]

        path_names = self._dijkstra(start_node.name, goal_node.name)
        if path_names is None:
            logger.warning(
                "GlobalPlanner: no path from '%s' to '%s'",
                start_node.name, goal_label,
            )
            return None

        path = [self._map.nodes[n] for n in path_names]
        self._current_path = path
        self._waypoint_index = 1  # skip start node
        return path

    def get_next_waypoint(self, current_pose: RobotPose, tolerance_m: float = 0.3) -> Optional[MapNode]:
        """Return the next waypoint to steer toward, advancing when within tolerance.

        Returns None when the final waypoint (goal) has been reached.
        """
        if not self._current_path or self._waypoint_index >= len(self._current_path):
            return None

        wp = self._current_path[self._waypoint_index]
        dist = math.hypot(wp.x - current_pose.x, wp.y - current_pose.y)

        if dist < tolerance_m and self._waypoint_index < len(self._current_path) - 1:
            self._waypoint_index += 1
            wp = self._current_path[self._waypoint_index]
            dist = math.hypot(wp.x - current_pose.x, wp.y - current_pose.y)
            logger.info("GlobalPlanner: advancing to waypoint '%s'", wp.name)

        if dist < tolerance_m and self._waypoint_index == len(self._current_path) - 1:
            return None  # goal reached

        return wp

    def clear(self):
        """Reset the current path and waypoint index."""
        self._current_path = []
        self._waypoint_index = 0

    def _dijkstra(self, start: str, goal: str) -> Optional[List[str]]:
        """Run Dijkstra's shortest path on the topological map adjacency graph."""
        dist: Dict[str, float] = {start: 0.0}
        prev: Dict[str, Optional[str]] = {start: None}
        heap = [(0.0, start)]

        while heap:
            d, u = heapq.heappop(heap)
            if u == goal:
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = prev[node]
                return list(reversed(path))

            if d > dist.get(u, float("inf")):
                continue

            for neighbor, weight in self._map._adjacency.get(u, []):
                new_dist = d + weight
                if new_dist < dist.get(neighbor, float("inf")):
                    dist[neighbor] = new_dist
                    prev[neighbor] = u
                    heapq.heappush(heap, (new_dist, neighbor))

        return None


# ---------------------------------------------------------------------------
# LocalPlanner (VFH+)
# ---------------------------------------------------------------------------

class LocalPlanner:
    """
    VFH+ (Vector Field Histogram Plus) local planner.

    Converts ObstacleGrid + goal direction into velocity commands.
    Tuned for indoor navigation at 0.1-0.3 m/s on the Unitree Go2.
    """

    HISTOGRAM_SECTORS = 72         # 5-degree sectors
    SECTOR_WIDTH_RAD = 2 * math.pi / 72
    SECTOR_THRESHOLD = 3           # obstacle cell count to mark sector as blocked
    WIDE_VALLEY_MIN_SECTORS = 6    # minimum sectors for a "wide" valley
    SMOOTHING_WEIGHT = 0.3         # heading change smoothing

    def __init__(
        self,
        max_linear_speed: float = 0.3,
        max_yaw_rate: float = 0.5,
        safety_distance: float = 0.4,
        avoidance_distance: float = 0.8,
    ):
        """Configure the local planner speed and distance thresholds.

        Args:
            max_linear_speed: Maximum forward speed in m/s.
            max_yaw_rate: Maximum rotation rate in rad/s.
            safety_distance: E-stop distance in meters (speed = 0 below this).
            avoidance_distance: Start slowing down at this distance in meters.
        """
        self.max_linear_speed = max_linear_speed
        self.max_yaw_rate = max_yaw_rate
        self.safety_distance = safety_distance
        self.avoidance_distance = avoidance_distance
        self._prev_heading = 0.0

    def compute_velocity(
        self,
        obstacle_grid: ObstacleGrid,
        goal_direction: float,
        goal_distance: float,
    ) -> VelocityCommand:
        """Compute velocity toward the goal while avoiding obstacles.

        Args:
            obstacle_grid: Current obstacle map from depth processor.
            goal_direction: Bearing to goal in radians (0=ahead, positive=left).
            goal_distance: Distance to goal in meters.

        Returns:
            VelocityCommand with speed modulated by obstacle proximity and goal distance.
        """
        histogram = self._build_histogram(obstacle_grid)
        free_sectors = self._find_free_sectors(histogram)

        if not free_sectors:
            return VelocityCommand(0.0, 0.0, 0.0)

        goal_sector = self._angle_to_sector(goal_direction)
        best_sector = self._select_best_sector(free_sectors, goal_sector)
        target_heading = self._sector_to_angle(best_sector)

        # Smooth heading changes
        target_heading = (
            self.SMOOTHING_WEIGHT * self._prev_heading
            + (1 - self.SMOOTHING_WEIGHT) * target_heading
        )
        self._prev_heading = target_heading

        # Speed modulation based on nearest obstacle
        nearest = obstacle_grid.nearest_obstacle_m
        base_speed = self._modulate_speed(self.max_linear_speed, nearest)

        # Slow down when close to goal
        if goal_distance < 0.5:
            base_speed = min(base_speed, 0.1)

        # Compute velocity command
        vyaw = np.clip(target_heading, -self.max_yaw_rate, self.max_yaw_rate)

        # Reduce forward speed when turning sharply
        turn_factor = 1.0 - min(abs(vyaw) / self.max_yaw_rate, 1.0) * 0.5
        vx = base_speed * turn_factor

        return VelocityCommand(vx=vx, vy=0.0, vyaw=float(vyaw))

    def compute_avoidance(self, obstacle_grid: ObstacleGrid) -> VelocityCommand:
        """Reactive avoidance: turn away from nearest obstacle."""
        bearing = obstacle_grid.nearest_obstacle_bearing
        # Turn away from obstacle
        vyaw = self.max_yaw_rate if bearing < 0 else -self.max_yaw_rate

        nearest = obstacle_grid.nearest_obstacle_m
        vx = self._modulate_speed(0.1, nearest)

        return VelocityCommand(vx=vx, vy=0.0, vyaw=vyaw)

    def _build_histogram(self, grid: ObstacleGrid) -> np.ndarray:
        """Build a polar obstacle density histogram (72 sectors, 5 degrees each)."""
        histogram = np.zeros(self.HISTOGRAM_SECTORS, dtype=np.float32)
        occupied = np.argwhere(grid.grid > 0)

        for row, col in occupied:
            dx = (grid.origin_row - row) * grid.resolution
            dy = (grid.origin_col - col) * grid.resolution
            angle = math.atan2(dy, dx)
            sector = self._angle_to_sector(angle)
            distance = math.hypot(dx, dy)
            # Weight inversely by distance: closer obstacles matter more
            weight = 1.0 / max(distance, 0.1)
            histogram[sector] += weight

        return histogram

    def _find_free_sectors(self, histogram: np.ndarray) -> List[int]:
        """Return sector indices with obstacle density below the threshold."""
        return [i for i in range(self.HISTOGRAM_SECTORS) if histogram[i] < self.SECTOR_THRESHOLD]

    def _select_best_sector(self, free_sectors: List[int], goal_sector: int) -> int:
        """Select the free sector closest to the goal direction (circular distance)."""
        best = free_sectors[0]
        best_cost = float("inf")

        for sector in free_sectors:
            diff = abs(sector - goal_sector)
            diff = min(diff, self.HISTOGRAM_SECTORS - diff)
            if diff < best_cost:
                best_cost = diff
                best = sector

        return best

    def _angle_to_sector(self, angle: float) -> int:
        """Convert a bearing angle (radians) to a histogram sector index."""
        angle = angle % (2 * math.pi)
        return int(angle / self.SECTOR_WIDTH_RAD) % self.HISTOGRAM_SECTORS

    def _sector_to_angle(self, sector: int) -> float:
        """Convert a sector index back to a bearing angle in [-pi, pi]."""
        angle = (sector + 0.5) * self.SECTOR_WIDTH_RAD
        if angle > math.pi:
            angle -= 2 * math.pi
        return angle

    def _modulate_speed(self, base_speed: float, nearest_obstacle_m: float) -> float:
        """Linear speed ramp: 0 at safety_distance, base_speed at avoidance_distance."""
        if nearest_obstacle_m <= self.safety_distance:
            return 0.0
        if nearest_obstacle_m >= self.avoidance_distance:
            return base_speed
        ratio = (nearest_obstacle_m - self.safety_distance) / (
            self.avoidance_distance - self.safety_distance
        )
        return base_speed * ratio


# ---------------------------------------------------------------------------
# SafetyMonitor
# ---------------------------------------------------------------------------

class SafetyMonitor:
    """
    Filters every velocity command through safety checks.
    Inspired by CMU ABS safety-supervisor pattern.
    """

    def __init__(
        self,
        safety_distance: float = 0.4,
        avoidance_distance: float = 0.8,
        stuck_timeout: float = 10.0,
    ):
        """Configure safety thresholds.

        Args:
            safety_distance: E-stop if obstacle closer than this (meters).
            avoidance_distance: Scale speed down between safety and this (meters).
            stuck_timeout: Trigger stuck event after this many seconds without progress.
        """
        self.safety_distance = safety_distance
        self.avoidance_distance = avoidance_distance
        self.stuck_timeout = stuck_timeout

    def filter_command(
        self,
        cmd: VelocityCommand,
        nearest_obstacle_m: float,
        ground_plane_valid: bool = True,
        seconds_since_progress: float = 0.0,
    ) -> Tuple[VelocityCommand, Optional[str]]:
        """Filter a velocity command through safety checks (priority-ordered).

        Returns:
            Tuple of (possibly zeroed command, optional event string).
            Event string is None when no safety condition triggered.
        """
        # Priority 1: E-STOP
        if nearest_obstacle_m <= self.safety_distance:
            return VelocityCommand(0.0, 0.0, 0.0), "e_stop:obstacle_too_close"

        # Priority 2: Cliff detection
        if not ground_plane_valid:
            return VelocityCommand(0.0, 0.0, 0.0), "e_stop:ground_plane_missing"

        # Priority 3: Stuck detection
        if seconds_since_progress >= self.stuck_timeout:
            return VelocityCommand(0.0, 0.0, 0.0), "stuck:no_progress"

        # Priority 5: Speed modulation in avoidance zone
        if nearest_obstacle_m < self.avoidance_distance:
            ratio = (nearest_obstacle_m - self.safety_distance) / (
                self.avoidance_distance - self.safety_distance
            )
            ratio = max(0.0, min(ratio, 1.0))
            cmd = VelocityCommand(
                vx=cmd.vx * ratio,
                vy=cmd.vy * ratio,
                vyaw=cmd.vyaw,
            )

        return cmd, None


# ---------------------------------------------------------------------------
# OdometryProvider
# ---------------------------------------------------------------------------

class OdometryProvider:
    """
    Tracks robot pose. In the initial implementation, uses dead-reckoning
    from velocity commands. Phase 3 adds DDS subscription to SDK odometry.
    """

    def __init__(self):
        """Initialize odometry at the origin (0, 0, 0)."""
        self._pose = RobotPose()
        self._lock = threading.Lock()
        self._last_update = time.monotonic()

    def update_from_velocity(self, cmd: VelocityCommand, dt: float):
        """Integrate a velocity command over dt seconds (dead-reckoning)."""
        with self._lock:
            self._pose.x += cmd.vx * math.cos(self._pose.yaw) * dt
            self._pose.y += cmd.vx * math.sin(self._pose.yaw) * dt
            self._pose.yaw += cmd.vyaw * dt
            self._pose.yaw = math.atan2(
                math.sin(self._pose.yaw), math.cos(self._pose.yaw)
            )
            self._pose.timestamp = time.time()

    def get_pose(self) -> RobotPose:
        """Return a copy of the current pose (thread-safe)."""
        with self._lock:
            return RobotPose(
                x=self._pose.x,
                y=self._pose.y,
                yaw=self._pose.yaw,
                timestamp=self._pose.timestamp,
            )

    def reset(self):
        """Reset pose to the origin."""
        with self._lock:
            self._pose = RobotPose()
            self._last_update = time.monotonic()


# ---------------------------------------------------------------------------
# NavCore (singleton)
# ---------------------------------------------------------------------------

_NAV_INIT_STATE = {
    "attempted": False,
    "available": True,
    "error": None,
}


class NavCore:
    """
    Main navigation engine for the Unitree Go2 EDU.

    Singleton: use NavCore.get_instance() to obtain the shared instance.
    Runs a background navigation loop at NAV_LOOP_HZ (default 10 Hz).
    """

    _instance: Optional["NavCore"] = None
    _instance_lock = threading.Lock()

    # Configuration (overridable via environment variables)
    NAV_LOOP_HZ: int = _env_int("NAV_LOOP_HZ", 10)
    SAFETY_DISTANCE_M: float = _env_float("NAV_SAFETY_DISTANCE", 0.4)
    AVOIDANCE_DISTANCE_M: float = _env_float("NAV_AVOIDANCE_DISTANCE", 0.8)
    MAX_LINEAR_SPEED: float = _env_float("NAV_MAX_LINEAR_SPEED", 0.3)
    MAX_YAW_RATE: float = _env_float("NAV_MAX_YAW_RATE", 0.5)
    GOAL_TOLERANCE_M: float = _env_float("NAV_GOAL_TOLERANCE", 0.3)
    STUCK_TIMEOUT_S: float = _env_float("NAV_STUCK_TIMEOUT", 10.0)

    @classmethod
    def get_instance(cls) -> "NavCore":
        """Return the shared NavCore singleton, creating it on first call."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        """Initialize all sub-components: depth processor, planners, safety, odometry."""
        self._state = NavState.IDLE
        self._goal: Optional[NavGoal] = None
        self._state_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Sub-components
        self._depth_processor = DepthProcessor()
        self._local_planner = LocalPlanner(
            max_linear_speed=self.MAX_LINEAR_SPEED,
            max_yaw_rate=self.MAX_YAW_RATE,
            safety_distance=self.SAFETY_DISTANCE_M,
            avoidance_distance=self.AVOIDANCE_DISTANCE_M,
        )
        self._safety = SafetyMonitor(
            safety_distance=self.SAFETY_DISTANCE_M,
            avoidance_distance=self.AVOIDANCE_DISTANCE_M,
            stuck_timeout=self.STUCK_TIMEOUT_S,
        )
        self._odometry = OdometryProvider()

        # Global planner (optional, requires map)
        self._topo_map = TopologicalMap()
        self._global_planner = GlobalPlanner(self._topo_map)

        # Progress tracking
        self._last_progress_pose = RobotPose()
        self._last_progress_time = time.monotonic()

        # Go2 macros (lazy init)
        self._go2 = None

        # Load map if configured
        map_file = os.environ.get("NAV_MAP_FILE", "")
        if map_file and Path(map_file).exists():
            self._topo_map.load_from_file(map_file)

        # Status callback (for agent notifications)
        self._on_status_change: Optional[Callable[[str], None]] = None

        # Start depth processor
        if self._depth_processor.is_available:
            self._depth_processor.start()

        logger.info(
            "NavCore: initialized (depth=%s, map=%s, loop=%d Hz)",
            self._depth_processor.backend,
            "loaded" if self._topo_map.is_loaded else "none",
            self.NAV_LOOP_HZ,
        )

    def _ensure_go2(self):
        """Lazily initialize the Go2Macros motor controller."""
        if self._go2 is None:
            self._go2 = _get_go2_macros()

    # ------------------------------------------------------------------
    # Public navigation commands
    # ------------------------------------------------------------------

    def navigate_to(self, destination: str) -> bool:
        """Navigate to a named location on the topological map.

        Plans a global path via Dijkstra, then the nav loop handles local avoidance.
        Returns False if no map loaded or destination unreachable.
        """
        if not self._topo_map.is_loaded:
            logger.warning("NavCore: no map loaded, cannot navigate to '%s'", destination)
            return False

        pose = self._odometry.get_pose()
        path = self._global_planner.plan_path(pose, destination)
        if path is None:
            return False

        with self._state_lock:
            self._goal = NavGoal(
                goal_type="semantic",
                x=path[-1].x,
                y=path[-1].y,
                label=destination,
            )
            self._state = NavState.NAVIGATING
            self._reset_progress_tracker()

        self._ensure_running()
        logger.info("NavCore: navigating to '%s' via %d waypoints", destination, len(path))
        return True

    def move_relative(self, distance: float, angle: float = 0.0) -> bool:
        """Move a given distance (meters) at a given angle offset (radians) from current heading."""
        pose = self._odometry.get_pose()
        target_yaw = pose.yaw + angle
        goal_x = pose.x + distance * math.cos(target_yaw)
        goal_y = pose.y + distance * math.sin(target_yaw)

        with self._state_lock:
            self._goal = NavGoal(
                goal_type="relative",
                x=goal_x,
                y=goal_y,
                yaw=target_yaw if abs(angle) > 0.01 else None,
            )
            self._state = NavState.NAVIGATING
            self._reset_progress_tracker()

        self._ensure_running()
        logger.info("NavCore: moving relative d=%.2f a=%.2f", distance, angle)
        return True

    def turn(self, angle_rad: float) -> bool:
        """Rotate in place by the given angle (radians, positive=left)."""
        pose = self._odometry.get_pose()
        with self._state_lock:
            self._goal = NavGoal(
                goal_type="relative",
                x=pose.x,
                y=pose.y,
                yaw=pose.yaw + angle_rad,
            )
            self._state = NavState.NAVIGATING
            self._reset_progress_tracker()

        self._ensure_running()
        return True

    def stop(self):
        """Cancel current navigation and send stop command to the robot."""
        with self._state_lock:
            self._state = NavState.IDLE
            self._goal = None
            self._global_planner.clear()
        self._ensure_go2()
        if self._go2 and getattr(self._go2, "available", False):
            self._go2.stop_move()
        logger.info("NavCore: navigation stopped")

    def resume(self):
        """Clear E-STOP state and return to IDLE. Does not restart previous goal."""
        with self._state_lock:
            if self._state == NavState.E_STOP:
                self._state = NavState.IDLE
                logger.info("NavCore: resumed from E-STOP")

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    @property
    def state(self) -> NavState:
        """Current navigation state (thread-safe)."""
        with self._state_lock:
            return self._state

    def is_initialized(self) -> bool:
        """True once NavCore has completed initialization."""
        return True

    def get_status_summary(self) -> str:
        """Human-readable summary of navigation state, pose, goal, and obstacles."""
        with self._state_lock:
            state = self._state
            goal = self._goal

        pose = self._odometry.get_pose()
        parts = [f"Navigation state: {state.value}"]
        parts.append(f"Position: ({pose.x:.1f}, {pose.y:.1f}), heading: {math.degrees(pose.yaw):.0f} deg")

        if goal:
            if goal.label:
                parts.append(f"Destination: {goal.label}")
            dist = math.hypot(goal.x - pose.x, goal.y - pose.y)
            parts.append(f"Distance to goal: {dist:.1f}m")

        grid = self._depth_processor.get_obstacle_grid()
        if grid:
            if grid.nearest_obstacle_m < float("inf"):
                parts.append(f"Nearest obstacle: {grid.nearest_obstacle_m:.2f}m")
            else:
                parts.append("No obstacles detected")

        return ". ".join(parts) + "."

    def get_nearest_obstacle_distance(self) -> float:
        """Return distance to nearest obstacle in meters, or inf if none detected."""
        grid = self._depth_processor.get_obstacle_grid()
        if grid:
            return grid.nearest_obstacle_m
        return float("inf")

    def get_obstacle_summary(self) -> str:
        """Human-readable obstacle summary from the depth processor."""
        return self._depth_processor.get_obstacle_summary()

    def list_destinations(self) -> str:
        """Return a string listing available map destinations, or a fallback message."""
        if not self._topo_map.is_loaded:
            return "No map loaded. Only relative navigation (move_forward, turn) is available."
        names = self._topo_map.list_destinations()
        return "Available destinations: " + ", ".join(names)

    # ------------------------------------------------------------------
    # Background navigation loop
    # ------------------------------------------------------------------

    def _ensure_running(self):
        """Start the background nav loop thread if not already running."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._nav_loop, daemon=True, name="nav-core")
        self._thread.start()
        logger.info("NavCore: navigation loop started")

    def _nav_loop(self):
        """Main navigation loop running at NAV_LOOP_HZ in a daemon thread."""
        self._ensure_go2()

        while self._running:
            cycle_start = time.monotonic()

            with self._state_lock:
                state = self._state
                goal = self._goal

            if state == NavState.IDLE or state == NavState.E_STOP or goal is None:
                time.sleep(0.1)
                continue

            try:
                self._nav_cycle(state, goal)
            except Exception as exc:
                logger.error("NavCore: nav cycle error: %s", exc)

            elapsed = time.monotonic() - cycle_start
            sleep_time = (1.0 / self.NAV_LOOP_HZ) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _nav_cycle(self, state: NavState, goal: NavGoal):
        """Execute one navigation cycle: sense -> plan -> safety filter -> actuate."""
        # 1. Read sensors
        grid = self._depth_processor.get_obstacle_grid()
        pose = self._odometry.get_pose()

        nearest_dist = grid.nearest_obstacle_m if grid else float("inf")

        # 2. Safety pre-check
        if nearest_dist <= self.SAFETY_DISTANCE_M:
            self._ensure_go2()
            if self._go2 and getattr(self._go2, "available", False):
                self._go2.stop_move()
            with self._state_lock:
                self._state = NavState.E_STOP
            logger.warning("NavCore: E-STOP triggered (obstacle at %.2fm)", nearest_dist)
            return

        # 3. Check if goal reached
        dist_to_goal = math.hypot(goal.x - pose.x, goal.y - pose.y)
        if dist_to_goal < self.GOAL_TOLERANCE_M:
            self._ensure_go2()
            if self._go2 and getattr(self._go2, "available", False):
                self._go2.stop_move()
            with self._state_lock:
                self._state = NavState.IDLE
                self._goal = None
                self._global_planner.clear()
            logger.info("NavCore: goal reached (dist=%.2fm)", dist_to_goal)
            return

        # 4. Compute velocity command
        if state == NavState.NAVIGATING:
            if goal.goal_type == "semantic":
                waypoint = self._global_planner.get_next_waypoint(pose, self.GOAL_TOLERANCE_M)
                if waypoint is None:
                    # Global path complete
                    with self._state_lock:
                        self._state = NavState.IDLE
                        self._goal = None
                    return
                target_x, target_y = waypoint.x, waypoint.y
            else:
                target_x, target_y = goal.x, goal.y

            goal_dir = math.atan2(target_y - pose.y, target_x - pose.x) - pose.yaw
            # Normalize to [-pi, pi]
            goal_dir = math.atan2(math.sin(goal_dir), math.cos(goal_dir))
            goal_dist = math.hypot(target_x - pose.x, target_y - pose.y)

            if grid:
                cmd = self._local_planner.compute_velocity(grid, goal_dir, goal_dist)
            else:
                # No depth data: drive carefully toward goal
                speed = min(0.1, self.MAX_LINEAR_SPEED)
                vyaw = np.clip(goal_dir, -self.MAX_YAW_RATE, self.MAX_YAW_RATE)
                cmd = VelocityCommand(vx=speed, vy=0.0, vyaw=float(vyaw))

        elif state == NavState.AVOIDING:
            if grid:
                cmd = self._local_planner.compute_avoidance(grid)
            else:
                cmd = VelocityCommand(0.0, 0.0, 0.0)

            if grid and grid.nearest_obstacle_m > self.AVOIDANCE_DISTANCE_M:
                with self._state_lock:
                    self._state = NavState.NAVIGATING

        elif state == NavState.STUCK:
            cmd = VelocityCommand(0.0, 0.0, 0.0)
        else:
            cmd = VelocityCommand(0.0, 0.0, 0.0)

        # 5. Safety filter
        seconds_since_progress = time.monotonic() - self._last_progress_time
        cmd, event = self._safety.filter_command(
            cmd,
            nearest_obstacle_m=nearest_dist,
            seconds_since_progress=seconds_since_progress,
        )

        if event:
            if event.startswith("e_stop"):
                with self._state_lock:
                    self._state = NavState.E_STOP
                self._ensure_go2()
                if self._go2 and getattr(self._go2, "available", False):
                    self._go2.stop_move()
                logger.warning("NavCore: safety event: %s", event)
                return
            elif event.startswith("stuck"):
                with self._state_lock:
                    self._state = NavState.STUCK
                logger.warning("NavCore: stuck detected")
                return

        # 6. Execute
        self._ensure_go2()
        if self._go2 and getattr(self._go2, "available", False):
            self._go2.move(vx=cmd.vx, vy=cmd.vy, vyaw=cmd.vyaw)

        # 7. Update odometry (dead-reckoning)
        dt = 1.0 / self.NAV_LOOP_HZ
        self._odometry.update_from_velocity(cmd, dt)

        # 8. Update progress tracker
        self._update_progress(pose)

    def _reset_progress_tracker(self):
        """Reset the stuck-detection timer to now."""
        self._last_progress_pose = self._odometry.get_pose()
        self._last_progress_time = time.monotonic()

    def _update_progress(self, current_pose: RobotPose):
        """Update progress tracker if robot has moved more than 0.1m since last check."""
        dist_moved = math.hypot(
            current_pose.x - self._last_progress_pose.x,
            current_pose.y - self._last_progress_pose.y,
        )
        if dist_moved > 0.1:
            self._last_progress_pose = current_pose
            self._last_progress_time = time.monotonic()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self):
        """Stop navigation, join the background thread, and release depth camera."""
        self.stop()
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._depth_processor.stop()
        logger.info("NavCore: shutdown complete")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
