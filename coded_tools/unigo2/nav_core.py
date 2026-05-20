
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

import difflib
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
    CenterDepthReading,
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
    aliases: List[str] = field(default_factory=list)


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
        self._node_aliases: Dict[str, str] = {}

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
        self._node_aliases.clear()

        for node_data in data.get("nodes", []):
            name = node_data["name"]
            node = MapNode(
                name=name,
                x=float(node_data.get("x", 0)),
                y=float(node_data.get("y", 0)),
                description=node_data.get("description", ""),
                tags=node_data.get("tags", []),
                aliases=node_data.get("aliases", []),
            )
            self.nodes[name] = node
            self._adjacency.setdefault(name, [])
            self._register_node_aliases(node)

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
                    "aliases": n.aliases,
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
        node = self.nodes.get(name)
        if node is not None:
            return node

        canonical_name = self._node_aliases.get(self._normalize_node_name(name))
        if canonical_name:
            return self.nodes.get(canonical_name)

        canonical_name = self._closest_node_alias(name)
        if canonical_name:
            logger.info("TopologicalMap: resolved destination '%s' to '%s'", name, canonical_name)
            return self.nodes.get(canonical_name)
        return None

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

    def list_destination_labels(self) -> List[str]:
        """Return sorted human-readable destination labels."""
        labels = []
        for node in self.nodes.values():
            labels.append(self.get_node_label(node))
        return sorted(labels, key=str.lower)

    @staticmethod
    def get_node_label(node: MapNode) -> str:
        """Return a human-readable label for a map node."""
        label = node.description.split(",", 1)[0].strip()
        return label or node.name.replace("_", " ")

    @property
    def is_loaded(self) -> bool:
        """True if the map has at least one node."""
        return len(self.nodes) > 0

    @staticmethod
    def _normalize_node_name(name: str) -> str:
        """Normalize map/node names for natural language destination matching."""
        return "".join(ch.lower() for ch in name if ch.isalnum())

    def _register_node_aliases(self, node: MapNode):
        """Register common natural-language aliases for a map node."""
        aliases = {
            node.name,
            node.name.replace("_", " "),
            node.description.split(",", 1)[0],
            *node.aliases,
        }
        if node.name == "charging_station":
            aliases.update({"base", "home", "charger", "charging dock", "docking station"})
        if node.name == "shrushtis_desk":
            aliases.update({
                "Xuxi's desk",
                "Xuxi desk",
                "Xushi's desk",
                "Xushi desk",
                "Shushti's death",
                "Shushti death",
                "Shushdi death",
                "Shush this death",
            })

        for alias in aliases:
            normalized = self._normalize_node_name(alias)
            if normalized:
                self._node_aliases.setdefault(normalized, node.name)

    def _closest_node_alias(self, name: str) -> Optional[str]:
        """Resolve small ASR/spelling mistakes in destination names."""
        normalized = self._normalize_node_name(name)
        if not normalized:
            return None

        best_alias = None
        best_score = 0.0
        for alias in self._node_aliases:
            score = difflib.SequenceMatcher(None, normalized, alias).ratio()
            if score > best_score:
                best_alias = alias
                best_score = score

        if best_alias is not None and best_score >= 0.90:
            return self._node_aliases[best_alias]
        return None


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
            self._current_path = [goal_node]
            self._waypoint_index = 0
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
    PIVOT_HEADING_ERROR_RAD = _env_float("NAV_PIVOT_HEADING_ERROR_RAD", math.radians(20.0))
    FORWARD_HAZARD_CONE_RAD = _env_float("NAV_FORWARD_HAZARD_CONE_RAD", math.radians(20.0))
    MIN_PIVOT_YAW_RATE = _env_float("NAV_MIN_PIVOT_YAW_RATE", 0.30)

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
        path_nearest = obstacle_grid.path_obstacle_m

        if path_nearest > self.avoidance_distance:
            return self._compute_direct_velocity(goal_direction, goal_distance, path_nearest)

        histogram = self._build_histogram(obstacle_grid)
        free_sectors = self._find_free_sectors(histogram)

        if not free_sectors:
            if path_nearest > self.safety_distance:
                vx = self._modulate_speed(min(self.max_linear_speed, 0.12), path_nearest)
                return VelocityCommand(vx=vx, vy=0.0, vyaw=0.0)
            return VelocityCommand(0.0, 0.0, 0.0)

        if (
            path_nearest <= self.safety_distance
            and abs(goal_direction) < self.PIVOT_HEADING_ERROR_RAD
        ):
            return VelocityCommand(0.0, 0.0, 0.0)

        if (
            goal_distance > 0.5
            and abs(goal_direction) >= self.PIVOT_HEADING_ERROR_RAD
        ):
            vyaw = self._pivot_yaw_rate(goal_direction)
            self._prev_heading = float(vyaw)
            return VelocityCommand(vx=0.0, vy=0.0, vyaw=float(vyaw))

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
        nearest = path_nearest
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

    def _compute_direct_velocity(
        self,
        goal_direction: float,
        goal_distance: float,
        path_nearest: float,
    ) -> VelocityCommand:
        """Drive the mapped path directly when the path corridor is clear."""
        if (
            goal_distance > 0.5
            and abs(goal_direction) >= self.PIVOT_HEADING_ERROR_RAD
        ):
            vyaw = self._pivot_yaw_rate(goal_direction)
            self._prev_heading = float(vyaw)
            return VelocityCommand(vx=0.0, vy=0.0, vyaw=float(vyaw))

        base_speed = self._modulate_speed(self.max_linear_speed, path_nearest)
        if goal_distance < 0.5:
            base_speed = min(base_speed, 0.1)

        vyaw = float(np.clip(goal_direction, -self.max_yaw_rate, self.max_yaw_rate))
        turn_factor = 1.0 - min(abs(vyaw) / max(self.max_yaw_rate, 1e-6), 1.0) * 0.5
        vx = base_speed * turn_factor
        self._prev_heading = vyaw
        return VelocityCommand(vx=vx, vy=0.0, vyaw=vyaw)

    def compute_avoidance(self, obstacle_grid: ObstacleGrid) -> VelocityCommand:
        """Reactive avoidance: turn away from nearest obstacle."""
        bearing = obstacle_grid.nearest_obstacle_bearing
        # Turn away from obstacle
        vyaw = self.max_yaw_rate if bearing < 0 else -self.max_yaw_rate

        nearest = obstacle_grid.nearest_obstacle_m
        vx = self._modulate_speed(0.1, nearest)

        return VelocityCommand(vx=vx, vy=0.0, vyaw=vyaw)

    def _pivot_yaw_rate(self, heading_error: float) -> float:
        """Return a decisive in-place turn rate for large heading corrections."""
        if abs(heading_error) < 1e-6:
            return 0.0
        yaw_limit = abs(self.max_yaw_rate)
        min_pivot_rate = min(self.MIN_PIVOT_YAW_RATE, yaw_limit)
        requested = min(abs(heading_error), yaw_limit)
        magnitude = max(requested, min_pivot_rate)
        return math.copysign(magnitude, heading_error)

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
        pivot_hard_stop_distance: float = 0.2,
        forward_hazard_cone_rad: float = math.radians(20.0),
    ):
        """Configure safety thresholds.

        Args:
            safety_distance: E-stop if obstacle closer than this (meters).
            avoidance_distance: Scale speed down between safety and this (meters).
            stuck_timeout: Trigger stuck event after this many seconds without progress.
            pivot_hard_stop_distance: Minimum distance allowed for in-place turning.
            forward_hazard_cone_rad: Bearing cone treated as forward path blockage.
        """
        self.safety_distance = safety_distance
        self.avoidance_distance = avoidance_distance
        self.stuck_timeout = stuck_timeout
        self.pivot_hard_stop_distance = pivot_hard_stop_distance
        self.forward_hazard_cone_rad = forward_hazard_cone_rad

    def filter_command(
        self,
        cmd: VelocityCommand,
        nearest_obstacle_m: float,
        nearest_obstacle_bearing: float = 0.0,
        ground_plane_valid: bool = True,
        seconds_since_progress: float = 0.0,
    ) -> Tuple[VelocityCommand, Optional[str]]:
        """Filter a velocity command through safety checks (priority-ordered).

        Returns:
            Tuple of (possibly zeroed command, optional event string).
            Event string is None when no safety condition triggered.
        """
        pivot_only = abs(cmd.vx) < 1e-3 and abs(cmd.vy) < 1e-3 and abs(cmd.vyaw) > 1e-3
        forward_hazard = abs(nearest_obstacle_bearing) <= self.forward_hazard_cone_rad
        hard_stop = nearest_obstacle_m <= self.pivot_hard_stop_distance

        # Priority 1: E-STOP. Close objects outside the forward cone are handled
        # by VFH steering unless they are inside the hard-stop distance.
        if (
            nearest_obstacle_m <= self.safety_distance
            and (hard_stop or forward_hazard)
            and not (pivot_only and nearest_obstacle_m > self.pivot_hard_stop_distance)
        ):
            return VelocityCommand(0.0, 0.0, 0.0), "e_stop:obstacle_too_close"

        # Priority 2: Cliff detection
        if not ground_plane_valid:
            return VelocityCommand(0.0, 0.0, 0.0), "e_stop:ground_plane_missing"

        # Priority 3: Stuck detection
        if seconds_since_progress >= self.stuck_timeout:
            return VelocityCommand(0.0, 0.0, 0.0), "stuck:no_progress"

        # Priority 5: Speed modulation in avoidance zone
        if nearest_obstacle_m < self.avoidance_distance and (hard_stop or forward_hazard):
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

    def set_pose(self, x: float, y: float, yaw: float = 0.0):
        """Set the dead-reckoned pose from a known map/location anchor."""
        with self._lock:
            self._pose = RobotPose(x=x, y=y, yaw=yaw, timestamp=time.time())
            self._last_update = time.monotonic()

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
    _global_status_callback: Optional[Callable[[str], None]] = None

    # Configuration (overridable via environment variables)
    NAV_LOOP_HZ: int = _env_int("NAV_LOOP_HZ", 10)
    SAFETY_DISTANCE_M: float = _env_float("NAV_SAFETY_DISTANCE", 0.4)
    AVOIDANCE_DISTANCE_M: float = _env_float("NAV_AVOIDANCE_DISTANCE", 0.8)
    MAX_LINEAR_SPEED: float = _env_float("NAV_MAX_LINEAR_SPEED", 0.3)
    MAX_YAW_RATE: float = _env_float("NAV_MAX_YAW_RATE", 0.5)
    GOAL_TOLERANCE_M: float = _env_float("NAV_GOAL_TOLERANCE", 0.3)
    STUCK_TIMEOUT_S: float = _env_float("NAV_STUCK_TIMEOUT", 10.0)
    PIVOT_HARD_STOP_DISTANCE_M: float = _env_float("NAV_PIVOT_HARD_STOP_DISTANCE", 0.2)
    FORWARD_HAZARD_CONE_RAD: float = _env_float(
        "NAV_FORWARD_HAZARD_CONE_RAD",
        math.radians(20.0),
    )
    CLOSE_OBSTACLE_CONFIRM_S: float = _env_float("NAV_CLOSE_OBSTACLE_CONFIRM_S", 0.7)
    CLOSE_OBSTACLE_CONFIRM_READINGS: int = _env_int("NAV_CLOSE_OBSTACLE_CONFIRM_READINGS", 3)
    FORWARD_SPEED: float = _env_float("NAV_FORWARD_SPEED", 0.45)
    FORWARD_STOP_DISTANCE_M: float = _env_float("NAV_FORWARD_STOP_DISTANCE", 0.50)
    FORWARD_MAX_SECONDS: float = _env_float("NAV_FORWARD_MAX_SECONDS", 15.0)
    FORWARD_MIN_SECONDS: float = _env_float("NAV_FORWARD_MIN_SECONDS", 0.50)
    FORWARD_COMMAND_PERIOD_S: float = _env_float("NAV_FORWARD_COMMAND_PERIOD", 0.20)
    FORWARD_ACTUAL_SPEED_RATIO: float = _env_float("NAV_FORWARD_ACTUAL_SPEED_RATIO", 1.40)
    DEPTH_READY_TIMEOUT_S: float = _env_float("NAV_DEPTH_READY_TIMEOUT", 2.0)
    OBSTACLE_LIMITED_TIMEOUT_S: float = _env_float("NAV_OBSTACLE_LIMITED_TIMEOUT", 8.0)
    OBSTACLE_LIMITED_SPEED_MPS: float = _env_float("NAV_OBSTACLE_LIMITED_SPEED", 0.12)
    YAW_PROGRESS_TOLERANCE_RAD: float = _env_float(
        "NAV_YAW_PROGRESS_TOLERANCE_RAD",
        math.radians(5.0),
    )
    ODOMETRY_LINEAR_SPEED_RATIO: float = _env_float(
        "NAV_ODOMETRY_LINEAR_SPEED_RATIO",
        0.70,
    )
    ODOMETRY_YAW_RATE_RATIO: float = _env_float("NAV_ODOMETRY_YAW_RATE_RATIO", 1.0)
    DEPTH_STOP_WHEN_IDLE: bool = _env_flag("NAV_DEPTH_STOP_WHEN_IDLE", True)

    @classmethod
    def get_instance(cls) -> "NavCore":
        """Return the shared NavCore singleton, creating it on first call."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def set_status_callback(cls, callback: Optional[Callable[[str], None]]) -> None:
        """Register a callback for terminal navigation status updates."""
        with cls._instance_lock:
            cls._global_status_callback = callback
            if cls._instance is not None:
                cls._instance._on_status_change = callback

    def __init__(self):
        """Initialize all sub-components: depth processor, planners, safety, odometry."""
        self._state = NavState.IDLE
        self._goal: Optional[NavGoal] = None
        self._last_stop_reason: Optional[str] = None
        self._on_status_change = type(self)._global_status_callback
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
            pivot_hard_stop_distance=self.PIVOT_HARD_STOP_DISTANCE_M,
            forward_hazard_cone_rad=self.FORWARD_HAZARD_CONE_RAD,
        )
        self._odometry = OdometryProvider()

        # Global planner (optional, requires map)
        self._topo_map = TopologicalMap()
        self._global_planner = GlobalPlanner(self._topo_map)

        # Progress tracking
        self._last_progress_pose = RobotPose()
        self._last_progress_time = time.monotonic()
        self._obstacle_limited_since: Optional[float] = None
        self._close_obstacle_first_seen_at: Optional[float] = None
        self._close_obstacle_count = 0
        self._close_obstacle_min_distance = float("inf")

        # Go2 macros (lazy init)
        self._go2 = None

        # Load map if configured
        map_file = os.environ.get("NAV_MAP_FILE", "")
        if map_file and Path(map_file).exists():
            if self._topo_map.load_from_file(map_file):
                self._anchor_initial_pose()

        logger.info(
            "NavCore: initialized (depth=%s, map=%s, loop=%d Hz)",
            self._depth_processor.backend,
            "loaded" if self._topo_map.is_loaded else "none",
            self.NAV_LOOP_HZ,
        )
        logger.info(
            "NavCore: config max_vx=%.2f max_vyaw=%.2f safety=%.2f "
            "avoidance=%.2f goal_tol=%.2f odom_linear_ratio=%.2f "
            "odom_yaw_ratio=%.2f",
            self.MAX_LINEAR_SPEED,
            self.MAX_YAW_RATE,
            self.SAFETY_DISTANCE_M,
            self.AVOIDANCE_DISTANCE_M,
            self.GOAL_TOLERANCE_M,
            self.ODOMETRY_LINEAR_SPEED_RATIO,
            self.ODOMETRY_YAW_RATE_RATIO,
        )

    def _anchor_initial_pose(self):
        """Set initial odometry to the configured map start location, if present."""
        initial_location = os.environ.get("NAV_INITIAL_LOCATION", "charging_station")
        node = self._topo_map.get_node(initial_location)
        if node is None:
            return

        heading_deg = _env_float("NAV_INITIAL_HEADING_DEGREES", 0.0)
        self._odometry.set_pose(node.x, node.y, math.radians(heading_deg))
        self._reset_progress_tracker()
        logger.info(
            "NavCore: initial pose anchored to '%s' at (%.2f, %.2f), heading %.0f deg",
            node.name,
            node.x,
            node.y,
            heading_deg,
        )

    def _ensure_go2(self):
        """Lazily initialize the Go2Macros motor controller."""
        if self._go2 is None:
            self._go2 = _get_go2_macros()

    def _ensure_depth_running(self) -> bool:
        """Start or restart depth capture before a command that requires it."""
        start = getattr(self._depth_processor, "start", None)
        if callable(start):
            start()
        return bool(getattr(self._depth_processor, "is_available", False))

    def _stop_depth_when_idle(self) -> None:
        """Release depth camera resources between navigation actions."""
        if not self.DEPTH_STOP_WHEN_IDLE:
            return
        stop = getattr(self._depth_processor, "stop", None)
        if callable(stop):
            stop()

    def _notify_status_change(self, message: str) -> None:
        """Notify the host application of a terminal navigation status change."""
        callback = self._on_status_change
        if callback is None:
            return
        try:
            callback(message)
        except Exception:
            logger.exception("NavCore: status callback failed")

    @staticmethod
    def _goal_display_name(goal: NavGoal) -> str:
        """Return a friendly destination name for status updates."""
        return goal.label or "the destination"

    def _describe_known_destinations(self) -> str:
        """Return a concise destination list for spoken failure reports."""
        labels = self._topo_map.list_destination_labels()
        if not labels:
            return ""
        if len(labels) <= 4:
            return ", ".join(labels)
        return ", ".join(labels[:4]) + ", and others"

    def _abort_active_navigation(
        self,
        goal: NavGoal,
        reason: str,
        message: str,
        state: NavState = NavState.STUCK,
    ) -> None:
        """Stop motion, clear the active goal, and report a terminal nav failure."""
        self._ensure_go2()
        if self._go2 and getattr(self._go2, "available", False):
            try:
                self._go2.stop_move()
            except Exception:
                logger.debug("NavCore: stop_move failed while aborting navigation", exc_info=True)

        with self._state_lock:
            self._state = state
            self._goal = None
            self._last_stop_reason = reason
            self._global_planner.clear()
            self._obstacle_limited_since = None
            self._reset_close_obstacle_confirmation()

        logger.warning("NavCore: %s", reason)
        self._notify_status_change(message)
        self._stop_depth_when_idle()

    def _send_motion_command(self, cmd: VelocityCommand, goal: NavGoal) -> bool:
        """Send a Go2 velocity command and fail loudly if motors are unavailable."""
        self._ensure_go2()
        if not self._go2 or not getattr(self._go2, "available", False):
            self._abort_active_navigation(
                goal,
                "Robot control unavailable",
                f"I did not move toward {self._goal_display_name(goal)} because "
                "robot motor control is unavailable.",
                state=NavState.E_STOP,
            )
            return False

        try:
            self._go2.move(vx=cmd.vx, vy=cmd.vy, vyaw=cmd.vyaw)
            return True
        except Exception:
            logger.exception("NavCore: failed to send Go2 move command")
            self._abort_active_navigation(
                goal,
                "Robot control command failed",
                f"I stopped before reaching {self._goal_display_name(goal)} because "
                "the robot motor command failed.",
                state=NavState.E_STOP,
            )
            return False

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
            self._notify_status_change(
                f"I could not navigate to {destination} because no navigation map is loaded."
            )
            return False

        pose = self._odometry.get_pose()
        goal_node = self._topo_map.get_node(destination)
        if goal_node is None:
            known_destinations = self._describe_known_destinations()
            with self._state_lock:
                self._state = NavState.IDLE
                self._goal = None
                self._last_stop_reason = f"Unknown destination: {destination}"
                self._global_planner.clear()
            logger.warning("NavCore: unknown destination '%s'", destination)
            if known_destinations:
                self._notify_status_change(
                    f"I did not move because I do not recognize {destination} "
                    f"as a mapped destination. I know destinations such as {known_destinations}."
                )
            else:
                self._notify_status_change(
                    f"I did not move because I do not recognize {destination} "
                    "as a mapped destination."
                )
            return False

        goal_label = self._topo_map.get_node_label(goal_node)
        dist_to_goal = math.hypot(goal_node.x - pose.x, goal_node.y - pose.y)
        if dist_to_goal <= self.GOAL_TOLERANCE_M:
            if self._go2 and getattr(self._go2, "available", False):
                try:
                    self._go2.stop_move()
                except Exception:
                    logger.debug(
                        "NavCore: stop_move failed while confirming current location",
                        exc_info=True,
                    )
            with self._state_lock:
                self._state = NavState.IDLE
                self._goal = None
                self._last_stop_reason = f"Already at {goal_label}"
                self._global_planner.clear()
            logger.info(
                "NavCore: already at '%s' (dist=%.2fm), no movement needed",
                goal_label,
                dist_to_goal,
            )
            self._notify_status_change(f"I am already at {goal_label}.")
            self._stop_depth_when_idle()
            return True

        path = self._global_planner.plan_path(pose, destination)
        if path is None:
            with self._state_lock:
                self._state = NavState.IDLE
                self._goal = None
                self._last_stop_reason = f"No route to {goal_label}"
                self._global_planner.clear()
            self._notify_status_change(
                f"I did not move because I do not have a mapped route to {goal_label}."
            )
            return False

        if not self._wait_for_depth_grid(self.DEPTH_READY_TIMEOUT_S):
            reason = "E-STOP: depth grid unavailable"
            self._ensure_go2()
            if self._go2 and getattr(self._go2, "available", False):
                self._go2.stop_move()
            with self._state_lock:
                self._state = NavState.E_STOP
                self._goal = None
                self._last_stop_reason = reason
                self._global_planner.clear()
            logger.warning("NavCore: %s before navigating to '%s'", reason, destination)
            self._notify_status_change(
                f"I did not move toward {goal_label} because my depth grid was not available."
            )
            self._stop_depth_when_idle()
            return False

        with self._state_lock:
            self._goal = NavGoal(
                goal_type="semantic",
                x=path[-1].x,
                y=path[-1].y,
                label=destination,
            )
            self._state = NavState.NAVIGATING
            self._last_stop_reason = None
            self._reset_progress_tracker()

        self._ensure_running()
        logger.info("NavCore: navigating to '%s' via %d waypoints", destination, len(path))
        return True

    def _wait_for_depth_grid(self, timeout_s: float) -> bool:
        """Wait briefly for the depth capture thread to publish its first grid."""
        if not self._ensure_depth_running():
            return False

        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            if self._depth_processor.get_obstacle_grid() is not None:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def set_location(self, location: str, heading_rad: float = 0.0) -> bool:
        """Anchor the dead-reckoned pose to a known map node after manual relocation."""
        if not self._topo_map.is_loaded:
            logger.warning("NavCore: no map loaded, cannot set location to '%s'", location)
            return False

        node = self._topo_map.get_node(location)
        if node is None:
            logger.warning("NavCore: unknown location anchor '%s'", location)
            return False

        with self._state_lock:
            self._state = NavState.IDLE
            self._goal = None
            self._last_stop_reason = None
            self._global_planner.clear()
            self._reset_close_obstacle_confirmation()

        self._odometry.set_pose(node.x, node.y, heading_rad)
        self._reset_progress_tracker()
        self._ensure_go2()
        if self._go2 and getattr(self._go2, "available", False):
            self._go2.stop_move()
        logger.info(
            "NavCore: location anchored to '%s' at (%.2f, %.2f), heading %.0f deg",
            node.name,
            node.x,
            node.y,
            math.degrees(heading_rad),
        )
        return True

    def move_relative(self, distance: float, angle: float = 0.0) -> bool:
        """Move a given distance (meters) at a given angle offset (radians) from current heading."""
        if not self._wait_for_depth_grid(self.DEPTH_READY_TIMEOUT_S):
            logger.warning("NavCore: E-STOP: depth grid unavailable before relative move")
            self._stop_depth_when_idle()
            return False

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
            self._last_stop_reason = None
            self._reset_progress_tracker()

        self._ensure_running()
        logger.info("NavCore: moving relative d=%.2f a=%.2f", distance, angle)
        return True

    def move_forward_guarded(
        self,
        max_distance_m: Optional[float] = None,
        stop_distance_m: Optional[float] = None,
        speed: Optional[float] = None,
        max_seconds: Optional[float] = None,
        command_period_s: Optional[float] = None,
    ) -> str:
        """Move forward continuously while a raw depth watchdog stays clear.

        This is the hardware-safe forward primitive validated on CAIL-E. It does
        not declare success from dead-reckoned distance. Instead, it keeps a
        continuous Go2 gait command active and stops immediately when the raw
        center depth band sees an obstacle at or inside stop_distance_m.

        Args:
            max_distance_m: Optional requested travel distance. Until real
                odometry is available, this is converted to a conservative time
                backstop rather than used as an arrival guarantee.
            stop_distance_m: Depth threshold at which to stop.
            speed: Forward command velocity.
            max_seconds: Safety timeout. If omitted, derived from max_distance_m
                when provided, otherwise NAV_FORWARD_MAX_SECONDS.
            command_period_s: How often to refresh the Go2 move command.

        Returns:
            Human-readable result for the agent/tool layer.
        """
        stop_distance = stop_distance_m or self.FORWARD_STOP_DISTANCE_M
        forward_speed = speed or self.FORWARD_SPEED
        command_period = (
            self.FORWARD_COMMAND_PERIOD_S
            if command_period_s is None
            else command_period_s
        )

        if max_seconds is None:
            if max_distance_m is not None:
                actual_speed_estimate = max(
                    forward_speed * self.FORWARD_ACTUAL_SPEED_RATIO,
                    0.05,
                )
                max_seconds = max(
                    self.FORWARD_MIN_SECONDS,
                    max_distance_m / actual_speed_estimate,
                )
            else:
                max_seconds = self.FORWARD_MAX_SECONDS

        if not self._ensure_depth_running():
            return "Cannot move forward: no depth camera is available."

        initial_reading = self._depth_processor.get_center_depth_reading()
        if initial_reading is None:
            self._stop_depth_when_idle()
            return "Cannot move forward: no reliable center depth reading is available."

        if initial_reading.distance_m <= stop_distance:
            self._stop_depth_when_idle()
            return (
                "Already stopped: obstacle is "
                f"{initial_reading.distance_m:.2f}m ahead."
            )

        self.stop(stop_depth=False)
        self._ensure_go2()
        if not self._go2 or not getattr(self._go2, "available", False):
            self._stop_depth_when_idle()
            return "Cannot move forward: robot motor control is unavailable."

        with self._state_lock:
            self._goal = NavGoal(
                goal_type="guarded_forward",
                x=max_distance_m or 0.0,
                y=0.0,
                label="guarded_forward",
            )
            self._state = NavState.NAVIGATING
            self._last_stop_reason = None
            self._reset_progress_tracker()

        start = time.monotonic()
        last_reading: CenterDepthReading = initial_reading
        stop_reason = "timeout"

        try:
            while time.monotonic() - start < max_seconds:
                reading = self._depth_processor.get_center_depth_reading()
                if reading is None:
                    stop_reason = "depth_unavailable"
                    with self._state_lock:
                        self._state = NavState.E_STOP
                        self._last_stop_reason = "E-STOP: center depth unavailable"
                    break

                last_reading = reading
                if reading.distance_m <= stop_distance:
                    stop_reason = "obstacle"
                    with self._state_lock:
                        self._last_stop_reason = f"Stopped: obstacle at {reading.distance_m:.2f}m"
                    break

                if self._go2 and getattr(self._go2, "available", False):
                    self._go2.move(vx=forward_speed, vy=0.0, vyaw=0.0)

                if command_period > 0:
                    time.sleep(command_period)
        finally:
            if self._go2 and getattr(self._go2, "available", False):
                self._go2.stop_move()

            with self._state_lock:
                if self._state != NavState.E_STOP:
                    self._state = NavState.IDLE
                    if stop_reason != "obstacle":
                        self._last_stop_reason = None
                self._goal = None
            self._stop_depth_when_idle()

        if stop_reason == "obstacle":
            return (
                "Stopped forward movement: obstacle is "
                f"{last_reading.distance_m:.2f}m ahead."
            )
        if stop_reason == "depth_unavailable":
            return "Emergency stopped: center depth reading became unavailable."
        return (
            "Stopped forward movement after the safety timeout; last center "
            f"depth was {last_reading.distance_m:.2f}m."
        )

    def turn(self, angle_rad: float) -> bool:
        """Rotate in place by the given angle (radians, positive=left)."""
        if not self._wait_for_depth_grid(self.DEPTH_READY_TIMEOUT_S):
            logger.warning("NavCore: E-STOP: depth grid unavailable before turn")
            self._stop_depth_when_idle()
            return False

        pose = self._odometry.get_pose()
        with self._state_lock:
            self._goal = NavGoal(
                goal_type="relative",
                x=pose.x,
                y=pose.y,
                yaw=pose.yaw + angle_rad,
            )
            self._state = NavState.NAVIGATING
            self._last_stop_reason = None
            self._reset_progress_tracker()

        self._ensure_running()
        return True

    def stop(self, stop_depth: bool = True):
        """Cancel current navigation and send stop command to the robot."""
        with self._state_lock:
            self._state = NavState.IDLE
            self._goal = None
            self._last_stop_reason = None
            self._global_planner.clear()
        self._ensure_go2()
        if self._go2 and getattr(self._go2, "available", False):
            self._go2.stop_move()
        if stop_depth:
            self._stop_depth_when_idle()
        logger.info("NavCore: navigation stopped")

    def resume(self):
        """Clear E-STOP state and return to IDLE. Does not restart previous goal."""
        with self._state_lock:
            if self._state == NavState.E_STOP:
                self._state = NavState.IDLE
                self._last_stop_reason = None
                self._reset_close_obstacle_confirmation()
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
            last_stop_reason = self._last_stop_reason

        pose = self._odometry.get_pose()
        parts = [f"Navigation state: {state.value}"]
        parts.append(f"Position: ({pose.x:.1f}, {pose.y:.1f}), heading: {math.degrees(pose.yaw):.0f} deg")

        if goal:
            if goal.label:
                parts.append(f"Destination: {goal.label}")
            dist = math.hypot(goal.x - pose.x, goal.y - pose.y)
            parts.append(f"Distance to goal: {dist:.1f}m")

        if last_stop_reason:
            parts.append(last_stop_reason)

        grid = self._depth_processor.get_obstacle_grid()
        if grid:
            if grid.nearest_obstacle_m < float("inf"):
                parts.append(f"Nearest obstacle anywhere: {grid.nearest_obstacle_m:.2f}m")
            else:
                parts.append("No obstacles detected")
            if grid.path_obstacle_m < float("inf"):
                parts.append(
                    f"Path obstacle: {grid.path_obstacle_m:.2f}m "
                    f"({grid.path_obstacle_points} depth points)"
                )
            else:
                parts.append("Path corridor clear")

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
        labels = self._topo_map.list_destination_labels()
        return "Available destinations: " + ", ".join(labels or names)

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

            if (
                state in {NavState.IDLE, NavState.STUCK, NavState.E_STOP}
                or goal is None
            ):
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
        if state in {NavState.IDLE, NavState.STUCK, NavState.E_STOP}:
            return

        # 1. Read sensors
        grid = self._depth_processor.get_obstacle_grid()
        pose = self._odometry.get_pose()
        self._update_progress(pose)

        path_dist = grid.path_obstacle_m if grid else float("inf")
        path_bearing = grid.path_obstacle_bearing if grid else 0.0

        # 2. Check if goal reached
        dist_to_goal = math.hypot(goal.x - pose.x, goal.y - pose.y)
        if dist_to_goal < self.GOAL_TOLERANCE_M:
            self._ensure_go2()
            if self._go2 and getattr(self._go2, "available", False):
                self._go2.stop_move()
            with self._state_lock:
                self._state = NavState.IDLE
                self._goal = None
                self._last_stop_reason = None
                self._global_planner.clear()
            logger.info("NavCore: goal reached (dist=%.2fm)", dist_to_goal)
            self._stop_depth_when_idle()
            if goal.goal_type == "semantic":
                self._notify_status_change(f"I arrived at {self._goal_display_name(goal)}.")
            return

        # 3. Compute velocity command
        if state == NavState.NAVIGATING:
            if goal.goal_type == "semantic":
                waypoint = self._global_planner.get_next_waypoint(pose, self.GOAL_TOLERANCE_M)
                if waypoint is None:
                    # Global path complete
                    with self._state_lock:
                        self._state = NavState.IDLE
                        self._goal = None
                        self._last_stop_reason = None
                        self._global_planner.clear()
                    self._stop_depth_when_idle()
                    self._notify_status_change(f"I arrived at {self._goal_display_name(goal)}.")
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
                self._abort_active_navigation(
                    goal,
                    "E-STOP: depth grid unavailable",
                    f"I stopped before reaching {self._goal_display_name(goal)} "
                    "because my depth grid became unavailable.",
                    state=NavState.E_STOP,
                )
                return

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

        # 4. Safety filter
        seconds_since_progress = time.monotonic() - self._last_progress_time
        cmd, event = self._safety.filter_command(
            cmd,
            nearest_obstacle_m=path_dist,
            nearest_obstacle_bearing=path_bearing,
            seconds_since_progress=seconds_since_progress,
        )

        if event:
            if event.startswith("e_stop"):
                if (
                    event == "e_stop:obstacle_too_close"
                    and self._should_defer_close_obstacle_stop(path_dist, path_bearing)
                ):
                    return

                if event == "e_stop:obstacle_too_close":
                    reason = f"E-STOP: path obstacle at {path_dist:.2f}m"
                else:
                    reason = f"E-STOP: {event.split(':', 1)[-1].replace('_', ' ')}"
                logger.warning("NavCore: safety event: %s (%s)", event, reason)
                if event == "e_stop:obstacle_too_close":
                    message = (
                        f"I stopped before reaching {self._goal_display_name(goal)} "
                        f"because my depth sensor reported something in my path "
                        f"at {path_dist:.2f} meters."
                    )
                else:
                    message = (
                        f"I stopped before reaching {self._goal_display_name(goal)} "
                        f"because {reason.lower()}."
                    )
                self._abort_active_navigation(goal, reason, message, state=NavState.E_STOP)
                return
            elif event.startswith("stuck"):
                reason = "Stuck: no progress toward the goal"
                self._abort_active_navigation(
                    goal,
                    reason,
                    f"I stopped before reaching {self._goal_display_name(goal)} "
                    "because I was not making progress.",
                )
                return

        self._reset_close_obstacle_confirmation()

        if self._handle_obstacle_limited_motion(cmd, path_dist, goal, dist_to_goal):
            return

        if (
            dist_to_goal > self.GOAL_TOLERANCE_M
            and abs(cmd.vx) < 1e-3
            and abs(cmd.vy) < 1e-3
            and abs(cmd.vyaw) < 1e-3
        ):
            self._abort_active_navigation(
                goal,
                "Blocked: no safe motion command",
                f"I did not move toward {self._goal_display_name(goal)} because "
                "my local planner could not find a safe motion command.",
            )
            self._obstacle_limited_since = None
            return

        # 5. Execute
        if not self._send_motion_command(cmd, goal):
            self._obstacle_limited_since = None
            return

        # 6. Update odometry (dead-reckoning)
        dt = 1.0 / self.NAV_LOOP_HZ
        odometry_cmd = VelocityCommand(
            vx=cmd.vx * self.ODOMETRY_LINEAR_SPEED_RATIO,
            vy=cmd.vy * self.ODOMETRY_LINEAR_SPEED_RATIO,
            vyaw=cmd.vyaw * self.ODOMETRY_YAW_RATE_RATIO,
        )
        self._odometry.update_from_velocity(odometry_cmd, dt)

        # 7. Update progress tracker
        self._update_progress(self._odometry.get_pose())

    def _should_defer_close_obstacle_stop(
        self,
        nearest_dist: float,
        nearest_bearing: float,
    ) -> bool:
        """Hold briefly on borderline close obstacles to reject transient frames."""
        if (
            nearest_dist <= self.PIVOT_HARD_STOP_DISTANCE_M
            or abs(nearest_bearing) > self.FORWARD_HAZARD_CONE_RAD
        ):
            self._reset_close_obstacle_confirmation()
            return False

        now = time.monotonic()
        if self._close_obstacle_first_seen_at is None:
            self._close_obstacle_first_seen_at = now
            self._close_obstacle_count = 0
            self._close_obstacle_min_distance = float("inf")
            logger.info(
                "NavCore: holding to confirm close obstacle at %.2fm before E-STOP",
                nearest_dist,
            )

        self._close_obstacle_count += 1
        self._close_obstacle_min_distance = min(
            self._close_obstacle_min_distance,
            nearest_dist,
        )

        confirmed_long_enough = (
            now - self._close_obstacle_first_seen_at >= self.CLOSE_OBSTACLE_CONFIRM_S
        )
        confirmed_readings = self._close_obstacle_count >= self.CLOSE_OBSTACLE_CONFIRM_READINGS
        if confirmed_long_enough and confirmed_readings:
            return False

        self._ensure_go2()
        if self._go2 and getattr(self._go2, "available", False):
            self._go2.stop_move()
        return True

    def _reset_close_obstacle_confirmation(self) -> None:
        """Clear transient close-obstacle confirmation state."""
        self._close_obstacle_first_seen_at = None
        self._close_obstacle_count = 0
        self._close_obstacle_min_distance = float("inf")

    def _handle_obstacle_limited_motion(
        self,
        cmd: VelocityCommand,
        nearest_dist: float,
        goal: NavGoal,
        dist_to_goal: float,
    ) -> bool:
        """Stop when the robot spends too long crawling near an obstacle."""
        is_motion_blocked = (
            nearest_dist < self.AVOIDANCE_DISTANCE_M
            and dist_to_goal > self.GOAL_TOLERANCE_M
            and (
                0.0 < cmd.vx <= self.OBSTACLE_LIMITED_SPEED_MPS
                or (abs(cmd.vx) < 1e-3 and abs(cmd.vyaw) < 1e-3)
            )
        )

        if not is_motion_blocked:
            self._obstacle_limited_since = None
            return False

        now = time.monotonic()
        if self._obstacle_limited_since is None:
            self._obstacle_limited_since = now
            return False

        if now - self._obstacle_limited_since < self.OBSTACLE_LIMITED_TIMEOUT_S:
            return False

        reason = f"Blocked: nearby obstacle or wall at {nearest_dist:.2f}m"
        self._abort_active_navigation(
            goal,
            reason,
            f"I stopped before reaching {self._goal_display_name(goal)} because "
            f"my depth sensor kept seeing something nearby at {nearest_dist:.2f} meters "
            "and I was only able to crawl.",
        )
        return True

    def _reset_progress_tracker(self):
        """Reset the stuck-detection timer to now."""
        self._last_progress_pose = self._odometry.get_pose()
        self._last_progress_time = time.monotonic()
        self._obstacle_limited_since = None
        self._reset_close_obstacle_confirmation()

    @staticmethod
    def _angular_delta(a: float, b: float) -> float:
        """Return the shortest absolute angular difference between two headings."""
        return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

    def _update_progress(self, current_pose: RobotPose):
        """Update stuck detection when position or heading has meaningfully changed."""
        dist_moved = math.hypot(
            current_pose.x - self._last_progress_pose.x,
            current_pose.y - self._last_progress_pose.y,
        )
        yaw_moved = self._angular_delta(current_pose.yaw, self._last_progress_pose.yaw)
        if dist_moved > 0.1 or yaw_moved > self.YAW_PROGRESS_TOLERANCE_RAD:
            self._last_progress_pose = current_pose
            self._last_progress_time = time.monotonic()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self):
        """Stop navigation, join the background thread, and release depth camera."""
        with self._state_lock:
            self._state = NavState.IDLE
            self._goal = None
            self._last_stop_reason = None
            self._global_planner.clear()

        if self._go2 and getattr(self._go2, "available", False):
            try:
                self._go2.stop_move()
            except Exception:
                logger.debug("NavCore: stop_move failed during shutdown", exc_info=True)

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
