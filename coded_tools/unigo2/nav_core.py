
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
import importlib
import json
import math
import os
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from coded_tools.unigo2.depth_processor import (
    CenterDepthReading,
    ObstacleGrid,
    _env_flag,
    _env_float,
    _env_int,
)
from coded_tools.unigo2.obstacle_confirmation import ObstacleConfirmationTracker
from coded_tools.unigo2.obstacle_grid_utils import (
    ObstacleGridSpec,
    build_obstacle_grid,
    is_transverse_wall,
    occupied_xy_points,
)
from coded_tools.unigo2.obstacle_provider import create_default_obstacle_provider

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_FILE = REPO_ROOT / "maps" / "cail_lab.json"


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


def _configured_map_file() -> str:
    """Return the map file for production navigation, with env only as an override."""
    configured = os.environ.get("NAV_MAP_FILE")
    if configured is not None:
        return configured.strip()
    if _env_flag("NAV_SIMULATION_MODE", False):
        return ""
    return str(DEFAULT_MAP_FILE)


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


class LocalObstacleMemory:
    """Retain recent wall geometry and align it to the current robot pose.

    Scalar safety metadata always comes from the newest sensor frame. Historical
    points are used only to stabilize local geometry for steering.
    """

    def __init__(self, ttl_s: float = 0.8, max_points_per_frame: int = 3000):
        self.ttl_s = max(0.0, ttl_s)
        self.max_points_per_frame = max(100, max_points_per_frame)
        self._samples: List[Tuple[float, RobotPose, np.ndarray]] = []
        self._lock = threading.Lock()

    def clear(self) -> None:
        """Forget all retained obstacle geometry."""
        with self._lock:
            self._samples.clear()

    def update(
        self,
        current: Optional[ObstacleGrid],
        pose: RobotPose,
        now: Optional[float] = None,
    ) -> Optional[ObstacleGrid]:
        """Merge recent observations into the current robot frame."""
        if current is None or self.ttl_s <= 0.0:
            return current

        sample_time = time.monotonic() if now is None else now
        cutoff = sample_time - self.ttl_s
        current_points = occupied_xy_points(current)
        if len(current_points) > self.max_points_per_frame:
            indices = np.linspace(
                0,
                len(current_points) - 1,
                self.max_points_per_frame,
                dtype=np.int32,
            )
            current_points = current_points[indices]

        with self._lock:
            self._samples = [sample for sample in self._samples if sample[0] >= cutoff]
            if current_points.size:
                self._samples.append(
                    (
                        sample_time,
                        RobotPose(pose.x, pose.y, pose.yaw, pose.timestamp),
                        current_points,
                    )
                )
            samples = list(self._samples)

        aligned_batches = [
            self._points_in_current_frame(points, sample_pose, pose)
            for _timestamp, sample_pose, points in samples
        ]
        aligned_batches = [points for points in aligned_batches if points.size]
        if not aligned_batches:
            return current

        memory_grid = build_obstacle_grid(
            np.vstack(aligned_batches),
            ObstacleGridSpec(
                rows=current.grid.shape[0],
                cols=current.grid.shape[1],
                resolution=current.resolution,
                origin_row=current.origin_row,
                origin_col=current.origin_col,
                # Metadata is replaced below with current-frame values.
                path_corridor_half_width=0.0,
                path_obstacle_min_points=max(current.grid.size, 1),
            ),
        )
        memory_grid.timestamp = current.timestamp
        memory_grid.nearest_obstacle_m = current.nearest_obstacle_m
        memory_grid.nearest_obstacle_bearing = current.nearest_obstacle_bearing
        memory_grid.path_obstacle_m = current.path_obstacle_m
        memory_grid.path_obstacle_bearing = current.path_obstacle_bearing
        memory_grid.path_obstacle_points = current.path_obstacle_points
        return memory_grid

    @staticmethod
    def _points_in_current_frame(
        points: np.ndarray,
        sample_pose: RobotPose,
        current_pose: RobotPose,
    ) -> np.ndarray:
        """Transform robot-frame points from a sample pose to the current pose."""
        sample_cos = math.cos(sample_pose.yaw)
        sample_sin = math.sin(sample_pose.yaw)
        world_x = sample_pose.x + sample_cos * points[:, 0] - sample_sin * points[:, 1]
        world_y = sample_pose.y + sample_sin * points[:, 0] + sample_cos * points[:, 1]

        delta_x = world_x - current_pose.x
        delta_y = world_y - current_pose.y
        current_cos = math.cos(current_pose.yaw)
        current_sin = math.sin(current_pose.yaw)
        forward = current_cos * delta_x + current_sin * delta_y
        lateral = -current_sin * delta_x + current_cos * delta_y
        return np.column_stack((forward, lateral)).astype(np.float32)


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
    heading_degrees: Optional[float] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    arrival_landmarks: List[Dict[str, Any]] = field(default_factory=list)


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
                heading_degrees=(
                    float(node_data["heading_degrees"])
                    if node_data.get("heading_degrees") is not None
                    else None
                ),
                description=node_data.get("description", ""),
                tags=node_data.get("tags", []),
                aliases=node_data.get("aliases", []),
                arrival_landmarks=node_data.get("arrival_landmarks", []),
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
                    "arrival_landmarks": n.arrival_landmarks,
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

    def get_next_waypoint(
        self,
        current_pose: RobotPose,
        tolerance_m: float = 0.3,
        on_advance: Optional[Callable[[MapNode, MapNode], None]] = None,
    ) -> Optional[MapNode]:
        """Return the next waypoint to steer toward, advancing when within tolerance.

        Returns None when the final waypoint (goal) has been reached.
        """
        if not self._current_path or self._waypoint_index >= len(self._current_path):
            return None

        wp = self._current_path[self._waypoint_index]
        dist = math.hypot(wp.x - current_pose.x, wp.y - current_pose.y)

        if dist < tolerance_m:
            advanced = self.advance_current_waypoint(on_advance=on_advance)
            return advanced[1] if advanced is not None else None

        return wp

    def current_segment(self) -> Optional[Tuple[MapNode, MapNode]]:
        """Return the active directed map edge."""
        if not self._current_path or not 0 < self._waypoint_index < len(self._current_path):
            return None
        return (
            self._current_path[self._waypoint_index - 1],
            self._current_path[self._waypoint_index],
        )

    def advance_current_waypoint(
        self,
        on_advance: Optional[Callable[[MapNode, MapNode], None]] = None,
    ) -> Optional[Tuple[MapNode, Optional[MapNode]]]:
        """Accept the active waypoint and return the following waypoint, if any."""
        if not self._current_path or self._waypoint_index >= len(self._current_path):
            return None

        reached = self._current_path[self._waypoint_index]
        self._waypoint_index += 1
        upcoming = (
            self._current_path[self._waypoint_index]
            if self._waypoint_index < len(self._current_path)
            else None
        )
        if upcoming is not None:
            logger.info("GlobalPlanner: advancing to waypoint '%s'", upcoming.name)
            if on_advance is not None:
                on_advance(reached, upcoming)
        return reached, upcoming

    def clear(self):
        """Reset the current path and waypoint index."""
        self._current_path = []
        self._waypoint_index = 0

    def project_onto_current_segment(
        self,
        pose: RobotPose,
    ) -> Optional[Tuple[RobotPose, MapNode, MapNode]]:
        """Project an operator-corrected pose onto the active route edge."""
        if not self._current_path or self._waypoint_index <= 0:
            return None
        if self._waypoint_index >= len(self._current_path):
            return None

        start = self._current_path[self._waypoint_index - 1]
        target = self._current_path[self._waypoint_index]
        edge_x = target.x - start.x
        edge_y = target.y - start.y
        edge_length_sq = edge_x * edge_x + edge_y * edge_y
        if edge_length_sq <= 1e-9:
            return None

        along = (
            (pose.x - start.x) * edge_x + (pose.y - start.y) * edge_y
        ) / edge_length_sq
        along = min(1.0, max(0.0, along))
        corrected = RobotPose(
            x=start.x + along * edge_x,
            y=start.y + along * edge_y,
            yaw=math.atan2(edge_y, edge_x),
            timestamp=time.time(),
        )
        return corrected, start, target

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
    DEFAULT_PIVOT_YAW_RATE = _env_float(
        "NAV_PIVOT_YAW_RATE",
        _env_float("NAV_MIN_PIVOT_YAW_RATE", 0.50),
    )
    CORRIDOR_MIN_POINTS_PER_SIDE = 8
    CORRIDOR_MIN_LENGTH_M = 0.45
    CORRIDOR_MAX_RESIDUAL_M = 0.12
    CORRIDOR_MAX_WALL_ANGLE_RAD = math.radians(55.0)
    CORRIDOR_MAX_PARALLEL_ERROR_RAD = math.radians(10.0)
    CORRIDOR_MIN_WIDTH_M = 0.55
    CORRIDOR_MAX_WIDTH_M = 2.50
    SINGLE_WALL_TARGET_CLEARANCE_M = _env_float("NAV_WALL_CLEARANCE", 0.55)

    def __init__(
        self,
        max_linear_speed: float = 0.4,
        max_yaw_rate: float = 0.08,
        pivot_yaw_rate: Optional[float] = None,
        safety_distance: float = 0.2,
        avoidance_distance: float = 0.6,
    ):
        """Configure the local planner speed and distance thresholds.

        Args:
            max_linear_speed: Maximum forward speed in m/s.
            max_yaw_rate: Maximum steering yaw rate while translating in rad/s.
            pivot_yaw_rate: In-place yaw rate for planned map turns in rad/s.
            safety_distance: E-stop distance in meters (speed = 0 below this).
            avoidance_distance: Start slowing down at this distance in meters.
        """
        self.max_linear_speed = max_linear_speed
        self.max_yaw_rate = max_yaw_rate
        self.pivot_yaw_rate = abs(
            self.DEFAULT_PIVOT_YAW_RATE if pivot_yaw_rate is None else pivot_yaw_rate
        )
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
            goal_sector = self._angle_to_sector(goal_direction)
            best_sector = self._select_best_sector(free_sectors, goal_sector)
            target_heading = self._sector_to_angle(best_sector)
            return VelocityCommand(
                vx=0.0,
                vy=0.0,
                vyaw=self._pivot_yaw_rate(target_heading),
            )

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

    def apply_corridor_course_correction(
        self,
        cmd: VelocityCommand,
        obstacle_grid: ObstacleGrid,
    ) -> VelocityCommand:
        """Add bounded steering from visible corridor or one-sided wall geometry."""
        if cmd.vx <= 0.05 or obstacle_grid.path_obstacle_m < self.avoidance_distance:
            return cmd

        wall_geometry = self._estimate_wall_geometry(obstacle_grid)
        if wall_geometry is None:
            return cmd

        wall_heading, lateral_error, geometry_type = wall_geometry
        if geometry_type == "corridor":
            correction = 0.30 * wall_heading + 0.20 * lateral_error
        else:
            # For a single wall, first align with it, then maintain clearance.
            # lateral_error is signed so a close left wall steers right and a
            # close right wall steers left.
            alignment = float(np.clip(0.20 * wall_heading, -0.02, 0.02))
            correction = alignment + 0.25 * lateral_error
        correction_limit = min(0.06, self.max_yaw_rate * 0.75)
        corrected_yaw = float(
            np.clip(
                cmd.vyaw + correction,
                -self.max_yaw_rate,
                self.max_yaw_rate,
            )
        )
        if abs(corrected_yaw - cmd.vyaw) > correction_limit:
            corrected_yaw = cmd.vyaw + math.copysign(
                correction_limit,
                corrected_yaw - cmd.vyaw,
            )
        return VelocityCommand(vx=cmd.vx, vy=cmd.vy, vyaw=corrected_yaw)

    def _estimate_wall_geometry(
        self,
        obstacle_grid: ObstacleGrid,
    ) -> Optional[Tuple[float, float, str]]:
        """Return heading/error for a corridor or a reliable one-sided wall."""
        walls = self._visible_wall_fits(obstacle_grid)
        left = walls.get("left")
        right = walls.get("right")

        if left is not None and right is not None:
            left_heading, left_at_reference = left
            right_heading, right_at_reference = right
            heading_delta = abs(
                math.atan2(
                    math.sin(left_heading - right_heading),
                    math.cos(left_heading - right_heading),
                )
            )
            corridor_width = left_at_reference - right_at_reference
            if (
                heading_delta <= self.CORRIDOR_MAX_PARALLEL_ERROR_RAD
                and self.CORRIDOR_MIN_WIDTH_M
                <= corridor_width
                <= self.CORRIDOR_MAX_WIDTH_M
            ):
                wall_heading = math.atan2(
                    math.sin(left_heading) + math.sin(right_heading),
                    math.cos(left_heading) + math.cos(right_heading),
                )
                center_offset = 0.5 * (left_at_reference + right_at_reference)
                return wall_heading, center_offset, "corridor"

        # A single visible wall is still useful. Prefer the side with the
        # smallest lateral clearance when both fits exist but are not parallel.
        candidates = []
        if left is not None:
            candidates.append((abs(left[1]), 1.0, left))
        if right is not None:
            candidates.append((abs(right[1]), -1.0, right))
        if not candidates:
            return None

        _clearance, side, (heading, lateral_at_reference) = min(candidates)
        clearance_error = side * (
            abs(lateral_at_reference) - self.SINGLE_WALL_TARGET_CLEARANCE_M
        )
        return heading, clearance_error, "single_wall"

    def _visible_wall_fits(
        self,
        obstacle_grid: ObstacleGrid,
    ) -> Dict[str, Tuple[float, float]]:
        """Fit reliable wall lines independently on the robot's left and right."""
        occupied = np.argwhere(obstacle_grid.grid > 0)
        if occupied.size == 0:
            return {}

        forward = (obstacle_grid.origin_row - occupied[:, 0]) * obstacle_grid.resolution
        lateral = (obstacle_grid.origin_col - occupied[:, 1]) * obstacle_grid.resolution
        usable = (
            (forward >= 0.25)
            & (forward <= 2.50)
            & (np.abs(lateral) >= 0.18)
            & (np.abs(lateral) <= 1.50)
        )
        forward = forward[usable]
        lateral = lateral[usable]

        fits: Dict[str, Tuple[float, float]] = {}
        left = self._fit_corridor_wall(forward[lateral > 0], lateral[lateral > 0])
        right = self._fit_corridor_wall(forward[lateral < 0], lateral[lateral < 0])
        if left is not None:
            fits["left"] = left
        if right is not None:
            fits["right"] = right
        return fits

    def _estimate_corridor_walls(
        self,
        obstacle_grid: ObstacleGrid,
    ) -> Optional[Tuple[float, float]]:
        """Return corridor heading and center offset when both walls are reliable."""
        walls = self._visible_wall_fits(obstacle_grid)
        left = walls.get("left")
        right = walls.get("right")
        if left is None or right is None:
            return None

        left_heading, left_at_reference = left
        right_heading, right_at_reference = right
        heading_delta = abs(
            math.atan2(
                math.sin(left_heading - right_heading),
                math.cos(left_heading - right_heading),
            )
        )
        corridor_width = left_at_reference - right_at_reference
        if (
            heading_delta > self.CORRIDOR_MAX_PARALLEL_ERROR_RAD
            or not self.CORRIDOR_MIN_WIDTH_M
            <= corridor_width
            <= self.CORRIDOR_MAX_WIDTH_M
        ):
            return None

        wall_heading = math.atan2(
            math.sin(left_heading) + math.sin(right_heading),
            math.cos(left_heading) + math.cos(right_heading),
        )
        center_offset = 0.5 * (left_at_reference + right_at_reference)
        return wall_heading, center_offset

    def _fit_corridor_wall(
        self,
        forward: np.ndarray,
        lateral: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        """Fit one longitudinal wall and reject short or scattered point sets."""
        if len(forward) < self.CORRIDOR_MIN_POINTS_PER_SIDE:
            return None
        if np.percentile(forward, 90) - np.percentile(forward, 10) < self.CORRIDOR_MIN_LENGTH_M:
            return None

        slope, intercept = np.polyfit(forward, lateral, 1)
        heading = math.atan(float(slope))
        if abs(heading) > self.CORRIDOR_MAX_WALL_ANGLE_RAD:
            return None
        residual = np.median(np.abs(lateral - (slope * forward + intercept)))
        if residual > self.CORRIDOR_MAX_RESIDUAL_M:
            return None
        return heading, float(slope * 0.75 + intercept)

    def _pivot_yaw_rate(self, heading_error: float) -> float:
        """Return a decisive in-place turn rate for large heading corrections."""
        if abs(heading_error) < 1e-6:
            return 0.0
        yaw_limit = max(self.pivot_yaw_rate, 0.0)
        if yaw_limit <= 1e-6:
            return 0.0
        magnitude = min(abs(heading_error), yaw_limit)
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
    Tracks robot pose with dead-reckoning from velocity commands.

    This remains the fallback when measured SDK odometry is unavailable.
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

    def shutdown(self):
        """Release provider resources. Base dead-reckoning has none."""

    @property
    def source_name(self) -> str:
        """Human-readable pose source name for diagnostics."""
        return "dead_reckoning"


class SdkSportModeOdometryProvider(OdometryProvider):
    """Align Unitree SportModeState pose into the loaded map frame.

    The SDK publishes position and IMU yaw in its own odometry frame. When the
    app anchors the robot to a known map node, we remember the SDK pose at that
    instant and transform subsequent SDK deltas into map coordinates. If SDK
    samples stop arriving, the provider falls back to the base dead-reckoning
    integration so navigation can still degrade gracefully.
    """

    MAX_SAMPLE_AGE_S = _env_float("NAV_SDK_ODOMETRY_MAX_AGE", 0.75)
    MIN_MOTION_DELTA_M = _env_float("NAV_SDK_ODOMETRY_MIN_DELTA_M", 0.02)
    MIN_MOTION_DELTA_YAW_RAD = _env_float("NAV_SDK_ODOMETRY_MIN_DELTA_YAW_RAD", 0.03)
    REMOTE_STICK_DEADZONE = 0.08
    REMOTE_RELEASE_GRACE_S = 0.35

    def __init__(
        self,
        topic: str = "rt/sportmodestate",
        network_interface: Optional[str] = None,
        start_subscriber: bool = True,
    ):
        super().__init__()
        self._topic = topic
        self._subscriber = None
        self._remote_subscriber = None
        self._latest_sdk_pose: Optional[Tuple[float, float, float, float]] = None
        self._sdk_anchor: Optional[Tuple[float, float, float]] = None
        self._map_anchor = RobotPose()
        self._subscriber_error: Optional[str] = None
        self._use_translation_odometry = _env_flag(
            "NAV_USE_SDK_TRANSLATION_ODOMETRY",
            True,
        )
        self._manual_control_until = 0.0
        self._sdk_translation_confirmed = False
        self._sdk_yaw_confirmed = False
        self._reported_static_fallback = False

        if start_subscriber:
            self._start_subscriber(network_interface)

    @property
    def source_name(self) -> str:
        """Human-readable pose source name for diagnostics."""
        if self._subscriber is not None:
            if self._sdk_translation_confirmed:
                return f"sdk_sportmodestate:{self._topic}"
            if self._sdk_yaw_confirmed:
                return f"sdk_sportmodestate_yaw_only:{self._topic}"
            return f"sdk_sportmodestate_pending_motion:{self._topic}"
        if self._subscriber_error:
            return "dead_reckoning_after_sdk_error"
        return "sdk_sportmodestate:manual"

    @property
    def subscriber_error(self) -> Optional[str]:
        """Return SDK subscriber startup error, if any."""
        return self._subscriber_error

    def _start_subscriber(self, network_interface: Optional[str]) -> None:
        """Initialize the Unitree SportModeState subscriber."""
        try:
            ChannelSubscriber, ChannelFactoryInitialize, SportModeState_, WirelessController_ = (
                self._import_unitree_sport_state()
            )

            if network_interface:
                ChannelFactoryInitialize(0, network_interface)
            else:
                ChannelFactoryInitialize(0)

            self._subscriber = ChannelSubscriber(self._topic, SportModeState_)
            self._subscriber.Init(self._handle_sample, 1)
            if WirelessController_ is not None:
                self._remote_subscriber = ChannelSubscriber(
                    "rt/wirelesscontroller",
                    WirelessController_,
                )
                self._remote_subscriber.Init(self._handle_remote_sample, 1)
                logger.info(
                    "SdkSportModeOdometryProvider: subscribed to %s and "
                    "rt/wirelesscontroller",
                    self._topic,
                )
            else:
                logger.warning(
                    "SdkSportModeOdometryProvider: wireless controller IDL unavailable; "
                    "remote override detection disabled"
                )
        except Exception as exc:
            self._subscriber = None
            self._subscriber_error = str(exc)
            logger.warning(
                "SdkSportModeOdometryProvider: unavailable, using dead-reckoning fallback: %s",
                exc,
            )

    @staticmethod
    def _import_unitree_sport_state():
        """Import Unitree SDK2 symbols across the two package layouts in use."""
        import_errors = []
        for root in ("unitree_sdk2_python.unitree_sdk2py", "unitree_sdk2py"):
            try:
                channel_mod = importlib.import_module(f"{root}.core.channel")
                dds_mod = importlib.import_module(f"{root}.idl.unitree_go.msg.dds_")
                return (
                    channel_mod.ChannelSubscriber,
                    channel_mod.ChannelFactoryInitialize,
                    dds_mod.SportModeState_,
                    getattr(dds_mod, "WirelessController_", None),
                )
            except Exception as exc:
                import_errors.append(f"{root}: {exc}")
        raise ImportError("; ".join(import_errors))

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize an angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _read_field(obj: Any, name: str) -> Any:
        """Read either a dataclass field or method-style generated IDL field."""
        value = getattr(obj, name, None)
        if callable(value):
            return value()
        return value

    @classmethod
    def _extract_sample_pose(
        cls,
        sample: Any,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Extract x, y, yaw, monotonic_timestamp from a SportModeState sample."""
        position = cls._read_field(sample, "position")
        imu_state = cls._read_field(sample, "imu_state")
        rpy = cls._read_field(imu_state, "rpy") if imu_state is not None else None

        if position is None or rpy is None or len(position) < 2 or len(rpy) < 3:
            return None

        return (
            float(position[0]),
            float(position[1]),
            cls._normalize_angle(float(rpy[2])),
            time.monotonic(),
        )

    def _handle_sample(self, sample: Any) -> None:
        """DDS callback for incoming SportModeState samples."""
        sdk_pose = self._extract_sample_pose(sample)
        if sdk_pose is None:
            return

        with self._lock:
            self._latest_sdk_pose = sdk_pose
            if self._sdk_anchor is None:
                self._sdk_anchor = (sdk_pose[0], sdk_pose[1], sdk_pose[2])
                self._map_anchor = RobotPose(
                    x=self._pose.x,
                    y=self._pose.y,
                    yaw=self._pose.yaw,
                    timestamp=self._pose.timestamp,
                )

            linear_delta, yaw_delta = self._sdk_pose_delta_from_anchor_locked(sdk_pose)
            if (
                self._use_translation_odometry
                and not self._sdk_translation_confirmed
                and linear_delta >= self.MIN_MOTION_DELTA_M
            ):
                self._sdk_translation_confirmed = True
                logger.info(
                    "SdkSportModeOdometryProvider: SDK translation odometry confirmed"
                )
            if not self._sdk_yaw_confirmed and yaw_delta >= self.MIN_MOTION_DELTA_YAW_RAD:
                self._sdk_yaw_confirmed = True
                logger.info("SdkSportModeOdometryProvider: SDK yaw odometry confirmed")

            if self._sdk_translation_confirmed:
                self._pose = self._aligned_pose_from_sdk_locked(sdk_pose)
            elif self._sdk_yaw_confirmed:
                aligned_pose = self._aligned_pose_from_sdk_locked(sdk_pose)
                self._pose.yaw = aligned_pose.yaw
                self._pose.timestamp = aligned_pose.timestamp

    def _handle_remote_sample(self, sample: Any) -> None:
        """Record recent human controller input without taking over locomotion."""
        sticks = (
            self._read_field(sample, "lx"),
            self._read_field(sample, "ly"),
            self._read_field(sample, "rx"),
            self._read_field(sample, "ry"),
        )
        keys = self._read_field(sample, "keys") or 0
        stick_active = any(
            isinstance(value, (int, float))
            and abs(float(value)) >= self.REMOTE_STICK_DEADZONE
            for value in sticks
        )
        try:
            button_active = int(keys) != 0
        except (TypeError, ValueError):
            button_active = False
        if stick_active or button_active:
            with self._lock:
                self._manual_control_until = (
                    time.monotonic() + self.REMOTE_RELEASE_GRACE_S
                )

    def is_manual_control_active(self) -> bool:
        """Return whether the wireless controller was used very recently."""
        with self._lock:
            return time.monotonic() < self._manual_control_until

    def _has_fresh_sdk_pose_locked(self) -> bool:
        """True if the last SDK sample is recent enough to trust."""
        if self._latest_sdk_pose is None:
            return False
        return time.monotonic() - self._latest_sdk_pose[3] <= self.MAX_SAMPLE_AGE_S

    def _sdk_pose_delta_from_anchor_locked(
        self,
        sdk_pose: Tuple[float, float, float, float],
    ) -> Tuple[float, float]:
        """Return linear and yaw deltas from the SDK anchor."""
        if self._sdk_anchor is None:
            return 0.0, 0.0

        dx = sdk_pose[0] - self._sdk_anchor[0]
        dy = sdk_pose[1] - self._sdk_anchor[1]
        linear_delta = math.hypot(dx, dy)
        yaw_delta = abs(
            self._normalize_angle(sdk_pose[2] - self._sdk_anchor[2])
        )
        return linear_delta, yaw_delta

    def _aligned_pose_from_sdk_locked(
        self,
        sdk_pose: Tuple[float, float, float, float],
    ) -> RobotPose:
        """Transform SDK odometry deltas into the current map-anchor frame."""
        if self._sdk_anchor is None:
            self._sdk_anchor = (sdk_pose[0], sdk_pose[1], sdk_pose[2])

        sdk_anchor_x, sdk_anchor_y, sdk_anchor_yaw = self._sdk_anchor
        dx = sdk_pose[0] - sdk_anchor_x
        dy = sdk_pose[1] - sdk_anchor_y
        frame_yaw = self._map_anchor.yaw - sdk_anchor_yaw
        cos_yaw = math.cos(frame_yaw)
        sin_yaw = math.sin(frame_yaw)

        map_dx = cos_yaw * dx - sin_yaw * dy
        map_dy = sin_yaw * dx + cos_yaw * dy
        map_yaw = self._normalize_angle(
            self._map_anchor.yaw + self._normalize_angle(sdk_pose[2] - sdk_anchor_yaw)
        )
        return RobotPose(
            x=self._map_anchor.x + map_dx,
            y=self._map_anchor.y + map_dy,
            yaw=map_yaw,
            timestamp=time.time(),
        )

    def update_from_velocity(self, cmd: VelocityCommand, dt: float):
        """Use proven SDK pose when fresh; otherwise fall back to dead-reckoning."""
        with self._lock:
            has_fresh_sdk_pose = self._has_fresh_sdk_pose_locked()
            if has_fresh_sdk_pose:
                aligned_pose = self._aligned_pose_from_sdk_locked(self._latest_sdk_pose)
                if self._sdk_translation_confirmed:
                    self._pose = aligned_pose
                    return
                if self._sdk_yaw_confirmed:
                    self._pose.yaw = aligned_pose.yaw
                    self._pose.timestamp = aligned_pose.timestamp

            command_active = (
                abs(cmd.vx) > 1e-3
                or abs(cmd.vy) > 1e-3
                or abs(cmd.vyaw) > 1e-3
            )
            if (
                command_active
                and has_fresh_sdk_pose
                and not self._sdk_translation_confirmed
                and not self._reported_static_fallback
            ):
                self._reported_static_fallback = True
                logger.info(
                    "SdkSportModeOdometryProvider: SDK translation is not confirmed; "
                    "using command-integrated fallback position"
                )

            self._pose.x += cmd.vx * math.cos(self._pose.yaw) * dt
            self._pose.y += cmd.vx * math.sin(self._pose.yaw) * dt
            if not (has_fresh_sdk_pose and self._sdk_yaw_confirmed):
                self._pose.yaw = self._normalize_angle(self._pose.yaw + cmd.vyaw * dt)
            self._pose.timestamp = time.time()

    def set_pose(self, x: float, y: float, yaw: float = 0.0):
        """Anchor the map pose and align future SDK odometry samples to it."""
        with self._lock:
            now = time.time()
            normalized_yaw = self._normalize_angle(yaw)
            self._pose = RobotPose(x=x, y=y, yaw=normalized_yaw, timestamp=now)
            self._map_anchor = RobotPose(x=x, y=y, yaw=normalized_yaw, timestamp=now)
            if self._latest_sdk_pose is not None:
                sdk_pose = self._latest_sdk_pose
                self._sdk_anchor = (sdk_pose[0], sdk_pose[1], sdk_pose[2])
            else:
                self._sdk_anchor = None
            self._sdk_translation_confirmed = False
            self._sdk_yaw_confirmed = False
            self._reported_static_fallback = False
            self._manual_control_until = 0.0
            self._last_update = time.monotonic()

    def get_pose(self) -> RobotPose:
        """Return measured pose when fresh, otherwise the fallback pose."""
        with self._lock:
            if self._has_fresh_sdk_pose_locked():
                aligned_pose = self._aligned_pose_from_sdk_locked(self._latest_sdk_pose)
                if self._sdk_translation_confirmed:
                    self._pose = aligned_pose
                elif self._sdk_yaw_confirmed:
                    self._pose.yaw = aligned_pose.yaw
                    self._pose.timestamp = aligned_pose.timestamp
            return RobotPose(
                x=self._pose.x,
                y=self._pose.y,
                yaw=self._pose.yaw,
                timestamp=self._pose.timestamp,
            )

    def reset(self):
        """Reset pose and SDK/map anchors."""
        with self._lock:
            self._pose = RobotPose()
            self._map_anchor = RobotPose()
            self._sdk_anchor = None
            self._latest_sdk_pose = None
            self._sdk_translation_confirmed = False
            self._sdk_yaw_confirmed = False
            self._reported_static_fallback = False
            self._last_update = time.monotonic()

    def shutdown(self):
        """Close the SDK subscriber if it was started."""
        for subscriber_name in ("_subscriber", "_remote_subscriber"):
            subscriber = getattr(self, subscriber_name)
            if subscriber is None:
                continue
            try:
                subscriber.Close()
            except Exception:
                logger.debug(
                    "SdkSportModeOdometryProvider: subscriber close failed",
                    exc_info=True,
                )
            finally:
                setattr(self, subscriber_name, None)


def _create_odometry_provider() -> OdometryProvider:
    """Create the preferred odometry provider with graceful fallback."""
    use_sdk_default = not _env_flag("NAV_SIMULATION_MODE", False)
    if not _env_flag("NAV_USE_SDK_ODOMETRY", use_sdk_default):
        return OdometryProvider()

    topic = os.environ.get("NAV_SPORT_MODE_STATE_TOPIC", "rt/sportmodestate")
    network_interface = (
        os.environ.get("GO2_NETWORK_INTERFACE")
        or os.environ.get("CYCLONEDDS_NETWORK_INTERFACE")
        or "eth0"
    )
    provider = SdkSportModeOdometryProvider(
        topic=topic,
        network_interface=network_interface,
    )
    if provider.subscriber_error:
        return provider
    return provider


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
    SAFETY_DISTANCE_M: float = _env_float("NAV_SAFETY_DISTANCE", 0.20)
    AVOIDANCE_DISTANCE_M: float = _env_float("NAV_AVOIDANCE_DISTANCE", 0.75)
    MAX_LINEAR_SPEED: float = _env_float("NAV_MAX_LINEAR_SPEED", 0.40)
    MAX_YAW_RATE: float = _env_float("NAV_MAX_YAW_RATE", 0.08)
    PIVOT_YAW_RATE: float = _env_float(
        "NAV_PIVOT_YAW_RATE",
        _env_float("NAV_MIN_PIVOT_YAW_RATE", 0.50),
    )
    GOAL_TOLERANCE_M: float = _env_float("NAV_GOAL_TOLERANCE", 0.15)
    STUCK_TIMEOUT_S: float = _env_float("NAV_STUCK_TIMEOUT", 10.0)
    MAX_STUCK_RECOVERY_ATTEMPTS: int = 2
    PIVOT_HARD_STOP_DISTANCE_M: float = _env_float("NAV_PIVOT_HARD_STOP_DISTANCE", 0.0)
    FORWARD_HAZARD_CONE_RAD: float = _env_float(
        "NAV_FORWARD_HAZARD_CONE_RAD",
        math.radians(20.0),
    )
    CLOSE_OBSTACLE_CONFIRM_S: float = _env_float("NAV_CLOSE_OBSTACLE_CONFIRM_S", 0.7)
    CLOSE_OBSTACLE_CONFIRM_READINGS: int = _env_int("NAV_CLOSE_OBSTACLE_CONFIRM_READINGS", 6)
    PATH_OBSTACLE_CONFIRM_S: float = _env_float("NAV_PATH_OBSTACLE_CONFIRM_S", 0.3)
    PATH_OBSTACLE_CONFIRM_READINGS: int = _env_int("NAV_PATH_OBSTACLE_CONFIRM_READINGS", 3)
    PATH_OBSTACLE_DISTANCE_TOLERANCE_M: float = _env_float(
        "NAV_PATH_OBSTACLE_DISTANCE_TOLERANCE",
        0.15,
    )
    PATH_OBSTACLE_BEARING_TOLERANCE_RAD: float = _env_float(
        "NAV_PATH_OBSTACLE_BEARING_TOLERANCE_RAD",
        math.radians(10.0),
    )
    PATH_OBSTACLE_CENTER_DEPTH_MARGIN_M: float = _env_float(
        "NAV_PATH_OBSTACLE_CENTER_DEPTH_MARGIN",
        0.15,
    )
    FORWARD_SPEED: float = _env_float("NAV_FORWARD_SPEED", 0.45)
    FORWARD_STOP_DISTANCE_M: float = _env_float("NAV_FORWARD_STOP_DISTANCE", 0.50)
    FORWARD_MAX_SECONDS: float = _env_float("NAV_FORWARD_MAX_SECONDS", 15.0)
    FORWARD_MIN_SECONDS: float = _env_float("NAV_FORWARD_MIN_SECONDS", 0.50)
    FORWARD_COMMAND_PERIOD_S: float = _env_float("NAV_FORWARD_COMMAND_PERIOD", 0.20)
    FORWARD_ACTUAL_SPEED_RATIO: float = _env_float("NAV_FORWARD_ACTUAL_SPEED_RATIO", 1.40)
    DEPTH_READY_TIMEOUT_S: float = _env_float("NAV_DEPTH_READY_TIMEOUT", 2.0)
    OBSTACLE_GRID_MAX_AGE_S: float = 0.50
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
    OBSTACLE_MEMORY_SECONDS: float = _env_float("NAV_OBSTACLE_MEMORY_SECONDS", 0.8)
    OBSTACLE_TELEMETRY_SECONDS: float = _env_float("NAV_OBSTACLE_TELEMETRY_SECONDS", 1.0)

    @classmethod
    def get_instance(cls) -> "NavCore":
        """Return the shared NavCore singleton, creating it on first call."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def set_status_callback(cls, callback: Optional[Callable[[str], None]]) -> None:
        """Register a callback for meaningful navigation state changes."""
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
        self._depth_processor = create_default_obstacle_provider()
        self._local_planner = LocalPlanner(
            max_linear_speed=self.MAX_LINEAR_SPEED,
            max_yaw_rate=self.MAX_YAW_RATE,
            pivot_yaw_rate=self.PIVOT_YAW_RATE,
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
        self._odometry = _create_odometry_provider()

        # Global planner (optional, requires map)
        self._topo_map = TopologicalMap()
        self._global_planner = GlobalPlanner(self._topo_map)

        # Progress tracking
        self._last_progress_pose = RobotPose()
        self._last_progress_time = time.monotonic()
        self._stuck_recovery_attempts = 0
        self._close_obstacle_confirmation = ObstacleConfirmationTracker(
            min_seconds=self.CLOSE_OBSTACLE_CONFIRM_S,
            min_readings=self.CLOSE_OBSTACLE_CONFIRM_READINGS,
        )
        self._path_obstacle_confirmation = ObstacleConfirmationTracker(
            min_seconds=self.PATH_OBSTACLE_CONFIRM_S,
            min_readings=self.PATH_OBSTACLE_CONFIRM_READINGS,
            distance_tolerance_m=self.PATH_OBSTACLE_DISTANCE_TOLERANCE_M,
            bearing_tolerance_rad=self.PATH_OBSTACLE_BEARING_TOLERANCE_RAD,
        )
        self._path_obstacle_active = False
        self._manual_override_active = False
        self._obstacle_memory = LocalObstacleMemory(self.OBSTACLE_MEMORY_SECONDS)
        self._last_obstacle_telemetry_time = 0.0

        # Go2 macros (lazy init)
        self._go2 = None

        # Load map if configured
        map_file = _configured_map_file()
        if map_file and Path(map_file).exists():
            if self._topo_map.load_from_file(map_file):
                self._anchor_initial_pose()

        logger.info(
            "NavCore: initialized (obstacle_sensors=%s, map=%s, loop=%d Hz)",
            self._depth_processor.backend,
            "loaded" if self._topo_map.is_loaded else "none",
            self.NAV_LOOP_HZ,
        )
        logger.info(
            "NavCore: config max_vx=%.2f steer_vyaw=%.2f pivot_vyaw=%.2f "
            "safety=%.2f avoidance=%.2f goal_tol=%.2f odom_source=%s "
            "odom_linear_ratio=%.2f odom_yaw_ratio=%.2f",
            self.MAX_LINEAR_SPEED,
            self.MAX_YAW_RATE,
            self.PIVOT_YAW_RATE,
            self.SAFETY_DISTANCE_M,
            self.AVOIDANCE_DISTANCE_M,
            self.GOAL_TOLERANCE_M,
            self._odometry.source_name,
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
        if os.environ.get("NAV_INITIAL_HEADING_DEGREES") is None:
            heading_deg = (
                node.heading_degrees
                if node.heading_degrees is not None
                else 0.0
            )
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
        """Start or restart obstacle sensing before a command that requires it."""
        start = getattr(self._depth_processor, "start", None)
        if callable(start):
            start()
            return True
        return bool(getattr(self._depth_processor, "is_available", False))

    def _stop_depth_when_idle(self) -> None:
        """Release obstacle-sensor resources between navigation actions."""
        if not self.DEPTH_STOP_WHEN_IDLE:
            return
        stop = getattr(self._depth_processor, "stop", None)
        if callable(stop):
            stop()

    def _notify_status_change(self, message: str) -> None:
        """Notify the agent of a meaningful navigation state change."""
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

    def _clear_planner_and_obstacle_state(self) -> None:
        """Clear the active route plus transient local-navigation state."""
        self._global_planner.clear()
        self._reset_close_obstacle_confirmation()
        self._reset_path_obstacle_confirmation()
        self._path_obstacle_active = False
        self._obstacle_memory.clear()

    def _notify_waypoint_advance(self, reached: MapNode, upcoming: MapNode) -> None:
        """Publish topological progress without exposing noisy coordinates."""
        reached_label = self._topo_map.get_node_label(reached)
        upcoming_label = self._topo_map.get_node_label(upcoming)
        self._notify_status_change(
            f"I reached {reached_label} and am continuing toward {upcoming_label}."
        )

    def _accept_expected_landmark(
        self,
        pose: RobotPose,
        grid: Optional[ObstacleGrid],
        waypoint: MapNode,
    ) -> Tuple[RobotPose, Optional[MapNode], bool]:
        """Use a mapped landmark to correct along-track pose and accept a waypoint."""
        segment = self._global_planner.current_segment()
        if segment is None:
            return pose, waypoint, False

        start, target = segment
        path_distance = grid.path_obstacle_m if grid is not None else float("inf")
        path_bearing = grid.path_obstacle_bearing if grid is not None else 0.0
        edge_x = target.x - start.x
        edge_y = target.y - start.y
        edge_length = math.hypot(edge_x, edge_y)
        if edge_length <= 1e-6:
            return pose, target, False

        unit_x = edge_x / edge_length
        unit_y = edge_y / edge_length
        remaining_x = target.x - pose.x
        remaining_y = target.y - pose.y
        longitudinal_error = remaining_x * unit_x + remaining_y * unit_y
        lateral_error = abs(remaining_x * unit_y - remaining_y * unit_x)

        for landmark in target.arrival_landmarks:
            if landmark.get("type") != "wall":
                continue
            approach_from = landmark.get("approach_from", [])
            if approach_from and start.name not in approach_from:
                continue
            if (
                abs(longitudinal_error) > float(landmark.get("max_pose_error_m", 1.0))
                or lateral_error > float(landmark.get("max_lateral_error_m", 0.5))
                or path_distance > float(landmark.get("max_detection_distance_m", 0.3))
                or abs(path_bearing) > self.FORWARD_HAZARD_CONE_RAD
            ):
                continue
            if grid is None or not is_transverse_wall(
                grid,
                path_distance,
                float(landmark.get("min_span_m", 0.3)),
            ):
                continue

            corrected_pose = RobotPose(
                x=pose.x + longitudinal_error * unit_x,
                y=pose.y + longitudinal_error * unit_y,
                yaw=pose.yaw,
                timestamp=time.time(),
            )
            self._odometry.set_pose(
                corrected_pose.x,
                corrected_pose.y,
                corrected_pose.yaw,
            )
            advanced = self._global_planner.advance_current_waypoint(
                on_advance=self._notify_waypoint_advance,
            )
            upcoming = advanced[1] if advanced is not None else None
            self._path_obstacle_active = False
            self._reset_progress_tracker()
            logger.info(
                "NavCore: accepted mapped wall landmark at '%s' from '%s' "
                "(longitudinal correction %.2fm)",
                target.name,
                start.name,
                longitudinal_error,
            )
            return corrected_pose, upcoming, True

        return pose, target, False

    def _update_path_obstacle_event(self, path_distance_m: float, goal: NavGoal) -> None:
        """Publish confirmed obstacle enter/clear transitions once each."""
        in_avoidance_band = (
            self.SAFETY_DISTANCE_M < path_distance_m < self.AVOIDANCE_DISTANCE_M
        )
        if in_avoidance_band and not self._path_obstacle_active:
            self._path_obstacle_active = True
            self._notify_status_change(
                f"I encountered an obstacle in my path at {path_distance_m:.2f} meters "
                f"while heading to {self._goal_display_name(goal)}. "
                "My local planner is navigating around it."
            )
        elif (
            path_distance_m >= self.AVOIDANCE_DISTANCE_M
            and self._path_obstacle_active
        ):
            self._path_obstacle_active = False
            self._notify_status_change(
                f"The path is clear again, and I am continuing toward "
                f"{self._goal_display_name(goal)}."
            )

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
            self._clear_planner_and_obstacle_state()

        logger.warning("NavCore: %s", reason)
        self._notify_status_change(message)
        self._stop_depth_when_idle()

    def _recover_from_stall(self, goal: NavGoal, pose: RobotPose) -> bool:
        """Replan in place after a transient stall while preserving the destination."""
        if self._stuck_recovery_attempts >= self.MAX_STUCK_RECOVERY_ATTEMPTS:
            return False

        if goal.goal_type == "semantic":
            path = self._global_planner.plan_path(pose, goal.label or "")
            if path is None:
                return False

        self._ensure_go2()
        if self._go2 and getattr(self._go2, "available", False):
            try:
                self._go2.stop_move()
            except Exception:
                logger.debug("NavCore: stop_move failed during stall recovery", exc_info=True)

        with self._state_lock:
            self._stuck_recovery_attempts += 1
            attempt = self._stuck_recovery_attempts
            self._state = NavState.NAVIGATING
            self._last_stop_reason = None
            self._path_obstacle_active = False
            self._reset_progress_tracker(reset_recovery_attempts=False)

        logger.warning(
            "NavCore: no-progress recovery %d/%d toward '%s'",
            attempt,
            self.MAX_STUCK_RECOVERY_ATTEMPTS,
            self._goal_display_name(goal),
        )
        self._notify_status_change(
            f"I stalled while heading to {self._goal_display_name(goal)}. "
            "I replanned from my current position and am continuing."
        )
        return True

    def _complete_navigation(self, goal: NavGoal, distance_m: Optional[float] = None) -> None:
        """Stop motion and consistently close a successful navigation action."""
        self._ensure_go2()
        if self._go2 and getattr(self._go2, "available", False):
            self._go2.stop_move()
        with self._state_lock:
            self._state = NavState.IDLE
            self._goal = None
            self._last_stop_reason = None
            self._clear_planner_and_obstacle_state()
        if distance_m is not None:
            logger.info("NavCore: goal reached (dist=%.2fm)", distance_m)
        self._stop_depth_when_idle()
        if goal.goal_type == "semantic":
            self._notify_status_change(f"I arrived at {self._goal_display_name(goal)}.")

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
                self._clear_planner_and_obstacle_state()
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
                self._clear_planner_and_obstacle_state()
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
                self._clear_planner_and_obstacle_state()
            self._notify_status_change(
                f"I did not move because I do not have a mapped route to {goal_label}."
            )
            return False

        if not self._wait_for_depth_grid(self.DEPTH_READY_TIMEOUT_S):
            reason = "E-STOP: obstacle grid unavailable"
            self._ensure_go2()
            if self._go2 and getattr(self._go2, "available", False):
                self._go2.stop_move()
            with self._state_lock:
                self._state = NavState.E_STOP
                self._goal = None
                self._last_stop_reason = reason
                self._clear_planner_and_obstacle_state()
            logger.warning("NavCore: %s before navigating to '%s'", reason, destination)
            self._notify_status_change(
                f"I did not move toward {goal_label} because my obstacle grid was not available."
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
        """Wait briefly for obstacle sensing to publish its first grid."""
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
            self._clear_planner_and_obstacle_state()

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
            logger.warning("NavCore: E-STOP: obstacle grid unavailable before relative move")
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
        """Move forward continuously while forward obstacle clearance stays clear.

        This is the hardware-safe forward primitive validated on CAIL-E. It does
        not declare success from dead-reckoned distance. Instead, it keeps a
        continuous Go2 gait command active and stops immediately when the active
        obstacle provider sees a path obstacle at or inside stop_distance_m.

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
        clearance_max_depth = max(self.AVOIDANCE_DISTANCE_M, stop_distance + 0.05)
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
            return "Cannot move forward: no obstacle sensor is available."

        initial_reading = self._read_forward_clearance(max_depth_m=clearance_max_depth)
        if initial_reading is None:
            self._stop_depth_when_idle()
            return "Cannot move forward: no reliable forward clearance reading is available."

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
                reading = self._read_forward_clearance(max_depth_m=clearance_max_depth)
                if reading is None:
                    stop_reason = "clearance_unavailable"
                    with self._state_lock:
                        self._state = NavState.E_STOP
                        self._last_stop_reason = "E-STOP: forward clearance unavailable"
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
        if stop_reason == "clearance_unavailable":
            return "Emergency stopped: forward clearance reading became unavailable."
        return (
            "Stopped forward movement after the safety timeout; last forward "
            f"clearance was {last_reading.distance_m:.2f}m."
        )

    def turn(self, angle_rad: float) -> bool:
        """Rotate in place by the given angle (radians, positive=left)."""
        if not self._wait_for_depth_grid(self.DEPTH_READY_TIMEOUT_S):
            logger.warning("NavCore: E-STOP: obstacle grid unavailable before turn")
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
            self._clear_planner_and_obstacle_state()
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
                self._clear_planner_and_obstacle_state()
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
                    f"({grid.path_obstacle_points} sensor points)"
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
        """Human-readable obstacle summary from the active obstacle provider."""
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

            if self._handle_manual_override(goal):
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

    def _handle_manual_override(self, goal: NavGoal) -> bool:
        """Pause for remote control, then replan from the measured robot pose."""
        is_active = getattr(self._odometry, "is_manual_control_active", None)
        manual_active = callable(is_active) and is_active()
        if manual_active:
            if not self._manual_override_active:
                self._manual_override_active = True
                logger.info("NavCore: autonomous navigation paused for remote control")
                self._notify_status_change(
                    f"Manual control is active. I paused autonomous navigation to "
                    f"{self._goal_display_name(goal)}."
                )
            return True

        if not self._manual_override_active:
            return False

        self._manual_override_active = False
        if goal.goal_type != "semantic":
            with self._state_lock:
                self._state = NavState.IDLE
                self._goal = None
                self._last_stop_reason = "Relative movement canceled after manual control"
                self._clear_planner_and_obstacle_state()
            self._notify_status_change(
                "Manual control ended, so I canceled the previous relative movement command."
            )
            self._stop_depth_when_idle()
            return True

        measured_pose = self._odometry.get_pose()
        correction = self._global_planner.project_onto_current_segment(measured_pose)
        if correction is None:
            corrected_pose = RobotPose(
                x=measured_pose.x,
                y=measured_pose.y,
                yaw=math.atan2(
                    goal.y - measured_pose.y,
                    goal.x - measured_pose.x,
                ),
            )
            status_message = (
                f"Manual control ended. I accepted the corrected position and am "
                f"continuing toward {self._goal_display_name(goal)}."
            )
            log_args = (self._goal_display_name(goal),)
            log_message = "NavCore: accepted remote correction toward '%s'"
        else:
            corrected_pose, segment_start, segment_target = correction
            status_message = (
                f"Manual control ended. I accepted the correction on the route from "
                f"{self._topo_map.get_node_label(segment_start)} to "
                f"{self._topo_map.get_node_label(segment_target)} and am "
                f"continuing toward {self._goal_display_name(goal)}."
            )
            log_args = (segment_start.name, segment_target.name)
            log_message = (
                "NavCore: accepted remote correction on route segment '%s' -> '%s'"
            )

        self._odometry.set_pose(
            corrected_pose.x,
            corrected_pose.y,
            corrected_pose.yaw,
        )
        self._reset_progress_tracker()
        logger.info(log_message, *log_args)
        self._notify_status_change(status_message)
        return False

    def _nav_cycle(self, state: NavState, goal: NavGoal):
        """Execute one navigation cycle: sense -> plan -> safety filter -> actuate."""
        if state in {NavState.IDLE, NavState.STUCK, NavState.E_STOP}:
            return

        # 1. Read sensors
        raw_grid = self._fresh_obstacle_grid(
            self._depth_processor.get_obstacle_grid()
        )
        pose = self._odometry.get_pose()
        geometry_grid = self._obstacle_memory.update(raw_grid, pose)
        grid = self._filter_transient_path_obstacle(geometry_grid)
        self._log_obstacle_telemetry(raw_grid, geometry_grid)
        self._update_progress(pose)

        path_dist = grid.path_obstacle_m if grid else float("inf")
        path_bearing = grid.path_obstacle_bearing if grid else 0.0
        safety_dist = raw_grid.path_obstacle_m if raw_grid else float("inf")
        safety_bearing = raw_grid.path_obstacle_bearing if raw_grid else 0.0

        # 2. Check if goal reached
        dist_to_goal = math.hypot(goal.x - pose.x, goal.y - pose.y)
        if dist_to_goal < self.GOAL_TOLERANCE_M:
            self._complete_navigation(goal, dist_to_goal)
            return

        # 3. Compute velocity command
        if state == NavState.NAVIGATING:
            accepted_landmark = False
            if goal.goal_type == "semantic":
                waypoint = self._global_planner.get_next_waypoint(
                    pose,
                    self.GOAL_TOLERANCE_M,
                    on_advance=self._notify_waypoint_advance,
                )
                if waypoint is None:
                    self._complete_navigation(goal)
                    return
                pose, waypoint, accepted_landmark = self._accept_expected_landmark(
                    pose,
                    grid,
                    waypoint,
                )
                if accepted_landmark and waypoint is None:
                    self._complete_navigation(goal)
                    return
                target_x, target_y = waypoint.x, waypoint.y
            else:
                target_x, target_y = goal.x, goal.y

            if not accepted_landmark:
                self._update_path_obstacle_event(path_dist, goal)

            goal_dir = math.atan2(target_y - pose.y, target_x - pose.x) - pose.yaw
            # Normalize to [-pi, pi]
            goal_dir = math.atan2(math.sin(goal_dir), math.cos(goal_dir))
            goal_dist = math.hypot(target_x - pose.x, target_y - pose.y)

            if grid:
                cmd = self._local_planner.compute_velocity(grid, goal_dir, goal_dist)
                if goal.goal_type == "semantic":
                    cmd = self._local_planner.apply_corridor_course_correction(cmd, grid)
            else:
                self._abort_active_navigation(
                    goal,
                    "E-STOP: obstacle grid unavailable",
                    f"I stopped before reaching {self._goal_display_name(goal)} "
                    "because my obstacle grid became unavailable.",
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
        safety_dist, safety_bearing = self._forward_clearance_for_safety(
            cmd,
            safety_dist,
            safety_bearing,
        )
        seconds_since_progress = time.monotonic() - self._last_progress_time
        cmd, event = self._safety.filter_command(
            cmd,
            nearest_obstacle_m=safety_dist,
            nearest_obstacle_bearing=safety_bearing,
            seconds_since_progress=seconds_since_progress,
        )

        if event:
            if event.startswith("e_stop"):
                if (
                    event == "e_stop:obstacle_too_close"
                    and self._should_defer_close_obstacle_stop(
                        safety_dist,
                        safety_bearing,
                    )
                ):
                    return

                if event == "e_stop:obstacle_too_close":
                    reason = f"E-STOP: path obstacle at {safety_dist:.2f}m"
                else:
                    reason = f"E-STOP: {event.split(':', 1)[-1].replace('_', ' ')}"
                logger.warning("NavCore: safety event: %s (%s)", event, reason)
                if event == "e_stop:obstacle_too_close":
                    message = (
                        f"I stopped before reaching {self._goal_display_name(goal)} "
                        f"because my depth sensor reported something in my path "
                        f"at {safety_dist:.2f} meters."
                    )
                else:
                    message = (
                        f"I stopped before reaching {self._goal_display_name(goal)} "
                        f"because {reason.lower()}."
                    )
                self._abort_active_navigation(goal, reason, message, state=NavState.E_STOP)
                return
            elif event.startswith("stuck"):
                if self._recover_from_stall(goal, pose):
                    return
                reason = "Stuck: no progress toward the goal"
                self._abort_active_navigation(
                    goal,
                    reason,
                    f"I stopped before reaching {self._goal_display_name(goal)} "
                    "because I was not making progress.",
                )
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
            return

        # 5. Execute
        if not self._send_motion_command(cmd, goal):
            return

        self._reset_close_obstacle_confirmation()

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

    def _forward_clearance_for_safety(
        self,
        cmd: VelocityCommand,
        path_dist: float,
        path_bearing: float,
    ) -> Tuple[float, float]:
        """Merge projected path clearance with raw center depth for one safety gate."""
        if cmd.vx <= 1e-3:
            return path_dist, path_bearing

        reading, supported = self._read_center_depth(
            max_depth_m=self.AVOIDANCE_DISTANCE_M,
            percentile=25.0,
        )
        if not supported or reading is None:
            return path_dist, path_bearing

        center_dist = reading.distance_m
        if not isinstance(center_dist, (int, float)) or not math.isfinite(center_dist):
            return path_dist, path_bearing
        center_dist = float(center_dist)

        if center_dist < path_dist:
            return center_dist, 0.0
        return path_dist, path_bearing

    def _log_obstacle_telemetry(
        self,
        current: Optional[ObstacleGrid],
        geometry: Optional[ObstacleGrid],
    ) -> None:
        """Periodically report directional clearance and fitted wall geometry."""
        now = time.monotonic()
        if (
            self.OBSTACLE_TELEMETRY_SECONDS <= 0.0
            or now - self._last_obstacle_telemetry_time
            < self.OBSTACLE_TELEMETRY_SECONDS
        ):
            return
        self._last_obstacle_telemetry_time = now

        if current is None or geometry is None:
            logger.info("NavCore: obstacle view unavailable")
            return

        points = occupied_xy_points(geometry)
        sector_centers = (-45, -30, -15, 0, 15, 30, 45)
        sector_values = []
        if points.size:
            distances = np.hypot(points[:, 0], points[:, 1])
            bearings = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
            for center in sector_centers:
                selected = distances[np.abs(bearings - center) <= 7.5]
                value = float(np.percentile(selected, 10)) if selected.size else float("inf")
                sector_values.append(
                    f"{center:+d}:{value:.2f}"
                    if math.isfinite(value)
                    else f"{center:+d}:clear"
                )
        else:
            sector_values = [f"{center:+d}:clear" for center in sector_centers]

        wall_geometry = self._local_planner._estimate_wall_geometry(geometry)
        if (
            not isinstance(wall_geometry, (tuple, list))
            or len(wall_geometry) != 3
        ):
            wall_text = "none"
        else:
            heading, lateral_error, geometry_type = wall_geometry
            wall_text = (
                f"{geometry_type}, heading={math.degrees(heading):+.0f}deg, "
                f"lateral_error={lateral_error:+.2f}m"
            )

        path_text = (
            f"{current.path_obstacle_m:.2f}m"
            if math.isfinite(current.path_obstacle_m)
            else "clear"
        )
        logger.info(
            "NavCore: obstacle view path=%s sectors_deg_m=[%s] wall=[%s]",
            path_text,
            " ".join(sector_values),
            wall_text,
        )

    def _fresh_obstacle_grid(
        self,
        grid: Optional[ObstacleGrid],
    ) -> Optional[ObstacleGrid]:
        """Reject stale obstacle data before it can authorize robot motion."""
        if grid is None:
            return None
        timestamp = getattr(grid, "timestamp", 0.0)
        age = time.time() - timestamp if timestamp > 0.0 else 0.0
        if age > self.OBSTACLE_GRID_MAX_AGE_S:
            logger.warning("NavCore: obstacle grid is stale by %.2fs", age)
            return None
        return grid

    def _read_forward_clearance(self, max_depth_m: float) -> Optional[CenterDepthReading]:
        """Read forward clearance from the active obstacle provider."""
        reading, _supported = self._read_center_depth(
            max_depth_m=max_depth_m,
            percentile=10.0,
        )
        if reading is not None:
            return reading

        grid = self._depth_processor.get_obstacle_grid()
        if grid is None:
            return None

        distance = getattr(grid, "path_obstacle_m", None)
        if not isinstance(distance, (int, float)):
            return None
        if not math.isfinite(distance):
            distance = max_depth_m
        return CenterDepthReading(
            distance_m=float(distance),
            coverage=1.0,
            timestamp=getattr(grid, "timestamp", time.time()),
        )

    def _read_center_depth(
        self,
        max_depth_m: float,
        percentile: float = 25.0,
    ) -> Tuple[Optional[CenterDepthReading], bool]:
        """Read the raw center depth band when the depth backend supports it."""
        if getattr(self._depth_processor, "supports_center_depth", True) is False:
            return None, False

        reader = getattr(self._depth_processor, "get_center_depth_reading", None)
        if not callable(reader):
            return None, False
        try:
            reading = reader(
                max_depth_m=max_depth_m,
                percentile=percentile,
                min_coverage=0.02,
            )
        except TypeError:
            reading = reader()
        except Exception:
            logger.debug("NavCore: center-depth read failed", exc_info=True)
            return None, False
        return reading, True

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

        confirmed, started_new_track = self._close_obstacle_confirmation.update(
            nearest_dist,
            nearest_bearing,
        )
        if started_new_track:
            logger.info(
                "NavCore: holding to confirm close obstacle at %.2fm before E-STOP",
                nearest_dist,
            )
        if confirmed:
            return False

        self._ensure_go2()
        if self._go2 and getattr(self._go2, "available", False):
            self._go2.stop_move()
        return True

    def _reset_close_obstacle_confirmation(self) -> None:
        """Clear transient close-obstacle confirmation state."""
        self._close_obstacle_confirmation.reset()

    def _filter_transient_path_obstacle(
        self,
        grid: Optional[ObstacleGrid],
    ) -> Optional[ObstacleGrid]:
        """Require repeat support before avoidance-band path obstacles affect planning."""
        if grid is None:
            self._reset_path_obstacle_confirmation()
            return None

        path_dist = grid.path_obstacle_m
        path_bearing = grid.path_obstacle_bearing
        if path_dist > self.AVOIDANCE_DISTANCE_M:
            self._reset_path_obstacle_confirmation()
            return grid

        if not self._path_obstacle_matches_center_depth(path_dist):
            self._reset_path_obstacle_confirmation()
            logger.debug(
                "NavCore: ignoring path obstacle at %.2fm because center depth is clear",
                path_dist,
            )
            return self._without_path_obstacle(grid)

        confirmed, _ = self._path_obstacle_confirmation.update(path_dist, path_bearing)
        if confirmed:
            return grid

        logger.debug(
            "NavCore: ignoring unconfirmed path obstacle at %.2fm "
            "(%d/%d readings)",
            path_dist,
            self._path_obstacle_confirmation.count,
            self.PATH_OBSTACLE_CONFIRM_READINGS,
        )
        return self._without_path_obstacle(grid)

    def _path_obstacle_matches_center_depth(self, distance_m: float) -> bool:
        """Cross-check slowdown-zone projected obstacles against raw center depth."""
        max_depth = self.AVOIDANCE_DISTANCE_M + self.PATH_OBSTACLE_CENTER_DEPTH_MARGIN_M
        reading, supported = self._read_center_depth(
            max_depth_m=max_depth,
            percentile=25.0,
        )
        if not supported:
            return True

        if reading is None:
            return False

        center_dist = getattr(reading, "distance_m", None)
        if not isinstance(center_dist, (int, float)) or not math.isfinite(center_dist):
            return True

        return float(center_dist) <= max(max_depth, distance_m)

    @staticmethod
    def _without_path_obstacle(grid: ObstacleGrid) -> ObstacleGrid:
        """Return a copy of the grid with the path corridor treated as clear."""
        return replace(
            grid,
            path_obstacle_m=float("inf"),
            path_obstacle_bearing=0.0,
            path_obstacle_points=0,
        )

    def _reset_path_obstacle_confirmation(self) -> None:
        """Clear transient avoidance-band path-obstacle confirmation state."""
        self._path_obstacle_confirmation.reset()

    def _reset_progress_tracker(self, *, reset_recovery_attempts: bool = True):
        """Reset the stuck-detection timer to now."""
        self._last_progress_pose = self._odometry.get_pose()
        self._last_progress_time = time.monotonic()
        if reset_recovery_attempts:
            self._stuck_recovery_attempts = 0
        self._reset_close_obstacle_confirmation()
        self._reset_path_obstacle_confirmation()

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
            self._stuck_recovery_attempts = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self):
        """Stop navigation, join the background thread, and release sensors."""
        with self._state_lock:
            self._state = NavState.IDLE
            self._goal = None
            self._last_stop_reason = None
            self._clear_planner_and_obstacle_state()

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
        self._odometry.shutdown()
        logger.info("NavCore: shutdown complete")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
