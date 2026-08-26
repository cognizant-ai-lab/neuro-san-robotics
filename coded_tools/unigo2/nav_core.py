
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

Provides autonomous navigation for the robot dog:
- Local reactive obstacle avoidance (VFH+ algorithm)
- Clearance-aware occupancy planning with live-obstacle overlays
- Semantic destinations with topological fallback for simulation maps
- Safety monitoring with emergency stop
- Odometry tracking via Unitree SDK2 DDS
- Background navigation loop at configurable frequency

Architecture:
  NavCore (singleton) runs a background thread at ~10 Hz.
  Each cycle: read sensors -> plan -> safety filter -> execute via Go2Macros.

See docs/nav_core_design.md for full architecture documentation.
"""

import difflib
from concurrent.futures import Future, ThreadPoolExecutor
import heapq
import importlib
import json
import math
import os
import logging
import random
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
from coded_tools.unigo2.metric_navigation import MetricOccupancyMap

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
# The San Francisco map that ships with the repo. This is NOT a default: every
# site sets NAV_MAP_FILE in its own setmyenv.sh. Kept as a named path because
# tests load it as a realistic map fixture.
CAIL_LAB_MAP_FILE = REPO_ROOT / "maps" / "cail_lab.json"


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
    """Return the map file for navigation, or "" when this site has no map.

    The map belongs to the site, not to the robot, so each site sets
    NAV_MAP_FILE in its own setmyenv.sh and every robot in that building
    shares the value. There is deliberately no in-code default: a robot that
    has not been told where it lives reports that it has no map, rather than
    loading another office and offering destinations that do not exist here.
    """
    configured = os.environ.get("NAV_MAP_FILE")
    if configured is not None:
        return configured.strip()
    return ""


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
    arrival_tolerance_m: Optional[float] = None
    pass_through_tolerance_m: Optional[float] = None


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
    """Semantic destinations plus an optional metric occupancy map."""

    def __init__(self):
        """Initialize an empty topological map."""
        self.name: str = ""
        self.nodes: Dict[str, MapNode] = {}
        self.edges: List[MapEdge] = []
        self._adjacency: Dict[str, List[Tuple[str, float]]] = {}
        self._node_aliases: Dict[str, str] = {}
        self.metric_map: Optional[MetricOccupancyMap] = None
        self.metric_map_required: bool = False
        self.metric_map_error: Optional[str] = None

    def load_from_file(self, path: str) -> bool:
        """Load map from a JSON file. Returns True if at least one node was loaded."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            parsed = self._parse(data)
            if parsed:
                self._load_metric_map(data, Path(path).resolve().parent)
            return self.is_loaded
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
        self.metric_map = None
        self.metric_map_error = None
        occupancy_config = data.get("occupancy_map", {})
        if not isinstance(occupancy_config, dict):
            occupancy_config = {}
        self.metric_map_required = bool(occupancy_config.get("required", False))

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
                arrival_tolerance_m=(
                    float(node_data["arrival_tolerance_m"])
                    if node_data.get("arrival_tolerance_m") is not None
                    else None
                ),
                pass_through_tolerance_m=(
                    float(node_data["pass_through_tolerance_m"])
                    if node_data.get("pass_through_tolerance_m") is not None
                    else None
                ),
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

    def _load_metric_map(self, data: Dict[str, Any], base_directory: Path) -> None:
        """Load the runtime occupancy grid declared by a map file."""
        config = data.get("occupancy_map")
        if not isinstance(config, dict) or not config.get("file"):
            return
        metric_path = base_directory / str(config["file"])
        try:
            self.metric_map = MetricOccupancyMap.load(
                metric_path,
                robot_clearance_m=(
                    float(config["robot_clearance_m"])
                    if config.get("robot_clearance_m") is not None
                    else None
                ),
                preferred_clearance_m=(
                    float(config["preferred_clearance_m"])
                    if config.get("preferred_clearance_m") is not None
                    else None
                ),
            )
            logger.info(
                "TopologicalMap: loaded metric occupancy %s (%dx%d at %.2fm)",
                metric_path,
                self.metric_map.occupied.shape[1],
                self.metric_map.occupied.shape[0],
                self.metric_map.resolution_m,
            )
        except Exception as exc:
            self.metric_map = None
            self.metric_map_error = str(exc)
            logger.error(
                "TopologicalMap: failed to load metric occupancy %s: %s",
                metric_path,
                exc,
            )

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
                    "arrival_tolerance_m": n.arrival_tolerance_m,
                    "pass_through_tolerance_m": n.pass_through_tolerance_m,
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
        return len(self.nodes) > 0 and (
            not self.metric_map_required or self.metric_map is not None
        )

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
    """Metric occupancy planning with topological fallback for simple maps."""

    MAX_DETOUR_LENGTH_RATIO = 1.35
    MAX_DETOUR_EXTRA_M = 5.0
    MAX_DETOUR_BEARING_CHANGE_RAD = math.radians(100.0)
    FINAL_METRIC_LONGITUDINAL_TOLERANCE_M = 0.30
    FINAL_METRIC_CROSS_TRACK_TOLERANCE_M = 0.40
    FINAL_METRIC_HEADING_TOLERANCE_RAD = math.radians(25.0)
    STRAIGHT_METRIC_PASS_TOLERANCE_M = 0.75
    STRAIGHT_METRIC_PASS_MAX_TURN_RAD = math.radians(20.0)

    def __init__(self, topo_map: TopologicalMap):
        """Initialize the global planner with a topological map reference."""
        self._map = topo_map
        self._current_path: List[MapNode] = []
        self._waypoint_index: int = 0

    def plan_path(
        self,
        current_pose: RobotPose,
        goal_label: str,
        dynamic_obstacles_xy: Optional[np.ndarray] = None,
    ) -> Optional[List[MapNode]]:
        """Plan a collision-free path from the current pose to a named goal.

        Returns:
            List of MapNodes from start to goal, or None if unreachable/unknown.
        """
        goal_node = self._map.get_node(goal_label)
        if goal_node is None:
            logger.warning("GlobalPlanner: unknown destination '%s'", goal_label)
            return None

        if self._map.metric_map is not None:
            return self._plan_metric_path(
                current_pose,
                goal_node,
                dynamic_obstacles_xy=dynamic_obstacles_xy,
            )

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

    def _plan_metric_path(
        self,
        current_pose: RobotPose,
        goal_node: MapNode,
        dynamic_obstacles_xy: Optional[np.ndarray] = None,
    ) -> Optional[List[MapNode]]:
        """Plan dense transient waypoints over free space to a semantic goal."""
        metric_map = self._map.metric_map
        if metric_map is None:
            return None
        points = metric_map.plan_path(
            (current_pose.x, current_pose.y),
            (goal_node.x, goal_node.y),
            dynamic_obstacles_xy=dynamic_obstacles_xy,
        )
        if not points:
            return None

        return self.install_metric_path_points(current_pose, goal_node.name, points)

    def install_metric_path_points(
        self,
        current_pose: RobotPose,
        goal_label: str,
        points: List[Tuple[float, float]],
    ) -> Optional[List[MapNode]]:
        """Install metric points produced without mutating the active route."""
        goal_node = self._map.get_node(goal_label)
        if goal_node is None or not points:
            return None
        path = [
            MapNode(
                name="__metric_start__",
                x=points[0][0],
                y=points[0][1],
                description="Current position",
                tags=["metric_transit"],
                arrival_tolerance_m=0.45,
                pass_through_tolerance_m=0.45,
            )
        ]
        for index, (x_m, y_m) in enumerate(points[1:-1], start=1):
            path.append(
                MapNode(
                    name=f"__metric_{index:03d}__",
                    x=x_m,
                    y=y_m,
                    description="Collision-free route point",
                    tags=["metric_transit"],
                    arrival_tolerance_m=0.45,
                    pass_through_tolerance_m=0.45,
                )
            )
        safe_goal_x, safe_goal_y = points[-1]
        path.append(replace(goal_node, x=safe_goal_x, y=safe_goal_y))
        self._current_path = path
        self._waypoint_index = 1 if len(path) > 1 else 0
        route_length = sum(
            math.hypot(second.x - first.x, second.y - first.y)
            for first, second in zip(path, path[1:])
        )
        logger.info(
            "GlobalPlanner: metric route to '%s' has %d targets over %.2fm",
            goal_node.name,
            max(0, len(path) - 1),
            route_length,
        )
        return path

    def remaining_metric_route(self, pose: RobotPose) -> Tuple[float, Optional[float]]:
        """Return remaining route length and bearing of its active segment."""
        if not self._current_path or self._waypoint_index >= len(self._current_path):
            return 0.0, None
        remaining = self._current_path[self._waypoint_index:]
        first = remaining[0]
        length = math.hypot(first.x - pose.x, first.y - pose.y)
        length += sum(
            math.hypot(second.x - first_node.x, second.y - first_node.y)
            for first_node, second in zip(remaining, remaining[1:])
        )
        return length, math.atan2(first.y - pose.y, first.x - pose.x)

    def replan_path_around_obstacles(
        self,
        current_pose: RobotPose,
        goal_label: str,
        dynamic_obstacles_xy: Optional[np.ndarray] = None,
    ) -> Optional[List[MapNode]]:
        """Replace the remaining metric route using the latest obstacle scan."""
        if self._map.metric_map is None:
            return self.replan_path_preserving_progress(current_pose, goal_label)
        goal_node = self._map.get_node(goal_label)
        if goal_node is None:
            return None
        reference_length, reference_bearing = self.remaining_metric_route(current_pose)
        points = self._map.metric_map.plan_path(
            (current_pose.x, current_pose.y),
            (goal_node.x, goal_node.y),
            dynamic_obstacles_xy=dynamic_obstacles_xy,
        )
        if not points:
            return None
        if not self.metric_detour_is_reasonable(
            current_pose,
            points,
            reference_length,
            reference_bearing,
        ):
            logger.warning(
                "GlobalPlanner: rejected stalled-route detour that reversed and "
                "greatly lengthened the remaining route"
            )
            return None
        path = self.install_metric_path_points(current_pose, goal_label, points)
        if path is not None:
            logger.info("GlobalPlanner: replaced stalled route with a fresh metric route")
        return path

    @classmethod
    def metric_detour_is_reasonable(
        cls,
        current_pose: RobotPose,
        points: List[Tuple[float, float]],
        reference_length: float,
        reference_bearing: Optional[float],
    ) -> bool:
        """Reject only detours that are both a major reversal and much longer."""
        if len(points) < 2 or reference_bearing is None or reference_length <= 0.0:
            return True
        first_target = next(
            (
                point
                for point in points[1:]
                if math.hypot(point[0] - current_pose.x, point[1] - current_pose.y) > 0.05
            ),
            None,
        )
        if first_target is None:
            return True
        candidate_bearing = math.atan2(
            first_target[1] - current_pose.y,
            first_target[0] - current_pose.x,
        )
        bearing_change = abs(
            math.atan2(
                math.sin(candidate_bearing - reference_bearing),
                math.cos(candidate_bearing - reference_bearing),
            )
        )
        candidate_length = sum(
            math.hypot(second[0] - first[0], second[1] - first[1])
            for first, second in zip(points, points[1:])
        )
        excessive_length = candidate_length > max(
            reference_length * cls.MAX_DETOUR_LENGTH_RATIO,
            reference_length + cls.MAX_DETOUR_EXTRA_M,
        )
        return not (
            excessive_length
            and bearing_change > cls.MAX_DETOUR_BEARING_CHANGE_RAD
        )

    def replan_path_preserving_progress(
        self,
        current_pose: RobotPose,
        goal_label: str,
    ) -> Optional[List[MapNode]]:
        """Replan without reinstating waypoints already completed on this route."""
        active = self.get_current_waypoint()
        completed_names = {
            node.name for node in self._current_path[: self._waypoint_index]
        }
        path = self.plan_path(current_pose, goal_label)
        if path is None:
            return None

        if active is not None:
            active_indices = [
                index for index, node in enumerate(path) if node.name == active.name
            ]
            if active_indices:
                self._waypoint_index = active_indices[0]

        while (
            self._waypoint_index < len(self._current_path)
            and self._current_path[self._waypoint_index].name in completed_names
        ):
            self._waypoint_index += 1

        logger.info(
            "GlobalPlanner: replanned while preserving progress; active waypoint='%s'",
            self._current_path[self._waypoint_index].name
            if self._waypoint_index < len(self._current_path)
            else "complete",
        )
        return path

    def get_next_waypoint(
        self,
        current_pose: RobotPose,
        tolerance_m: float = 0.3,
        on_advance: Optional[Callable[[MapNode, MapNode], None]] = None,
        *,
        final_arrival_sensor_confirmed: bool = False,
    ) -> Optional[MapNode]:
        """Return the next waypoint to steer toward, advancing when within tolerance.

        Returns None when the final waypoint (goal) has been reached.
        """
        if not self._current_path or self._waypoint_index >= len(self._current_path):
            return None

        wp = self._current_path[self._waypoint_index]
        dist = math.hypot(wp.x - current_pose.x, wp.y - current_pose.y)
        arrival_tolerance = (
            tolerance_m
            if wp.arrival_tolerance_m is None
            else wp.arrival_tolerance_m
        )

        metric_final = (
            self._waypoint_index == len(self._current_path) - 1
            and self._waypoint_index > 0
            and "metric_transit"
            in self._current_path[self._waypoint_index - 1].tags
        )
        route_arrival_consistent = (
            self._metric_final_arrival_is_consistent(current_pose)
            if metric_final
            else True
        )
        # Inside the destination's configured radius, repeated depth-to-map
        # agreement is stronger evidence than dead-reckoned progress along the
        # last short segment.  The latter can disagree after a safe detour or a
        # recovery replan.  Requiring both creates a deadlock: the robot is close
        # enough to slow down, but can never satisfy the stricter route gate.
        arrival_consistent = (
            not metric_final or final_arrival_sensor_confirmed
        )

        if dist < arrival_tolerance and arrival_consistent:
            if metric_final and not route_arrival_consistent:
                logger.info(
                    "GlobalPlanner: accepting sensor-confirmed final waypoint "
                    "'%s' inside %.2fm arrival region despite stale final-segment "
                    "progress",
                    wp.name,
                    arrival_tolerance,
                )
            logger.info(
                "GlobalPlanner: accepted waypoint '%s' within %.2fm arrival region",
                wp.name,
                arrival_tolerance,
            )
            advanced = self.advance_current_waypoint(on_advance=on_advance)
            return advanced[1] if advanced is not None else None
        if dist < arrival_tolerance and metric_final:
            logger.info(
                "GlobalPlanner: deferred final waypoint '%s' despite %.2fm proximity; "
                "final route gate or sensor-to-map validation is incomplete",
                wp.name,
                dist,
            )

        pass_tolerance = self._effective_pass_through_tolerance(wp)
        if pass_tolerance is not None and self._passed_waypoint_plane(
            current_pose,
            lateral_tolerance_m=pass_tolerance,
        ):
            logger.info(
                "GlobalPlanner: accepted waypoint '%s' after passing its arrival plane",
                wp.name,
            )
            advanced = self.advance_current_waypoint(on_advance=on_advance)
            return advanced[1] if advanced is not None else None

        return wp

    def _effective_pass_through_tolerance(
        self,
        waypoint: MapNode,
    ) -> Optional[float]:
        """Allow a wider pass gate only for straight metric transit points.

        Dense metric waypoints are guidance samples, not destinations.  A robot
        that crosses a sample just outside its ordinary arrival radius while
        continuing along the same corridor must advance to the next sample;
        otherwise the missed point moves behind it and provokes a reverse
        replan.  Corners retain their configured, tighter pass tolerance.
        """
        tolerance = waypoint.pass_through_tolerance_m
        if (
            tolerance is None
            or "metric_transit" not in waypoint.tags
            or not 0 < self._waypoint_index < len(self._current_path) - 1
        ):
            return tolerance

        previous = self._current_path[self._waypoint_index - 1]
        upcoming = self._current_path[self._waypoint_index + 1]
        incoming_heading = math.atan2(
            waypoint.y - previous.y,
            waypoint.x - previous.x,
        )
        outgoing_heading = math.atan2(
            upcoming.y - waypoint.y,
            upcoming.x - waypoint.x,
        )
        turn = abs(
            math.atan2(
                math.sin(outgoing_heading - incoming_heading),
                math.cos(outgoing_heading - incoming_heading),
            )
        )
        if turn <= self.STRAIGHT_METRIC_PASS_MAX_TURN_RAD:
            return max(tolerance, self.STRAIGHT_METRIC_PASS_TOLERANCE_M)
        return tolerance

    def _metric_final_arrival_is_consistent(self, pose: RobotPose) -> bool:
        """Require the final route corridor and entrance gate, not radius alone."""
        segment = self.current_segment()
        if segment is None:
            return False
        start, target = segment
        edge_x = target.x - start.x
        edge_y = target.y - start.y
        edge_length = math.hypot(edge_x, edge_y)
        if edge_length <= 1e-6:
            return False

        unit_x, unit_y = edge_x / edge_length, edge_y / edge_length
        relative_x = pose.x - start.x
        relative_y = pose.y - start.y
        along = relative_x * unit_x + relative_y * unit_y
        cross_track = abs(relative_x * unit_y - relative_y * unit_x)
        longitudinal_remaining = edge_length - along
        route_heading = math.atan2(edge_y, edge_x)
        heading_error = abs(
            math.atan2(
                math.sin(pose.yaw - route_heading),
                math.cos(pose.yaw - route_heading),
            )
        )
        return (
            longitudinal_remaining
            <= self.FINAL_METRIC_LONGITUDINAL_TOLERANCE_M
            and cross_track <= self.FINAL_METRIC_CROSS_TRACK_TOLERANCE_M
            and heading_error <= self.FINAL_METRIC_HEADING_TOLERANCE_RAD
        )

    def get_current_waypoint(self) -> Optional[MapNode]:
        """Return the active waypoint without changing route progress."""
        if not self._current_path or self._waypoint_index >= len(self._current_path):
            return None
        return self._current_path[self._waypoint_index]

    def _passed_waypoint_plane(
        self,
        pose: RobotPose,
        lateral_tolerance_m: float,
    ) -> bool:
        """Return whether pose passed the active waypoint along the route edge."""
        segment = self.current_segment()
        if segment is None:
            return False
        start, target = segment
        edge_x = target.x - start.x
        edge_y = target.y - start.y
        edge_length = math.hypot(edge_x, edge_y)
        if edge_length <= 1e-6:
            return False

        unit_x = edge_x / edge_length
        unit_y = edge_y / edge_length
        relative_x = pose.x - start.x
        relative_y = pose.y - start.y
        along = relative_x * unit_x + relative_y * unit_y
        lateral = abs(relative_x * unit_y - relative_y * unit_x)
        return along >= edge_length and lateral <= lateral_tolerance_m

    def current_segment(self) -> Optional[Tuple[MapNode, MapNode]]:
        """Return the active directed map edge."""
        if not self._current_path or not 0 < self._waypoint_index < len(self._current_path):
            return None
        return (
            self._current_path[self._waypoint_index - 1],
            self._current_path[self._waypoint_index],
        )

    def current_segment_heading(self) -> Optional[float]:
        """Return the active mapped segment's world-frame heading."""
        segment = self.current_segment()
        if segment is None:
            return None
        start, target = segment
        dx = target.x - start.x
        dy = target.y - start.y
        if math.hypot(dx, dy) <= 1e-6:
            return None
        return math.atan2(dy, dx)

    def current_segment_guidance(
        self,
        pose: RobotPose,
        *,
        lookahead_m: float = 1.50,
        max_correction_rad: float = math.radians(10.0),
    ) -> Optional[float]:
        """Follow the active straight segment with bounded cross-track correction."""
        segment = self.current_segment()
        if segment is None:
            return None
        start, target = segment
        dx = target.x - start.x
        dy = target.y - start.y
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return None
        unit_x, unit_y = dx / length, dy / length
        relative_x = pose.x - start.x
        relative_y = pose.y - start.y
        cross_track = unit_x * relative_y - unit_y * relative_x
        correction = float(
            np.clip(
                -math.atan2(cross_track, max(lookahead_m, 0.10)),
                -abs(max_correction_rad),
                abs(max_correction_rad),
            )
        )
        guided_heading = math.atan2(dy, dx) + correction
        return math.atan2(math.sin(guided_heading), math.cos(guided_heading))

    def upcoming_turn(self) -> Optional[Tuple[MapNode, MapNode, float]]:
        """Return the next node and signed heading change after the active segment."""
        if not self._current_path or not 0 < self._waypoint_index < len(self._current_path) - 1:
            return None
        start = self._current_path[self._waypoint_index - 1]
        corner = self._current_path[self._waypoint_index]
        following = self._current_path[self._waypoint_index + 1]
        incoming = math.atan2(corner.y - start.y, corner.x - start.x)
        outgoing = math.atan2(following.y - corner.y, following.x - corner.x)
        change = math.atan2(math.sin(outgoing - incoming), math.cos(outgoing - incoming))
        return corner, following, change

    def current_turn_change(self) -> Optional[float]:
        """Return the heading change just entered after advancing a waypoint."""
        if not self._current_path or not 1 < self._waypoint_index < len(self._current_path):
            return None
        previous = self._current_path[self._waypoint_index - 2]
        corner = self._current_path[self._waypoint_index - 1]
        target = self._current_path[self._waypoint_index]
        incoming = math.atan2(corner.y - previous.y, corner.x - previous.x)
        outgoing = math.atan2(target.y - corner.y, target.x - corner.x)
        return math.atan2(math.sin(outgoing - incoming), math.cos(outgoing - incoming))

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
    PIVOT_ENTER_HEADING_ERROR_RAD = _env_float(
        "NAV_PIVOT_ENTER_HEADING_ERROR_RAD",
        math.radians(35.0),
    )
    PIVOT_EXIT_HEADING_ERROR_RAD = _env_float(
        "NAV_PIVOT_EXIT_HEADING_ERROR_RAD",
        math.radians(6.0),
    )
    FORWARD_HAZARD_CONE_RAD = _env_float("NAV_FORWARD_HAZARD_CONE_RAD", math.radians(20.0))
    DEFAULT_PIVOT_YAW_RATE = _env_float(
        "NAV_PIVOT_YAW_RATE",
        _env_float("NAV_MIN_PIVOT_YAW_RATE", 0.50),
    )
    MIN_EFFECTIVE_PIVOT_YAW_RATE = _env_float(
        "NAV_MIN_EFFECTIVE_PIVOT_YAW_RATE",
        0.35,
    )
    CORRIDOR_MIN_POINTS_PER_SIDE = 8
    CORRIDOR_MIN_LENGTH_M = 0.45
    CORRIDOR_MAX_RESIDUAL_M = 0.12
    CORRIDOR_MAX_WALL_ANGLE_RAD = math.radians(55.0)
    CORRIDOR_MAX_PARALLEL_ERROR_RAD = math.radians(10.0)
    CORRIDOR_MIN_WIDTH_M = 0.55
    CORRIDOR_MAX_WIDTH_M = 2.50
    SINGLE_WALL_TARGET_CLEARANCE_M = _env_float("NAV_WALL_CLEARANCE", 0.55)
    EARLY_WALL_ALIGNMENT_CLEARANCE_M = _env_float(
        "NAV_EARLY_WALL_ALIGNMENT_CLEARANCE",
        0.70,
    )
    EARLY_WALL_CONVERGENCE_RAD = _env_float(
        "NAV_EARLY_WALL_CONVERGENCE_RAD",
        math.radians(3.0),
    )
    EARLY_WALL_MAX_ALIGNMENT_YAW_RPS = _env_float(
        "NAV_EARLY_WALL_MAX_ALIGNMENT_YAW",
        0.06,
    )
    EARLY_WALL_MIN_AWAY_YAW_RPS = _env_float(
        "NAV_EARLY_WALL_MIN_AWAY_YAW",
        0.02,
    )
    ROUTE_CENTER_LOOKAHEAD_M = _env_float("NAV_ROUTE_CENTER_LOOKAHEAD", 1.50)
    ROUTE_CENTER_HALF_WIDTH_M = _env_float("NAV_ROUTE_CENTER_HALF_WIDTH", 0.42)
    ROUTE_CENTER_MAX_HEADING_RAD = _env_float(
        "NAV_ROUTE_CENTER_MAX_HEADING_RAD",
        math.radians(40.0),
    )
    ROUTE_CENTER_GOAL_PRIORITY_RAD = _env_float(
        "NAV_ROUTE_CENTER_GOAL_PRIORITY_RAD",
        math.radians(25.0),
    )
    ROUTE_CENTER_MAX_CHANGE_RAD = _env_float(
        "NAV_ROUTE_CENTER_MAX_CHANGE_RAD",
        math.radians(10.0),
    )
    ROUTE_CENTER_STEP_RAD = math.radians(5.0)
    ROUTE_CENTER_CLEARANCE_MARGIN_M = 0.12
    ROUTE_CENTER_MIN_BLOCKING_POINTS = 3
    MIN_TRANSIT_SPEED_MPS = _env_float("NAV_MIN_TRANSIT_SPEED", 0.28)
    MIN_EFFECTIVE_APPROACH_SPEED_MPS = _env_float(
        "NAV_MIN_EFFECTIVE_APPROACH_SPEED",
        0.18,
    )
    MIN_EFFECTIVE_APPROACH_CLEARANCE_M = _env_float(
        "NAV_MIN_EFFECTIVE_APPROACH_CLEARANCE",
        0.55,
    )
    PARALLEL_WALL_MAX_HEADING_ERROR_RAD = _env_float(
        "NAV_PARALLEL_WALL_MAX_HEADING_ERROR_RAD",
        math.radians(15.0),
    )
    PARALLEL_WALL_MIN_CLEARANCE_M = _env_float(
        "NAV_PARALLEL_WALL_MIN_CLEARANCE",
        0.45,
    )
    STEERING_REVERSAL_CONFIRM_CYCLES = _env_int(
        "NAV_STEERING_REVERSAL_CONFIRM_CYCLES",
        12,
    )
    STEERING_DEADBAND_RPS = _env_float("NAV_STEERING_DEADBAND_RPS", 0.015)

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
        self._route_center_heading: Optional[float] = None
        self._pivoting = False
        self._pivot_direction = 0
        self._steering_sign = 0
        self._pending_steering_sign = 0
        self._pending_steering_cycles = 0
        self._wall_alignment_override_active = False

    def reset_navigation_state(self) -> None:
        """Forget steering history after a new route or waypoint transition."""
        self._prev_heading = 0.0
        self._route_center_heading = None
        self._pivoting = False
        self._pivot_direction = 0
        self._steering_sign = 0
        self._pending_steering_sign = 0
        self._pending_steering_cycles = 0
        self._wall_alignment_override_active = False

    def stabilize_translating_steering(self, cmd: VelocityCommand) -> VelocityCommand:
        """Require a persistent request before reversing translating steering."""
        if cmd.vx <= 0.05 or abs(cmd.vyaw) < self.STEERING_DEADBAND_RPS:
            if abs(cmd.vyaw) < self.STEERING_DEADBAND_RPS:
                self._pending_steering_sign = 0
                self._pending_steering_cycles = 0
            return cmd

        requested_sign = 1 if cmd.vyaw > 0.0 else -1
        if self._steering_sign == 0 or requested_sign == self._steering_sign:
            self._steering_sign = requested_sign
            self._pending_steering_sign = 0
            self._pending_steering_cycles = 0
            return cmd

        if requested_sign != self._pending_steering_sign:
            self._pending_steering_sign = requested_sign
            self._pending_steering_cycles = 1
        else:
            self._pending_steering_cycles += 1

        if self._pending_steering_cycles < self.STEERING_REVERSAL_CONFIRM_CYCLES:
            return VelocityCommand(vx=cmd.vx, vy=cmd.vy, vyaw=0.0)

        self._steering_sign = requested_sign
        self._pending_steering_sign = 0
        self._pending_steering_cycles = 0
        return cmd

    def maintain_effective_final_approach(
        self,
        cmd: VelocityCommand,
        obstacle_grid: ObstacleGrid,
        goal_distance: float,
        arrival_tolerance: float,
    ) -> VelocityCommand:
        """Keep a clear, steering final approach above the Go2 walking threshold.

        Arrival braking can otherwise combine with the translating turn factor to
        produce roughly 0.10 m/s commands.  The Go2 may accept those commands
        without measurably translating, causing a clear route to be abandoned as
        a locomotion failure.  Preserve the requested small yaw correction while
        raising only the forward component, and never do so inside arrival range
        or when forward clearance is tight.
        """
        if (
            goal_distance <= arrival_tolerance
            or obstacle_grid.path_obstacle_m
            < self.MIN_EFFECTIVE_APPROACH_CLEARANCE_M
            or cmd.vx < 0.03
            or cmd.vx >= self.MIN_EFFECTIVE_APPROACH_SPEED_MPS
        ):
            return cmd
        return replace(cmd, vx=self.MIN_EFFECTIVE_APPROACH_SPEED_MPS)

    def parallel_wall_supports_route(
        self,
        obstacle_grid: Optional[ObstacleGrid],
        route_heading: float,
    ) -> bool:
        """Return true when a long side wall is safely parallel to the route."""
        if obstacle_grid is None or abs(route_heading) > math.radians(30.0):
            return False
        for wall_heading, lateral_at_reference in self._visible_wall_fits(
            obstacle_grid
        ).values():
            heading_error = abs(
                math.atan2(
                    math.sin(wall_heading - route_heading),
                    math.cos(wall_heading - route_heading),
                )
            )
            if (
                heading_error <= self.PARALLEL_WALL_MAX_HEADING_ERROR_RAD
                and abs(lateral_at_reference) >= self.PARALLEL_WALL_MIN_CLEARANCE_M
            ):
                return True
        return False

    def _should_pivot(self, heading_error: float, goal_distance: float) -> bool:
        """Use hysteresis so ordinary route corrections cannot oscillate in place."""
        if goal_distance <= 0.5:
            self._pivoting = False
            self._pivot_direction = 0
            return False
        error = abs(heading_error)
        if self._pivoting:
            if error <= self.PIVOT_EXIT_HEADING_ERROR_RAD:
                self._pivoting = False
                self._pivot_direction = 0
        elif error >= self.PIVOT_ENTER_HEADING_ERROR_RAD:
            self._pivoting = True
            self._pivot_direction = 1 if heading_error > 0.0 else -1
        return self._pivoting

    def compute_velocity(
        self,
        obstacle_grid: ObstacleGrid,
        goal_direction: float,
        goal_distance: float,
        *,
        slow_for_arrival: bool = True,
        pivot_heading: Optional[float] = None,
    ) -> VelocityCommand:
        """Compute velocity toward the goal while avoiding obstacles.

        Args:
            obstacle_grid: Current obstacle map from depth processor.
            goal_direction: Bearing to goal in radians (0=ahead, positive=left).
            goal_distance: Distance to goal in meters.
            slow_for_arrival: Brake inside 0.5m for a stopping destination. False
                for intermediate route points that should be passed through.

        Returns:
            VelocityCommand with speed modulated by obstacle proximity and goal distance.
        """
        path_nearest = obstacle_grid.path_obstacle_m
        mapped_heading = goal_direction if pivot_heading is None else pivot_heading
        route_direction = self._centered_route_heading(
            obstacle_grid,
            goal_direction,
        )

        if path_nearest > self.avoidance_distance:
            return self._compute_direct_velocity(
                route_direction,
                goal_distance,
                path_nearest,
                slow_for_arrival=slow_for_arrival,
                pivot_heading=mapped_heading,
            )

        histogram = self._build_histogram(obstacle_grid)
        free_sectors = self._find_free_sectors(histogram)

        if not free_sectors:
            if path_nearest > self.safety_distance:
                vx = self._modulate_speed(min(self.max_linear_speed, 0.12), path_nearest)
                return VelocityCommand(vx=vx, vy=0.0, vyaw=0.0)
            return VelocityCommand(0.0, 0.0, 0.0)

        if (
            path_nearest <= self.safety_distance
            and abs(route_direction) < self.PIVOT_HEADING_ERROR_RAD
        ):
            goal_sector = self._angle_to_sector(route_direction)
            best_sector = self._select_best_sector(free_sectors, goal_sector)
            target_heading = self._sector_to_angle(best_sector)
            return VelocityCommand(
                vx=0.0,
                vy=0.0,
                vyaw=self._pivot_yaw_rate(target_heading),
            )

        # Open-space centering is a small translating correction, not a reason
        # to turn away from the mapped route.  Only the actual waypoint bearing
        # may initiate or drive an in-place route pivot.
        if self._should_pivot(mapped_heading, goal_distance):
            vyaw = self._pivot_yaw_rate(mapped_heading)
            self._prev_heading = float(vyaw)
            return VelocityCommand(vx=0.0, vy=0.0, vyaw=float(vyaw))

        goal_sector = self._angle_to_sector(route_direction)
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
        if slow_for_arrival and goal_distance < 1.0:
            base_speed = min(base_speed, max(0.10, 0.20 * goal_distance))

        # Compute velocity command
        vyaw = np.clip(target_heading, -self.max_yaw_rate, self.max_yaw_rate)

        # Reduce forward speed when turning sharply
        turn_factor = 1.0 - min(abs(vyaw) / self.max_yaw_rate, 1.0) * 0.5
        vx = base_speed * turn_factor

        return VelocityCommand(vx=vx, vy=0.0, vyaw=float(vyaw))

    def _centered_route_heading(
        self,
        obstacle_grid: ObstacleGrid,
        goal_direction: float,
    ) -> float:
        """Aim through the middle of visible open space unless the goal lane is wide."""
        max_heading = self.ROUTE_CENTER_MAX_HEADING_RAD
        if abs(goal_direction) >= self.ROUTE_CENTER_GOAL_PRIORITY_RAD:
            self._route_center_heading = None
            return goal_direction

        points = occupied_xy_points(obstacle_grid)
        if points.size == 0:
            self._route_center_heading = None
            return goal_direction

        direct_clearance = self._swept_route_clearance(points, goal_direction)
        if direct_clearance >= self.ROUTE_CENTER_LOOKAHEAD_M:
            self._route_center_heading = None
            return goal_direction

        headings = np.arange(
            -max_heading,
            max_heading + 0.5 * self.ROUTE_CENTER_STEP_RAD,
            self.ROUTE_CENTER_STEP_RAD,
        )
        clearances = np.asarray(
            [self._swept_route_clearance(points, float(heading)) for heading in headings]
        )
        best_clearance = float(np.max(clearances))
        open_enough = (
            clearances >= best_clearance - self.ROUTE_CENTER_CLEARANCE_MARGIN_M
        )

        runs: List[Tuple[int, int]] = []
        run_start: Optional[int] = None
        for index, is_open in enumerate(open_enough):
            if is_open and run_start is None:
                run_start = index
            if run_start is not None and (not is_open or index == len(open_enough) - 1):
                run_end = index if is_open else index - 1
                runs.append((run_start, run_end))
                run_start = None

        if not runs:
            return goal_direction

        def run_rank(run: Tuple[int, int]) -> Tuple[int, float]:
            start, end = run
            center = 0.5 * (float(headings[start]) + float(headings[end]))
            return end - start + 1, -abs(center - goal_direction)

        best_run = max(runs, key=run_rank)
        centered_heading = 0.5 * (
            float(headings[best_run[0]]) + float(headings[best_run[1]])
        )
        if abs(goal_direction) < self.PIVOT_HEADING_ERROR_RAD:
            translating_limit = 0.95 * self.PIVOT_HEADING_ERROR_RAD
            centered_heading = float(
                np.clip(centered_heading, -translating_limit, translating_limit)
            )
        centered_heading = float(np.clip(centered_heading, -max_heading, max_heading))
        if self._route_center_heading is not None:
            change = float(
                np.clip(
                    centered_heading - self._route_center_heading,
                    -self.ROUTE_CENTER_MAX_CHANGE_RAD,
                    self.ROUTE_CENTER_MAX_CHANGE_RAD,
                )
            )
            centered_heading = self._route_center_heading + change
        self._route_center_heading = centered_heading
        return centered_heading

    def _swept_route_clearance(
        self,
        points: np.ndarray,
        heading: float,
    ) -> float:
        """Return clearance for the robot-width corridor along a candidate heading."""
        cos_heading = math.cos(heading)
        sin_heading = math.sin(heading)
        forward = points[:, 0] * cos_heading + points[:, 1] * sin_heading
        lateral = -points[:, 0] * sin_heading + points[:, 1] * cos_heading
        blocking = forward[
            (forward >= 0.10)
            & (forward <= self.ROUTE_CENTER_LOOKAHEAD_M)
            & (np.abs(lateral) <= self.ROUTE_CENTER_HALF_WIDTH_M)
        ]
        if blocking.size < self.ROUTE_CENTER_MIN_BLOCKING_POINTS:
            return self.ROUTE_CENTER_LOOKAHEAD_M
        return float(np.percentile(blocking, 10))

    def _compute_direct_velocity(
        self,
        goal_direction: float,
        goal_distance: float,
        path_nearest: float,
        *,
        slow_for_arrival: bool = True,
        pivot_heading: Optional[float] = None,
    ) -> VelocityCommand:
        """Drive the mapped path directly when the path corridor is clear."""
        mapped_heading = goal_direction if pivot_heading is None else pivot_heading
        if self._should_pivot(mapped_heading, goal_distance):
            vyaw = self._pivot_yaw_rate(mapped_heading)
            self._prev_heading = float(vyaw)
            return VelocityCommand(vx=0.0, vy=0.0, vyaw=float(vyaw))

        base_speed = self._modulate_speed(self.max_linear_speed, path_nearest)
        if slow_for_arrival and goal_distance < 1.0:
            base_speed = min(base_speed, max(0.10, 0.20 * goal_distance))

        vyaw = float(np.clip(goal_direction, -self.max_yaw_rate, self.max_yaw_rate))
        turn_factor = 1.0 - min(abs(vyaw) / max(self.max_yaw_rate, 1e-6), 1.0) * 0.5
        vx = base_speed * turn_factor
        if not slow_for_arrival and goal_distance > 0.5:
            vx = max(vx, min(self.max_linear_speed, self.MIN_TRANSIT_SPEED_MPS))
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
        *,
        correction_limit: Optional[float] = None,
        align_only: bool = False,
    ) -> VelocityCommand:
        """Add bounded steering from visible corridor or one-sided wall geometry."""
        if cmd.vx <= 0.05:
            self._wall_alignment_override_active = False
            return cmd

        wall_geometry = self._estimate_wall_geometry(obstacle_grid)
        if wall_geometry is None:
            self._wall_alignment_override_active = False
            return cmd

        wall_heading, lateral_error, geometry_type = wall_geometry
        if geometry_type == "single_wall":
            walls = self._visible_wall_fits(obstacle_grid)
            candidates = []
            if "left" in walls:
                candidates.append((abs(walls["left"][1]), 1.0, walls["left"]))
            if "right" in walls:
                candidates.append((abs(walls["right"][1]), -1.0, walls["right"]))
            if candidates:
                clearance, side, (heading, _lateral) = min(candidates)
                converging = (
                    side * heading <= -self.EARLY_WALL_CONVERGENCE_RAD
                )
                too_close = clearance < self.SINGLE_WALL_TARGET_CLEARANCE_M
                if (
                    not align_only
                    and clearance <= self.EARLY_WALL_ALIGNMENT_CLEARANCE_M
                    and (converging or too_close)
                ):
                    # Route cross-track error can be wrong when localization has
                    # drifted. A repeatedly fitted wall that is visibly converging
                    # is direct physical evidence: align with it now rather than
                    # allowing a stronger route command to keep aiming into it.
                    desired_yaw = float(
                        np.clip(
                            0.45 * heading,
                            -self.EARLY_WALL_MAX_ALIGNMENT_YAW_RPS,
                            self.EARLY_WALL_MAX_ALIGNMENT_YAW_RPS,
                        )
                    )
                    if too_close:
                        desired_yaw += float(
                            np.clip(0.50 * lateral_error, -0.04, 0.04)
                        )
                        away_sign = -side
                        if desired_yaw * away_sign < self.EARLY_WALL_MIN_AWAY_YAW_RPS:
                            desired_yaw = (
                                away_sign * self.EARLY_WALL_MIN_AWAY_YAW_RPS
                            )
                    desired_yaw = float(
                        np.clip(
                            desired_yaw,
                            -self.EARLY_WALL_MAX_ALIGNMENT_YAW_RPS,
                            self.EARLY_WALL_MAX_ALIGNMENT_YAW_RPS,
                        )
                    )
                    if not self._wall_alignment_override_active:
                        logger.info(
                            "LocalPlanner: early wall alignment override side=%s "
                            "clearance=%.2fm heading=%+.0fdeg route_yaw=%+.3f "
                            "corrected_yaw=%+.3f",
                            "left" if side > 0.0 else "right",
                            clearance,
                            math.degrees(heading),
                            cmd.vyaw,
                            desired_yaw,
                        )
                    self._wall_alignment_override_active = True
                    return VelocityCommand(
                        vx=cmd.vx,
                        vy=cmd.vy,
                        vyaw=desired_yaw,
                    )
        self._wall_alignment_override_active = False
        if align_only:
            # On the mapped final approach, lateral odometry is less reliable
            # than the known route.  Use the wall only as a heading reference;
            # chasing a nominal wall clearance can gradually steer out of the
            # mapped entrance corridor even while the forward path is clear.
            correction = 0.25 * wall_heading
        elif geometry_type == "corridor":
            correction = 0.30 * wall_heading + 0.20 * lateral_error
        else:
            # For a single wall, first align with it, then maintain clearance.
            # lateral_error is signed so a close left wall steers right and a
            # close right wall steers left.
            alignment = float(np.clip(0.20 * wall_heading, -0.02, 0.02))
            correction = alignment + 0.25 * lateral_error
        if correction_limit is None:
            correction_limit = min(0.06, self.max_yaw_rate * 0.75)
        else:
            correction_limit = min(abs(correction_limit), self.max_yaw_rate)
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
        if self._pivoting:
            magnitude = max(
                magnitude,
                min(self.MIN_EFFECTIVE_PIVOT_YAW_RATE, yaw_limit),
            )
        direction = self._pivot_direction if self._pivoting else 0
        if direction:
            return direction * magnitude
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
        pivot_sweep_blocked = pivot_only and hard_stop
        forward_motion_blocked = (
            nearest_obstacle_m <= self.safety_distance
            and (hard_stop or forward_hazard)
            and not pivot_only
        )
        if pivot_sweep_blocked or forward_motion_blocked:
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
            cos_yaw = math.cos(self._pose.yaw)
            sin_yaw = math.sin(self._pose.yaw)
            self._pose.x += (cmd.vx * cos_yaw - cmd.vy * sin_yaw) * dt
            self._pose.y += (cmd.vx * sin_yaw + cmd.vy * cos_yaw) * dt
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

    def apply_pose_correction(self, x: float, y: float, yaw: float) -> None:
        """Apply a localization correction without changing pose-source trust."""
        self.set_pose(x, y, yaw)

    def has_confirmed_translation(self) -> bool:
        """Whether translation is measured rather than command-integrated."""
        return True

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
    REMOTE_RELEASE_GRACE_S = _env_float("NAV_REMOTE_RELEASE_GRACE", 1.5)

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

            cos_yaw = math.cos(self._pose.yaw)
            sin_yaw = math.sin(self._pose.yaw)
            self._pose.x += (cmd.vx * cos_yaw - cmd.vy * sin_yaw) * dt
            self._pose.y += (cmd.vx * sin_yaw + cmd.vy * cos_yaw) * dt
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

    def apply_pose_correction(self, x: float, y: float, yaw: float) -> None:
        """Re-anchor a corrected map pose while preserving confirmed SDK trust."""
        with self._lock:
            now = time.time()
            normalized_yaw = self._normalize_angle(yaw)
            self._pose = RobotPose(x=x, y=y, yaw=normalized_yaw, timestamp=now)
            self._map_anchor = RobotPose(x=x, y=y, yaw=normalized_yaw, timestamp=now)
            if self._latest_sdk_pose is not None:
                sdk_pose = self._latest_sdk_pose
                self._sdk_anchor = (sdk_pose[0], sdk_pose[1], sdk_pose[2])
            self._last_update = time.monotonic()

    def has_confirmed_translation(self) -> bool:
        """Return true only while fresh SDK translation is established."""
        with self._lock:
            return self._sdk_translation_confirmed and self._has_fresh_sdk_pose_locked()

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
    SEMANTIC_ARRIVAL_TOLERANCE_M: float = _env_float(
        "NAV_SEMANTIC_ARRIVAL_TOLERANCE",
        0.65,
    )
    STUCK_TIMEOUT_S: float = _env_float("NAV_STUCK_TIMEOUT", 6.0)
    CLEAR_MOTION_ACK_TIMEOUT_S: float = _env_float(
        "NAV_CLEAR_MOTION_ACK_TIMEOUT",
        3.0,
    )
    MAX_STUCK_RECOVERY_ATTEMPTS: int = 2
    MAX_CLEAR_MOTION_RECOVERY_ATTEMPTS: int = _env_int(
        "NAV_MAX_CLEAR_MOTION_RECOVERY_ATTEMPTS",
        2,
    )
    LOCOMOTION_VERIFICATION_SPEED_MPS: float = _env_float(
        "NAV_LOCOMOTION_VERIFICATION_SPEED",
        0.28,
    )
    LOCOMOTION_MIN_VERIFICATION_COMMAND_MPS: float = _env_float(
        "NAV_LOCOMOTION_MIN_VERIFICATION_COMMAND",
        0.20,
    )
    FINAL_ROUTE_MAX_CROSS_TRACK_CORRECTION_RAD: float = _env_float(
        "NAV_FINAL_ROUTE_MAX_CROSS_TRACK_CORRECTION_RAD",
        math.radians(3.0),
    )
    ARRIVAL_MAP_MAX_SCORE_M: float = _env_float(
        "NAV_ARRIVAL_MAP_MAX_SCORE",
        0.16,
    )
    ARRIVAL_MAP_MIN_MATCHED_FRACTION: float = _env_float(
        "NAV_ARRIVAL_MAP_MIN_MATCHED_FRACTION",
        0.35,
    )
    ARRIVAL_MAP_CONFIRM_READINGS: int = _env_int(
        "NAV_ARRIVAL_MAP_CONFIRM_READINGS",
        2,
    )
    CLEAR_STALL_TRANSIT_SKIP_M: float = _env_float(
        "NAV_CLEAR_STALL_TRANSIT_SKIP",
        0.65,
    )
    # In-place turns sweep the Go2's body and legs through a much wider area
    # than straight motion.  Reserve enough visible clearance before pivoting.
    PIVOT_HARD_STOP_DISTANCE_M: float = _env_float("NAV_PIVOT_HARD_STOP_DISTANCE", 0.40)
    PIVOT_CLEARANCE_PERCENTILE: float = _env_float(
        "NAV_PIVOT_CLEARANCE_PERCENTILE",
        10.0,
    )
    PIVOT_GRID_INFLATION_M: float = _env_float(
        "NAV_PIVOT_GRID_INFLATION_M",
        0.15,
    )
    FORWARD_HAZARD_CONE_RAD: float = _env_float(
        "NAV_FORWARD_HAZARD_CONE_RAD",
        math.radians(20.0),
    )
    CLOSE_OBSTACLE_CONFIRM_S: float = _env_float("NAV_CLOSE_OBSTACLE_CONFIRM_S", 0.7)
    CLOSE_OBSTACLE_CONFIRM_READINGS: int = _env_int("NAV_CLOSE_OBSTACLE_CONFIRM_READINGS", 6)
    CENTER_ONLY_CLOSE_CONFIRM_S: float = _env_float(
        "NAV_CENTER_ONLY_CLOSE_CONFIRM_S",
        0.2,
    )
    CENTER_ONLY_CLOSE_CONFIRM_READINGS: int = _env_int(
        "NAV_CENTER_ONLY_CLOSE_CONFIRM_READINGS",
        3,
    )
    CENTER_ONLY_GRID_MARGIN_M: float = _env_float(
        "NAV_CENTER_ONLY_GRID_MARGIN",
        0.15,
    )
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
    PATH_OBSTACLE_MIN_CENTER_COVERAGE: float = _env_float(
        "NAV_PATH_OBSTACLE_MIN_CENTER_COVERAGE",
        0.06,
    )
    PATH_OBSTACLE_CLEAR_CONFIRM_S: float = _env_float(
        "NAV_PATH_OBSTACLE_CLEAR_CONFIRM_S",
        1.0,
    )
    FORWARD_SPEED: float = _env_float("NAV_FORWARD_SPEED", 0.45)
    FORWARD_STOP_DISTANCE_M: float = _env_float("NAV_FORWARD_STOP_DISTANCE", 0.50)
    FORWARD_MAX_SECONDS: float = _env_float("NAV_FORWARD_MAX_SECONDS", 15.0)
    FORWARD_MIN_SECONDS: float = _env_float("NAV_FORWARD_MIN_SECONDS", 0.50)
    FORWARD_COMMAND_PERIOD_S: float = _env_float("NAV_FORWARD_COMMAND_PERIOD", 0.20)
    FORWARD_ACTUAL_SPEED_RATIO: float = _env_float("NAV_FORWARD_ACTUAL_SPEED_RATIO", 1.40)
    DEPTH_READY_TIMEOUT_S: float = _env_float("NAV_DEPTH_READY_TIMEOUT", 2.0)
    OBSTACLE_GRID_MAX_AGE_S: float = 0.50
    OBSTACLE_GRID_LOSS_GRACE_S: float = _env_float(
        "NAV_OBSTACLE_GRID_LOSS_GRACE",
        3.0,
    )
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
    STALL_ESCAPE_SPEED_MPS: float = _env_float("NAV_STALL_ESCAPE_SPEED", 0.15)
    STALL_ESCAPE_DISTANCE_M: float = _env_float("NAV_STALL_ESCAPE_DISTANCE", 0.15)
    STALL_ESCAPE_CLEARANCE_M: float = _env_float("NAV_STALL_ESCAPE_CLEARANCE", 0.40)
    STALL_ESCAPE_ROBOT_HALF_LENGTH_M: float = _env_float(
        "NAV_STALL_ESCAPE_ROBOT_HALF_LENGTH",
        0.35,
    )
    STALL_BACKUP_SPEED_MPS: float = _env_float("NAV_STALL_BACKUP_SPEED", 0.12)
    STALL_BACKUP_DISTANCE_M: float = _env_float("NAV_STALL_BACKUP_DISTANCE", 0.15)
    STALL_BACKUP_CLEARANCE_DROP_M: float = _env_float(
        "NAV_STALL_BACKUP_CLEARANCE_DROP",
        0.08,
    )
    STALL_SCAN_YAW_RATE_RPS: float = _env_float("NAV_STALL_SCAN_YAW_RATE", 0.35)
    STALL_SCAN_MIN_ANGLE_RAD: float = _env_float(
        "NAV_STALL_SCAN_MIN_ANGLE_RAD",
        math.radians(20.0),
    )
    STALL_SCAN_MAX_ANGLE_RAD: float = _env_float(
        "NAV_STALL_SCAN_MAX_ANGLE_RAD",
        math.radians(50.0),
    )
    STALL_ESCAPE_COMMAND_PERIOD_S: float = _env_float(
        "NAV_STALL_ESCAPE_COMMAND_PERIOD",
        0.10,
    )
    # Depth readings averaged before turning on a measured wall angle. One
    # frame's line fit is noisy -- readings inside a single stall have spread
    # nearly 20 degrees -- so a pivot chosen from one of them is still a guess.
    STALL_WALL_ALIGN_SAMPLES: int = int(_env_float("NAV_STALL_WALL_ALIGN_SAMPLES", 5))
    # Reject the estimate when those readings disagree by more than this; a
    # wall that cannot be measured consistently is not one to turn against.
    STALL_WALL_ALIGN_MAX_SPREAD_RAD: float = _env_float(
        "NAV_STALL_WALL_ALIGN_MAX_SPREAD_RAD",
        math.radians(25.0),
    )
    # Below this the robot is already near enough to parallel that alignment is
    # not what is blocking it.
    STALL_WALL_ALIGN_MIN_ANGLE_RAD: float = _env_float(
        "NAV_STALL_WALL_ALIGN_MIN_ANGLE_RAD",
        math.radians(8.0),
    )
    # Turn a little past parallel so the shoulder clears rather than grazing.
    STALL_WALL_ALIGN_MARGIN_RAD: float = _env_float(
        "NAV_STALL_WALL_ALIGN_MARGIN_RAD",
        math.radians(10.0),
    )
    METRIC_LOCALIZATION_INTERVAL_S: float = _env_float(
        "NAV_METRIC_LOCALIZATION_INTERVAL",
        5.0,
    )
    METRIC_LOCALIZATION_MAX_TRANSLATION_M: float = _env_float(
        "NAV_METRIC_LOCALIZATION_MAX_TRANSLATION",
        0.08,
    )
    METRIC_LOCALIZATION_MAX_YAW_RAD: float = _env_float(
        "NAV_METRIC_LOCALIZATION_MAX_YAW_RAD",
        math.radians(1.0),
    )
    METRIC_LOCALIZATION_MIN_TRAVEL_M: float = _env_float(
        "NAV_METRIC_LOCALIZATION_MIN_TRAVEL",
        0.50,
    )
    ROUTE_WALL_HEADING_CONFIRM_READINGS: int = _env_int(
        "NAV_ROUTE_WALL_HEADING_CONFIRM_READINGS",
        3,
    )
    ROUTE_WALL_MAX_OBSERVED_HEADING_RAD: float = _env_float(
        "NAV_ROUTE_WALL_MAX_OBSERVED_HEADING_RAD",
        math.radians(12.0),
    )
    ROUTE_WALL_MIN_YAW_CORRECTION_RAD: float = _env_float(
        "NAV_ROUTE_WALL_MIN_YAW_CORRECTION_RAD",
        math.radians(10.0),
    )
    ROUTE_WALL_MAX_YAW_CORRECTION_RAD: float = _env_float(
        "NAV_ROUTE_WALL_MAX_YAW_CORRECTION_RAD",
        math.radians(35.0),
    )
    EXPECTED_CORNER_MIN_WALL_DISTANCE_M: float = _env_float(
        "NAV_EXPECTED_CORNER_MIN_WALL_DISTANCE",
        0.55,
    )
    EXPECTED_CORNER_MAX_WALL_DISTANCE_M: float = _env_float(
        "NAV_EXPECTED_CORNER_MAX_WALL_DISTANCE",
        1.50,
    )
    EXPECTED_CORNER_MAX_POSE_ERROR_M: float = _env_float(
        "NAV_EXPECTED_CORNER_MAX_POSE_ERROR",
        1.25,
    )
    EXPECTED_CORNER_MIN_TURN_RAD: float = _env_float(
        "NAV_EXPECTED_CORNER_MIN_TURN_RAD",
        math.radians(45.0),
    )
    MAPPED_SIDE_ROUTE_MIN_CLEARANCE_M: float = _env_float(
        "NAV_MAPPED_SIDE_ROUTE_MIN_CLEARANCE",
        0.50,
    )
    METRIC_BLOCKED_REPLAN_DELAY_S: float = _env_float(
        "NAV_METRIC_BLOCKED_REPLAN_DELAY",
        2.0,
    )
    METRIC_REPLAN_COOLDOWN_S: float = _env_float(
        "NAV_METRIC_REPLAN_COOLDOWN",
        8.0,
    )
    METRIC_REPLAN_MAX_HEADING_ERROR_RAD: float = _env_float(
        "NAV_METRIC_REPLAN_MAX_HEADING_ERROR_RAD",
        math.radians(18.0),
    )
    METRIC_REPLAN_MIN_ROUTE_CHANGE_RAD: float = _env_float(
        "NAV_METRIC_REPLAN_MIN_ROUTE_CHANGE_RAD",
        math.radians(15.0),
    )
    METRIC_ROUTE_REGRESSION_DISTANCE_M: float = _env_float(
        "NAV_METRIC_ROUTE_REGRESSION_DISTANCE",
        0.65,
    )
    METRIC_ROUTE_REGRESSION_CONFIRM_S: float = _env_float(
        "NAV_METRIC_ROUTE_REGRESSION_CONFIRM_S",
        1.0,
    )
    REMOTE_HEADING_REALIGN_MIN_TURN_RAD: float = _env_float(
        "NAV_REMOTE_HEADING_REALIGN_MIN_TURN_RAD",
        math.radians(20.0),
    )
    REMOTE_HEADING_REALIGN_MIN_ERROR_RAD: float = _env_float(
        "NAV_REMOTE_HEADING_REALIGN_MIN_ERROR_RAD",
        math.radians(45.0),
    )

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
        self._current_location_name: Optional[str] = None
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
        self._last_translation_progress_pose = RobotPose()
        self._last_translation_progress_time = time.monotonic()
        self._stuck_recovery_attempts = 0
        self._clear_motion_recovery_attempts = 0
        self._locomotion_recovery_verification_pending: Optional[str] = None
        self._locomotion_recovery_error: Optional[str] = None
        self._arrival_map_consistency_readings = 0
        self._last_stall_scan_direction: Optional[float] = None
        self._close_obstacle_confirmation = ObstacleConfirmationTracker(
            min_seconds=self.CLOSE_OBSTACLE_CONFIRM_S,
            min_readings=self.CLOSE_OBSTACLE_CONFIRM_READINGS,
        )
        self._center_only_close_confirmation = ObstacleConfirmationTracker(
            min_seconds=self.CENTER_ONLY_CLOSE_CONFIRM_S,
            min_readings=self.CENTER_ONLY_CLOSE_CONFIRM_READINGS,
            distance_tolerance_m=0.10,
        )
        self._center_only_close_pending = False
        self._path_obstacle_confirmation = ObstacleConfirmationTracker(
            min_seconds=self.PATH_OBSTACLE_CONFIRM_S,
            min_readings=self.PATH_OBSTACLE_CONFIRM_READINGS,
            distance_tolerance_m=self.PATH_OBSTACLE_DISTANCE_TOLERANCE_M,
            bearing_tolerance_rad=self.PATH_OBSTACLE_BEARING_TOLERANCE_RAD,
        )
        self._path_obstacle_active = False
        self._path_obstacle_active_since: Optional[float] = None
        self._path_obstacle_clear_since: Optional[float] = None
        self._obstacle_grid_unavailable_since: Optional[float] = None
        self._obstacle_grid_unavailable_notified = False
        self._route_wall_heading_candidate: Optional[float] = None
        self._route_wall_heading_readings = 0
        self._manual_override_active = False
        self._manual_override_start_pose: Optional[RobotPose] = None
        self._obstacle_memory = LocalObstacleMemory(self.OBSTACLE_MEMORY_SECONDS)
        self._last_obstacle_telemetry_time = 0.0
        self._last_metric_localization_time = time.monotonic()
        self._last_metric_localization_pose = RobotPose()
        self._last_metric_replan_time = 0.0
        self._metric_replan_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="metric-replan",
        )
        self._metric_replan_future: Optional[Future] = None
        self._metric_replan_context: Optional[Dict[str, Any]] = None
        self._last_motion_command = VelocityCommand()
        self._route_progress_waypoint_name: Optional[str] = None
        self._route_progress_best_distance = float("inf")
        self._route_regression_since: Optional[float] = None

        # Go2 macros (lazy init)
        self._go2 = None

        # Load map if configured
        map_file = _configured_map_file()
        if map_file and Path(map_file).exists():
            if self._topo_map.load_from_file(map_file):
                self._anchor_initial_pose()
        self._last_metric_localization_pose = self._odometry.get_pose()

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
            logger.warning(
                "NavCore: initial location '%s' is not in map '%s'; starting unanchored. "
                "Set NAV_INITIAL_LOCATION to one of: %s",
                initial_location,
                self._topo_map.name or "unnamed",
                ", ".join(self._topo_map.list_destinations()) or "(none)",
            )
            return

        heading_deg = _env_float("NAV_INITIAL_HEADING_DEGREES", 0.0)
        if os.environ.get("NAV_INITIAL_HEADING_DEGREES") is None:
            heading_deg = (
                node.heading_degrees
                if node.heading_degrees is not None
                else 0.0
            )
        self._odometry.set_pose(node.x, node.y, math.radians(heading_deg))
        self._current_location_name = node.name
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
        pending_replan = getattr(self, "_metric_replan_future", None)
        if pending_replan is not None:
            pending_replan.cancel()
        self._metric_replan_future = None
        self._metric_replan_context = None
        self._last_motion_command = VelocityCommand()
        self._global_planner.clear()
        self._reset_close_obstacle_confirmation()
        self._reset_center_only_close_confirmation()
        self._reset_path_obstacle_confirmation()
        self._path_obstacle_active = False
        self._path_obstacle_active_since = None
        self._path_obstacle_clear_since = None
        self._obstacle_grid_unavailable_since = None
        self._obstacle_grid_unavailable_notified = False
        self._last_stall_scan_direction = None
        self._route_progress_waypoint_name = None
        self._route_progress_best_distance = float("inf")
        self._route_regression_since = None
        self._locomotion_recovery_verification_pending = None
        self._locomotion_recovery_error = None
        self._arrival_map_consistency_readings = 0
        self._obstacle_memory.clear()
        self._local_planner.reset_navigation_state()
        self._reset_route_wall_heading_confirmation()

    def _notify_waypoint_advance(self, reached: MapNode, upcoming: MapNode) -> None:
        """Publish topological progress without exposing noisy coordinates."""
        self._route_progress_waypoint_name = None
        self._route_progress_best_distance = float("inf")
        self._route_regression_since = None
        if "metric_transit" in reached.tags or "metric_transit" in upcoming.tags:
            planner = getattr(self, "_global_planner", None)
            turn_change = (
                planner.current_turn_change()
                if planner is not None
                else None
            )
            if isinstance(turn_change, (int, float)) and abs(turn_change) >= math.radians(20.0):
                self._local_planner.reset_navigation_state()
            return
        self._local_planner.reset_navigation_state()
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
            self._path_obstacle_active_since = None
            self._path_obstacle_clear_since = None
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
        """Publish obstacle transitions without chattering on intermittent depth frames."""
        in_avoidance_band = (
            self.SAFETY_DISTANCE_M < path_distance_m < self.AVOIDANCE_DISTANCE_M
        )
        now = time.monotonic()
        if in_avoidance_band and not self._path_obstacle_active:
            self._path_obstacle_active = True
            self._path_obstacle_active_since = now
            self._path_obstacle_clear_since = None
            self._notify_status_change(
                f"I encountered an obstacle in my path at {path_distance_m:.2f} meters "
                f"while heading to {self._goal_display_name(goal)}. "
                "My local planner is navigating around it."
            )
            return

        if in_avoidance_band:
            self._path_obstacle_clear_since = None
            return

        if not self._path_obstacle_active:
            self._path_obstacle_clear_since = None
            return

        if path_distance_m < self.AVOIDANCE_DISTANCE_M:
            self._path_obstacle_clear_since = None
            return

        if self._path_obstacle_clear_since is None:
            self._path_obstacle_clear_since = now
            return

        if now - self._path_obstacle_clear_since >= self.PATH_OBSTACLE_CLEAR_CONFIRM_S:
            self._path_obstacle_active = False
            self._path_obstacle_active_since = None
            self._path_obstacle_clear_since = None
            self._notify_status_change(
                f"The path is clear again, and I am continuing toward "
                f"{self._goal_display_name(goal)}."
            )

    def _maybe_replan_blocked_metric_route(
        self,
        goal: NavGoal,
        pose: RobotPose,
        grid: Optional[ObstacleGrid],
    ) -> bool:
        """Schedule a route search only after aligned forward motion stalls."""
        metric_map = self._topo_map.metric_map
        active_since = self._path_obstacle_active_since
        if (
            metric_map is None
            or goal.goal_type != "semantic"
            or grid is None
            or not self._path_obstacle_active
            or active_since is None
            or getattr(self, "_metric_replan_future", None) is not None
        ):
            return False
        now = time.monotonic()
        if now - active_since < self.METRIC_BLOCKED_REPLAN_DELAY_S:
            return False
        if now - self._last_metric_replan_time < self.METRIC_REPLAN_COOLDOWN_S:
            return False
        waypoint = self._global_planner.get_current_waypoint()
        if waypoint is None:
            return False
        route_bearing = math.atan2(waypoint.y - pose.y, waypoint.x - pose.x)
        if self._angular_delta(route_bearing, pose.yaw) > self.METRIC_REPLAN_MAX_HEADING_ERROR_RAD:
            return False
        last_cmd = self._last_motion_command
        if last_cmd.vx <= 0.03 or abs(last_cmd.vyaw) > 0.12:
            return False
        if now - self._last_translation_progress_time < self.METRIC_BLOCKED_REPLAN_DELAY_S:
            return False
        self._last_metric_replan_time = now

        world_obstacles = metric_map.robot_points_to_world(
            occupied_xy_points(grid),
            pose.x,
            pose.y,
            pose.yaw,
        )
        goal_node = self._topo_map.get_node(goal.label or "")
        if goal_node is None:
            return False
        old_length, old_bearing = self._global_planner.remaining_metric_route(pose)
        request_pose = RobotPose(pose.x, pose.y, pose.yaw, pose.timestamp)
        self._metric_replan_context = {
            "goal_label": goal.label or "",
            "pose": request_pose,
            "old_length": old_length,
            "old_bearing": old_bearing,
        }
        self._metric_replan_future = self._metric_replan_executor.submit(
            metric_map.plan_path,
            (pose.x, pose.y),
            (goal_node.x, goal_node.y),
            dynamic_obstacles_xy=world_obstacles,
        )
        logger.info(
            "NavCore: scheduled background metric replan toward '%s'",
            self._goal_display_name(goal),
        )
        return True

    def _poll_metric_replan(self, goal: NavGoal, pose: RobotPose) -> bool:
        """Install a completed background route only if it is current and different."""
        future = self._metric_replan_future
        context = self._metric_replan_context
        if future is None or context is None or not future.done():
            return False
        self._metric_replan_future = None
        self._metric_replan_context = None
        try:
            points = future.result()
        except Exception:
            logger.exception("NavCore: background metric replan failed")
            return False
        request_pose = context["pose"]
        if (
            goal.label != context["goal_label"]
            or not self._path_obstacle_active
            or math.hypot(pose.x - request_pose.x, pose.y - request_pose.y) > 0.40
            or not points
            or len(points) < 2
        ):
            logger.info("NavCore: discarded stale background metric route")
            return False
        candidate_length = sum(
            math.hypot(second[0] - first[0], second[1] - first[1])
            for first, second in zip(points, points[1:])
        )
        candidate_bearing = math.atan2(
            points[1][1] - pose.y,
            points[1][0] - pose.x,
        )
        old_bearing = context["old_bearing"]
        if not self._global_planner.metric_detour_is_reasonable(
            pose,
            points,
            context["old_length"],
            old_bearing,
        ):
            logger.warning(
                "NavCore: rejected background metric route that reversed and "
                "greatly lengthened the remaining route"
            )
            return False
        if (
            old_bearing is not None
            and self._angular_delta(candidate_bearing, old_bearing)
            < self.METRIC_REPLAN_MIN_ROUTE_CHANGE_RAD
            and candidate_length >= context["old_length"] * 0.95
        ):
            logger.info("NavCore: rejected equivalent background metric route")
            return False
        path = self._global_planner.install_metric_path_points(
            pose,
            context["goal_label"],
            points,
        )
        if path is None:
            return False
        goal.x, goal.y = path[-1].x, path[-1].y
        self._path_obstacle_active_since = time.monotonic()
        self._path_obstacle_clear_since = None
        self._local_planner.reset_navigation_state()
        self._reset_progress_tracker(reset_recovery_attempts=False)
        logger.info(
            "NavCore: installed distinct background metric route toward '%s'",
            self._goal_display_name(goal),
        )
        return True

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

    def _pause_for_obstacle_grid_recovery(self, goal: NavGoal) -> None:
        """Stop safely during a brief depth dropout while preserving the route."""
        self._ensure_go2()
        if self._go2 and getattr(self._go2, "available", False):
            try:
                self._go2.stop_move()
            except Exception:
                logger.debug(
                    "NavCore: stop_move failed during obstacle-grid pause",
                    exc_info=True,
                )

        now = time.monotonic()
        if self._obstacle_grid_unavailable_since is None:
            self._obstacle_grid_unavailable_since = now
            logger.warning(
                "NavCore: obstacle grid unavailable; holding route for up to %.1fs",
                self.OBSTACLE_GRID_LOSS_GRACE_S,
            )
        if not self._obstacle_grid_unavailable_notified:
            self._obstacle_grid_unavailable_notified = True
            self._notify_status_change(
                f"My obstacle view paused while heading to "
                f"{self._goal_display_name(goal)}. I stopped safely and am waiting "
                "for it to recover."
            )

    def _obstacle_grid_grace_expired(self) -> bool:
        """Return whether the current depth dropout exceeded its retry window."""
        if self._obstacle_grid_unavailable_since is None:
            return False
        return (
            time.monotonic() - self._obstacle_grid_unavailable_since
            >= self.OBSTACLE_GRID_LOSS_GRACE_S
        )

    def _resume_after_obstacle_grid_recovery(self, goal: NavGoal) -> None:
        """Clear a transient depth pause and prevent it from becoming a stall."""
        if self._obstacle_grid_unavailable_since is None:
            return
        outage_s = time.monotonic() - self._obstacle_grid_unavailable_since
        self._obstacle_grid_unavailable_since = None
        was_notified = self._obstacle_grid_unavailable_notified
        self._obstacle_grid_unavailable_notified = False
        self._reset_progress_tracker(reset_recovery_attempts=False)
        logger.info(
            "NavCore: obstacle grid recovered after %.2fs; resuming '%s'",
            outage_s,
            self._goal_display_name(goal),
        )
        if was_notified:
            self._notify_status_change(
                f"My obstacle view recovered, and I am continuing toward "
                f"{self._goal_display_name(goal)}."
            )

    def _lateral_escape_clearance(self, grid: ObstacleGrid, direction: float) -> float:
        """Return visible clearance on one side of the robot for a short strafe."""
        points = occupied_xy_points(grid)
        if points.size == 0:
            return float("inf")
        signed_lateral = points[:, 1] * direction
        nearby = points[
            (signed_lateral > 0.05)
            & (points[:, 0] >= 0.0)
            # A lateral step only sweeps the robot's own front-to-back
            # footprint.  Including farther points made a wall in front look
            # like it blocked both sides, even when one side was wide open.
            & (points[:, 0] <= self.STALL_ESCAPE_ROBOT_HALF_LENGTH_M)
        ]
        if nearby.size == 0:
            return float("inf")
        return float(np.min(np.abs(nearby[:, 1])))

    def _choose_stall_escape_direction(self, grid: ObstacleGrid) -> Optional[float]:
        """Choose a safe lateral direction, preferring motion away from the obstacle."""
        clearances = {
            1.0: self._lateral_escape_clearance(grid, 1.0),
            -1.0: self._lateral_escape_clearance(grid, -1.0),
        }
        safe_directions = [
            direction
            for direction, clearance in clearances.items()
            if clearance >= self.STALL_ESCAPE_CLEARANCE_M
        ]
        if not safe_directions:
            return None

        bearing = grid.path_obstacle_bearing
        if (
            not math.isfinite(grid.path_obstacle_m)
            or not isinstance(bearing, (int, float))
            or not math.isfinite(bearing)
        ):
            bearing = grid.nearest_obstacle_bearing
        if isinstance(bearing, (int, float)) and abs(float(bearing)) >= math.radians(5.0):
            away = -math.copysign(1.0, float(bearing))
            if away in safe_directions:
                return away

        best_clearance = max(clearances[direction] for direction in safe_directions)
        best_directions = [
            direction
            for direction in safe_directions
            if clearances[direction] >= best_clearance - 0.10
        ]
        return random.choice(best_directions)

    def _execute_lateral_stall_escape(
        self,
        goal: NavGoal,
        direction: float,
    ) -> Optional[Tuple[RobotPose, str]]:
        """Take one short lateral step while continuously checking clearance."""
        self._ensure_go2()
        if not self._go2 or not getattr(self._go2, "available", False):
            return None

        speed = max(0.05, abs(self.STALL_ESCAPE_SPEED_MPS)) * direction
        distance = max(0.05, self.STALL_ESCAPE_DISTANCE_M) * random.uniform(0.8, 1.2)
        period = max(0.05, self.STALL_ESCAPE_COMMAND_PERIOD_S)
        deadline = time.monotonic() + distance / abs(speed)
        side_name = "left" if direction > 0.0 else "right"
        command = VelocityCommand(vx=0.0, vy=speed, vyaw=0.0)

        logger.warning(
            "NavCore: attempting %.2fm %s escape step before rerouting to '%s'",
            distance,
            side_name,
            self._goal_display_name(goal),
        )
        try:
            while time.monotonic() < deadline:
                latest = self._fresh_obstacle_grid(
                    self._depth_processor.get_obstacle_grid()
                )
                if (
                    latest is None
                    or self._lateral_escape_clearance(latest, direction)
                    < self.STALL_ESCAPE_CLEARANCE_M
                ):
                    logger.warning(
                        "NavCore: cancelled %s escape step; clearance changed",
                        side_name,
                    )
                    return None
                self._go2.move(vx=0.0, vy=speed, vyaw=0.0)
                self._odometry.update_from_velocity(command, period)
                time.sleep(period)
        except Exception:
            logger.exception("NavCore: %s escape step failed", side_name)
            return None
        finally:
            try:
                self._go2.stop_move()
            except Exception:
                logger.debug("NavCore: stop_move failed after escape step", exc_info=True)

        return self._odometry.get_pose(), side_name

    def _execute_stall_backup(
        self,
        goal: NavGoal,
        grid: ObstacleGrid,
    ) -> Optional[RobotPose]:
        """Backtrack briefly over recently traversed floor and verify front clearance."""
        self._ensure_go2()
        if not self._go2 or not getattr(self._go2, "available", False):
            return None

        speed = max(0.05, abs(self.STALL_BACKUP_SPEED_MPS))
        distance = max(0.05, self.STALL_BACKUP_DISTANCE_M) * random.uniform(0.8, 1.2)
        period = max(0.05, self.STALL_ESCAPE_COMMAND_PERIOD_S)
        deadline = time.monotonic() + distance / speed
        command = VelocityCommand(vx=-speed, vy=0.0, vyaw=0.0)
        previous_clearance = grid.path_obstacle_m

        logger.warning(
            "NavCore: lateral escape blocked; attempting %.2fm backward escape "
            "before rerouting to '%s'",
            distance,
            self._goal_display_name(goal),
        )
        try:
            while time.monotonic() < deadline:
                latest = self._fresh_obstacle_grid(
                    self._depth_processor.get_obstacle_grid()
                )
                if latest is None:
                    logger.warning(
                        "NavCore: cancelled backward escape; obstacle view unavailable"
                    )
                    return None

                latest_clearance = latest.path_obstacle_m
                if (
                    math.isfinite(previous_clearance)
                    and math.isfinite(latest_clearance)
                    and latest_clearance
                    < previous_clearance - self.STALL_BACKUP_CLEARANCE_DROP_M
                ):
                    logger.warning(
                        "NavCore: cancelled backward escape; front clearance worsened "
                        "from %.2fm to %.2fm",
                        previous_clearance,
                        latest_clearance,
                    )
                    return None

                self._go2.move(vx=-speed, vy=0.0, vyaw=0.0)
                self._odometry.update_from_velocity(command, period)
                previous_clearance = latest_clearance
                time.sleep(period)
        except Exception:
            logger.exception("NavCore: backward escape step failed")
            return None
        finally:
            try:
                self._go2.stop_move()
            except Exception:
                logger.debug(
                    "NavCore: stop_move failed after backward escape",
                    exc_info=True,
                )

        return self._odometry.get_pose()

    def _execute_stall_escape(
        self,
        goal: NavGoal,
        grid: Optional[ObstacleGrid],
    ) -> Optional[Tuple[RobotPose, str]]:
        """Change position before replanning, backing up if both sides are blocked."""
        if grid is None:
            return None

        direction = self._choose_stall_escape_direction(grid)
        if direction is not None:
            return self._execute_lateral_stall_escape(goal, direction)

        logger.warning(
            "NavCore: no safe lateral direction; trying a guarded backward escape"
        )
        backed_pose = self._execute_stall_backup(goal, grid)
        if backed_pose is None:
            return None

        latest = self._fresh_obstacle_grid(self._depth_processor.get_obstacle_grid())
        direction = (
            self._choose_stall_escape_direction(latest)
            if latest is not None
            else None
        )
        if direction is None:
            logger.warning(
                "NavCore: lateral directions remain blocked after backing up; "
                "rerouting from the backed-up pose"
            )
            return backed_pose, "backward"

        lateral = self._execute_lateral_stall_escape(goal, direction)
        if lateral is None:
            logger.warning(
                "NavCore: lateral follow-up became unsafe; rerouting from the "
                "backed-up pose"
            )
            return backed_pose, "backward"
        pose, side_name = lateral
        return pose, f"backward then {side_name}"

    @staticmethod
    def _stall_side_view_clearance(grid: ObstacleGrid, direction: float) -> float:
        """Estimate visible open distance in the left or right turning sector."""
        points = occupied_xy_points(grid)
        if points.size == 0:
            return float("inf")
        distances = np.hypot(points[:, 0], points[:, 1])
        bearings = np.arctan2(points[:, 1], points[:, 0]) * direction
        selected = distances[
            (bearings >= math.radians(15.0))
            & (bearings <= math.radians(60.0))
        ]
        if selected.size == 0:
            return float("inf")
        return float(np.percentile(selected, 10))

    def _choose_stall_scan_direction(self, grid: ObstacleGrid) -> float:
        """Choose an exploratory turn and force the next attempt to try the other side."""
        if self._last_stall_scan_direction in {-1.0, 1.0}:
            return -self._last_stall_scan_direction

        clearances = {
            1.0: self._stall_side_view_clearance(grid, 1.0),
            -1.0: self._stall_side_view_clearance(grid, -1.0),
        }
        best = max(clearances.values())
        candidates = [
            direction
            for direction, clearance in clearances.items()
            if (
                clearance == best
                or (
                    math.isfinite(clearance)
                    and math.isfinite(best)
                    and clearance >= best - 0.10
                )
            )
        ]
        return random.choice(candidates)

    def _measure_wall_alignment(self) -> Optional[float]:
        """
        Return how far the robot is turned into a wall, in radians.

        The obstacle grid already carries a wall line fit, but a single frame's
        estimate is noisy, so several are taken and the median returned. None
        means no wall was measured consistently enough to turn against, and the
        caller should fall back to searching for an opening.

        Sign matches the steering correction: applying yaw of this sign rotates
        the robot towards parallel with the wall.
        """
        # The wall fit lives on the local planner, which already computes it
        # every cycle to steer by. Missing either collaborator is not an error
        # here: the caller falls back to searching for an opening.
        planner = getattr(self, "_local_planner", None)
        depth = getattr(self, "_depth_processor", None)
        if planner is None or depth is None:
            return None

        readings: List[float] = []
        period = max(0.05, self.STALL_ESCAPE_COMMAND_PERIOD_S)
        for _ in range(max(1, self.STALL_WALL_ALIGN_SAMPLES)):
            try:
                grid = self._fresh_obstacle_grid(depth.get_obstacle_grid())
                if grid is not None:
                    geometry = planner._estimate_wall_geometry(grid)
                    if geometry is not None:
                        readings.append(float(geometry[0]))
            except Exception:
                logger.debug("NavCore: wall angle reading failed", exc_info=True)
            time.sleep(period)

        if len(readings) < max(2, (self.STALL_WALL_ALIGN_SAMPLES + 1) // 2):
            return None

        spread = max(readings) - min(readings)
        if spread > self.STALL_WALL_ALIGN_MAX_SPREAD_RAD:
            logger.warning(
                "NavCore: wall angle unreliable across %d readings (spread %.0fdeg)",
                len(readings),
                math.degrees(spread),
            )
            return None

        return float(np.median(readings))

    def _plan_stall_turn(
        self,
        grid: ObstacleGrid,
        min_angle: float,
        max_angle: float,
    ) -> Tuple[float, float, str]:
        """
        Decide which way to pivot out of a stall, and by how much.

        Prefers the measured wall angle: nosed into a wall at a known angle,
        the way out is to turn by that angle rather than to guess and re-check.
        Falls back to the exploratory scan when no wall can be measured.
        """
        wall_angle = self._measure_wall_alignment()
        if wall_angle is not None and abs(wall_angle) >= self.STALL_WALL_ALIGN_MIN_ANGLE_RAD:
            direction = 1.0 if wall_angle > 0.0 else -1.0
            target = min(
                abs(wall_angle) + self.STALL_WALL_ALIGN_MARGIN_RAD,
                max_angle,
            )
            logger.warning(
                "NavCore: wall measured %.0fdeg off parallel; turning %.0fdeg %s to align",
                math.degrees(wall_angle),
                math.degrees(target),
                "left" if direction > 0.0 else "right",
            )
            return direction, target, "wall alignment"

        direction = self._choose_stall_scan_direction(grid)
        return direction, random.uniform(min_angle, max_angle), "opening scan"

    def _execute_stall_turn_scan(
        self,
        goal: NavGoal,
        grid: Optional[ObstacleGrid],
    ) -> Optional[Tuple[RobotPose, str]]:
        """Pivot to align with a measured wall, or to inspect an opening."""
        if grid is None:
            return None
        self._ensure_go2()
        if not self._go2 or not getattr(self._go2, "available", False):
            return None

        min_angle = max(0.0, self.STALL_SCAN_MIN_ANGLE_RAD)
        max_angle = max(min_angle, self.STALL_SCAN_MAX_ANGLE_RAD)
        direction, target_angle, basis = self._plan_stall_turn(
            grid, min_angle, max_angle
        )
        self._last_stall_scan_direction = direction
        side_name = "left" if direction > 0.0 else "right"
        yaw_rate = max(0.10, abs(self.STALL_SCAN_YAW_RATE_RPS)) * direction
        period = max(0.05, self.STALL_ESCAPE_COMMAND_PERIOD_S)
        # An alignment turn has a known target, so it must not be cut short by
        # the early exit that an exploratory scan relies on.
        early_exit_after = min_angle if basis == "opening scan" else target_angle
        period = max(0.05, self.STALL_ESCAPE_COMMAND_PERIOD_S)
        turned = 0.0
        command = VelocityCommand(vx=0.0, vy=0.0, vyaw=yaw_rate)

        logger.warning(
            "NavCore: turning up to %.0fdeg %s by %s toward '%s'",
            math.degrees(target_angle),
            side_name,
            basis,
            self._goal_display_name(goal),
        )
        try:
            while turned < target_angle:
                latest = self._fresh_obstacle_grid(
                    self._depth_processor.get_obstacle_grid()
                )
                if latest is None:
                    logger.warning(
                        "NavCore: paused %s turn scan; obstacle view unavailable",
                        side_name,
                    )
                    return None
                if (
                    turned >= early_exit_after
                    and latest.path_obstacle_m >= self.AVOIDANCE_DISTANCE_M
                ):
                    logger.warning(
                        "NavCore: found an open path after turning %.0fdeg %s",
                        math.degrees(turned),
                        side_name,
                    )
                    break
                self._go2.move(vx=0.0, vy=0.0, vyaw=yaw_rate)
                self._odometry.update_from_velocity(command, period)
                time.sleep(period)
                turned += abs(yaw_rate) * period
        except Exception:
            logger.exception("NavCore: %s turn scan failed", side_name)
            return None
        finally:
            try:
                self._go2.stop_move()
            except Exception:
                logger.debug("NavCore: stop failed after turn scan", exc_info=True)

        return self._odometry.get_pose(), side_name

    def _recover_from_stall(
        self,
        goal: NavGoal,
        pose: RobotPose,
        grid: Optional[ObstacleGrid],
        *,
        allow_locomotion_recovery: bool = True,
    ) -> bool:
        """Change position safely, then replan while preserving the destination."""
        planner = getattr(self, "_global_planner", None)
        waypoint = planner.get_current_waypoint() if planner is not None else None
        waypoint_distance = (
            math.hypot(waypoint.x - pose.x, waypoint.y - pose.y)
            if isinstance(waypoint, MapNode)
            else float("inf")
        )
        if (
            allow_locomotion_recovery
            and grid is not None
            and grid.path_obstacle_m >= self.AVOIDANCE_DISTANCE_M
        ):
            clear_attempts = getattr(self, "_clear_motion_recovery_attempts", 0)
            if (
                clear_attempts
                >= self.MAX_CLEAR_MOTION_RECOVERY_ATTEMPTS
            ):
                if not getattr(self, "_locomotion_recovery_error", None):
                    stage = getattr(
                        self,
                        "_locomotion_recovery_verification_pending",
                        None,
                    )
                    self._locomotion_recovery_error = (
                        f"{stage or 'locomotion'} recovery completed, but subsequent "
                        "commands still produced no measured translation"
                    )
                return False

            advanced_name = None
            if (
                isinstance(waypoint, MapNode)
                and "metric_transit" in waypoint.tags
                and waypoint_distance <= self.CLEAR_STALL_TRANSIT_SKIP_M
            ):
                advanced = planner.advance_current_waypoint(
                    on_advance=self._notify_waypoint_advance,
                )
                if advanced is not None and advanced[1] is not None:
                    advanced_name = waypoint.name

            self._ensure_go2()
            recovered = False
            recovery_stage = "soft locomotion-mode reset"
            if self._go2:
                try:
                    if clear_attempts == 0 and getattr(
                        self._go2,
                        "available",
                        False,
                    ):
                        recover = getattr(self._go2, "recover_locomotion", None)
                        if callable(recover):
                            recovered = bool(recover())
                        else:
                            self._go2.stop_move()
                        # A rejected mode reset falls through immediately to a
                        # fresh client rather than spending a watchdog cycle on
                        # a recovery operation known to have failed.
                        if not recovered:
                            recovery_stage = "SportClient reinitialization"
                            reinitialize = getattr(
                                self._go2,
                                "reinitialize_locomotion",
                                None,
                            )
                            recovered = bool(
                                reinitialize() if callable(reinitialize) else False
                            )
                    else:
                        recovery_stage = "SportClient reinitialization"
                        reinitialize = getattr(
                            self._go2,
                            "reinitialize_locomotion",
                            None,
                        )
                        recovered = bool(
                            reinitialize() if callable(reinitialize) else False
                        )
                except Exception:
                    logger.exception("NavCore: locomotion-mode recovery failed")

            if not recovered:
                detail = getattr(self._go2, "last_recovery_error", None)
                self._locomotion_recovery_error = (
                    f"{recovery_stage} failed"
                    + (f": {detail}" if detail else "")
                )
                self._clear_motion_recovery_attempts = (
                    self.MAX_CLEAR_MOTION_RECOVERY_ATTEMPTS
                )
                logger.error(
                    "NavCore: locomotion recovery could not continue: %s",
                    self._locomotion_recovery_error,
                )
                return False

            with self._state_lock:
                self._clear_motion_recovery_attempts = clear_attempts + 1
                attempt = self._clear_motion_recovery_attempts
                self._locomotion_recovery_verification_pending = recovery_stage
                self._locomotion_recovery_error = None
                self._state = NavState.NAVIGATING
                self._last_stop_reason = None
                self._last_motion_command = VelocityCommand()
                self._local_planner.reset_navigation_state()
                self._reset_progress_tracker(reset_recovery_attempts=False)
            logger.warning(
                "NavCore: clear path but commanded translation was not measured; "
                "%s %d/%d completed and is awaiting measured-motion verification%s",
                recovery_stage,
                attempt,
                self.MAX_CLEAR_MOTION_RECOVERY_ATTEMPTS,
                (
                    f"; advanced past nearby transit point '{advanced_name}'"
                    if advanced_name
                    else ""
                ),
            )
            if attempt == 1:
                self._notify_status_change(
                    "Locomotion stopped responding. I preserved my position and "
                    "route, reset the locomotion mode, and am using a bounded "
                    "route-aligned command to verify motion "
                    f"before continuing toward {self._goal_display_name(goal)}."
                )
            else:
                self._notify_status_change(
                    "The locomotion-mode reset was not verified, so I replaced the "
                    "robot motion-service client while preserving my position and "
                    "route. I am verifying motion again with a bounded route-aligned "
                    "command."
                )
            return True

        if self._stuck_recovery_attempts >= self.MAX_STUCK_RECOVERY_ATTEMPTS:
            return False

        escaped = self._execute_stall_escape(goal, grid)
        escape_direction = None
        scan_direction = None
        if escaped is None:
            # A close obstacle can prevent every translational escape even when
            # one turning sector is open.  Do not spend all recovery attempts
            # waiting in the same pose: pivot to inspect that sector, then
            # replan from the new heading.
            latest_grid = self._fresh_obstacle_grid(
                self._depth_processor.get_obstacle_grid()
            )
            scanned = self._execute_stall_turn_scan(goal, latest_grid)
            if scanned is not None:
                pose, scan_direction = scanned

        if escaped is None and scan_direction is None:
            with self._state_lock:
                self._stuck_recovery_attempts += 1
                attempt = self._stuck_recovery_attempts
                self._state = NavState.NAVIGATING
                self._reset_progress_tracker(reset_recovery_attempts=False)
            logger.warning(
                "NavCore: stall recovery %d/%d could not move safely; pausing "
                "before retrying toward '%s'",
                attempt,
                self.MAX_STUCK_RECOVERY_ATTEMPTS,
                self._goal_display_name(goal),
            )
            self._notify_status_change(
                f"I stalled while heading to {self._goal_display_name(goal)}. "
                "I could not make a safe recovery move yet, so I will pause and "
                "try again."
            )
            return True
        if escaped is not None:
            pose, escape_direction = escaped
            latest_grid = self._fresh_obstacle_grid(
                self._depth_processor.get_obstacle_grid()
            )
            scanned = self._execute_stall_turn_scan(goal, latest_grid)
            if scanned is not None:
                pose, scan_direction = scanned

        # Use a view captured in the recovered orientation when projecting
        # depth points into the map.  Reusing the pre-turn view rotates dynamic
        # obstacles into the wrong world locations and can reject a valid path.
        latest_grid = self._fresh_obstacle_grid(
            self._depth_processor.get_obstacle_grid()
        )

        if goal.goal_type == "semantic":
            dynamic_obstacles = None
            metric_map = self._topo_map.metric_map
            if metric_map is not None and latest_grid is not None:
                robot_points = occupied_xy_points(latest_grid)
                dynamic_obstacles = metric_map.robot_points_to_world(
                    robot_points,
                    pose.x,
                    pose.y,
                    pose.yaw,
                )
            if metric_map is None:
                path = self._global_planner.replan_path_preserving_progress(
                    pose,
                    goal.label or "",
                )
            else:
                path = self._global_planner.replan_path_around_obstacles(
                    pose,
                    goal.label or "",
                    dynamic_obstacles,
                )
            if path is None:
                with self._state_lock:
                    self._stuck_recovery_attempts += 1
                    attempt = self._stuck_recovery_attempts
                    self._state = NavState.NAVIGATING
                    self._last_stop_reason = None
                    self._reset_progress_tracker(reset_recovery_attempts=False)
                logger.warning(
                    "NavCore: recovery %d/%d rejected an unsafe detour; retaining "
                    "the route and trying the other opening next",
                    attempt,
                    self.MAX_STUCK_RECOVERY_ATTEMPTS,
                )
                self._notify_status_change(
                    f"I avoided an unsafe turn away from {self._goal_display_name(goal)}. "
                    "I will try the other opening and reroute again."
                )
                return True
            goal.x = path[-1].x
            goal.y = path[-1].y
            self._local_planner.reset_navigation_state()

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
            self._path_obstacle_active_since = None
            self._path_obstacle_clear_since = None
            self._reset_progress_tracker(reset_recovery_attempts=False)

        waypoint = self._global_planner.get_current_waypoint()
        remaining = (
            math.hypot(waypoint.x - pose.x, waypoint.y - pose.y)
            if isinstance(waypoint, MapNode)
            else math.hypot(goal.x - pose.x, goal.y - pose.y)
        )
        waypoint_name = waypoint.name if isinstance(waypoint, MapNode) else "goal"
        logger.warning(
            "NavCore: no-progress recovery %d/%d toward '%s'; "
            "pose=(%.2f, %.2f, %.0fdeg) waypoint='%s' remaining=%.2fm",
            attempt,
            self.MAX_STUCK_RECOVERY_ATTEMPTS,
            self._goal_display_name(goal),
            pose.x,
            pose.y,
            math.degrees(pose.yaw),
            waypoint_name,
            remaining,
        )
        self._notify_status_change(
            f"I stalled while heading to {self._goal_display_name(goal)}. "
            + (
                f"I stepped {escape_direction}"
                if escape_direction
                else f"I turned {scan_direction}"
            )
            + (
                f", turned {scan_direction}"
                if escape_direction and scan_direction
                else ""
            )
            + ", rerouted, and am continuing."
        )
        return True

    def _complete_navigation(self, goal: NavGoal, distance_m: Optional[float] = None) -> None:
        """Stop motion and consistently close a successful navigation action."""
        if distance_m is None:
            pose = self._odometry.get_pose()
            distance_m = math.hypot(goal.x - pose.x, goal.y - pose.y)
        self._ensure_go2()
        if self._go2 and getattr(self._go2, "available", False):
            self._go2.stop_move()
        arrived_node = (
            self._topo_map.get_node(goal.label or "")
            if goal.goal_type == "semantic"
            else None
        )
        with self._state_lock:
            self._state = NavState.IDLE
            self._goal = None
            self._last_stop_reason = None
            if arrived_node is not None:
                self._current_location_name = arrived_node.name
            self._clear_planner_and_obstacle_state()
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
            self._last_motion_command = cmd
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

        Plans over the static occupancy map, then the nav loop handles live
        obstacle avoidance and confirmed blocked-route replanning.
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
        arrival_tolerance = (
            goal_node.arrival_tolerance_m
            if goal_node.arrival_tolerance_m is not None
            else self.SEMANTIC_ARRIVAL_TOLERANCE_M
        )
        immediate_tolerance = (
            min(
                arrival_tolerance,
                GlobalPlanner.FINAL_METRIC_LONGITUDINAL_TOLERANCE_M,
            )
            if self._topo_map.metric_map is not None
            else arrival_tolerance
        )
        if (
            dist_to_goal <= immediate_tolerance
            and (
                self._topo_map.metric_map is None
                or self._current_location_name == goal_node.name
            )
        ):
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
                self._current_location_name = goal_node.name
                self._clear_planner_and_obstacle_state()
            logger.info(
                "NavCore: already at '%s' (dist=%.2fm), no movement needed",
                goal_label,
                dist_to_goal,
            )
            self._notify_status_change(f"I am already at {goal_label}.")
            self._stop_depth_when_idle()
            return True

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

        # Do not let a single robot-frame scan redefine the global route at
        # startup. Nearby mapped walls and furniture are already represented in
        # the occupancy map, while pose/heading error can project them across the
        # correct exit corridor. Live depth still gates every motion cycle and is
        # included by the confirmed blocked-route replanner.
        path = self._global_planner.plan_path(
            pose,
            destination,
            dynamic_obstacles_xy=None,
        )
        if path is None:
            with self._state_lock:
                self._state = NavState.IDLE
                self._goal = None
                self._last_stop_reason = f"No route to {goal_label}"
                self._clear_planner_and_obstacle_state()
            self._notify_status_change(
                f"I did not move because I do not have a collision-free route to "
                f"{goal_label}."
            )
            self._stop_depth_when_idle()
            return False

        # A full-map search can outlive the frame it started from.  Do not begin
        # moving until the safety loop has a current view again.
        if not self._wait_for_depth_grid(self.DEPTH_READY_TIMEOUT_S):
            with self._state_lock:
                self._state = NavState.E_STOP
                self._goal = None
                self._last_stop_reason = "E-STOP: obstacle grid stale after route planning"
                self._clear_planner_and_obstacle_state()
            self._notify_status_change(
                f"I did not move toward {goal_label} because my obstacle view "
                "did not refresh after route planning."
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
            self._current_location_name = None
            self._reset_progress_tracker()
            self._local_planner.reset_navigation_state()

        self._ensure_running()
        logger.info("NavCore: navigating to '%s' via %d waypoints", destination, len(path))
        return True

    def _wait_for_depth_grid(self, timeout_s: float) -> bool:
        """Wait briefly for obstacle sensing to publish a fresh grid."""
        if not self._ensure_depth_running():
            return False

        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            if self._fresh_obstacle_grid(
                self._depth_processor.get_obstacle_grid()
            ) is not None:
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
            self._current_location_name = node.name
            self._clear_planner_and_obstacle_state()

        self._odometry.set_pose(node.x, node.y, heading_rad)
        self._last_metric_localization_pose = self._odometry.get_pose()
        self._last_metric_localization_time = time.monotonic()
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
            self._current_location_name = None
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

        This is the hardware-safe forward primitive validated on the robot. It does
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
            self._current_location_name = None
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
            current_location_name = getattr(self, "_current_location_name", None)

        pose = self._odometry.get_pose()
        parts = [f"Navigation state: {state.value}"]
        parts.append(f"Position: ({pose.x:.1f}, {pose.y:.1f}), heading: {math.degrees(pose.yaw):.0f} deg")

        if current_location_name:
            current_node = self._topo_map.get_node(current_location_name)
            if current_node is not None:
                parts.append(
                    f"Current mapped location: {self._topo_map.get_node_label(current_node)}"
                )
        else:
            parts.append("Current mapped location: unverified")

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
                self._manual_override_start_pose = self._odometry.get_pose()
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
            self._manual_override_start_pose = None
            self._notify_status_change(
                "Manual control ended, so I canceled the previous relative movement command."
            )
            self._stop_depth_when_idle()
            return True

        measured_pose = self._odometry.get_pose()
        path = self._global_planner.plan_path(
            measured_pose,
            goal.label or "",
            dynamic_obstacles_xy=None,
        )
        heading_realigned = False
        if path is not None and len(path) > 1:
            first_target = path[1]
            route_bearing = math.atan2(
                first_target.y - measured_pose.y,
                first_target.x - measured_pose.x,
            )
            route_heading_error = abs(
                math.atan2(
                    math.sin(route_bearing - measured_pose.yaw),
                    math.cos(route_bearing - measured_pose.yaw),
                )
            )
            start_pose = getattr(self, "_manual_override_start_pose", None)
            manual_turn = (
                abs(
                    math.atan2(
                        math.sin(measured_pose.yaw - start_pose.yaw),
                        math.cos(measured_pose.yaw - start_pose.yaw),
                    )
                )
                if isinstance(start_pose, RobotPose)
                else 0.0
            )
            if (
                manual_turn >= self.REMOTE_HEADING_REALIGN_MIN_TURN_RAD
                and route_heading_error >= self.REMOTE_HEADING_REALIGN_MIN_ERROR_RAD
            ):
                # A substantial human course correction is authoritative.  If
                # SDK yaw still disagrees sharply with the newly planned route,
                # re-anchor the map yaw so autonomous control continues in the
                # physical direction selected with the remote instead of
                # immediately undoing the correction.
                measured_pose = RobotPose(
                    x=measured_pose.x,
                    y=measured_pose.y,
                    yaw=route_bearing,
                    timestamp=time.time(),
                )
                self._odometry.apply_pose_correction(
                    measured_pose.x,
                    measured_pose.y,
                    measured_pose.yaw,
                )
                heading_realigned = True
        if path is not None:
            goal.x, goal.y = path[-1].x, path[-1].y
            self._local_planner.reset_navigation_state()
        self._reset_progress_tracker()
        self._manual_override_start_pose = None
        logger.info(
            "NavCore: preserved remote correction pose=(%.2f, %.2f, %.0fdeg)%s and %s "
            "a fresh route toward '%s'",
            measured_pose.x,
            measured_pose.y,
            math.degrees(measured_pose.yaw),
            " with route-aligned heading" if heading_realigned else "",
            "planned" if path is not None else "could not plan",
            self._goal_display_name(goal),
        )
        self._notify_status_change(
            f"Manual control ended. I preserved the corrected position and heading and "
            f"am continuing toward {self._goal_display_name(goal)}."
        )
        return False

    def _reset_route_wall_heading_confirmation(self) -> None:
        self._route_wall_heading_candidate = None
        self._route_wall_heading_readings = 0

    def _maybe_align_heading_to_route_wall(
        self,
        grid: Optional[ObstacleGrid],
        pose: RobotPose,
    ) -> RobotPose:
        """Use a repeatedly observed parallel wall to correct map-frame yaw."""
        segment_heading = self._global_planner.current_segment_heading()
        waypoint = self._global_planner.get_current_waypoint()
        if (
            grid is None
            or segment_heading is None
            or waypoint is None
            or "metric_transit" not in waypoint.tags
            or waypoint.name not in {"__metric_001__", "__metric_002__"}
        ):
            # Wall yaw re-anchoring corrects a bad starting orientation only.
            # Once route motion is established, short or slanted wall fragments
            # must not rewrite map heading and steer the robot off its route.
            self._reset_route_wall_heading_confirmation()
            return pose
        if grid.path_obstacle_m <= self.AVOIDANCE_DISTANCE_M:
            self._reset_route_wall_heading_confirmation()
            return pose
        last_cmd = getattr(self, "_last_motion_command", VelocityCommand())
        if (
            abs(last_cmd.vx) < 0.03
            and abs(last_cmd.vy) < 0.03
            and abs(last_cmd.vyaw) > 0.12
        ):
            self._reset_route_wall_heading_confirmation()
            return pose

        wall_fits = self._local_planner._visible_wall_fits(grid)
        candidates = [
            heading
            for heading, _lateral in wall_fits.values()
            if abs(heading) <= self.ROUTE_WALL_MAX_OBSERVED_HEADING_RAD
        ]
        if not candidates:
            self._reset_route_wall_heading_confirmation()
            return pose
        observed_heading = min(candidates, key=abs)
        corrected_yaw = math.atan2(
            math.sin(segment_heading - observed_heading),
            math.cos(segment_heading - observed_heading),
        )
        correction = math.atan2(
            math.sin(corrected_yaw - pose.yaw),
            math.cos(corrected_yaw - pose.yaw),
        )
        if not (
            self.ROUTE_WALL_MIN_YAW_CORRECTION_RAD
            <= abs(correction)
            <= self.ROUTE_WALL_MAX_YAW_CORRECTION_RAD
        ):
            self._reset_route_wall_heading_confirmation()
            return pose

        previous = self._route_wall_heading_candidate
        if previous is None or abs(
            math.atan2(
                math.sin(corrected_yaw - previous),
                math.cos(corrected_yaw - previous),
            )
        ) > math.radians(5.0):
            self._route_wall_heading_candidate = corrected_yaw
            self._route_wall_heading_readings = 1
            return pose

        self._route_wall_heading_candidate = corrected_yaw
        self._route_wall_heading_readings += 1
        if self._route_wall_heading_readings < self.ROUTE_WALL_HEADING_CONFIRM_READINGS:
            return pose

        corrected = RobotPose(
            x=pose.x,
            y=pose.y,
            yaw=corrected_yaw,
            timestamp=time.time(),
        )
        self._odometry.apply_pose_correction(corrected.x, corrected.y, corrected.yaw)
        self._local_planner.reset_navigation_state()
        self._reset_route_wall_heading_confirmation()
        logger.info(
            "NavCore: route-parallel wall corrected map heading by %+.0fdeg",
            math.degrees(correction),
        )
        return corrected

    def _accept_expected_metric_corner(
        self,
        pose: RobotPose,
        grid: Optional[ObstacleGrid],
    ) -> Tuple[RobotPose, bool]:
        """Advance a mapped sharp corner when its expected transverse wall appears."""
        upcoming = self._global_planner.upcoming_turn()
        segment = self._global_planner.current_segment()
        if grid is None or upcoming is None or segment is None:
            return pose, False
        corner, _following, turn = upcoming
        if (
            "metric_transit" not in corner.tags
            or abs(turn) < self.EXPECTED_CORNER_MIN_TURN_RAD
            or not (
                self.EXPECTED_CORNER_MIN_WALL_DISTANCE_M
                <= grid.path_obstacle_m
                <= self.EXPECTED_CORNER_MAX_WALL_DISTANCE_M
            )
            or abs(grid.path_obstacle_bearing) > self.FORWARD_HAZARD_CONE_RAD
            or not is_transverse_wall(grid, grid.path_obstacle_m, 0.50)
        ):
            return pose, False

        start, target = segment
        edge_x = target.x - start.x
        edge_y = target.y - start.y
        edge_length = math.hypot(edge_x, edge_y)
        if edge_length <= 1e-6:
            return pose, False
        unit_x, unit_y = edge_x / edge_length, edge_y / edge_length
        remaining_x = target.x - pose.x
        remaining_y = target.y - pose.y
        longitudinal_error = remaining_x * unit_x + remaining_y * unit_y
        lateral_error = abs(remaining_x * unit_y - remaining_y * unit_x)
        if not (
            0.0 <= longitudinal_error <= self.EXPECTED_CORNER_MAX_POSE_ERROR_M
            and lateral_error <= 0.55
        ):
            return pose, False

        corrected = RobotPose(
            x=pose.x + longitudinal_error * unit_x,
            y=pose.y + longitudinal_error * unit_y,
            yaw=pose.yaw,
            timestamp=time.time(),
        )
        self._odometry.apply_pose_correction(corrected.x, corrected.y, corrected.yaw)
        advanced = self._global_planner.advance_current_waypoint(
            on_advance=self._notify_waypoint_advance,
        )
        if advanced is None:
            return pose, False
        self._local_planner.reset_navigation_state()
        self._reset_progress_tracker()
        logger.info(
            "NavCore: expected transverse wall accepted metric corner %.2fm early; "
            "starting the %+.0fdeg route turn",
            longitudinal_error,
            math.degrees(turn),
        )
        return corrected, True

    def _maybe_correct_metric_pose(
        self,
        grid: Optional[ObstacleGrid],
        pose: RobotPose,
    ) -> RobotPose:
        """Apply a small, high-confidence depth-to-map odometry correction."""
        metric_map = self._topo_map.metric_map
        if metric_map is None or grid is None:
            return pose
        last_cmd = self._last_motion_command
        if last_cmd.vx <= 0.05 or abs(last_cmd.vyaw) > 0.12:
            return pose
        if not self._odometry.has_confirmed_translation():
            return pose
        distance_since_correction = math.hypot(
            pose.x - self._last_metric_localization_pose.x,
            pose.y - self._last_metric_localization_pose.y,
        )
        if distance_since_correction < self.METRIC_LOCALIZATION_MIN_TRAVEL_M:
            return pose
        now = time.monotonic()
        if (
            self.METRIC_LOCALIZATION_INTERVAL_S > 0.0
            and now - self._last_metric_localization_time
            < self.METRIC_LOCALIZATION_INTERVAL_S
        ):
            return pose
        scan_points = occupied_xy_points(grid)
        useful = scan_points[
            (scan_points[:, 0] >= 0.20)
            & (scan_points[:, 0] <= 4.0)
            & (np.abs(scan_points[:, 1]) <= 2.5)
        ]
        if len(useful) < 30:
            return pose
        covariance = np.cov(useful, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues[-1] <= 1e-6 or eigenvalues[0] / eigenvalues[-1] < 0.06:
            logger.debug("NavCore: skipped ambiguous single-wall pose correction")
            return pose
        correction = metric_map.match_pose(
            scan_points,
            pose.x,
            pose.y,
            pose.yaw,
            minimum_improvement_m=0.08,
            maximum_score_m=0.12,
            minimum_matched_fraction=0.45,
        )
        if correction is None:
            return pose

        dx = correction.x - pose.x
        dy = correction.y - pose.y
        distance = math.hypot(dx, dy)
        if distance > self.METRIC_LOCALIZATION_MAX_TRANSLATION_M > 0.0:
            scale = self.METRIC_LOCALIZATION_MAX_TRANSLATION_M / distance
            dx *= scale
            dy *= scale
        yaw_delta = math.atan2(
            math.sin(correction.yaw - pose.yaw),
            math.cos(correction.yaw - pose.yaw),
        )
        yaw_delta = float(
            np.clip(
                yaw_delta,
                -self.METRIC_LOCALIZATION_MAX_YAW_RAD,
                self.METRIC_LOCALIZATION_MAX_YAW_RAD,
            )
        )
        corrected = RobotPose(
            x=pose.x + dx,
            y=pose.y + dy,
            yaw=pose.yaw + yaw_delta,
            timestamp=time.time(),
        )
        self._odometry.apply_pose_correction(corrected.x, corrected.y, corrected.yaw)
        self._last_metric_localization_time = now
        self._last_metric_localization_pose = corrected
        logger.info(
            "NavCore: metric pose correction dx=%+.2fm dy=%+.2fm yaw=%+.1fdeg "
            "score=%.2fm improvement=%.2fm matched=%.0f%%",
            dx,
            dy,
            math.degrees(yaw_delta),
            correction.score_m,
            correction.improvement_m,
            100.0 * correction.matched_fraction,
        )
        return corrected

    def _parallel_wall_projection_is_clear(
        self,
        grid: Optional[ObstacleGrid],
        pose: RobotPose,
    ) -> bool:
        """Ignore a side-wall edge when the mapped route runs parallel to it."""
        if (
            grid is None
            or grid.path_obstacle_m > self.AVOIDANCE_DISTANCE_M
            or abs(grid.path_obstacle_bearing) < math.radians(10.0)
            or grid.path_obstacle_m < self.MAPPED_SIDE_ROUTE_MIN_CLEARANCE_M
        ):
            return False
        guidance_heading = self._global_planner.current_segment_guidance(pose)
        if isinstance(guidance_heading, (int, float)) and math.isfinite(
            guidance_heading
        ):
            route_heading = guidance_heading - pose.yaw
        else:
            waypoint = self._global_planner.get_current_waypoint()
            if not isinstance(waypoint, MapNode):
                return False
            route_heading = math.atan2(
                waypoint.y - pose.y,
                waypoint.x - pose.x,
            ) - pose.yaw
        route_heading = math.atan2(
            math.sin(route_heading),
            math.cos(route_heading),
        )
        supported = self._local_planner.parallel_wall_supports_route(
            grid,
            route_heading,
        )
        mapped_side_clear = (
            grid.path_obstacle_m >= self.MAPPED_SIDE_ROUTE_MIN_CLEARANCE_M
            and abs(grid.path_obstacle_bearing) >= math.radians(30.0)
        )
        supported = supported or mapped_side_clear
        if supported:
            logger.debug(
                "NavCore: treating parallel mapped wall at %+.0fdeg as side "
                "geometry, not a blocked route",
                math.degrees(grid.path_obstacle_bearing),
            )
        return supported

    def _metric_arrival_sensor_is_confirmed(
        self,
        grid: Optional[ObstacleGrid],
        pose: RobotPose,
    ) -> bool:
        """Confirm a claimed destination pose against independent depth geometry."""
        metric_map = self._topo_map.metric_map
        if metric_map is None or grid is None:
            self._arrival_map_consistency_readings = 0
            return False
        consistency = metric_map.pose_consistency(
            occupied_xy_points(grid),
            pose.x,
            pose.y,
            pose.yaw,
        )
        if consistency is None:
            self._arrival_map_consistency_readings = 0
            logger.info(
                "NavCore: destination arrival deferred; insufficient structural "
                "depth geometry for map validation"
            )
            return False

        score_m, matched_fraction = consistency
        consistent = (
            score_m <= self.ARRIVAL_MAP_MAX_SCORE_M
            and matched_fraction >= self.ARRIVAL_MAP_MIN_MATCHED_FRACTION
        )
        if consistent:
            self._arrival_map_consistency_readings += 1
        else:
            self._arrival_map_consistency_readings = 0
        logger.info(
            "NavCore: destination map validation score=%.2fm matched=%.0f%% "
            "consistent=%s readings=%d/%d",
            score_m,
            100.0 * matched_fraction,
            consistent,
            self._arrival_map_consistency_readings,
            self.ARRIVAL_MAP_CONFIRM_READINGS,
        )
        return (
            self._arrival_map_consistency_readings
            >= self.ARRIVAL_MAP_CONFIRM_READINGS
        )

    def _recover_regressing_metric_route(
        self,
        goal: NavGoal,
        pose: RobotPose,
        waypoint: MapNode,
        waypoint_distance: float,
    ) -> bool:
        """Stop and replan when measured motion is persistently leaving the route."""
        if "metric_transit" not in waypoint.tags:
            self._route_progress_waypoint_name = None
            self._route_progress_best_distance = float("inf")
            self._route_regression_since = None
            return False

        if self._route_progress_waypoint_name != waypoint.name:
            self._route_progress_waypoint_name = waypoint.name
            self._route_progress_best_distance = waypoint_distance
            self._route_regression_since = None
            return False

        if waypoint_distance < self._route_progress_best_distance:
            self._route_progress_best_distance = waypoint_distance
            self._route_regression_since = None
            return False

        regression = waypoint_distance - self._route_progress_best_distance
        if regression < self.METRIC_ROUTE_REGRESSION_DISTANCE_M:
            self._route_regression_since = None
            return False

        now = time.monotonic()
        if self._route_regression_since is None:
            self._route_regression_since = now
            logger.warning(
                "NavCore: route regression detected at '%s': best=%.2fm now=%.2fm; "
                "holding for confirmation",
                waypoint.name,
                self._route_progress_best_distance,
                waypoint_distance,
            )
            return False
        if now - self._route_regression_since < self.METRIC_ROUTE_REGRESSION_CONFIRM_S:
            return False

        self._ensure_go2()
        if self._go2 and getattr(self._go2, "available", False):
            self._go2.stop_move()
        path = self._global_planner.plan_path(
            pose,
            goal.label or "",
            dynamic_obstacles_xy=None,
        )
        if path is None:
            self._abort_active_navigation(
                goal,
                "Localization/route disagreement",
                f"I stopped before reaching {self._goal_display_name(goal)} because "
                "my measured position was moving away from the mapped route and I "
                "could not establish a safe replacement route.",
                state=NavState.STUCK,
            )
            return True

        goal.x, goal.y = path[-1].x, path[-1].y
        self._local_planner.reset_navigation_state()
        self._reset_progress_tracker()
        self._route_progress_waypoint_name = None
        self._route_progress_best_distance = float("inf")
        self._route_regression_since = None
        logger.warning(
            "NavCore: replaced regressing route to '%s' from pose=(%.2f, %.2f, %.0fdeg)",
            self._goal_display_name(goal),
            pose.x,
            pose.y,
            math.degrees(pose.yaw),
        )
        return True

    def _nav_cycle(self, state: NavState, goal: NavGoal):
        """Execute one navigation cycle: sense -> plan -> safety filter -> actuate."""
        if state in {NavState.IDLE, NavState.STUCK, NavState.E_STOP}:
            return

        # 1. Read sensors
        raw_grid = self._fresh_obstacle_grid(
            self._depth_processor.get_obstacle_grid()
        )
        pose = self._odometry.get_pose()
        pose = self._maybe_correct_metric_pose(raw_grid, pose)
        geometry_grid = self._obstacle_memory.update(raw_grid, pose)
        pose = self._maybe_align_heading_to_route_wall(raw_grid, pose)
        grid = self._filter_transient_path_obstacle(geometry_grid)
        self._log_obstacle_telemetry(raw_grid, geometry_grid, pose)
        self._update_progress(pose)

        parallel_wall_clear = self._parallel_wall_projection_is_clear(
            grid,
            pose,
        )
        if parallel_wall_clear:
            grid = self._without_path_obstacle(grid)

        path_dist = grid.path_obstacle_m if grid else float("inf")
        path_bearing = grid.path_obstacle_bearing if grid else 0.0
        # Forward safety uses the temporally confirmed, center-corroborated
        # path projection. Raw center depth is merged below as an independent
        # safety channel. Feeding raw path metadata here let a single stray
        # off-axis depth cluster bypass both filters and abort a clear route.
        safety_dist = grid.path_obstacle_m if grid else float("inf")
        safety_bearing = grid.path_obstacle_bearing if grid else 0.0
        if (
            raw_grid is not None
            and raw_grid.path_obstacle_m <= self.SAFETY_DISTANCE_M
            and abs(raw_grid.path_obstacle_bearing) <= self.FORWARD_HAZARD_CONE_RAD
            and self._path_obstacle_matches_center_depth(
                raw_grid.path_obstacle_m
            )
        ):
            # A center-corroborated imminent obstacle holds motion immediately
            # while the temporal tracker decides whether it is persistent.
            safety_dist = raw_grid.path_obstacle_m
            safety_bearing = raw_grid.path_obstacle_bearing

        # 2. Check if goal reached
        dist_to_goal = math.hypot(goal.x - pose.x, goal.y - pose.y)
        metric_semantic_route = (
            goal.goal_type == "semantic"
            and self._topo_map.metric_map is not None
            and self._global_planner.get_current_waypoint() is not None
        )
        if dist_to_goal < self.GOAL_TOLERANCE_M and not metric_semantic_route:
            self._complete_navigation(goal, dist_to_goal)
            return

        if raw_grid is None:
            self._pause_for_obstacle_grid_recovery(goal)
            if self._obstacle_grid_grace_expired():
                self._abort_active_navigation(
                    goal,
                    "E-STOP: obstacle grid unavailable",
                    f"I stopped before reaching {self._goal_display_name(goal)} "
                    "because my obstacle view did not recover.",
                    state=NavState.E_STOP,
                )
            return
        self._resume_after_obstacle_grid_recovery(goal)

        # 3. Compute velocity command
        if state == NavState.NAVIGATING:
            accepted_landmark = False
            slow_for_arrival = True
            metric_route = (
                goal.goal_type == "semantic"
                and self._topo_map.metric_map is not None
            )
            if metric_route:
                self._update_path_obstacle_event(path_dist, goal)
                self._poll_metric_replan(goal, pose)
                self._maybe_replan_blocked_metric_route(goal, pose, grid)
            if goal.goal_type == "semantic":
                if metric_route:
                    pose, _ = self._accept_expected_metric_corner(
                        pose,
                        geometry_grid,
                    )
                active_waypoint = self._global_planner.get_current_waypoint()
                final_sensor_confirmed = True
                if (
                    metric_route
                    and isinstance(active_waypoint, MapNode)
                    and "metric_transit" not in active_waypoint.tags
                    and math.hypot(
                        active_waypoint.x - pose.x,
                        active_waypoint.y - pose.y,
                    )
                    <= (
                        active_waypoint.arrival_tolerance_m
                        or self.SEMANTIC_ARRIVAL_TOLERANCE_M
                    )
                ):
                    final_sensor_confirmed = (
                        self._metric_arrival_sensor_is_confirmed(raw_grid, pose)
                    )
                else:
                    self._arrival_map_consistency_readings = 0
                waypoint = self._global_planner.get_next_waypoint(
                    pose,
                    self.SEMANTIC_ARRIVAL_TOLERANCE_M,
                    on_advance=self._notify_waypoint_advance,
                    final_arrival_sensor_confirmed=final_sensor_confirmed,
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
                slow_for_arrival = "metric_transit" not in waypoint.tags
            else:
                target_x, target_y = goal.x, goal.y

            if not accepted_landmark and not metric_route:
                self._update_path_obstacle_event(path_dist, goal)

            goal_dir = math.atan2(target_y - pose.y, target_x - pose.x) - pose.yaw
            # Normalize to [-pi, pi]
            goal_dir = math.atan2(math.sin(goal_dir), math.cos(goal_dir))
            goal_dist = math.hypot(target_x - pose.x, target_y - pose.y)
            if (
                metric_route
                and isinstance(waypoint, MapNode)
                and self._recover_regressing_metric_route(
                    goal,
                    pose,
                    waypoint,
                    goal_dist,
                )
            ):
                return
            pivot_heading = goal_dir
            if metric_route:
                segment_heading = self._global_planner.current_segment_heading()
                if isinstance(segment_heading, (int, float)) and math.isfinite(
                    segment_heading
                ):
                    pivot_heading = math.atan2(
                        math.sin(segment_heading - pose.yaw),
                        math.cos(segment_heading - pose.yaw),
                    )
                guidance_heading = self._global_planner.current_segment_guidance(
                    pose,
                    max_correction_rad=(
                        self.FINAL_ROUTE_MAX_CROSS_TRACK_CORRECTION_RAD
                        if slow_for_arrival
                        else math.radians(10.0)
                    ),
                )
                if isinstance(guidance_heading, (int, float)) and math.isfinite(
                    guidance_heading
                ):
                    goal_dir = math.atan2(
                        math.sin(guidance_heading - pose.yaw),
                        math.cos(guidance_heading - pose.yaw),
                    )

            cmd = self._local_planner.compute_velocity(
                grid,
                goal_dir,
                goal_dist,
                slow_for_arrival=slow_for_arrival,
                pivot_heading=pivot_heading,
            )
            stabilized = self._local_planner.stabilize_translating_steering(cmd)
            if isinstance(stabilized, VelocityCommand):
                cmd = stabilized
            if goal.goal_type == "semantic":
                # Stabilize route steering first.  Wall/corner clearance is a
                # safety correction and must be able to take effect immediately.
                corrected = self._local_planner.apply_corridor_course_correction(
                    cmd,
                    grid,
                    correction_limit=0.02 if metric_route else None,
                    align_only=metric_route and slow_for_arrival,
                )
                if isinstance(corrected, VelocityCommand):
                    cmd = corrected
                if metric_route and slow_for_arrival:
                    cmd = self._local_planner.maintain_effective_final_approach(
                        cmd,
                        grid,
                        goal_dist,
                        min(
                            self.SEMANTIC_ARRIVAL_TOLERANCE_M,
                            GlobalPlanner.FINAL_METRIC_LONGITUDINAL_TOLERANCE_M,
                        ),
                    )

            if (
                getattr(self, "_locomotion_recovery_verification_pending", None)
                and grid.path_obstacle_m >= self.AVOIDANCE_DISTANCE_M
                and raw_grid.path_obstacle_m >= self.AVOIDANCE_DISTANCE_M
                and cmd.vx >= 0.03
                and cmd.vx < self.LOCOMOTION_VERIFICATION_SPEED_MPS
            ):
                # Verification must use a command known to initiate a Go2 gait;
                # repeating a marginal low-speed command can falsely make a
                # healthy, freshly reset service look unresponsive.  Keep the
                # planner's route-aligned yaw and all ordinary safety filtering.
                cmd = replace(cmd, vx=self.LOCOMOTION_VERIFICATION_SPEED_MPS)

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
            parallel_wall_clear=parallel_wall_clear,
        )
        now = time.monotonic()
        # A commanded pivot is making useful progress when measured yaw changes;
        # a translating command must produce measured position change. Mixing the
        # two clocks made valid turns expire on the translation-only timeout.
        pivot_only = (
            abs(cmd.vx) < 1e-3
            and abs(cmd.vy) < 1e-3
            and abs(cmd.vyaw) > 1e-3
        )
        if pivot_only and raw_grid is not None:
            # A forward-path projection is insufficient for turns in place:
            # the rear legs sweep sideways around the robot.  Use a supported
            # low percentile of occupied cells rather than the minimum raw
            # depth pixel: the latter repeatedly reported a 0.19m peripheral
            # speck while every populated view sector was 0.36-0.95m away.
            nearest, nearest_bearing = self._robust_pivot_clearance(raw_grid)
            if math.isfinite(nearest) and nearest < safety_dist:
                safety_dist = nearest
                safety_bearing = nearest_bearing
        seconds_since_progress = (
            now - self._last_progress_time
            if pivot_only
            else now - self._last_translation_progress_time
        )
        locomotion_verification_requested = (
            abs(cmd.vx) >= self.LOCOMOTION_MIN_VERIFICATION_COMMAND_MPS
            or abs(cmd.vy) >= self.LOCOMOTION_MIN_VERIFICATION_COMMAND_MPS
        )
        if pivot_only:
            # Turning is not failed translation. Keep the translation watchdog
            # fresh throughout a planned pivot so the first forward command
            # afterward receives a complete acknowledgement window.
            self._last_translation_progress_time = now
            self._last_translation_progress_pose = pose
        measured_translation = self._odometry.has_confirmed_translation()
        if (
            locomotion_verification_requested
            and measured_translation
            and grid is not None
            and grid.path_obstacle_m >= self.AVOIDANCE_DISTANCE_M
            and raw_grid.path_obstacle_m >= self.AVOIDANCE_DISTANCE_M
            and seconds_since_progress >= self.CLEAR_MOTION_ACK_TIMEOUT_S
        ):
            # The SDK accepted the command, but measured odometry did not follow
            # it. Trigger the clear-path locomotion recovery promptly rather than
            # waiting for the longer spatial-obstruction timeout.
            seconds_since_progress = max(
                seconds_since_progress,
                self.STUCK_TIMEOUT_S,
            )
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
                        pivot_only=pivot_only,
                    )
                ):
                    return

                if event == "e_stop:obstacle_too_close":
                    reason = f"E-STOP: path obstacle at {safety_dist:.2f}m"
                else:
                    reason = f"E-STOP: {event.split(':', 1)[-1].replace('_', ' ')}"
                logger.warning("NavCore: safety event: %s (%s)", event, reason)
                if (
                    event == "e_stop:obstacle_too_close"
                    and goal.goal_type == "semantic"
                ):
                    self._ensure_go2()
                    if self._go2 and getattr(self._go2, "available", False):
                        self._go2.stop_move()
                    self._reset_close_obstacle_confirmation()
                    logger.warning(
                        "NavCore: confirmed close obstacle; attempting autonomous "
                        "semantic-route recovery before giving up"
                    )
                    # An obstacle-triggered safety stop is not evidence that the
                    # SportClient is unresponsive.  Preserve the measured hazard
                    # in the recovery grid even when temporal filtering removed
                    # it from route planning, so this path tries a guarded escape
                    # rather than resetting locomotion and then testing forward
                    # motion against the same obstacle.
                    recovery_grid = replace(
                        grid if grid is not None else raw_grid,
                        path_obstacle_m=safety_dist,
                        path_obstacle_bearing=safety_bearing,
                    )
                    if self._recover_from_stall(
                        goal,
                        pose,
                        recovery_grid,
                        allow_locomotion_recovery=False,
                    ):
                        return
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
                # Only diagnose an unresponsive locomotion service after a
                # meaningful translation command was sent on a genuinely clear
                # route.  Obstacle-slowed commands below gait-start speed (the
                # failing run reached vx=0.024m/s) cannot verify locomotion.
                clear_motion_candidate = (
                    locomotion_verification_requested
                    and grid is not None
                    and grid.path_obstacle_m >= self.AVOIDANCE_DISTANCE_M
                    and raw_grid is not None
                    and raw_grid.path_obstacle_m >= self.AVOIDANCE_DISTANCE_M
                )
                recovery_grid = grid
                if not clear_motion_candidate and raw_grid is not None:
                    recovery_grid = raw_grid
                if self._recover_from_stall(
                    goal,
                    pose,
                    recovery_grid,
                    allow_locomotion_recovery=clear_motion_candidate,
                ):
                    return
                clear_motion_failure = (
                    grid is not None
                    and grid.path_obstacle_m >= self.AVOIDANCE_DISTANCE_M
                    and self._clear_motion_recovery_attempts
                    >= self.MAX_CLEAR_MOTION_RECOVERY_ATTEMPTS
                )
                reason = (
                    "Locomotion recovery failed: "
                    + (
                        getattr(self, "_locomotion_recovery_error", None)
                        or "accepted commands produced no measured motion"
                    )
                    if clear_motion_failure
                    else "Stuck: no progress toward the goal"
                )
                message = (
                    f"I stopped before reaching {self._goal_display_name(goal)} "
                    "because the path was clear, but locomotion recovery could not "
                    "be verified. "
                    + (
                        getattr(self, "_locomotion_recovery_error", None)
                        or "Commands were accepted without measured movement."
                    )
                    if clear_motion_failure
                    else f"I stopped before reaching {self._goal_display_name(goal)} "
                    "because I was not making progress."
                )
                self._abort_active_navigation(
                    goal,
                    reason,
                    message,
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
        *,
        parallel_wall_clear: bool = False,
    ) -> Tuple[float, float]:
        """Merge projected path clearance with raw center depth for one safety gate."""
        if cmd.vx <= 1e-3:
            self._reset_center_only_close_confirmation()
            return path_dist, path_bearing

        if parallel_wall_clear and abs(path_bearing) > math.radians(10.0):
            # A mapped route parallel to a fitted side wall is not blocked by
            # that wall merely because the swept corridor includes its edge.
            # Raw center depth remains authoritative for anything truly ahead.
            path_dist = max(path_dist, self.AVOIDANCE_DISTANCE_M)

        reading, supported = self._read_center_depth(
            max_depth_m=self.AVOIDANCE_DISTANCE_M,
            percentile=25.0,
        )
        if not supported or reading is None:
            self._reset_center_only_close_confirmation()
            return path_dist, path_bearing

        center_dist = reading.distance_m
        if not isinstance(center_dist, (int, float)) or not math.isfinite(center_dist):
            self._reset_center_only_close_confirmation()
            return path_dist, path_bearing
        center_dist = float(center_dist)

        center_only_close = (
            center_dist <= self.SAFETY_DISTANCE_M
            and path_dist > self.SAFETY_DISTANCE_M + self.CENTER_ONLY_GRID_MARGIN_M
        )
        if center_only_close:
            confirmed, started_new_track = self._center_only_close_confirmation.update(
                center_dist,
                0.0,
            )
            self._center_only_close_pending = not confirmed
            if started_new_track:
                logger.info(
                    "NavCore: checking uncorroborated center-depth reading at %.2fm",
                    center_dist,
                )
            if not confirmed:
                # Hold for safety while deciding whether this grid-disputed reading
                # is persistent.  The hold is excluded from stall accounting below.
                return center_dist, 0.0
            logger.warning(
                "NavCore: persistent center-depth hazard at %.2fm despite clear grid",
                center_dist,
            )
        else:
            self._reset_center_only_close_confirmation()

        if center_dist < path_dist:
            return center_dist, 0.0
        return path_dist, path_bearing

    def _robust_pivot_clearance(
        self,
        grid: ObstacleGrid,
    ) -> Tuple[float, float]:
        """Return supported front-hemisphere clearance for an in-place turn."""
        points = occupied_xy_points(grid)
        if points.size == 0:
            return float("inf"), 0.0

        distances = np.hypot(points[:, 0], points[:, 1])
        bearings = np.arctan2(points[:, 1], points[:, 0])
        visible = (
            (points[:, 0] >= 0.0)
            & (np.abs(bearings) <= math.radians(60.0))
        )
        if not np.any(visible):
            return float("inf"), 0.0

        visible_distances = distances[visible]
        visible_bearings = bearings[visible]
        percentile = float(
            np.clip(self.PIVOT_CLEARANCE_PERCENTILE, 0.0, 50.0)
        )
        inflated_clearance = float(
            np.percentile(visible_distances, percentile)
        )
        # Occupied cells are already dilated by the robot radius.  Convert the
        # percentile back to an approximate sensor-to-surface range before the
        # SafetyMonitor applies its physical pivot clearance threshold.
        raw_nearest = grid.nearest_obstacle_m
        if (
            isinstance(raw_nearest, (int, float))
            and math.isfinite(raw_nearest)
            and abs(float(raw_nearest) - inflated_clearance)
            <= max(0.10, grid.resolution * 2.0)
        ):
            # Synthetic/non-inflated providers may already express the same
            # supported surface range directly.
            surface_clearance = float(raw_nearest)
        else:
            surface_clearance = inflated_clearance + max(
                0.0,
                self.PIVOT_GRID_INFLATION_M,
            )
        band = max(grid.resolution * 2.0, 0.08)
        near = np.abs(visible_distances - inflated_clearance) <= band
        bearing = (
            float(np.median(visible_bearings[near]))
            if np.any(near)
            else 0.0
        )
        return surface_clearance, bearing

    def _log_obstacle_telemetry(
        self,
        current: Optional[ObstacleGrid],
        geometry: Optional[ObstacleGrid],
        pose: RobotPose,
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
        waypoint = self._global_planner.get_current_waypoint()
        if isinstance(waypoint, MapNode):
            waypoint_text = (
                f"{waypoint.name}, "
                f"remaining={math.hypot(waypoint.x - pose.x, waypoint.y - pose.y):.2f}m"
            )
        else:
            waypoint_text = "none"
        logger.info(
            "NavCore: obstacle view pose=(%.2f, %.2f, %.0fdeg) waypoint=[%s] "
            "path=%s sectors_deg_m=[%s] wall=[%s]",
            pose.x,
            pose.y,
            math.degrees(pose.yaw),
            waypoint_text,
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
        *,
        pivot_only: bool = False,
    ) -> bool:
        """Hold briefly on borderline close obstacles to reject transient frames."""
        if self._center_only_close_pending:
            self._last_progress_time = time.monotonic()
            self._ensure_go2()
            if self._go2 and getattr(self._go2, "available", False):
                self._go2.stop_move()
            return True

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

        # A deliberate safety hold is not evidence that navigation is stuck.
        self._last_progress_time = time.monotonic()
        self._ensure_go2()
        if self._go2 and getattr(self._go2, "available", False):
            self._go2.stop_move()
        return True

    def _reset_close_obstacle_confirmation(self) -> None:
        """Clear transient close-obstacle confirmation state."""
        self._close_obstacle_confirmation.reset()

    def _reset_center_only_close_confirmation(self) -> None:
        """Clear confirmation state for a close raw reading absent from the grid."""
        self._center_only_close_confirmation.reset()
        self._center_only_close_pending = False

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

        # The raw-depth percentile is computed only from samples nearer than
        # max_depth.  Without a coverage floor, a tiny patch of invalid/edge
        # pixels can therefore report 0.19m even while the actual center view is
        # 0.5-1.0m clear.  The grid remains the primary detector; this check only
        # decides whether the independent center channel corroborates it.
        coverage = getattr(reading, "coverage", 0.0)
        if (
            not isinstance(coverage, (int, float))
            or not math.isfinite(coverage)
            or float(coverage) < self.PATH_OBSTACLE_MIN_CENTER_COVERAGE
        ):
            logger.debug(
                "NavCore: rejecting sparse center-depth corroboration at %.2fm "
                "(coverage=%.1f%%)",
                float(center_dist),
                100.0 * float(coverage or 0.0),
            )
            return False

        return float(center_dist) <= min(
            max_depth,
            distance_m + self.PATH_OBSTACLE_CENTER_DEPTH_MARGIN_M,
        )

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
        pose = self._odometry.get_pose()
        now = time.monotonic()
        self._last_progress_pose = pose
        self._last_progress_time = now
        self._last_translation_progress_pose = pose
        self._last_translation_progress_time = now
        self._route_progress_waypoint_name = None
        self._route_progress_best_distance = float("inf")
        self._route_regression_since = None
        if reset_recovery_attempts:
            self._stuck_recovery_attempts = 0
            self._clear_motion_recovery_attempts = 0
            self._last_stall_scan_direction = None
        self._reset_close_obstacle_confirmation()
        self._reset_center_only_close_confirmation()
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

        translation_moved = math.hypot(
            current_pose.x - self._last_translation_progress_pose.x,
            current_pose.y - self._last_translation_progress_pose.y,
        )
        if translation_moved > 0.1:
            self._last_translation_progress_pose = current_pose
            self._last_translation_progress_time = time.monotonic()
            self._stuck_recovery_attempts = 0
            self._clear_motion_recovery_attempts = 0
            self._last_stall_scan_direction = None
            pending_recovery = getattr(
                self,
                "_locomotion_recovery_verification_pending",
                None,
            )
            if pending_recovery:
                self._locomotion_recovery_verification_pending = None
                self._locomotion_recovery_error = None
                logger.info(
                    "NavCore: %s verified by %.2fm measured translation; route resumed",
                    pending_recovery,
                    translation_moved,
                )
                self._notify_status_change(
                    "Locomotion recovery was verified by measured movement, and I am "
                    "continuing the existing route."
                )

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
        executor = getattr(self, "_metric_replan_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._metric_replan_executor = None
        logger.info("NavCore: shutdown complete")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
