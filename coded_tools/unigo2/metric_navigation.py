"""Metric occupancy planning and conservative depth-to-map localization.

This module deliberately has no robot, agent-network, or image-processing
dependencies.  Production loads a compact occupancy grid generated from the
floor plan; tests and simulations can construct the same map from an array.
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoseCorrection:
    """A depth-to-map pose match returned in world coordinates."""

    x: float
    y: float
    yaw: float
    score_m: float
    improvement_m: float
    matched_fraction: float


class MetricOccupancyMap:
    """A 2-D occupancy map with clearance-aware A* path planning."""

    CARDINAL_COST = 1.0
    DIAGONAL_COST = math.sqrt(2.0)
    _NEIGHBORS = (
        (-1, 0, CARDINAL_COST),
        (1, 0, CARDINAL_COST),
        (0, -1, CARDINAL_COST),
        (0, 1, CARDINAL_COST),
        (-1, -1, DIAGONAL_COST),
        (-1, 1, DIAGONAL_COST),
        (1, -1, DIAGONAL_COST),
        (1, 1, DIAGONAL_COST),
    )

    def __init__(
        self,
        occupied: np.ndarray,
        *,
        resolution_m: float,
        origin_x_m: float = 0.0,
        origin_y_m: float = 0.0,
        robot_clearance_m: float = 0.34,
        preferred_clearance_m: float = 0.75,
    ):
        grid = np.asarray(occupied, dtype=bool)
        if grid.ndim != 2 or min(grid.shape) < 2:
            raise ValueError("occupied must be a two-dimensional grid")
        if resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive")

        self.occupied = grid
        self.resolution_m = float(resolution_m)
        self.origin_x_m = float(origin_x_m)
        self.origin_y_m = float(origin_y_m)
        self.robot_clearance_m = max(0.0, float(robot_clearance_m))
        self.preferred_clearance_m = max(
            self.robot_clearance_m,
            float(preferred_clearance_m),
        )
        self.distance_cells = self._chamfer_distance(self.occupied)
        self.distance_m = self.distance_cells * self.resolution_m
        self.inflated = self.distance_m <= self.robot_clearance_m

    @property
    def width_m(self) -> float:
        return self.occupied.shape[1] * self.resolution_m

    @property
    def height_m(self) -> float:
        return self.occupied.shape[0] * self.resolution_m

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        robot_clearance_m: Optional[float] = None,
        preferred_clearance_m: Optional[float] = None,
    ) -> "MetricOccupancyMap":
        """Load a compressed map produced by ``build_occupancy_map.py``."""
        with np.load(path, allow_pickle=False) as data:
            occupied = data["occupied"].astype(bool)
            resolution = float(data["resolution_m"])
            origin_x = float(data.get("origin_x_m", 0.0))
            origin_y = float(data.get("origin_y_m", 0.0))
            stored_clearance = float(data.get("robot_clearance_m", 0.34))
            stored_preferred = float(data.get("preferred_clearance_m", 0.75))
        return cls(
            occupied,
            resolution_m=resolution,
            origin_x_m=origin_x,
            origin_y_m=origin_y,
            robot_clearance_m=(
                stored_clearance
                if robot_clearance_m is None
                else robot_clearance_m
            ),
            preferred_clearance_m=(
                stored_preferred
                if preferred_clearance_m is None
                else preferred_clearance_m
            ),
        )

    def save(self, path: str | Path) -> None:
        """Save the raw occupancy data and planner defaults."""
        np.savez_compressed(
            path,
            occupied=self.occupied.astype(np.uint8),
            resolution_m=np.asarray(self.resolution_m),
            origin_x_m=np.asarray(self.origin_x_m),
            origin_y_m=np.asarray(self.origin_y_m),
            robot_clearance_m=np.asarray(self.robot_clearance_m),
            preferred_clearance_m=np.asarray(self.preferred_clearance_m),
        )

    def world_to_cell(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert a world coordinate to ``(row, column)``."""
        col = int(math.floor((x_m - self.origin_x_m) / self.resolution_m))
        row = int(math.floor((y_m - self.origin_y_m) / self.resolution_m))
        return row, col

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Return the world coordinate at a cell center."""
        return (
            self.origin_x_m + (col + 0.5) * self.resolution_m,
            self.origin_y_m + (row + 0.5) * self.resolution_m,
        )

    def plan_path(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        *,
        dynamic_obstacles_xy: Optional[np.ndarray] = None,
        dynamic_clearance_m: float = 0.45,
        waypoint_spacing_m: float = 0.80,
    ) -> Optional[List[Tuple[float, float]]]:
        """Plan and smooth a collision-free path between two world positions.

        Static clearance is enforced as a hard constraint.  A softer clearance
        cost keeps the route near the middle of open space.  RealSense points
        supplied during a replan are overlaid as temporary obstacles.
        """
        blocked = self.inflated.copy()
        if dynamic_obstacles_xy is not None:
            self._overlay_dynamic_obstacles(
                blocked,
                dynamic_obstacles_xy,
                dynamic_clearance_m,
            )

        start = self._nearest_free_cell(self.world_to_cell(*start_xy), blocked)
        goal = self._nearest_free_cell(self.world_to_cell(*goal_xy), blocked)
        if start is None or goal is None:
            logger.warning(
                "MetricOccupancyMap: no free cell near start=%s or goal=%s",
                start_xy,
                goal_xy,
            )
            return None

        cells = self._astar(start, goal, blocked)
        if cells is None:
            logger.warning(
                "MetricOccupancyMap: no metric path from %s to %s",
                start_xy,
                goal_xy,
            )
            return None

        cells = self._smooth_cells(cells, blocked)
        points = [self.cell_to_world(row, col) for row, col in cells]
        # Preserve the measured start exactly. The goal is represented by its
        # nearest safe cell so an annotated marker inside a wall cannot be used.
        points[0] = (float(start_xy[0]), float(start_xy[1]))
        return self._densify(points, max(self.resolution_m, waypoint_spacing_m))

    def robot_points_to_world(
        self,
        points_robot_xy: np.ndarray,
        pose_x: float,
        pose_y: float,
        pose_yaw: float,
    ) -> np.ndarray:
        """Transform ``(forward, left)`` depth points into map coordinates."""
        points = np.asarray(points_robot_xy, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            return np.empty((0, 2), dtype=np.float32)
        cos_yaw = math.cos(pose_yaw)
        sin_yaw = math.sin(pose_yaw)
        world_x = pose_x + cos_yaw * points[:, 0] - sin_yaw * points[:, 1]
        world_y = pose_y + sin_yaw * points[:, 0] + cos_yaw * points[:, 1]
        return np.column_stack((world_x, world_y)).astype(np.float32)

    def match_pose(
        self,
        points_robot_xy: np.ndarray,
        pose_x: float,
        pose_y: float,
        pose_yaw: float,
        *,
        translation_window_m: float = 0.24,
        translation_step_m: float = 0.08,
        yaw_window_rad: float = math.radians(8.0),
        yaw_step_rad: float = math.radians(2.0),
        minimum_improvement_m: float = 0.04,
        maximum_score_m: float = 0.20,
        minimum_matched_fraction: float = 0.22,
    ) -> Optional[PoseCorrection]:
        """Conservatively align a depth scan to mapped structural obstacles.

        Only a strong, bounded improvement is returned.  The robust score uses
        the closest 60 percent of points so chairs and people absent from the
        static floor plan do not dominate localization.
        """
        points = np.asarray(points_robot_xy, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            return None
        useful = (
            (points[:, 0] >= 0.20)
            & (points[:, 0] <= 4.0)
            & (np.abs(points[:, 1]) <= 2.5)
        )
        points = points[useful]
        if len(points) < 30:
            return None
        if len(points) > 400:
            indexes = np.linspace(0, len(points) - 1, 400, dtype=np.int32)
            points = points[indexes]

        baseline_score, baseline_fraction = self._pose_score(
            points,
            pose_x,
            pose_y,
            pose_yaw,
        )
        best = (
            baseline_score,
            baseline_fraction,
            pose_x,
            pose_y,
            pose_yaw,
            baseline_score,
        )
        translations = self._centered_samples(
            translation_window_m,
            translation_step_m,
        )
        yaw_offsets = self._centered_samples(yaw_window_rad, yaw_step_rad)

        for yaw_offset in yaw_offsets:
            candidate_yaw = self._normalize_angle(pose_yaw + yaw_offset)
            for dx in translations:
                for dy in translations:
                    score, matched_fraction = self._pose_score(
                        points,
                        pose_x + dx,
                        pose_y + dy,
                        candidate_yaw,
                    )
                    # Prefer the smallest correction when a long straight wall
                    # makes motion along that wall geometrically ambiguous.
                    objective = (
                        score
                        + 0.05 * math.hypot(float(dx), float(dy))
                        + 0.02 * abs(float(yaw_offset))
                    )
                    if objective < best[5]:
                        best = (
                            score,
                            matched_fraction,
                            pose_x + dx,
                            pose_y + dy,
                            candidate_yaw,
                            objective,
                        )

        improvement = baseline_score - best[0]
        if (
            improvement < minimum_improvement_m
            or best[0] > maximum_score_m
            or best[1] < minimum_matched_fraction
        ):
            return None
        return PoseCorrection(
            x=best[2],
            y=best[3],
            yaw=best[4],
            score_m=best[0],
            improvement_m=improvement,
            matched_fraction=best[1],
        )

    def pose_consistency(
        self,
        points_robot_xy: np.ndarray,
        pose_x: float,
        pose_y: float,
        pose_yaw: float,
        *,
        minimum_points: int = 30,
    ) -> Optional[Tuple[float, float]]:
        """Score whether a live structural scan agrees with the claimed map pose.

        Unlike ``match_pose``, this does not search for or apply a correction. It
        provides an independent arrival check so odometry proximity alone cannot
        certify a destination from a physically different corridor.
        """
        points = np.asarray(points_robot_xy, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            return None
        useful = (
            (points[:, 0] >= 0.20)
            & (points[:, 0] <= 4.0)
            & (np.abs(points[:, 1]) <= 2.5)
        )
        points = points[useful]
        if len(points) < max(1, int(minimum_points)):
            return None
        if len(points) > 400:
            indexes = np.linspace(0, len(points) - 1, 400, dtype=np.int32)
            points = points[indexes]
        return self._pose_score(points, pose_x, pose_y, pose_yaw)

    def _pose_score(
        self,
        points: np.ndarray,
        x_m: float,
        y_m: float,
        yaw: float,
    ) -> Tuple[float, float]:
        world = self.robot_points_to_world(points, x_m, y_m, yaw)
        cols = np.floor(
            (world[:, 0] - self.origin_x_m) / self.resolution_m
        ).astype(np.int32)
        rows = np.floor(
            (world[:, 1] - self.origin_y_m) / self.resolution_m
        ).astype(np.int32)
        inside = (
            (rows >= 0)
            & (rows < self.distance_m.shape[0])
            & (cols >= 0)
            & (cols < self.distance_m.shape[1])
        )
        distances = np.full(len(points), 1.0, dtype=np.float32)
        distances[inside] = self.distance_m[rows[inside], cols[inside]]
        keep_count = max(1, int(0.60 * len(distances)))
        robust = np.partition(distances, keep_count - 1)[:keep_count]
        score = float(np.mean(np.minimum(robust, 1.0)))
        matched_fraction = float(np.mean(distances <= 0.18))
        return score, matched_fraction

    @staticmethod
    def _centered_samples(window: float, step: float) -> np.ndarray:
        if window <= 0.0 or step <= 0.0:
            return np.asarray([0.0])
        count = max(1, int(math.floor(window / step)))
        return np.linspace(-count * step, count * step, 2 * count + 1)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _overlay_dynamic_obstacles(
        self,
        blocked: np.ndarray,
        points_xy: np.ndarray,
        clearance_m: float,
    ) -> None:
        points = np.asarray(points_xy, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
            return
        radius = max(1, int(math.ceil(clearance_m / self.resolution_m)))
        offsets = [
            (dr, dc)
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if dr * dr + dc * dc <= radius * radius
        ]
        for x_m, y_m in points:
            row, col = self.world_to_cell(float(x_m), float(y_m))
            for dr, dc in offsets:
                rr, cc = row + dr, col + dc
                if 0 <= rr < blocked.shape[0] and 0 <= cc < blocked.shape[1]:
                    blocked[rr, cc] = True

    def _nearest_free_cell(
        self,
        cell: Tuple[int, int],
        blocked: np.ndarray,
        search_radius_m: float = 1.50,
    ) -> Optional[Tuple[int, int]]:
        row = min(max(cell[0], 0), blocked.shape[0] - 1)
        col = min(max(cell[1], 0), blocked.shape[1] - 1)
        if not blocked[row, col]:
            return row, col
        max_radius = max(1, int(math.ceil(search_radius_m / self.resolution_m)))
        best = None
        best_distance = float("inf")
        for radius in range(1, max_radius + 1):
            for rr in range(max(0, row - radius), min(blocked.shape[0], row + radius + 1)):
                for cc in (col - radius, col + radius):
                    if 0 <= cc < blocked.shape[1] and not blocked[rr, cc]:
                        distance = (rr - row) ** 2 + (cc - col) ** 2
                        if distance < best_distance:
                            best, best_distance = (rr, cc), distance
            for cc in range(max(0, col - radius), min(blocked.shape[1], col + radius + 1)):
                for rr in (row - radius, row + radius):
                    if 0 <= rr < blocked.shape[0] and not blocked[rr, cc]:
                        distance = (rr - row) ** 2 + (cc - col) ** 2
                        if distance < best_distance:
                            best, best_distance = (rr, cc), distance
            if best is not None:
                return best
        return None

    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        blocked: np.ndarray,
    ) -> Optional[List[Tuple[int, int]]]:
        if start == goal:
            return [start]
        rows, cols = blocked.shape
        g_score = np.full((rows, cols), np.inf, dtype=np.float32)
        closed = np.zeros((rows, cols), dtype=bool)
        parents: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score[start] = 0.0
        open_heap = [(self._heuristic(start, goal), 0.0, start)]

        while open_heap:
            _f_score, current_g, current = heapq.heappop(open_heap)
            row, col = current
            if closed[row, col]:
                continue
            if current == goal:
                path = [current]
                while current in parents:
                    current = parents[current]
                    path.append(current)
                path.reverse()
                return path
            closed[row, col] = True

            for dr, dc, step_cost in self._NEIGHBORS:
                rr, cc = row + dr, col + dc
                if not (0 <= rr < rows and 0 <= cc < cols):
                    continue
                if blocked[rr, cc] or closed[rr, cc]:
                    continue
                if dr and dc and (blocked[row, cc] or blocked[rr, col]):
                    continue

                clearance = float(self.distance_m[rr, cc])
                clearance_deficit = max(0.0, self.preferred_clearance_m - clearance)
                clearance_penalty = 2.0 * clearance_deficit / max(
                    self.preferred_clearance_m,
                    self.resolution_m,
                )
                tentative = current_g + step_cost * (1.0 + clearance_penalty)
                if tentative >= float(g_score[rr, cc]):
                    continue
                g_score[rr, cc] = tentative
                parents[(rr, cc)] = current
                priority = tentative + self._heuristic((rr, cc), goal)
                heapq.heappush(open_heap, (priority, tentative, (rr, cc)))
        return None

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        return max(dr, dc) + (math.sqrt(2.0) - 1.0) * min(dr, dc)

    def _smooth_cells(
        self,
        path: Sequence[Tuple[int, int]],
        blocked: np.ndarray,
    ) -> List[Tuple[int, int]]:
        if len(path) <= 2:
            return list(path)
        smoothed = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            candidate = len(path) - 1
            while candidate > anchor + 1:
                if self._line_is_free(path[anchor], path[candidate], blocked):
                    break
                candidate -= 1
            smoothed.append(path[candidate])
            anchor = candidate
        return smoothed

    @staticmethod
    def _line_is_free(
        start: Tuple[int, int],
        end: Tuple[int, int],
        blocked: np.ndarray,
    ) -> bool:
        dr = end[0] - start[0]
        dc = end[1] - start[1]
        samples = max(abs(dr), abs(dc)) * 2 + 1
        rows = np.rint(np.linspace(start[0], end[0], samples)).astype(np.int32)
        cols = np.rint(np.linspace(start[1], end[1], samples)).astype(np.int32)
        return not bool(np.any(blocked[rows, cols]))

    @staticmethod
    def _densify(
        points: Sequence[Tuple[float, float]],
        spacing_m: float,
    ) -> List[Tuple[float, float]]:
        if len(points) <= 1:
            return list(points)
        dense = [points[0]]
        for start, end in zip(points, points[1:]):
            distance = math.hypot(end[0] - start[0], end[1] - start[1])
            segments = max(1, int(math.ceil(distance / spacing_m)))
            for index in range(1, segments + 1):
                fraction = index / segments
                dense.append(
                    (
                        start[0] + fraction * (end[0] - start[0]),
                        start[1] + fraction * (end[1] - start[1]),
                    )
                )
        return dense

    @staticmethod
    def _chamfer_distance(occupied: np.ndarray) -> np.ndarray:
        """Approximate Euclidean distance to occupancy without SciPy."""
        rows, cols = occupied.shape
        distance = np.full((rows, cols), np.inf, dtype=np.float32)
        distance[occupied] = 0.0
        diagonal = np.float32(math.sqrt(2.0))

        for row in range(rows):
            for col in range(cols):
                value = distance[row, col]
                if row > 0:
                    value = min(value, distance[row - 1, col] + 1.0)
                    if col > 0:
                        value = min(value, distance[row - 1, col - 1] + diagonal)
                    if col + 1 < cols:
                        value = min(value, distance[row - 1, col + 1] + diagonal)
                if col > 0:
                    value = min(value, distance[row, col - 1] + 1.0)
                distance[row, col] = value

        for row in range(rows - 1, -1, -1):
            for col in range(cols - 1, -1, -1):
                value = distance[row, col]
                if row + 1 < rows:
                    value = min(value, distance[row + 1, col] + 1.0)
                    if col > 0:
                        value = min(value, distance[row + 1, col - 1] + diagonal)
                    if col + 1 < cols:
                        value = min(value, distance[row + 1, col + 1] + diagonal)
                if col + 1 < cols:
                    value = min(value, distance[row, col + 1] + 1.0)
                distance[row, col] = value
        return distance
