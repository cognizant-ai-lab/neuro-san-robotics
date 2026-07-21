"""Confidence-gated depth alignment against fixed map structure."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from coded_tools.unigo2.depth_processor import ObstacleGrid


@dataclass(frozen=True)
class WallSegment:
    """A fixed wall segment in the navigation-map frame."""

    name: str
    start: Tuple[float, float]
    end: Tuple[float, float]


@dataclass(frozen=True)
class PoseCorrection:
    """A depth-supported map-pose correction."""

    pose_x: float
    pose_y: float
    pose_yaw: float
    score: float
    support_points: int
    support_segments: int


class StructuralMap:
    """Fixed wall geometry loaded from a floor-plan sidecar file."""

    def __init__(self, segments: Sequence[WallSegment] = ()):
        self.segments = list(segments)

    @property
    def is_loaded(self) -> bool:
        return bool(self.segments)

    @classmethod
    def load_from_file(
        cls,
        path: Path,
        coordinate_system: Dict[str, Any],
    ) -> "StructuralMap":
        """Load source-image wall segments and transform them into map meters."""
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        bbox = coordinate_system.get("source_floor_bbox_px", {})
        scale = float(coordinate_system.get("scale_m_per_px", 0.0))
        right = float(bbox.get("right", 0.0))
        bottom = float(bbox.get("bottom", 0.0))
        if scale <= 0.0 or right <= 0.0 or bottom <= 0.0:
            raise ValueError("map coordinate system has no usable source-pixel transform")

        def to_world(point: Sequence[float]) -> Tuple[float, float]:
            pixel_x, pixel_y = float(point[0]), float(point[1])
            return ((bottom - pixel_y) * scale, (right - pixel_x) * scale)

        segments = []
        for item in data.get("source_pixel_segments", []):
            start, end = item.get("start"), item.get("end")
            if not _valid_point(start) or not _valid_point(end):
                continue
            segments.append(
                WallSegment(
                    name=str(item.get("name", f"wall_{len(segments)}")),
                    start=to_world(start),
                    end=to_world(end),
                )
            )
        return cls(segments)


class DepthMapLocalizer:
    """Search a local pose neighborhood for depth support from mapped walls."""

    MIN_RANGE_M = 0.35
    MAX_RANGE_M = 3.5
    MAX_POINTS = 180
    MATCH_DISTANCE_M = 0.22
    MIN_SUPPORT_POINTS = 18
    MIN_SUPPORT_SEGMENTS = 2
    MIN_SCORE = 0.12
    MIN_SCORE_IMPROVEMENT = 0.03
    MIN_CORRECTION_M = 0.08
    MIN_CORRECTION_YAW_RAD = math.radians(3.0)

    def __init__(self, structural_map: StructuralMap):
        self._map = structural_map

    def correct_pose(
        self,
        grid: ObstacleGrid,
        pose_x: float,
        pose_y: float,
        pose_yaw: float,
    ) -> Optional[PoseCorrection]:
        """Return a correction only when nearby mapped structure clearly supports it."""
        points = self._occupied_points(grid)
        if len(points) < self.MIN_SUPPORT_POINTS or not self._map.is_loaded:
            return None

        baseline = self._evaluate(points, pose_x, pose_y, pose_yaw)
        candidates = []
        for dx in np.linspace(-0.75, 0.75, 11):
            for dy in np.linspace(-0.75, 0.75, 11):
                for dyaw in np.radians(np.arange(-20, 21, 5)):
                    candidates.append(
                        self._evaluate(points, pose_x + dx, pose_y + dy, pose_yaw + dyaw)
                    )

        best = max(candidates, key=lambda candidate: candidate[0])
        best_score, best_support, best_segments, best_pose = best
        if (
            best_score < self.MIN_SCORE
            or best_support < self.MIN_SUPPORT_POINTS
            or best_segments < self.MIN_SUPPORT_SEGMENTS
            or best_score < baseline[0] + self.MIN_SCORE_IMPROVEMENT
        ):
            return None

        correction_m = math.hypot(best_pose[0] - pose_x, best_pose[1] - pose_y)
        correction_yaw = _angle_delta(best_pose[2], pose_yaw)
        if (
            correction_m < self.MIN_CORRECTION_M
            and abs(correction_yaw) < self.MIN_CORRECTION_YAW_RAD
        ):
            return None

        return PoseCorrection(
            pose_x=best_pose[0],
            pose_y=best_pose[1],
            pose_yaw=_normalize_angle(best_pose[2]),
            score=best_score,
            support_points=best_support,
            support_segments=best_segments,
        )

    def _occupied_points(self, grid: ObstacleGrid) -> np.ndarray:
        occupied = np.argwhere(grid.grid > 0.0)
        if not len(occupied):
            return np.empty((0, 2), dtype=np.float32)

        forward = (grid.origin_row - occupied[:, 0]) * grid.resolution
        left = (grid.origin_col - occupied[:, 1]) * grid.resolution
        distance = np.hypot(forward, left)
        valid = (forward > 0.0) & (distance >= self.MIN_RANGE_M) & (distance <= self.MAX_RANGE_M)
        points = np.column_stack((forward[valid], left[valid])).astype(np.float32)
        if len(points) > self.MAX_POINTS:
            points = points[np.linspace(0, len(points) - 1, self.MAX_POINTS, dtype=int)]
        return points

    def _evaluate(
        self,
        local_points: np.ndarray,
        pose_x: float,
        pose_y: float,
        pose_yaw: float,
    ) -> Tuple[float, int, int, Tuple[float, float, float]]:
        cos_yaw, sin_yaw = math.cos(pose_yaw), math.sin(pose_yaw)
        world_points = np.empty_like(local_points)
        world_points[:, 0] = pose_x + cos_yaw * local_points[:, 0] - sin_yaw * local_points[:, 1]
        world_points[:, 1] = pose_y + sin_yaw * local_points[:, 0] + cos_yaw * local_points[:, 1]

        distances = np.full(len(world_points), np.inf, dtype=np.float32)
        segment_ids = np.full(len(world_points), -1, dtype=np.int32)
        for index, segment in enumerate(self._map.segments):
            candidate = _distance_to_segment(world_points, segment)
            replace = candidate < distances
            distances[replace] = candidate[replace]
            segment_ids[replace] = index

        matched = distances <= self.MATCH_DISTANCE_M
        support = int(np.count_nonzero(matched))
        segment_count = len(set(segment_ids[matched])) if support else 0
        score = support / max(len(world_points), 1)
        return score, support, segment_count, (pose_x, pose_y, _normalize_angle(pose_yaw))


def _distance_to_segment(points: np.ndarray, segment: WallSegment) -> np.ndarray:
    """Return the Euclidean distance from every point to one finite segment."""
    start = np.asarray(segment.start, dtype=np.float32)
    end = np.asarray(segment.end, dtype=np.float32)
    vector = end - start
    length_squared = float(np.dot(vector, vector))
    if length_squared <= 1e-9:
        return np.linalg.norm(points - start, axis=1)
    projection = np.clip(((points - start) @ vector) / length_squared, 0.0, 1.0)
    closest = start + projection[:, None] * vector
    return np.linalg.norm(points - closest, axis=1)


def _valid_point(point: Any) -> bool:
    return isinstance(point, (list, tuple)) and len(point) == 2


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _angle_delta(first: float, second: float) -> float:
    return _normalize_angle(first - second)
