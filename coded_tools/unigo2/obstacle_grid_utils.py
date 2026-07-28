"""Shared helpers for local obstacle grids."""

import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from coded_tools.unigo2.depth_processor import ObstacleGrid


@dataclass
class ObstacleGridSpec:
    """Geometry and filtering parameters for a local obstacle grid."""

    rows: int = 160
    cols: int = 160
    resolution: float = 0.05
    origin_row: Optional[int] = None
    origin_col: Optional[int] = None
    path_corridor_half_width: float = 0.27
    path_obstacle_min_points: int = 6
    inflation_radius_m: float = 0.0

    @property
    def resolved_origin_row(self) -> int:
        return self.rows // 2 if self.origin_row is None else self.origin_row

    @property
    def resolved_origin_col(self) -> int:
        return self.cols // 2 if self.origin_col is None else self.origin_col


def build_obstacle_grid(
    xy_points_m: np.ndarray,
    spec: ObstacleGridSpec,
) -> ObstacleGrid:
    """Project robot-frame xy points into an ObstacleGrid.

    Robot frame convention matches DepthProcessor and LocalPlanner:
    x is forward, y is left, bearing 0 is forward, positive is left.
    """
    grid = np.zeros((spec.rows, spec.cols), dtype=np.float32)
    origin_row = spec.resolved_origin_row
    origin_col = spec.resolved_origin_col

    points = _as_xy_array(xy_points_m)
    if points.size:
        rows = origin_row - (points[:, 0] / spec.resolution).astype(np.int32)
        cols = origin_col - (points[:, 1] / spec.resolution).astype(np.int32)
        in_bounds = (
            (rows >= 0)
            & (rows < spec.rows)
            & (cols >= 0)
            & (cols < spec.cols)
        )
        if np.any(in_bounds):
            np.add.at(grid, (rows[in_bounds], cols[in_bounds]), 1.0)
            grid = np.clip(grid, 0.0, 1.0)

    if spec.inflation_radius_m > 0.0 and np.any(grid > 0):
        grid = _inflate_grid(grid, spec.inflation_radius_m, spec.resolution)

    return grid_with_metadata(
        grid=grid,
        resolution=spec.resolution,
        origin_row=origin_row,
        origin_col=origin_col,
        path_corridor_half_width=spec.path_corridor_half_width,
        path_obstacle_min_points=spec.path_obstacle_min_points,
        metadata_points=points,
    )


def merge_obstacle_grids(
    grids: Iterable[Optional[ObstacleGrid]],
    spec: ObstacleGridSpec,
) -> Optional[ObstacleGrid]:
    """Merge grids into one geometry, preserving occupied cells."""
    candidates = [grid for grid in grids if grid is not None]
    xy_batches = [occupied_xy_points(grid) for grid in candidates]
    xy_batches = [xy for xy in xy_batches if xy.size]
    if not xy_batches:
        if not candidates:
            return None
        return build_obstacle_grid(np.empty((0, 2), dtype=np.float32), spec)

    merged = build_obstacle_grid(
        np.vstack(xy_batches),
        ObstacleGridSpec(
            rows=spec.rows,
            cols=spec.cols,
            resolution=spec.resolution,
            origin_row=spec.origin_row,
            origin_col=spec.origin_col,
            path_corridor_half_width=spec.path_corridor_half_width,
            path_obstacle_min_points=spec.path_obstacle_min_points,
            inflation_radius_m=0.0,
        ),
    )
    _copy_nearest_source_metadata(merged, candidates)
    return merged


def occupied_xy_points(grid: ObstacleGrid) -> np.ndarray:
    """Return occupied grid cells as robot-frame xy points."""
    occupied = np.argwhere(grid.grid > 0)
    if occupied.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    x = (grid.origin_row - occupied[:, 0]) * grid.resolution
    y = (grid.origin_col - occupied[:, 1]) * grid.resolution
    return np.column_stack((x, y)).astype(np.float32)


def is_transverse_wall(
    grid: ObstacleGrid,
    distance_m: float,
    min_span_m: float,
) -> bool:
    """Return whether occupied cells form a broad wall across the path."""
    points = occupied_xy_points(grid)
    if not points.size:
        return False

    forward, lateral = points.T
    depth_band_m = max(0.10, grid.resolution * 2.0)
    across = np.abs(forward - distance_m) <= depth_band_m
    forward, lateral = forward[across], lateral[across]
    if len(lateral) < 6:
        return False

    low, high = np.percentile(lateral, (10, 90))
    if high - low < min_span_m or low >= -0.10 or high <= 0.10:
        return False

    slope, intercept = np.polyfit(lateral, forward, 1)
    residual = np.median(np.abs(forward - (slope * lateral + intercept)))
    return abs(math.atan(float(slope))) <= math.radians(20.0) and residual <= 0.10


def grid_with_metadata(
    grid: np.ndarray,
    resolution: float,
    origin_row: int,
    origin_col: int,
    path_corridor_half_width: float,
    path_obstacle_min_points: int,
    metadata_points: Optional[np.ndarray] = None,
) -> ObstacleGrid:
    """Build an ObstacleGrid and recompute nearest/path obstacle metadata."""
    if metadata_points is None:
        xy_points = occupied_xy_points(
            ObstacleGrid(
                grid=grid,
                resolution=resolution,
                origin_row=origin_row,
                origin_col=origin_col,
            )
        )
    else:
        xy_points = _as_xy_array(metadata_points)

    nearest_dist = float("inf")
    nearest_bearing = 0.0
    path_dist = float("inf")
    path_bearing = 0.0
    path_points = 0

    if xy_points.size:
        x = xy_points[:, 0]
        y = xy_points[:, 1]
        distances = np.hypot(x, y)
        nearest_idx = int(np.argmin(distances))
        nearest_dist = float(distances[nearest_idx])
        nearest_bearing = float(math.atan2(y[nearest_idx], x[nearest_idx]))

        in_path = (x > 0.0) & (np.abs(y) <= path_corridor_half_width)
        if np.any(in_path):
            path_x = x[in_path]
            path_y = y[in_path]
            path_distances = distances[in_path]
            range_bins = np.floor(path_x / resolution).astype(np.int32)
            for range_bin in np.unique(range_bins):
                bin_mask = range_bins == range_bin
                support = int(np.sum(bin_mask))
                if support < path_obstacle_min_points:
                    continue
                candidate_dist = float(np.percentile(path_distances[bin_mask], 25))
                if candidate_dist >= path_dist:
                    continue
                path_dist = candidate_dist
                path_bearing = float(
                    math.atan2(
                        float(np.median(path_y[bin_mask])),
                        float(np.median(path_x[bin_mask])),
                    )
                )
                path_points = support

    return ObstacleGrid(
        grid=grid.astype(np.float32, copy=False),
        resolution=resolution,
        origin_row=origin_row,
        origin_col=origin_col,
        timestamp=time.time(),
        nearest_obstacle_m=nearest_dist,
        nearest_obstacle_bearing=nearest_bearing,
        path_obstacle_m=path_dist,
        path_obstacle_bearing=path_bearing,
        path_obstacle_points=path_points,
    )


def summarize_obstacle_grid(grid: Optional[ObstacleGrid]) -> str:
    """Human-readable obstacle summary for agent/status tools."""
    if grid is None:
        return "No obstacle data available."

    occupied_cells = int(np.sum(grid.grid > 0))
    occupied_pct = (occupied_cells / max(grid.grid.size, 1)) * 100
    parts = [f"Obstacle grid: {occupied_pct:.0f}% occupied"]

    if grid.nearest_obstacle_m < float("inf"):
        bearing_deg = math.degrees(grid.nearest_obstacle_bearing)
        if abs(bearing_deg) < 10:
            direction = "directly ahead"
        elif bearing_deg > 0:
            direction = f"{abs(bearing_deg):.0f} degrees to the left"
        else:
            direction = f"{abs(bearing_deg):.0f} degrees to the right"
        parts.append(
            f"Nearest obstacle anywhere: {grid.nearest_obstacle_m:.2f}m {direction}"
        )
    else:
        parts.append("No obstacles within range")

    if grid.path_obstacle_m < float("inf"):
        parts.append(
            f"Path obstacle: {grid.path_obstacle_m:.2f}m "
            f"({grid.path_obstacle_points} points)"
        )
    else:
        parts.append("Path corridor clear")

    return ". ".join(parts) + "."


def _as_xy_array(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return points.reshape((-1, 2))[:, :2]


def _copy_nearest_source_metadata(
    merged: ObstacleGrid,
    sources: Iterable[ObstacleGrid],
) -> None:
    nearest_source = min(
        sources,
        key=lambda grid: grid.nearest_obstacle_m,
        default=None,
    )
    path_source = min(
        sources,
        key=lambda grid: grid.path_obstacle_m,
        default=None,
    )
    if nearest_source is not None:
        merged.nearest_obstacle_m = nearest_source.nearest_obstacle_m
        merged.nearest_obstacle_bearing = nearest_source.nearest_obstacle_bearing
    if path_source is not None:
        merged.path_obstacle_m = path_source.path_obstacle_m
        merged.path_obstacle_bearing = path_source.path_obstacle_bearing
        merged.path_obstacle_points = path_source.path_obstacle_points


def _inflate_grid(
    grid: np.ndarray,
    radius_m: float,
    resolution: float,
) -> np.ndarray:
    radius_cells = max(1, int(math.ceil(radius_m / resolution)))
    binary = grid > 0
    padded = np.pad(
        binary,
        ((radius_cells, radius_cells), (radius_cells, radius_cells)),
        mode="constant",
    )
    dilated = np.zeros_like(binary, dtype=bool)

    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue
            y0 = radius_cells + dy
            x0 = radius_cells + dx
            dilated |= padded[y0 : y0 + binary.shape[0], x0 : x0 + binary.shape[1]]

    return dilated.astype(np.float32)
