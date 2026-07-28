"""Obstacle-grid providers and sensor fusion for navigation."""

import os
from typing import Iterable, Optional

from coded_tools.unigo2.depth_processor import (
    CenterDepthReading,
    DepthProcessor,
    ObstacleGrid,
    _env_flag,
)
from coded_tools.unigo2.lidar_processor import LidarPerimeterService
from coded_tools.unigo2.obstacle_grid_utils import (
    ObstacleGridSpec,
    merge_obstacle_grids,
    summarize_obstacle_grid,
)


_DEFAULT_DEPTH = object()


class FusedObstacleProvider:
    """Expose depth and LiDAR as one ObstacleGrid source for NavCore."""

    def __init__(
        self,
        depth=_DEFAULT_DEPTH,
        perimeter_sources: Optional[Iterable[LidarPerimeterService]] = None,
    ):
        self._depth = DepthProcessor() if depth is _DEFAULT_DEPTH else depth
        self._perimeter_sources = list(perimeter_sources or [])
        depth_config = getattr(self._depth, "_config", None)
        self._fusion_spec = ObstacleGridSpec(
            rows=160,
            cols=160,
            resolution=getattr(depth_config, "grid_resolution", 0.05),
            path_corridor_half_width=getattr(depth_config, "path_corridor_half_width", 0.27),
            path_obstacle_min_points=getattr(depth_config, "path_obstacle_min_points", 6),
        )

    def start(self) -> None:
        if self._depth is not None:
            start = getattr(self._depth, "start", None)
            if callable(start):
                start()
        for source in self._perimeter_sources:
            source.start()

    def stop(self) -> None:
        for source in self._perimeter_sources:
            source.stop()
        if self._depth is not None:
            stop = getattr(self._depth, "stop", None)
            if callable(stop):
                stop()

    @property
    def backend(self) -> str:
        backends = []
        if self._depth is not None:
            backends.append(getattr(self._depth, "backend", "depth:none"))
        backends.extend(source.backend for source in self._perimeter_sources)
        return "+".join(backends) or "none"

    @property
    def is_available(self) -> bool:
        if self._depth is not None and getattr(self._depth, "is_available", False):
            return True
        return any(source.is_available for source in self._perimeter_sources)

    @property
    def is_running(self) -> bool:
        if self._depth is not None and getattr(self._depth, "is_running", False):
            return True
        return any(source.get_obstacle_grid() is not None for source in self._perimeter_sources)

    @property
    def supports_center_depth(self) -> bool:
        return self._depth is not None

    def get_obstacle_grid(self) -> Optional[ObstacleGrid]:
        grids = []
        if self._depth is not None:
            grids.append(self._depth.get_obstacle_grid())
        grids.extend(source.get_obstacle_grid() for source in self._perimeter_sources)
        present = [grid for grid in grids if grid is not None]
        if not present:
            return None
        if len(present) == 1:
            return present[0]
        return merge_obstacle_grids(present, self._fusion_spec)

    def get_single_frame_grid(self) -> Optional[ObstacleGrid]:
        depth_grid = (
            self._depth.get_single_frame_grid()
            if self._depth is not None and hasattr(self._depth, "get_single_frame_grid")
            else None
        )
        lidar_grids = [source.get_obstacle_grid() for source in self._perimeter_sources]
        present = [grid for grid in [depth_grid, *lidar_grids] if grid is not None]
        if not present:
            return None
        if len(present) == 1:
            return present[0]
        return merge_obstacle_grids(present, self._fusion_spec)

    def get_center_depth_reading(self, *args, **kwargs) -> Optional[CenterDepthReading]:
        if self._depth is None:
            return None
        return self._depth.get_center_depth_reading(*args, **kwargs)

    def get_obstacle_summary(self) -> str:
        return summarize_obstacle_grid(self.get_obstacle_grid())


def create_default_obstacle_provider() -> FusedObstacleProvider:
    """Create the production obstacle provider used by NavCore."""
    default_source = "depth"
    source = os.environ.get("NAV_OBSTACLE_SOURCE", default_source).strip().lower()
    use_depth = source in {"depth", "camera", "realsense", "fused", "all"}
    use_lidar = source in {"lidar", "fused", "all"}

    return FusedObstacleProvider(
        depth=DepthProcessor() if use_depth else None,
        perimeter_sources=[LidarPerimeterService()] if use_lidar else [],
    )
