"""LiDAR perimeter service for navigation obstacle grids."""

import importlib
import logging
import math
import os
import struct
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import numpy as np

from coded_tools.unigo2.depth_processor import ObstacleGrid, _env_flag, _env_float, _env_int
from coded_tools.unigo2.obstacle_grid_utils import ObstacleGridSpec, build_obstacle_grid

logger = logging.getLogger(__name__)


@dataclass
class LidarPerimeterConfig:
    """Configuration for converting Unitree LiDAR samples into ObstacleGrid."""

    enabled: bool = False
    topic: str = "rt/utlidar/cloud"
    grid_rows: int = 160
    grid_cols: int = 160
    grid_resolution: float = 0.05
    min_range_m: float = 0.05
    max_range_m: float = 4.0
    min_height_m: float = -0.25
    max_height_m: float = 0.80
    robot_half_width: float = 0.15
    path_corridor_half_width: float = 0.12
    path_obstacle_min_points: int = 6
    angle_offset_rad: float = 0.0
    range_scale: float = 1.0
    max_sample_age_s: float = 0.75


class LidarPerimeterService:
    """Subscribe to LiDAR samples and expose a navigation ObstacleGrid.

    The hardware-specific DDS subscriber is isolated here. NavCore only sees
    the same small sensor interface it already uses: start/stop/is_available
    and get_obstacle_grid().
    """

    _MESSAGE_CANDIDATES = (
        ("idl.sensor_msgs.msg.dds_._PointCloud2_", "PointCloud2_"),
        ("idl.unitree_go.msg.dds_", "RangeData_"),
        ("idl.unitree_go.msg.dds_", "PointCloud2_"),
        ("idl.unitree_go.msg.dds_._LidarState_", "LidarState_"),
        ("idl.unitree_go.msg.dds_", "PointCloud_"),
        ("idl.unitree_go.msg.dds_", "LaserScan_"),
        ("idl.unitree_lidar.msg.dds_", "RangeData_"),
        ("idl.unitree_lidar.msg.dds_", "PointCloud2_"),
        ("idl.unitree_lidar.msg.dds_", "PointCloud_"),
        ("idl.unitree_lidar.msg.dds_", "LaserScan_"),
    )

    def __init__(self, config: Optional[LidarPerimeterConfig] = None):
        self._config = config or self._config_from_env()
        self._lock = threading.Lock()
        self._latest_grid: Optional[ObstacleGrid] = None
        self._latest_sample_at = 0.0
        self._subscriber = None
        self._backend = "disabled" if not self._config.enabled else "none"
        self._subscriber_error: Optional[str] = None

    @staticmethod
    def _config_from_env() -> LidarPerimeterConfig:
        enabled_by_default = not _env_flag("NAV_SIMULATION_MODE", False)
        return LidarPerimeterConfig(
            enabled=_env_flag("NAV_USE_LIDAR", enabled_by_default),
            topic=os.environ.get("NAV_LIDAR_TOPIC", "rt/utlidar/cloud"),
            grid_rows=_env_int("NAV_LIDAR_GRID_ROWS", 160),
            grid_cols=_env_int("NAV_LIDAR_GRID_COLS", 160),
            grid_resolution=_env_float("NAV_GRID_RESOLUTION", 0.05),
            min_range_m=_env_float("NAV_LIDAR_MIN_RANGE", 0.05),
            max_range_m=_env_float("NAV_LIDAR_MAX_RANGE", 4.0),
            min_height_m=_env_float("NAV_LIDAR_MIN_HEIGHT", -0.25),
            max_height_m=_env_float("NAV_LIDAR_MAX_HEIGHT", 0.80),
            robot_half_width=_env_float("NAV_ROBOT_HALF_WIDTH", 0.15),
            path_corridor_half_width=_env_float("NAV_PATH_CORRIDOR_HALF_WIDTH", 0.12),
            path_obstacle_min_points=_env_int("NAV_PATH_OBSTACLE_MIN_POINTS", 6),
            angle_offset_rad=_env_float("NAV_LIDAR_ANGLE_OFFSET_RAD", 0.0),
            range_scale=_env_float("NAV_LIDAR_RANGE_SCALE", 1.0),
            max_sample_age_s=_env_float("NAV_LIDAR_MAX_SAMPLE_AGE", 0.75),
        )

    def start(self) -> None:
        """Start the LiDAR subscriber if LiDAR is enabled."""
        if not self._config.enabled or self._subscriber is not None:
            return

        try:
            ChannelSubscriber, ChannelFactoryInitialize = self._import_channel()
            message_type = self._import_lidar_message_type()
            network_interface = (
                os.environ.get("GO2_NETWORK_INTERFACE")
                or os.environ.get("CYCLONEDDS_NETWORK_INTERFACE")
            )
            if network_interface:
                ChannelFactoryInitialize(0, network_interface)
            else:
                ChannelFactoryInitialize(0)

            self._subscriber = ChannelSubscriber(self._config.topic, message_type)
            self._subscriber.Init(self._handle_sample, 1)
            self._backend = f"lidar:{self._config.topic}"
            logger.info("LidarPerimeterService: subscribed to %s", self._config.topic)
        except Exception as exc:
            self._subscriber = None
            self._backend = "none"
            self._subscriber_error = str(exc)
            logger.warning("LidarPerimeterService: unavailable: %s", exc)

    def stop(self) -> None:
        """Close the LiDAR subscriber."""
        subscriber = self._subscriber
        self._subscriber = None
        if subscriber is not None:
            try:
                subscriber.Close()
            except Exception:
                logger.debug("LidarPerimeterService: subscriber close failed", exc_info=True)

        with self._lock:
            self._latest_grid = None
            self._latest_sample_at = 0.0
        self._backend = "disabled" if not self._config.enabled else "none"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_available(self) -> bool:
        return self.get_obstacle_grid() is not None

    @property
    def subscriber_error(self) -> Optional[str]:
        return self._subscriber_error

    def get_obstacle_grid(self) -> Optional[ObstacleGrid]:
        """Return the latest fresh LiDAR grid."""
        with self._lock:
            if self._latest_grid is None:
                return None
            if time.monotonic() - self._latest_sample_at > self._config.max_sample_age_s:
                return None
            return self._latest_grid

    def get_obstacle_summary(self) -> str:
        from coded_tools.unigo2.obstacle_grid_utils import summarize_obstacle_grid

        return summarize_obstacle_grid(self.get_obstacle_grid())

    def grid_from_ranges(
        self,
        ranges_m: Sequence[float],
        angle_min_rad: float = -math.pi,
        angle_increment_rad: Optional[float] = None,
    ) -> Optional[ObstacleGrid]:
        """Convert a polar LiDAR scan into an ObstacleGrid."""
        ranges = np.asarray(ranges_m, dtype=np.float32) * self._config.range_scale
        if ranges.size == 0:
            return None

        if angle_increment_rad is None:
            angle_increment_rad = (2.0 * math.pi) / max(int(ranges.size), 1)

        angles = (
            angle_min_rad
            + np.arange(ranges.size, dtype=np.float32) * float(angle_increment_rad)
            + self._config.angle_offset_rad
        )
        valid = (
            np.isfinite(ranges)
            & (ranges >= self._config.min_range_m)
            & (ranges <= self._config.max_range_m)
        )
        if not np.any(valid):
            return self._grid_from_xy(np.empty((0, 2), dtype=np.float32))

        xy = np.column_stack(
            (
                ranges[valid] * np.cos(angles[valid]),
                ranges[valid] * np.sin(angles[valid]),
            )
        )
        return self._grid_from_xy(xy)

    def grid_from_points(self, points: Iterable[Any]) -> Optional[ObstacleGrid]:
        """Convert an iterable of point-like objects/sequences into an ObstacleGrid."""
        xy_points = []
        for point in points:
            xyz = self._point_to_xyz(point)
            if xyz is None:
                continue
            x, y, z = xyz
            distance = math.hypot(x, y)
            if not (
                self._config.min_range_m <= distance <= self._config.max_range_m
                and self._config.min_height_m <= z <= self._config.max_height_m
            ):
                continue
            xy_points.append((x, y))
        return self._grid_from_xy(np.asarray(xy_points, dtype=np.float32))

    def _handle_sample(self, sample: Any) -> None:
        """DDS callback for LiDAR samples."""
        grid = self._grid_from_sample(sample)
        if grid is None:
            return
        with self._lock:
            self._latest_grid = grid
            self._latest_sample_at = time.monotonic()

    def _grid_from_sample(self, sample: Any) -> Optional[ObstacleGrid]:
        pointcloud_grid = self._grid_from_pointcloud2_sample(sample)
        if pointcloud_grid is not None:
            return pointcloud_grid

        points = self._read_first_field(sample, ("points", "point", "cloud", "cloud_points"))
        if points is not None:
            grid = self.grid_from_points(points)
            if grid is not None:
                return grid

        ranges = self._read_first_field(sample, ("ranges", "range", "range_data"))
        if ranges is None:
            return None

        angle_min = self._read_number(sample, ("angle_min", "min_angle"), -math.pi)
        angle_increment = self._read_number(
            sample,
            ("angle_increment", "angle_step", "resolution"),
            None,
        )
        return self.grid_from_ranges(ranges, angle_min, angle_increment)

    def _grid_from_pointcloud2_sample(self, sample: Any) -> Optional[ObstacleGrid]:
        """Convert a ROS/sensor_msgs-style PointCloud2 sample into an ObstacleGrid."""
        raw_data = self._read_first_field(sample, ("data", "data_"))
        if raw_data is None:
            return None

        data = self._bytes_from_data(raw_data)
        if not data:
            return None

        fields = self._read_first_field(sample, ("fields", "fields_"))
        point_step = self._read_int(sample, ("point_step", "point_step_"), 0)
        width = self._read_int(sample, ("width", "width_"), 0)
        height = self._read_int(sample, ("height", "height_"), 1)
        if point_step <= 0:
            point_step = self._infer_point_step(fields) or 16

        count = width * max(height, 1) if width > 0 else len(data) // point_step
        if count <= 0:
            return None

        is_bigendian = bool(self._read_first_field(sample, ("is_bigendian", "is_bigendian_")) or False)
        endian = ">" if is_bigendian else "<"
        offsets = self._pointcloud_xyz_offsets(fields)
        if offsets is None:
            offsets = (0, 4, 8)

        points = []
        x_offset, y_offset, z_offset = offsets
        max_offset = max(offsets) + 4
        for index in range(count):
            base = index * point_step
            if base + max_offset > len(data):
                break
            try:
                x = struct.unpack_from(endian + "f", data, base + x_offset)[0]
                y = struct.unpack_from(endian + "f", data, base + y_offset)[0]
                z = struct.unpack_from(endian + "f", data, base + z_offset)[0]
            except struct.error:
                break
            points.append((x, y, z))

        return self.grid_from_points(points)

    def _grid_from_xy(self, xy: np.ndarray) -> ObstacleGrid:
        cfg = self._config
        return build_obstacle_grid(
            xy,
            ObstacleGridSpec(
                rows=cfg.grid_rows,
                cols=cfg.grid_cols,
                resolution=cfg.grid_resolution,
                path_corridor_half_width=cfg.path_corridor_half_width,
                path_obstacle_min_points=cfg.path_obstacle_min_points,
                inflation_radius_m=cfg.robot_half_width,
            ),
        )

    @classmethod
    def _import_channel(cls):
        errors = []
        for root in ("unitree_sdk2_python.unitree_sdk2py", "unitree_sdk2py"):
            try:
                mod = importlib.import_module(f"{root}.core.channel")
                return mod.ChannelSubscriber, mod.ChannelFactoryInitialize
            except Exception as exc:
                errors.append(f"{root}: {exc}")
        raise ImportError("; ".join(errors))

    @classmethod
    def _import_lidar_message_type(cls):
        errors = []
        for root in ("unitree_sdk2_python.unitree_sdk2py", "unitree_sdk2py"):
            for module_suffix, class_name in cls._MESSAGE_CANDIDATES:
                module_name = f"{root}.{module_suffix}"
                try:
                    mod = importlib.import_module(module_name)
                    msg_type = getattr(mod, class_name)
                    logger.info(
                        "LidarPerimeterService: using DDS message %s.%s",
                        module_name,
                        class_name,
                    )
                    return msg_type
                except Exception as exc:
                    errors.append(f"{module_name}.{class_name}: {exc}")
        raise ImportError("; ".join(errors))

    @staticmethod
    def _bytes_from_data(data: Any) -> bytes:
        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        if isinstance(data, memoryview):
            return data.tobytes()
        try:
            return bytes(data)
        except (TypeError, ValueError):
            return b""

    @staticmethod
    def _read_field(obj: Any, name: str) -> Any:
        value = getattr(obj, name, None)
        if callable(value):
            return value()
        return value

    @classmethod
    def _read_first_field(cls, obj: Any, names: Sequence[str]) -> Any:
        for name in names:
            value = cls._read_field(obj, name)
            if value is not None:
                return value
        return None

    @classmethod
    def _read_number(
        cls,
        obj: Any,
        names: Sequence[str],
        default: Optional[float],
    ) -> Optional[float]:
        value = cls._read_first_field(obj, names)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _read_int(cls, obj: Any, names: Sequence[str], default: int) -> int:
        value = cls._read_first_field(obj, names)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _pointcloud_xyz_offsets(cls, fields: Any) -> Optional[Tuple[int, int, int]]:
        if fields is None:
            return None

        offsets = {}
        for field in fields:
            name = cls._read_first_field(field, ("name", "name_"))
            if isinstance(name, bytes):
                name = name.decode(errors="ignore")
            if name not in {"x", "y", "z"}:
                continue
            datatype = cls._read_int(field, ("datatype", "datatype_"), 7)
            if datatype != 7:
                continue
            offsets[name] = cls._read_int(field, ("offset", "offset_"), -1)

        if {"x", "y", "z"} <= offsets.keys() and min(offsets.values()) >= 0:
            return offsets["x"], offsets["y"], offsets["z"]
        return None

    @classmethod
    def _infer_point_step(cls, fields: Any) -> Optional[int]:
        if fields is None:
            return None

        max_end = 0
        for field in fields:
            offset = cls._read_int(field, ("offset", "offset_"), -1)
            datatype = cls._read_int(field, ("datatype", "datatype_"), 7)
            count = max(cls._read_int(field, ("count", "count_"), 1), 1)
            size = {
                1: 1,  # INT8
                2: 1,  # UINT8
                3: 2,  # INT16
                4: 2,  # UINT16
                5: 4,  # INT32
                6: 4,  # UINT32
                7: 4,  # FLOAT32
                8: 8,  # FLOAT64
            }.get(datatype, 4)
            if offset >= 0:
                max_end = max(max_end, offset + size * count)
        return max_end or None

    @classmethod
    def _point_to_xyz(cls, point: Any) -> Optional[Tuple[float, float, float]]:
        x = cls._read_field(point, "x")
        y = cls._read_field(point, "y")
        z = cls._read_field(point, "z")
        if x is not None and y is not None:
            return float(x), float(y), float(z or 0.0)

        if isinstance(point, Sequence) and len(point) >= 2:
            z_value = point[2] if len(point) >= 3 else 0.0
            return float(point[0]), float(point[1]), float(z_value)
        return None
