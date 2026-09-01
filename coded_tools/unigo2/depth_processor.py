
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
DepthProcessor - Depth Camera and LiDAR Processing for Navigation

Converts depth camera frames and optional LiDAR data into 2D obstacle grids
that the local planner uses for reactive obstacle avoidance.

Supports:
1. Intel RealSense depth cameras (via pyrealsense2)
2. Generic USB depth cameras (via OpenCV)
3. Simulation mode with synthetic obstacles (for desktop testing)

Processing pipeline (~10ms on Orin Nano CPU):
  Depth frame -> Downsample -> Height threshold -> 2D projection -> Inflate -> ObstacleGrid
"""

import os
import math
import time
import logging
import platform
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

# ---------------------------------------------------------------------------
# Environment helpers (same pattern as vision_core.py)
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean from environment variable. Accepts '1', 'true', 'yes', 'on'."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read an integer from environment variable, returning default on missing or invalid."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float from environment variable, returning default on missing or invalid."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Optional imports with graceful fallback
# ---------------------------------------------------------------------------

try:
    import pyrealsense2 as rs
    _HAS_REALSENSE = True
except ImportError:
    rs = None
    _HAS_REALSENSE = False

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ObstacleGrid:
    """2D local obstacle map centered on the robot."""
    grid: np.ndarray           # shape (rows, cols), dtype float32, 0.0=free, 1.0=occupied
    resolution: float          # meters per cell (e.g., 0.05 = 5cm)
    origin_row: int            # robot's row in the grid
    origin_col: int            # robot's col in the grid
    timestamp: float = 0.0
    nearest_obstacle_m: float = float("inf")
    nearest_obstacle_bearing: float = 0.0  # radians, 0=ahead, positive=left
    path_obstacle_m: float = float("inf")
    path_obstacle_bearing: float = 0.0
    path_obstacle_points: int = 0


@dataclass
class CenterDepthReading:
    """Raw center-band depth estimate used as a forward-motion watchdog."""
    distance_m: float
    coverage: float
    timestamp: float = 0.0


@dataclass
class DepthProcessorConfig:
    """Configuration for depth processing pipeline."""
    # Grid parameters
    grid_rows: int = 80
    grid_cols: int = 80
    grid_resolution: float = 0.05   # meters per cell -> 4m x 4m FOV

    # Height thresholds (meters, relative to camera mount height)
    ground_height: float = 0.05     # below this = ground, ignore
    obstacle_max_height: float = 0.60  # above this = overhead, ignore
    camera_mount_height: float = 0.30  # Go2 front camera height from ground

    # Obstacle inflation
    robot_half_width: float = 0.15  # meters, for obstacle dilation
    # Robot half-width plus lateral clearance.  The old 0.12m center ray was
    # narrower than the Go2 itself and could miss a slanted wall until contact.
    path_corridor_half_width: float = 0.27
    # The Go2's right legs need a little more room than the nominal symmetric
    # footprint.  Keep this robot-relative so it protects the same physical
    # side while travelling in either direction.
    right_side_clearance_margin: float = 0.08
    path_obstacle_min_points: int = 6

    # Depth camera parameters
    depth_width: int = 640
    depth_height: int = 480
    depth_fps: int = 30
    process_width: int = 640        # depth processing target
    process_height: int = 480

    # Range limits
    min_depth_m: float = 0.1
    max_depth_m: float = 4.0

    # Simulation
    simulation_mode: bool = False


# ---------------------------------------------------------------------------
# DepthProcessor
# ---------------------------------------------------------------------------

class DepthProcessor:
    """
    Processes depth camera input into obstacle grids for the local planner.

    Thread-safe: get_obstacle_grid() can be called from any thread.
    The depth capture runs in a background thread at the configured FPS.
    """

    def __init__(self, config: Optional[DepthProcessorConfig] = None):
        """Initialize the depth processor with given or environment-based config.

        Args:
            config: Processing configuration. If None, reads from NAV_* env vars.
        """
        self._config = config or self._config_from_env()
        self._lock = threading.Lock()
        self._latest_grid: Optional[ObstacleGrid] = None
        self._latest_depth_m: Optional[np.ndarray] = None
        self._latest_depth_timestamp: float = 0.0
        self._pipeline = None          # RealSense pipeline
        self._cv_capture = None        # OpenCV VideoCapture fallback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()
        self._backend = "none"
        self._inflation_kernel = self._build_inflation_kernel()

        if self._config.simulation_mode:
            self._backend = "simulation"
            logger.info("DepthProcessor: simulation mode (synthetic obstacles)")
            return

        self._init_camera()

    @staticmethod
    def _config_from_env() -> DepthProcessorConfig:
        """Build a DepthProcessorConfig from NAV_* environment variables."""
        return DepthProcessorConfig(
            grid_rows=_env_int("NAV_GRID_ROWS", 80),
            grid_cols=_env_int("NAV_GRID_COLS", 80),
            grid_resolution=_env_float("NAV_GRID_RESOLUTION", 0.05),
            ground_height=_env_float("NAV_GROUND_HEIGHT", 0.05),
            obstacle_max_height=_env_float("NAV_OBSTACLE_MAX_HEIGHT", 0.60),
            camera_mount_height=_env_float("NAV_CAMERA_MOUNT_HEIGHT", 0.30),
            robot_half_width=_env_float("NAV_ROBOT_HALF_WIDTH", 0.15),
            path_corridor_half_width=_env_float("NAV_PATH_CORRIDOR_HALF_WIDTH", 0.27),
            right_side_clearance_margin=_env_float(
                "NAV_RIGHT_SIDE_CLEARANCE_MARGIN",
                0.08,
            ),
            path_obstacle_min_points=_env_int("NAV_PATH_OBSTACLE_MIN_POINTS", 6),
            process_width=_env_int("NAV_DEPTH_PROCESS_WIDTH", 640),
            process_height=_env_int("NAV_DEPTH_PROCESS_HEIGHT", 480),
            min_depth_m=_env_float("NAV_MIN_DEPTH", 0.1),
            max_depth_m=_env_float("NAV_MAX_DEPTH", 4.0),
            simulation_mode=_env_flag("NAV_SIMULATION_MODE", False),
        )

    # ------------------------------------------------------------------
    # Camera initialization
    # ------------------------------------------------------------------

    def _init_camera(self):
        """Detect and initialize the depth camera backend (RealSense -> OpenCV -> none)."""
        source = os.environ.get("NAV_DEPTH_CAMERA_SOURCE", "auto")

        if source != "auto" and not source.startswith("realsense"):
            self._init_opencv_depth(source)
            return

        if _HAS_REALSENSE:
            if self._init_realsense():
                return

        if source == "auto":
            self._try_opencv_depth_auto()

        if self._backend == "none":
            logger.warning(
                "DepthProcessor: no depth camera found. "
                "Navigation will use vision-only fallback or be disabled."
            )

    def _init_realsense(self) -> bool:
        """Initialize Intel RealSense pipeline. Caches depth_scale and intrinsics."""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                logger.info("DepthProcessor: no RealSense devices found")
                return False

            dev = devices[0]
            dev_name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "unknown"
            dev_serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else ""
            logger.info("DepthProcessor: found RealSense device: %s (S/N: %s)", dev_name, dev_serial)

            self._pipeline = rs.pipeline()
            cfg = rs.config()
            if dev_serial:
                cfg.enable_device(dev_serial)
            cfg.enable_stream(
                rs.stream.depth,
                self._config.depth_width,
                self._config.depth_height,
                rs.format.z16,
                self._config.depth_fps,
            )

            profile = self._pipeline.start(cfg)

            # Cache depth scale (meters per depth unit) for accurate conversion
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = depth_sensor.get_depth_scale()

            # Cache intrinsics for 3D projection
            depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self._intrinsics = depth_stream.get_intrinsics()

            self._backend = "realsense"
            logger.info(
                "DepthProcessor: RealSense initialized (%dx%d @ %d fps, depth_scale=%.6f)",
                self._config.depth_width,
                self._config.depth_height,
                self._config.depth_fps,
                self._depth_scale,
            )
            return True
        except Exception as exc:
            logger.warning("DepthProcessor: RealSense init failed: %s", exc)
            self._pipeline = None
            return False

    def _init_opencv_depth(self, source: str):
        """Open a specific depth camera via OpenCV (device index or path)."""
        if not _HAS_CV2:
            logger.warning("DepthProcessor: OpenCV is not installed")
            return
        try:
            idx = int(source)
            cap = cv2.VideoCapture(idx)
        except ValueError:
            cap = cv2.VideoCapture(source)

        if cap.isOpened():
            self._cv_capture = cap
            self._backend = "opencv"
            logger.info("DepthProcessor: OpenCV depth camera opened: %s", source)
        else:
            cap.release()
            logger.warning("DepthProcessor: failed to open depth camera: %s", source)

    def _try_opencv_depth_auto(self):
        """Scan /dev/video* for a single-channel depth camera, skipping RGB devices."""
        if not _HAS_CV2:
            return
        from pathlib import Path
        video_devices = sorted(
            (p for p in Path("/dev").glob("video*") if p.name.removeprefix("video").isdigit()),
            key=lambda p: int(p.name.removeprefix("video")),
        )
        for dev_path in video_devices:
            try:
                cap = cv2.VideoCapture(str(dev_path), cv2.CAP_V4L2)
            except Exception:
                cap = cv2.VideoCapture(str(dev_path))
            if not cap.isOpened():
                continue
            ret, frame = cap.read()
            if ret and frame is not None:
                # Depth cameras typically produce single-channel (grayscale)
                # or 16-bit frames.  Skip 3-channel BGR (likely an RGB camera).
                if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                    self._cv_capture = cap
                    self._backend = "opencv"
                    logger.info("DepthProcessor: found depth camera at %s", dev_path)
                    return
            cap.release()

    # ------------------------------------------------------------------
    # Inflation kernel
    # ------------------------------------------------------------------

    def _build_inflation_kernel(self) -> np.ndarray:
        """Build a circular dilation kernel sized to the robot's half-width."""
        radius_cells = max(1, int(self._config.robot_half_width / self._config.grid_resolution))
        size = 2 * radius_cells + 1
        kernel = np.zeros((size, size), dtype=np.uint8)
        if _HAS_CV2:
            cv2.circle(kernel, (radius_cells, radius_cells), radius_cells, 1, -1)
        else:
            yy, xx = np.ogrid[:size, :size]
            mask = (yy - radius_cells) ** 2 + (xx - radius_cells) ** 2 <= radius_cells ** 2
            kernel[mask] = 1
        return kernel

    # ------------------------------------------------------------------
    # Background capture thread
    # ------------------------------------------------------------------

    def _camera_resources_ready(self) -> bool:
        """Return True when the selected backend has live camera resources."""
        if self._backend == "simulation":
            return True
        if self._backend == "realsense":
            return self._pipeline is not None
        if self._backend == "opencv":
            return self._cv_capture is not None and self._cv_capture.isOpened()
        return False

    def start(self):
        """Start the background depth capture thread."""
        with self._thread_lock:
            if self._running:
                return

            if not self._camera_resources_ready() and not self._config.simulation_mode:
                self._init_camera()

            if self._backend == "none":
                return

            self._running = True
            thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="depth-capture",
            )
            self._thread = thread
            try:
                thread.start()
            except Exception:
                self._running = False
                self._thread = None
                raise
        logger.info("DepthProcessor: capture thread started (%s backend)", self._backend)

    def stop(self):
        """Stop the background capture thread and release camera resources."""
        with self._thread_lock:
            self._running = False
            thread = self._thread
            self._thread = None

            if thread:
                try:
                    if thread.ident is not None and thread.is_alive():
                        thread.join(timeout=2.0)
                except RuntimeError as exc:
                    logger.debug(
                        "DepthProcessor: ignored stop/start race while joining capture thread: %s",
                        exc,
                    )
            self._release_camera()

    def _release_camera(self):
        """Release RealSense pipeline or OpenCV capture."""
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        if self._cv_capture:
            self._cv_capture.release()
            self._cv_capture = None
        with self._lock:
            self._latest_grid = None
            self._latest_depth_m = None
            self._latest_depth_timestamp = 0.0
        if self._backend in {"realsense", "opencv"}:
            self._backend = "none"

    def _capture_loop(self):
        """Background thread loop: read depth frames and update the latest grid."""
        while self._running:
            cycle_start = time.monotonic()
            try:
                depth_frame = self._read_depth_frame()
                if depth_frame is not None:
                    grid = self._process_depth_to_grid(depth_frame)
                    with self._lock:
                        self._latest_depth_m = depth_frame
                        self._latest_depth_timestamp = time.time()
                        self._latest_grid = grid
            except Exception as exc:
                logger.error("DepthProcessor: capture error: %s", exc)

            elapsed = time.monotonic() - cycle_start
            sleep_time = (1.0 / self._config.depth_fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Depth frame reading
    # ------------------------------------------------------------------

    def _read_depth_frame(self) -> Optional[np.ndarray]:
        """Read one depth frame from the active backend. Returns float32 array in meters."""
        if self._backend == "realsense":
            return self._read_realsense()
        elif self._backend == "opencv":
            return self._read_opencv()
        elif self._backend == "simulation":
            return self._generate_synthetic_depth()
        return None

    def _read_realsense(self) -> Optional[np.ndarray]:
        """Read from RealSense, converting raw uint16 to float32 meters via cached depth_scale."""
        frames = self._pipeline.wait_for_frames(timeout_ms=500)
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return None
        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_image.astype(np.float32) * self._depth_scale

    def _read_opencv(self) -> Optional[np.ndarray]:
        """Read from OpenCV capture, assuming millimeter units (divided by 1000)."""
        ret, frame = self._cv_capture.read()
        if not ret or frame is None:
            return None
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame.astype(np.float32) / 1000.0

    def _generate_synthetic_depth(self) -> np.ndarray:
        """Generate a synthetic depth frame with a wall at 2m for simulation mode."""
        h, w = self._config.process_height, self._config.process_width
        depth = np.full((h, w), 3.0, dtype=np.float32)

        # Simulated wall 2 meters ahead, spanning the middle third
        wall_col_start = w // 3
        wall_col_end = 2 * w // 3
        depth[h // 4 : 3 * h // 4, wall_col_start:wall_col_end] = 2.0

        return depth

    # ------------------------------------------------------------------
    # Depth -> ObstacleGrid processing
    # ------------------------------------------------------------------

    def _process_depth_to_grid(self, depth_m: np.ndarray) -> ObstacleGrid:
        """Convert a depth frame (float32, meters) into a 2D ObstacleGrid.

        Pipeline: downsample -> mask invalid -> 3D projection -> height filter ->
        grid binning -> inflate -> nearest obstacle computation.

        Args:
            depth_m: Depth image in meters, shape (H, W), dtype float32.

        Returns:
            ObstacleGrid with occupied cells, nearest obstacle distance and bearing.
        """
        cfg = self._config
        now = time.time()

        # Step 1: Downsample
        if depth_m.shape[0] != cfg.process_height or depth_m.shape[1] != cfg.process_width:
            if _HAS_CV2:
                depth_m = cv2.resize(
                    depth_m,
                    (cfg.process_width, cfg.process_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                row_idx = np.linspace(0, depth_m.shape[0] - 1, cfg.process_height).astype(np.int32)
                col_idx = np.linspace(0, depth_m.shape[1] - 1, cfg.process_width).astype(np.int32)
                depth_m = depth_m[np.ix_(row_idx, col_idx)]

        # Step 2: Mask invalid depths
        valid = (depth_m > cfg.min_depth_m) & (depth_m < cfg.max_depth_m)

        # Step 3: For each pixel, compute 3D position in robot frame
        # Use real intrinsics from RealSense if available, else approximate
        if hasattr(self, "_intrinsics") and self._intrinsics is not None:
            intr = self._intrinsics
            scale_x = cfg.process_width / intr.width
            scale_y = cfg.process_height / intr.height
            fx = intr.fx * scale_x
            fy = intr.fy * scale_y
            cx = intr.ppx * scale_x
            cy = intr.ppy * scale_y
        else:
            fx = cfg.process_width * 0.6
            fy = cfg.process_height * 0.6
            cx = cfg.process_width / 2.0
            cy = cfg.process_height / 2.0

        v_coords, u_coords = np.mgrid[0:cfg.process_height, 0:cfg.process_width]
        z = depth_m  # depth = distance along optical axis

        # Camera frame: x=right, y=down, z=forward
        x_cam = (u_coords.astype(np.float32) - cx) * z / fx
        y_cam = (v_coords.astype(np.float32) - cy) * z / fy

        # Robot frame: x=forward, y=left, height=up
        # Camera is mounted facing forward, so camera-z = robot-x, camera-x = robot-(-y)
        x_robot = z                    # forward distance
        y_robot = -x_cam               # left/right (camera-x is right, robot-y is left)
        height = -(y_cam - cfg.camera_mount_height)  # height above ground

        # Step 4: Height threshold to find obstacles
        is_obstacle = valid & (height > cfg.ground_height) & (height < cfg.obstacle_max_height)

        # Step 5: Project obstacle points to 2D grid
        grid = np.zeros((cfg.grid_rows, cfg.grid_cols), dtype=np.float32)
        origin_row = cfg.grid_rows - 1  # robot at bottom center
        origin_col = cfg.grid_cols // 2

        obs_x = x_robot[is_obstacle]
        obs_y = y_robot[is_obstacle]

        grid_row = origin_row - (obs_x / cfg.grid_resolution).astype(np.int32)
        grid_col = origin_col - (obs_y / cfg.grid_resolution).astype(np.int32)

        in_bounds = (
            (grid_row >= 0) & (grid_row < cfg.grid_rows) &
            (grid_col >= 0) & (grid_col < cfg.grid_cols)
        )
        grid_row = grid_row[in_bounds]
        grid_col = grid_col[in_bounds]

        if len(grid_row) > 0:
            np.add.at(grid, (grid_row, grid_col), 1.0)
            grid = np.clip(grid, 0.0, 1.0)

        # Step 6: Inflate obstacles by robot radius
        if np.any(grid > 0):
            if _HAS_CV2:
                grid_u8 = (grid * 255).astype(np.uint8)
                grid_u8 = cv2.dilate(grid_u8, self._inflation_kernel, iterations=1)
                grid = (grid_u8 > 0).astype(np.float32)
            else:
                grid = self._dilate_grid_numpy(grid)

        # Step 7: Compute nearest obstacle distance and bearing
        nearest_dist = float("inf")
        nearest_bearing = 0.0
        path_dist = float("inf")
        path_bearing = 0.0
        path_points = 0
        if len(obs_x) > 0:
            distances = np.sqrt(obs_x ** 2 + obs_y ** 2)
            min_idx = np.argmin(distances)
            nearest_dist = float(distances[min_idx])
            nearest_bearing = float(math.atan2(obs_y[min_idx], obs_x[min_idx]))

            # Robot-frame +y is left and -y is right.  Extend only the right
            # edge of the protected path so the vulnerable right legs do not
            # brush furniture; the left edge and narrow-route behavior remain
            # unchanged.
            in_path = (
                (obs_x > 0.0)
                & (obs_y <= cfg.path_corridor_half_width)
                & (
                    obs_y
                    >= -(
                        cfg.path_corridor_half_width
                        + cfg.right_side_clearance_margin
                    )
                )
            )
            if np.any(in_path):
                path_x = obs_x[in_path]
                path_y = obs_y[in_path]
                path_distances = distances[in_path]
                range_bins = np.floor(path_x / cfg.grid_resolution).astype(np.int32)

                for range_bin in np.unique(range_bins):
                    bin_mask = range_bins == range_bin
                    support = int(np.sum(bin_mask))
                    if support < cfg.path_obstacle_min_points:
                        continue

                    bin_distances = path_distances[bin_mask]
                    candidate_dist = float(np.percentile(bin_distances, 25))
                    if candidate_dist >= path_dist:
                        continue

                    candidate_y = float(np.median(path_y[bin_mask]))
                    candidate_x = float(np.median(path_x[bin_mask]))
                    path_dist = candidate_dist
                    path_bearing = float(math.atan2(candidate_y, candidate_x))
                    path_points = support

        # Step 8: Ground plane validity check
        # If very few depth pixels are valid in the lower half of the frame,
        # the ground plane may be missing (cliff/step ahead)
        lower_half_valid = valid[cfg.process_height // 2 :, :]
        ground_valid_ratio = np.sum(lower_half_valid) / max(lower_half_valid.size, 1)
        if ground_valid_ratio < 0.1:
            logger.warning("DepthProcessor: ground plane may be missing (%.1f%% valid)", ground_valid_ratio * 100)

        return ObstacleGrid(
            grid=grid,
            resolution=cfg.grid_resolution,
            origin_row=origin_row,
            origin_col=origin_col,
            timestamp=now,
            nearest_obstacle_m=nearest_dist,
            nearest_obstacle_bearing=nearest_bearing,
            path_obstacle_m=path_dist,
            path_obstacle_bearing=path_bearing,
            path_obstacle_points=path_points,
        )

    def _dilate_grid_numpy(self, grid: np.ndarray) -> np.ndarray:
        """Numpy fallback for one binary dilation iteration."""
        binary = grid > 0
        kernel = self._inflation_kernel > 0
        pad_y = kernel.shape[0] // 2
        pad_x = kernel.shape[1] // 2
        padded = np.pad(binary, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
        dilated = np.zeros_like(binary, dtype=bool)

        for ky, kx in np.argwhere(kernel):
            y0 = ky
            x0 = kx
            dilated |= padded[y0:y0 + binary.shape[0], x0:x0 + binary.shape[1]]

        return dilated.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_obstacle_grid(self) -> Optional[ObstacleGrid]:
        """Return the latest obstacle grid (thread-safe)."""
        if self._backend == "simulation":
            depth = self._generate_synthetic_depth()
            return self._process_depth_to_grid(depth)

        with self._lock:
            return self._latest_grid

    def get_single_frame_grid(self) -> Optional[ObstacleGrid]:
        """Capture and process a single depth frame (blocking). Useful for scanning."""
        depth = self._read_depth_frame()
        if depth is None:
            return None
        with self._lock:
            self._latest_depth_m = depth
            self._latest_depth_timestamp = time.time()
        return self._process_depth_to_grid(depth)

    def get_center_depth_reading(
        self,
        vertical_band: Tuple[float, float] = (0.375, 0.708),
        half_width_ratio: float = 0.22,
        percentile: float = 10.0,
        max_depth_m: float = 8.0,
        min_coverage: float = 0.02,
    ) -> Optional[CenterDepthReading]:
        """Return a raw center-band depth estimate for forward obstacle stopping.

        This intentionally bypasses the ground-plane projection used by
        get_obstacle_grid(). On the Go2, the projected grid can include floor or
        side false positives during walking; a raw mid-image band gave the most
        stable "object straight ahead" signal during robot testing.
        """
        depth = self._get_latest_depth_frame()
        if depth is None:
            return None
        return self._center_depth_reading_from_frame(
            depth,
            vertical_band=vertical_band,
            half_width_ratio=half_width_ratio,
            percentile=percentile,
            max_depth_m=max_depth_m,
            min_coverage=min_coverage,
        )

    def _get_latest_depth_frame(self) -> Optional[np.ndarray]:
        """Return the latest raw depth frame, capturing one if the thread has none."""
        if self._backend == "simulation":
            return self._generate_synthetic_depth()

        with self._lock:
            depth = None if self._latest_depth_m is None else self._latest_depth_m.copy()

        if depth is not None:
            return depth

        depth = self._read_depth_frame()
        if depth is not None:
            with self._lock:
                self._latest_depth_m = depth
                self._latest_depth_timestamp = time.time()
        return depth

    def _center_depth_reading_from_frame(
        self,
        depth_m: np.ndarray,
        vertical_band: Tuple[float, float] = (0.375, 0.708),
        half_width_ratio: float = 0.22,
        percentile: float = 10.0,
        max_depth_m: float = 8.0,
        min_coverage: float = 0.02,
    ) -> Optional[CenterDepthReading]:
        """Compute a robust low-percentile depth from the center image band."""
        if depth_m is None or depth_m.size == 0:
            return None

        h, w = depth_m.shape[:2]
        row_start = max(0, min(h - 1, int(h * vertical_band[0])))
        row_end = max(row_start + 1, min(h, int(h * vertical_band[1])))
        half_width = max(1, int(w * half_width_ratio))
        col_center = w // 2
        col_start = max(0, col_center - half_width)
        col_end = min(w, col_center + half_width)

        roi = depth_m[row_start:row_end, col_start:col_end]
        valid = roi[(roi > self._config.min_depth_m) & (roi < max_depth_m)]
        coverage = float(valid.size / max(roi.size, 1))

        if coverage < min_coverage or valid.size == 0:
            return None

        return CenterDepthReading(
            distance_m=float(np.percentile(valid, percentile)),
            coverage=coverage,
            timestamp=time.time(),
        )

    @property
    def backend(self) -> str:
        """Active backend: 'realsense', 'opencv', 'simulation', or 'none'."""
        return self._backend

    @property
    def is_available(self) -> bool:
        """True if any depth source is active (including simulation)."""
        return self._backend != "none"

    @property
    def is_running(self) -> bool:
        """True while the background capture thread is active."""
        return self._running

    def get_obstacle_summary(self) -> str:
        """Human-readable obstacle summary for agent consumption."""
        grid = self.get_obstacle_grid()
        if grid is None:
            return "No depth data available."

        occupied_cells = int(np.sum(grid.grid > 0))
        total_cells = grid.grid.size
        occupied_pct = (occupied_cells / total_cells) * 100

        parts = [f"Obstacle grid: {occupied_pct:.0f}% occupied"]

        if grid.nearest_obstacle_m < float("inf"):
            bearing_deg = math.degrees(grid.nearest_obstacle_bearing)
            if abs(bearing_deg) < 10:
                direction = "directly ahead"
            elif bearing_deg > 0:
                direction = f"{abs(bearing_deg):.0f} degrees to the left"
            else:
                direction = f"{abs(bearing_deg):.0f} degrees to the right"
            parts.append(f"Nearest obstacle anywhere: {grid.nearest_obstacle_m:.2f}m {direction}")
        else:
            parts.append("No obstacles within range")

        if grid.path_obstacle_m < float("inf"):
            parts.append(
                f"Path obstacle: {grid.path_obstacle_m:.2f}m "
                f"({grid.path_obstacle_points} depth points)"
            )
        else:
            parts.append("Path corridor clear")

        return ". ".join(parts) + "."

    def __del__(self):
        self.stop()


# ---------------------------------------------------------------------------
# Vision-based fallback distance estimation
# ---------------------------------------------------------------------------

def estimate_obstacle_distance_from_bbox(
    bbox: List[int],
    image_height: int,
    camera_fov_v: float = 0.78,
    camera_height: float = 0.30,
) -> float:
    """
    Rough distance estimate from a YOLO bounding box bottom edge.
    Objects whose bottom edge is lower in the image are closer.

    Uses simplified pinhole camera geometry. Accuracy is approximately +/- 50%.
    Useful only as a last-resort fallback when no depth camera is available.
    """
    bottom_y = max(bbox[1], bbox[3])
    image_center_y = image_height / 2.0

    pixel_below_center = bottom_y - image_center_y
    if pixel_below_center <= 0:
        return 10.0

    angle_below_horizon = (pixel_below_center / image_height) * camera_fov_v
    if angle_below_horizon <= 0.01:
        return 10.0

    distance = camera_height / math.tan(angle_below_horizon)
    return max(0.1, min(distance, 10.0))
