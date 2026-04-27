# Nav Core: Navigation Module Design for CAIL-E (Unitree Go2 EDU)

## 1. Research Summary & Approach Selection

This section evaluates existing navigation approaches for quadruped robots and selects the architecture best suited to our hardware constraints (Nvidia Jetson Orin Nano, 40 TOPS) and software stack (standalone Neuro SAN + CycloneDDS, no ROS2).

### Navigation Approaches Evaluated

| Approach | Source | Verdict | Rationale |
|----------|--------|---------|-----------|
| **CMU ABS (Agile But Safe)** | [agile-but-safe.github.io](https://agile-but-safe.github.io/), [LeCAR-Lab/ABS](https://github.com/LeCAR-Lab/ABS) | Borrow safety-supervisor pattern only | RL locomotion policies require Orin NX (16GB), not Nano. Focus is locomotion, not high-level navigation. But the concept of an agile policy supervised by a safety monitor is excellent and adopted here. |
| **ROS2 Nav2** | [ros-navigation/navigation2](https://github.com/ros-navigation/navigation2), [Issue #5512](https://github.com/ros-navigation/navigation2/issues/5512) | Use architecture concepts, not the stack | No native quadruped support (Issue #5512 confirms this). Project is standalone DDS, not ROS2. But the global planner -> local planner -> controller layering is the gold standard architecture. |
| **BotBrain** | [botbotrobotics/BotBrain](https://github.com/botbotrobotics/BotBrain) | Reference implementation | Proven Go2 + Orin Nano + dual RealSense D435i system. Uses RTABMap SLAM + Nav2. ROS2-dependent so not directly integrable, but validates the sensor and algorithm combination on our target hardware. |
| **NVIDIA Isaac ROS** | [Zhefan-Xu/isaac-go2-ros2](https://github.com/Zhefan-Xu/isaac-go2-ros2) | Simulation + future GPU acceleration | Good for sim-to-real testing. nvblox for 3D reconstruction is interesting for future phases. Heavy ROS2 dependency limits direct use. |
| **Terrain-Aware Locomotion** | Research papers (elevation mapping + CNN terrain classification) | Future Phase 5+ | 95% navigation success with 50% fewer falls on rough terrain. Relevant for outdoor/unstructured environments. Defer until indoor navigation is solid. |

### Selected Approach

Lightweight standalone navigation stack inspired by **Nav2's layered architecture** (global planner -> local planner -> controller) combined with **CMU ABS's safety-supervisor pattern**. Key characteristics:

- **No ROS2 dependency**: Fits our existing standalone DDS stack
- **Topological maps** instead of metric SLAM: avoids heavy compute on Orin Nano
- **VFH+ local planner**: proven, efficient, debuggable
- **Safety monitor**: every velocity command filtered before reaching the robot
- **Incremental build**: start with local obstacle avoidance, add global planning later

---

## 2. Universal Scene Description

### Do we need a universal scene description?

**Yes, but incrementally.** Vision_core already produces object lists (`{class_name, confidence, bbox}`). Nav_core adds **spatial context** to this, creating a richer understanding of the environment for the conscious_agent.

### Combined Scene Description Format

```
Current Scene Description:
- Objects: [person at ~1.5m ahead, chair at ~2.0m right-front]
- Obstacles: [wall 0.8m left, doorframe 1.2m ahead]
- Navigation: NAVIGATING toward "kitchen", next waypoint 3.2m ahead
- Terrain: flat indoor floor, clear path ahead
```

### How it's built

The scene description combines three data sources:

1. **vision_core** detections: What objects are present (class names, confidence, bounding boxes)
2. **depth_processor** obstacle grid: Where obstacles are spatially (distances, bearings)
3. **nav_core** state: Where the robot is going, what it's avoiding, navigation progress

This enriched scene is fed to the conscious_agent via the existing `scene_observer.py` pipeline. The `build_scene_input()` output gains spatial annotations like "person at ~1.5m ahead" instead of just "saw: person".

### Integration Point

`scene_observer.py` can be extended with an optional `observe_with_navigation()` method:

```python
def observe_with_navigation(self) -> Optional[Dict[str, Any]]:
    observation = self.observe()  # existing vision observation
    if observation is None:
        return None

    nav = NavCore.get_instance()
    if nav and nav.is_initialized():
        observation["nav_state"] = nav.state.name
        observation["nearest_obstacle_m"] = nav.get_nearest_obstacle_distance()
        observation["obstacle_directions"] = nav.get_obstacle_bearing_summary()
    return observation
```

---

## 3. Depth Camera: Why and How

### Do we need the depth camera?

**YES** - it is the single most important sensor for navigation.

### Why it's essential

1. **Obstacle distance**: RGB cameras see objects but don't know how far they are. Depth gives exact distances (0.1m - 10m range for RealSense D435i).
2. **Ground plane detection**: Depth reveals floor discontinuities (steps, stairs, curbs) that RGB alone cannot detect.
3. **3D obstacle mapping**: Project depth pixels to a 2D top-down obstacle grid - the local planner's primary input.
4. **Works in low light**: Infrared-based depth (RealSense active IR) works where RGB cameras fail.
5. **Complements LiDAR**: The built-in LiDAR gives 360-degree coverage but coarse vertical resolution. The depth camera gives dense forward-facing data ideal for what's directly in front of the robot.

### Depth Processing Pipeline

```
Depth Frame (640x480 @ 30Hz from RealSense D435i)
    |
    v
[1] Downsample to 320x240 (~2ms on CPU)
    |
    v
[2] Ground plane removal via height thresholding (~3ms)
    - Points below 5cm above ground: ground surface (ignore)
    - Points 5cm - 60cm above ground: obstacles (robot body height zone)
    - Points above 60cm: overhead clearance (ignore for ground robot)
    |
    v
[3] Project to 2D top-down grid (~3ms)
    - Each depth pixel -> (x, z) in robot frame using camera intrinsics
    - Accumulate into ObstacleGrid cells
    |
    v
[4] Inflate obstacles by robot half-width (~1ms)
    - Morphological dilation with circular kernel (~0.15m radius)
    |
    v
ObstacleGrid (80x80 cells, 5cm resolution = 4m x 4m field of view)
Total: ~10ms per frame on Orin Nano CPU. No GPU needed.
```

### Camera Access Strategy

The depth camera (Intel RealSense D435i or similar USB depth camera) is accessed via `pyrealsense2`, independently from the Unitree front RGB camera. This avoids contention with `scene_observer.py` which uses the front camera through `open_camera()`.

### Fallback: Vision-Based Distance Estimation

When no depth camera is available, nav_core degrades to approximate distance estimation using YOLO bounding boxes from vision_core:

```python
def estimate_obstacle_distance_from_bbox(
    bbox: List[int],
    class_name: str,
    image_height: int,
    camera_fov_v: float = 0.78,   # ~45 degrees vertical FOV
    camera_height: float = 0.3,    # Go2 front camera height
) -> float:
    """
    Rough distance estimate from bounding box bottom edge.
    Objects whose bottom edge is lower in the image are closer.
    Approximate: +/- 50% accuracy. Usable for basic avoidance only.
    """
```

---

## 4. High-Level vs Low-Level Planning

### Do we need both?

**YES.** Navigation requires two distinct planning layers operating at different scales:

### High-Level (Global) Planning

- **Input**: Semantic goal ("go to the kitchen")
- **Output**: Waypoint sequence through the environment
- **Algorithm**: Dijkstra's shortest path on a topological graph map
- **Frequency**: Once per goal (replan on failure)
- **Compute**: ~1ms (negligible)

### Low-Level (Local) Planning

- **Input**: Next waypoint direction + ObstacleGrid from depth camera
- **Output**: Velocity commands `(vx, vy, vyaw)` for Go2Macros.move()
- **Algorithm**: VFH+ (Vector Field Histogram Plus)
- **Frequency**: 10 Hz (every 100ms)
- **Compute**: ~1ms per cycle

### Why This Split Works

- **Separation of concerns**: Global planner handles the "where to go" without worrying about dynamic obstacles. Local planner handles "how to get there safely" without needing to know the full route.
- **Independent development and testing**: Each planner can be unit tested in isolation.
- **Robustness**: Global plan stays valid even when local planner detours around obstacles. Local planner works even without a global plan (e.g., "move forward 2 meters").
- **Proven architecture**: This is the same layered design used by Nav2, the industry standard.

---

## 5. Global Planning: Topological Maps

### Why topological maps instead of metric SLAM?

| Factor | Topological Map | Metric SLAM (e.g., RTABMap) |
|--------|----------------|---------------------------|
| Compute cost | ~0 (JSON file) | High (continuous processing, 2-4GB RAM) |
| Orin Nano feasibility | Easy | Tight alongside YOLO TensorRT |
| Creation | Hand-editable JSON or robot-learned | Automatic but needs calibration |
| Maintenance | Simple edits | Complex loop closure management |
| Accuracy | Approximate (sufficient for room-level) | Precise (centimeter-level) |
| Sufficient for "go to room X" | Yes | Overkill |

**Decision**: Topological maps for now. Can upgrade to metric maps in a future phase if centimeter-level precision becomes necessary.

### Map Structure

Maps are stored as JSON files in a `maps/` directory. Each map is a graph of named locations (nodes) connected by traversable paths (edges).

```json
{
    "name": "CAIL Lab - San Francisco",
    "version": "1.0",
    "nodes": [
        {
            "name": "charging_station",
            "x": 0.0, "y": 0.0,
            "description": "Home base near the entrance",
            "tags": ["home", "charging"]
        },
        {
            "name": "main_desk_area",
            "x": 3.0, "y": 0.0,
            "description": "Open desk area with workstations",
            "tags": ["room", "work"]
        },
        {
            "name": "kitchen",
            "x": 3.0, "y": 4.0,
            "description": "Kitchen and break area",
            "tags": ["room", "kitchen"]
        },
        {
            "name": "entrance",
            "x": -2.0, "y": 0.0,
            "description": "Lab main entrance",
            "tags": ["door", "entrance"]
        }
    ],
    "edges": [
        {"from": "charging_station", "to": "main_desk_area", "distance": 3.0, "description": "straight path"},
        {"from": "main_desk_area", "to": "kitchen", "distance": 4.0, "description": "through hallway"},
        {"from": "charging_station", "to": "entrance", "distance": 2.0, "description": "turn left"}
    ]
}
```

### Path Finding

Dijkstra's algorithm on the graph. Example: "go to kitchen" from charging_station -> `[charging_station, main_desk_area, kitchen]`. Each node has (x, y) coordinates in the odometry frame, so the local planner knows which direction to steer.

---

## 6. Local Planning: VFH+ Algorithm

### Algorithm Selection

| Algorithm | Pros | Cons | Verdict |
|-----------|------|------|---------|
| **VFH+** (Vector Field Histogram Plus) | Light compute (~1ms), handles narrow passages, proven on mobile robots | Requires parameter tuning | **Selected** |
| DWA (Dynamic Window Approach) | Kinematically optimal trajectories | More compute-intensive, designed for differential-drive | Not ideal for Go2's holonomic `move(vx,vy,vyaw)` |
| Potential Fields | Conceptually simple | Gets stuck in local minima (U-shaped obstacles) | Too fragile for real environments |
| RL Policies (like ABS) | Can be very agile | Requires training infrastructure, GPU-heavy, hard to debug | Overkill for 0.3 m/s indoor navigation |
| Bug Algorithms | Guaranteed convergence | Too primitive for cluttered indoor environments | No |

### Why VFH+ Fits the Go2

The Unitree Go2 provides a high-level `move(vx, vy, vyaw)` interface - not joint torques or trajectory tracking. VFH+ maps naturally to this:

1. Build a **polar histogram** of obstacle density (72 sectors of 5 degrees each) from the ObstacleGrid
2. Identify **candidate free sectors** (obstacle count below threshold)
3. Select the **sector closest to goal direction**
4. Compute a smooth **velocity command** toward the selected sector
5. **Modulate speed** based on nearest obstacle distance

### Speed Modulation Curve

```
Speed (m/s)
  |
  |  MAX_SPEED (0.3) ───────────────┐
  |                                  \
  |                                   \  (linear ramp)
  |                                    \
  |  0 ─────────────────────────────────┘
  +──────────────────────────────────────> Distance (m)
     0     SAFETY (0.4m)  AVOIDANCE (0.8m)
```

- **Beyond 0.8m**: Full speed toward goal
- **0.4m - 0.8m**: Linearly decreasing speed (caution zone)
- **Below 0.4m**: Emergency stop (safety zone)

---

## 7. Safety Architecture

Inspired by CMU ABS's safety-supervisor pattern. The key principle: **every velocity command passes through a SafetyMonitor before reaching Go2Macros**. No unsafe command can ever reach the robot, regardless of bugs in the planner.

### Safety Hierarchy

| Priority | Condition | Action | Recovery |
|----------|-----------|--------|----------|
| 1 - E-STOP | Obstacle within 0.4m | Immediate `stop_move()`, state -> E_STOP | Explicit `resume()` call |
| 2 - CLIFF | Ground plane missing ahead (step/stair detected by depth) | Stop, state -> E_STOP | Manual override or path replan |
| 3 - STUCK | No progress for 10 seconds | Stop, state -> STUCK | `recovery_stand()` + replan or cancel |
| 4 - TIMEOUT | Goal timeout exceeded | Stop, state -> IDLE | Report failure to agent |
| 5 - BOUNDARY | Robot leaving known map area | Reduce speed, warn agent | Agent decides next action |

### Implementation Pattern

```python
class SafetyMonitor:
    def filter_command(
        self,
        cmd: VelocityCommand,
        nearest_obstacle_m: float,
        ground_plane_valid: bool,
        seconds_since_progress: float,
    ) -> Tuple[VelocityCommand, Optional[str]]:
        """
        Filter a velocity command through safety checks.
        Returns (filtered_command, safety_event_or_None).
        If a safety event fires, the command is zeroed out.
        """
```

The safety monitor is called on **every iteration** of the 10 Hz navigation loop, forming an inescapable gate between the planner and the actuator.

---

## 8. 4D LiDAR Integration

The Go2 EDU has a built-in **Unitree L1 4D LiDAR** that is not yet accessed in the codebase.

### LiDAR Specifications

| Parameter | Value |
|-----------|-------|
| Coverage | 360 degrees |
| Range | ~25 meters |
| Update rate | ~20 Hz |
| Points per scan | ~21,600 |
| Access | DDS topic `rt/utlidar/range_data` or `ObstaclesClient` |

### Integration Plan (Phase 4)

1. **Subscribe** to the LiDAR DDS topic via `unitree_sdk2py.go2.obstacles.ObstaclesClient`
2. **Project** 3D point cloud to 2D obstacle grid (same format as depth camera output)
3. **Merge** with depth camera grid:
   - LiDAR: 360-degree coarse coverage (detects obstacles behind and to the sides)
   - Depth camera: Dense forward-facing coverage (~87 degree FOV, high resolution)
4. **Result**: Comprehensive obstacle awareness around the entire robot

### Why Defer to Phase 4

- The depth camera alone provides sufficient forward-facing obstacle detection for initial navigation
- LiDAR DDS subscription requires SDK exploration and testing on actual hardware
- Adding LiDAR later is additive (merge into existing ObstacleGrid) - no architecture changes needed

---

## 9. Compute Budget (Orin Nano 40 TOPS)

### Hardware Specs

- 1024-core Ampere GPU, 40 INT8 TOPS
- 8GB unified LPDDR5 memory
- 6 ARM Cortex-A78AE CPU cores

### Workload Analysis

| Component | Time per cycle | Resource | Notes |
|-----------|---------------|----------|-------|
| YOLO TensorRT (existing) | 5-10ms | GPU, ~2GB VRAM | Already running for vision_core |
| DeepFace (existing, periodic) | 30-50ms | GPU + CPU | Every 10th detection cycle |
| **Depth processing** | **~10ms** | **CPU (numpy)** | Downsample + threshold + project + inflate |
| **LiDAR processing** | **~5ms** | **CPU** | Point cloud filter + project (Phase 4) |
| **Local planner (VFH+)** | **~1ms** | **CPU** | Histogram build + sector selection |
| **Global planner (Dijkstra)** | **~1ms** | **CPU** | Only on new goal or replan |
| **Odometry subscription** | **~0ms** | **DDS callback** | Async, negligible processing |
| **Safety monitor** | **~0.1ms** | **CPU** | Simple distance comparisons |
| **Total nav_core** | **~17ms** | **CPU only** | **Fits comfortably in 10 Hz (100ms) loop** |

### Key Insight

Nav_core is **CPU-only**. It leaves the GPU entirely available for vision_core's YOLO TensorRT inference. The 10 Hz navigation loop and the vision pipeline can run concurrently without resource contention.

---

## 10. Architecture Overview

```
                        +-----------------------+
                        |   conscious_agent     |
                        |   (HOCON front-man)   |
                        +-----------+-----------+
                                    |
              +----------+----------+----------+----------+
              |          |          |          |          |
         robot_macros  learn_face  memory   nav_planner  nav_status
              |                               |           |
              |                    +----------+-----------+
              |                    |
              |              NavCore (singleton, background thread)
              |                    |
              |         +----------+----------+-----------+
              |         |          |          |           |
              |    LocalPlanner  Global   Safety     Odometry
              |     (VFH+)     Planner   Monitor    Provider
              |         |       (Dijkstra)    |
              |         |          |          |
              |    ObstacleGrid  Topological  |
              |         |         Map         |
              |    +----+----+               |
              |    |         |               |
              | DepthProc  LidarProc         |
              |    |         |               |
              | Depth Cam  4D LiDAR     SportModeState
              |    (USB)   (built-in)    (DDS topic)
              |
              +-----> Go2Macros.move(vx, vy, vyaw)
```

### Data Flow Summary

1. **Sensors** (depth camera, LiDAR, odometry) feed raw data into processors
2. **DepthProcessor** and **LidarProcessor** produce an **ObstacleGrid**
3. **GlobalPlanner** produces waypoint sequences from the **TopologicalMap**
4. **LocalPlanner** (VFH+) combines ObstacleGrid + waypoint direction into velocity commands
5. **SafetyMonitor** filters every velocity command before it reaches the robot
6. **Go2Macros.move(vx, vy, vyaw)** sends the command to the Unitree SDK

---

## 11. Module Decomposition

### New Files

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `coded_tools/unigo2/nav_core.py` | Main navigation engine: state machine, nav loop, LocalPlanner (VFH+), GlobalPlanner (Dijkstra), SafetyMonitor, OdometryProvider | ~1500-2000 |
| `coded_tools/unigo2/depth_processor.py` | Depth camera access (pyrealsense2), obstacle grid generation, LiDAR processing, ground plane detection | ~600-800 |
| `coded_tools/unigo2/nav_planner.py` | Neuro SAN CodedTool wrapper for agent-initiated navigation commands | ~150-200 |
| `coded_tools/unigo2/nav_status.py` | Neuro SAN CodedTool wrapper for querying navigation state | ~80-120 |
| `maps/cail_lab.json` | Initial topological map for the CAIL lab | ~50 |
| `tests/test_nav_core.py` | Unit tests: local planner, global planner, state machine, safety monitor | ~400-600 |
| `tests/test_depth_processor.py` | Unit tests: depth frame processing, obstacle grid generation | ~200-300 |

### Modified Files

| File | Change |
|------|--------|
| `registries/conscious_agent.hocon` | Add `nav_planner` and `nav_status` tool registrations, update agent instructions |
| `requirements.txt` | Add `pyrealsense2>=2.50` (Linux only) and `scipy>=1.10` |
| `apps/conscious_assistant/scene_observer.py` | Optional: enrich observations with navigation spatial data |

---

## 12. Key Data Structures

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import numpy as np


class NavState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    AVOIDING = "avoiding"
    STUCK = "stuck"
    E_STOP = "e_stop"


@dataclass
class RobotPose:
    """Robot position in local odometry frame."""
    x: float = 0.0       # meters forward from start
    y: float = 0.0       # meters left from start
    yaw: float = 0.0     # radians, CCW from start heading
    timestamp: float = 0.0


@dataclass
class NavGoal:
    """A navigation target."""
    goal_type: str        # "pose", "relative", "semantic"
    x: float = 0.0        # target x (for pose/relative goals)
    y: float = 0.0        # target y (for pose/relative goals)
    yaw: Optional[float] = None  # target heading (optional)
    label: str = ""       # semantic label (e.g., "kitchen")
    timeout_s: float = 60.0


@dataclass
class ObstacleGrid:
    """2D local obstacle map centered on the robot."""
    grid: np.ndarray      # shape (rows, cols), dtype float32, 0.0=free, 1.0=occupied
    resolution: float     # meters per cell (e.g., 0.05 = 5cm)
    origin_row: int       # robot's row in grid
    origin_col: int       # robot's col in grid
    timestamp: float


@dataclass
class VelocityCommand:
    """Velocity command to send to Go2Macros."""
    vx: float = 0.0      # forward m/s
    vy: float = 0.0      # lateral m/s
    vyaw: float = 0.0    # yaw rate rad/s


@dataclass
class MapNode:
    """A named location in the environment."""
    name: str
    x: float
    y: float
    description: str = ""
    tags: List[str] = None


@dataclass
class MapEdge:
    """A traversable connection between two nodes."""
    from_node: str
    to_node: str
    distance: float
    traversable: bool = True
    description: str = ""
```

---

## 13. Navigation State Machine

```
                    +------------+
          start --> |   IDLE     | <-- goal reached / cancel
                    +-----+------+
                          |
                    navigate_to() or move_relative()
                          |
                    +-----v------+
             +----> | NAVIGATING | ----+
             |      +-----+------+     |
             |            |            obstacle within avoidance zone
          replanned       |            |
             |      +-----v------+    |
             +------| AVOIDING   |<---+
                    +-----+------+
                          |
                    cannot clear for STUCK_TIMEOUT (10s)
                          |
                    +-----v------+
                    |  STUCK     | --> recovery_stand() + replan or cancel
                    +------------+

        any state --> E_STOP (safety trigger) --> requires explicit resume()
```

### State Transitions

| From | To | Trigger |
|------|----|---------|
| IDLE | NAVIGATING | `navigate_to()` or `move_relative()` called |
| NAVIGATING | IDLE | Goal reached (within 0.3m tolerance) |
| NAVIGATING | AVOIDING | Obstacle detected in avoidance zone (0.4m - 0.8m) |
| AVOIDING | NAVIGATING | Obstacle cleared, path to goal open |
| AVOIDING | STUCK | Unable to clear obstacle for 10 seconds |
| STUCK | NAVIGATING | After recovery_stand() + successful replan |
| STUCK | IDLE | Navigation cancelled |
| Any | E_STOP | Obstacle within 0.4m safety distance |
| E_STOP | IDLE | Explicit `resume()` call |
| Any | IDLE | `stop()` called |

---

## 14. Navigation Loop (10 Hz)

```python
def _nav_loop(self):
    """Background navigation loop running at NAV_LOOP_HZ."""
    while self._running:
        cycle_start = time.monotonic()

        # 1. Read sensors
        obstacle_grid = self._depth_processor.get_obstacle_grid()
        pose = self._odometry.get_pose()

        # 2. Safety pre-check
        nearest_dist = self._get_nearest_obstacle(obstacle_grid)
        if nearest_dist <= SAFETY_DISTANCE_M:
            self._go2.stop_move()
            self._state = NavState.E_STOP
            self._log("E-STOP: obstacle at %.2fm", nearest_dist)
            continue

        # 3. Plan based on current state
        if self._state == NavState.NAVIGATING:
            waypoint = self._global_planner.get_next_waypoint(pose)
            if waypoint is None:
                self._state = NavState.IDLE  # goal reached
                self._go2.stop_move()
                self._log("Goal reached")
                continue
            goal_dir = math.atan2(waypoint.y - pose.y, waypoint.x - pose.x) - pose.yaw
            goal_dist = math.hypot(waypoint.x - pose.x, waypoint.y - pose.y)
            cmd = self._local_planner.compute_velocity(obstacle_grid, goal_dir, goal_dist)

        elif self._state == NavState.AVOIDING:
            cmd = self._local_planner.compute_avoidance(obstacle_grid)
            if self._avoidance_clear(obstacle_grid):
                self._state = NavState.NAVIGATING

        else:
            cmd = VelocityCommand(0.0, 0.0, 0.0)

        # 4. Safety filter (final gate before actuator)
        cmd, event = self._safety.filter_command(cmd, nearest_dist)
        if event:
            self._log("Safety event: %s", event)

        # 5. Execute
        if self._state in (NavState.NAVIGATING, NavState.AVOIDING):
            self._go2.move(cmd.vx, cmd.vy, cmd.vyaw)

        # 6. Rate limit to NAV_LOOP_HZ
        elapsed = time.monotonic() - cycle_start
        sleep_time = (1.0 / NAV_LOOP_HZ) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
```

---

## 15. CodedTool Integration (HOCON)

### NavPlanner Tool Registration

Add to `registries/conscious_agent.hocon`:

```hocon
{
    "name": "nav_planner",
    "function": {
        "description": "Navigate the robot to a destination or control its movement through the environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Navigation command: 'navigate_to', 'move_forward', 'turn', 'stop', 'status'"
                },
                "target": {
                    "type": "string",
                    "description": "Target destination name (for navigate_to) or direction (for turn: 'left', 'right')"
                },
                "distance": {
                    "type": "number",
                    "description": "Distance in meters (for move_forward) or angle in degrees (for turn)"
                }
            },
            "required": ["command"]
        }
    },
    "class": "unigo2.nav_planner.NavPlannerTool"
}
```

### NavStatus Tool Registration

```hocon
{
    "name": "nav_status",
    "function": {
        "description": "Query the robot's navigation state, available destinations, and obstacle information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to query: 'status', 'destinations', 'obstacles'"
                }
            },
            "required": ["query"]
        }
    },
    "class": "unigo2.nav_status.NavStatusTool"
}
```

### Agent Instructions Addition

Add to the conscious_agent instructions:

```
If the user asks you to go somewhere, move to a location, or navigate, use your nav_planner tool.
You can check your surroundings and navigation progress using nav_status.
Available navigation commands: navigate_to (go to a named place), move_forward (go straight),
turn (rotate), stop (cancel navigation).
When navigating, you will be informed when you arrive or if navigation fails.
```

### Tools List Update

```hocon
"tools": ["commit_to_memory", "recall_memory", "list_topics", "reorganize_memory",
          "robot_macros", "learn_face", "nav_planner", "nav_status"]
```

---

## 16. Graceful Degradation

Following the project's existing pattern where every module handles failures gracefully (vision_core works without TensorRT, go2_macros works in simulation mode):

| Missing Component | Fallback Behavior |
|---|---|
| Depth camera unavailable | Fall back to YOLO-based distance estimates from vision_core (approximate, bbox bottom-edge heuristic) |
| LiDAR unavailable | Depth camera only (forward-facing ~87 degree FOV) |
| Both depth + LiDAR unavailable | Navigation disabled; nav_planner returns error message to agent |
| Odometry (SDK SportModeState) | Dead-reckoning from velocity commands and time (very approximate, drift-prone) |
| Topological map not loaded | Only relative navigation ("move forward 2m", "turn left") available; semantic goals return error |
| Go2Macros not initialized | Simulation mode: commands logged but not executed (consistent with go2_macros.py offline behavior) |
| pyrealsense2 not installed | Depth processing falls back to OpenCV depth stream or vision-only mode |

---

## 17. Environment Variables

Following existing patterns from `vision_core.py` (`_env_flag()`, `_env_float()`):

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `NAV_ENABLED` | bool | `False` | Master enable for navigation subsystem |
| `NAV_LOOP_HZ` | int | `10` | Navigation loop frequency |
| `NAV_SAFETY_DISTANCE` | float | `0.4` | E-stop trigger distance (meters) |
| `NAV_AVOIDANCE_DISTANCE` | float | `0.8` | Avoidance start distance (meters) |
| `NAV_MAX_LINEAR_SPEED` | float | `0.3` | Maximum forward speed (m/s) |
| `NAV_MAX_YAW_RATE` | float | `0.5` | Maximum rotation speed (rad/s) |
| `NAV_DEPTH_CAMERA_SOURCE` | str | `"auto"` | Depth camera device identifier |
| `NAV_USE_LIDAR` | bool | `True` | Attempt to subscribe to LiDAR DDS topic |
| `NAV_MAP_FILE` | str | `""` | Path to topological map JSON file |
| `NAV_SIMULATION_MODE` | bool | `False` | Desktop testing with synthetic obstacles |
| `NAV_GOAL_TOLERANCE` | float | `0.3` | Distance to consider goal reached (meters) |
| `NAV_STUCK_TIMEOUT` | float | `10.0` | Seconds without progress before STUCK state |

---

## 18. Dependencies

### New Python Packages

Add to `requirements.txt`:

```
# Navigation dependencies (nav_core)
pyrealsense2>=2.50; sys_platform == "linux"   # Intel RealSense depth camera (Jetson only)
scipy>=1.10                                    # Spatial algorithms (Dijkstra, KDTree)
```

### Notes

- `scipy` is likely already present as a transitive dependency of `ultralytics`
- `pyrealsense2` is Linux-only and required only when using an Intel RealSense depth camera
- For macOS development, all sensor inputs are mockable - no hardware-specific packages needed
- `numpy` and `opencv-python` are already in `requirements.txt`

---

## 19. Phased Development Plan

### Phase 1: Foundation - Safe Local Navigation (Week 1-2)

**Goal**: Robot can move forward while avoiding obstacles detected by the depth camera.

**Deliverables**:
- `depth_processor.py`: Depth camera access + ObstacleGrid generation
- `nav_core.py`: NavCore state machine, LocalPlanner (VFH+), SafetyMonitor
- `tests/test_nav_core.py` and `tests/test_depth_processor.py`
- Working `NavCore.move_relative(distance, angle)` command
- Emergency stop via `NavCore.stop()`

**Testing**: On-robot: `NavCore.move_relative(2.0, 0.0)` - move 2m forward, stopping if obstacles appear.

### Phase 2: Agent Integration (Week 3)

**Goal**: The conscious_agent can command navigation through Neuro SAN tool interface.

**Deliverables**:
- `nav_planner.py`: CodedTool wrapper for navigation commands
- `nav_status.py`: CodedTool wrapper for status queries
- Updated `conscious_agent.hocon` with nav tools
- Agent can say "move forward 2 meters" and it works end-to-end

### Phase 3: Global Planning (Week 4-5)

**Goal**: Robot can navigate to named locations using a topological map.

**Deliverables**:
- TopologicalMap and GlobalPlanner classes in nav_core
- OdometryProvider (DDS subscription to `rt/sportmodestate`)
- `maps/cail_lab.json` initial map
- "Go to the kitchen" produces waypoint-following navigation

### Phase 4: LiDAR Integration (Week 6)

**Goal**: Merge LiDAR data with depth camera for 360-degree obstacle awareness.

**Deliverables**:
- LidarProcessor in depth_processor.py
- LiDAR + depth grid merging
- Obstacles detected behind and to the sides of the robot

### Phase 5: Advanced Features (Week 7+)

**Potential additions** (prioritize based on need):
- **Map learning**: Robot explores environment and builds topological map by recording positions
- **Visual landmarks**: Associate YOLO detections with map nodes ("the node near the potted plant")
- **Step/stair detection**: Depth camera ground plane discontinuity analysis
- **Follow person**: Use YOLO person detection + depth to follow at a fixed distance
- **Return to charger**: Navigate to charging station on low battery (if battery state available via SDK LowState topic)

---

## 20. Testing Strategy

### Unit Tests (No Hardware Required)

Following existing test patterns from `tests/test_scene_observer.py`:

```python
class TestLocalPlanner(unittest.TestCase):
    def test_drives_toward_goal_in_clear_space(self):
        """With no obstacles, planner drives straight toward goal."""
        grid = ObstacleGrid(
            grid=np.zeros((80, 80), dtype=np.float32),
            resolution=0.05, origin_row=40, origin_col=40, timestamp=0.0
        )
        planner = LocalPlanner()
        cmd = planner.compute_velocity(grid, goal_direction=0.0, goal_distance=2.0)
        assert cmd.vx > 0.0
        assert abs(cmd.vyaw) < 0.1

    def test_stops_when_obstacle_at_safety_distance(self):
        """E-stop when obstacle is within safety distance."""
        grid = _create_grid_with_wall_ahead(distance_m=0.3)
        planner = LocalPlanner()
        cmd = planner.compute_velocity(grid, goal_direction=0.0, goal_distance=2.0)
        assert cmd.vx == 0.0

    def test_avoids_left_when_wall_on_right(self):
        """Steers left when wall blocks right side."""
        grid = _create_grid_with_wall_right()
        planner = LocalPlanner()
        cmd = planner.compute_velocity(grid, goal_direction=0.3, goal_distance=2.0)
        assert cmd.vyaw > 0.0  # positive vyaw = turn left (CCW)

class TestGlobalPlanner(unittest.TestCase):
    def test_dijkstra_finds_shortest_path(self):
        topo_map = _create_test_map()
        planner = GlobalPlanner(topo_map)
        path = planner.plan_path(RobotPose(0, 0, 0), "kitchen")
        assert path is not None
        assert path[-1].name == "kitchen"

    def test_unreachable_returns_none(self):
        topo_map = _create_disconnected_map()
        planner = GlobalPlanner(topo_map)
        path = planner.plan_path(RobotPose(0, 0, 0), "isolated_room")
        assert path is None

class TestSafetyMonitor(unittest.TestCase):
    def test_estop_at_close_distance(self):
        safety = SafetyMonitor()
        cmd = VelocityCommand(vx=0.3, vy=0.0, vyaw=0.0)
        filtered, event = safety.filter_command(cmd, nearest_obstacle_m=0.2)
        assert filtered.vx == 0.0
        assert event is not None

    def test_speed_ramp_in_avoidance_zone(self):
        safety = SafetyMonitor()
        cmd = VelocityCommand(vx=0.3, vy=0.0, vyaw=0.0)
        filtered, _ = safety.filter_command(cmd, nearest_obstacle_m=0.6)
        assert 0.0 < filtered.vx < 0.3  # reduced but not zero
```

### Integration Tests (On Robot)

Gated by `NAV_INTEGRATION_TEST=1` environment variable:

```python
@unittest.skipUnless(os.environ.get("NAV_INTEGRATION_TEST"), "Robot required")
class TestNavCoreOnRobot(unittest.TestCase):
    def test_move_forward_1_meter(self):
        nav = NavCore.get_instance()
        nav.move_relative(1.0, 0.0)
        time.sleep(15)
        assert nav.state == NavState.IDLE

    def test_emergency_stop(self):
        nav = NavCore.get_instance()
        nav.move_relative(5.0, 0.0)
        time.sleep(1)
        nav.stop()
        assert nav.state == NavState.IDLE
```

---

## 21. Open Source Repos Reference

These repositories serve as **implementation references**, not direct integrations:

| Repository | What to learn from it |
|------------|----------------------|
| **[BotBrain](https://github.com/botbotrobotics/BotBrain)** | Go2 + Orin Nano + dual RealSense proven combination; RTABMap SLAM configuration; Nav2 parameter tuning for quadrupeds |
| **[isaac-go2-ros2](https://github.com/Zhefan-Xu/isaac-go2-ros2)** | Isaac Sim setup for Go2; RL agent integration; sensor simulation for testing |
| **[ABS](https://github.com/LeCAR-Lab/ABS)** | Safety-supervisor architecture; reach-avoid value network concept; sim-to-real deployment on quadrupeds |
| **[Nav2](https://github.com/ros-navigation/navigation2)** | Layered planner architecture; VFH/DWA algorithm reference implementations; behavior tree patterns for complex navigation |
| **[unitree_ros2](https://github.com/unitreerobotics)** | Official ROS2 wrapper for Unitree SDK; DDS topic names and message types; SportModeState field definitions |
