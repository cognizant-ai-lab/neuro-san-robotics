# Nav Core: Navigation Module Design for CAIL-E (Unitree Go2 EDU)

## 0. Design Principle: Real-Time Autonomy

Nav_core runs as an **independent real-time loop at 10 Hz**. Once a goal is set (e.g., "go to kitchen"), the navigation loop makes all obstacle avoidance and path-following decisions autonomously — it never waits for an LLM agent response. The Neuro SAN agent layer (nav_planner CodedTool) is a thin command interface that sets goals and queries status, but does not participate in real-time move decisions. Nav_core is fully usable as a standalone Python module without the agent framework.

---

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
- **Hybrid semantic + metric map**: named destinations remain simple JSON nodes,
  while a compact floor-plan occupancy grid prevents routes through walls
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

The depth camera (Intel RealSense D435i or similar USB depth camera) is accessed via `pyrealsense2`. The scene observer prefers the RealSense color camera through `open_camera()` so the visual scene and depth grid are aligned during navigation tests.

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
- **Output**: Collision-free metric targets ending at the semantic destination
- **Algorithm**: Clearance-aware A* over the static floor plan, with current
  RealSense obstacles overlaid during initial planning and replanning
- **Frequency**: Once per goal, proactively after persistent blockage, and after
  a no-progress recovery
- **Compute**: Normally under 200ms for the full office map

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

## 5. Global Planning: Hybrid Semantic and Occupancy Maps

### Why a floor-plan occupancy map instead of waypoint-only routing or metric SLAM?

| Factor | Topological Map | Metric SLAM (e.g., RTABMap) |
|--------|----------------|---------------------------|
| Compute cost | ~0 (JSON file) | High (continuous processing, 2-4GB RAM) |
| Orin Nano feasibility | Easy | Tight alongside YOLO TensorRT |
| Creation | Hand-editable JSON or robot-learned | Automatic but needs calibration |
| Maintenance | Simple edits | Complex loop closure management |
| Accuracy | Approximate (sufficient for room-level) | Precise (centimeter-level) |
| Sufficient for "go to room X" | Yes | Overkill |

**Decision**: Use named topological nodes only for destinations and human-readable
status. Use a 10cm static occupancy grid for actual global routing. This supplies
wall geometry without the compute and operational complexity of continuous SLAM.
The local depth grid handles people, chairs, calibration error, and other live
changes. Scan-to-map matching applies only small, high-confidence odometry
corrections.

### Map Structure

Maps are stored as JSON files in a `maps/` directory. Each map contains named
locations and references a generated NPZ occupancy grid. Edges remain available
as a fallback for small test/simulation maps that do not declare occupancy.

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

Clearance-aware A* finds a route through free floor-plan cells and penalizes
cells close to walls, naturally favoring the middle of an opening. Line-of-sight
smoothing and targets spaced about 0.8m apart keep the robot moving efficiently.
Dijkstra remains the compatibility fallback for maps without an occupancy grid.

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
  |  MAX_SPEED (0.40) ──────────────┐
  |                                  \
  |                                   \  (linear ramp)
  |                                    \
  |  0 ─────────────────────────────────┘
  +──────────────────────────────────────> Distance (m)
     0     SAFETY (0.20m)  AVOIDANCE (0.60m)
```

- **Beyond 0.60m**: Full planned speed toward goal
- **0.20m - 0.60m**: Linearly decreasing speed in the path corridor
- **At or below 0.20m**: Stop after close-obstacle confirmation

---

## 7. Safety Architecture

Inspired by CMU ABS's safety-supervisor pattern. The key principle: **every velocity command passes through a SafetyMonitor before reaching Go2Macros**. No unsafe command can ever reach the robot, regardless of bugs in the planner.

### Safety Hierarchy

| Priority | Condition | Action | Recovery |
|----------|-----------|--------|----------|
| 1 - E-STOP | Confirmed path obstacle within 0.20m | `stop_move()`, state -> E_STOP | Explicit `resume()` call |
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

## 8. Glass Wall and Transparent Obstacle Detection

Glass walls are a known challenge for robotic navigation — they are invisible to most sensors.

### Sensor Capabilities Against Glass

| Sensor | Detects Glass? | Notes |
|--------|---------------|-------|
| RGB camera (YOLO) | No | Glass is visually transparent |
| Depth camera (RealSense IR) | Partially | Active IR reflects off glass at some angles; shows as noisy/flickering depth or invalid regions |
| 4D LiDAR | No | Laser passes through glass |
| Ultrasonic | **Yes** | Sound waves reflect off glass reliably — this is the primary solution |
| Foot force sensors | **Yes** (reactive) | Detects unexpected contact after the fact |

### Detection Strategy (Multi-Layered)

1. **Depth camera anomaly detection** (primary, proactive): When the depth camera returns "no valid depth" in a region where floor should be visible (based on camera geometry), treat it as a potential transparent obstacle. Glass causes characteristic patterns: valid depth on either side but a void in the middle. This heuristic catches most indoor glass walls.

2. **Ultrasonic cross-check** (if accessible via Go2 SDK): The Go2 EDU has built-in ultrasonic sensors. Ultrasound reliably reflects off glass. If the depth camera shows "clear" but ultrasonic shows "blocked," flag it as a transparent obstacle.

3. **Map annotations** (static): Known glass walls can be marked in the topological map with a `"transparent_wall"` tag, so the global planner routes around them.

4. **Contact-based learning** (reactive): If foot force sensors detect unexpected contact with no obstacle in the grid, the robot stops, backs up, and marks the location as a transparent obstacle in its spatial memory for future avoidance.

### No-Bot Zones

A complementary approach: define **exclusion zones** where the robot should never navigate, regardless of sensor readings. Useful for:
- Known glass walls and transparent barriers
- Restricted areas (server rooms, executive offices)
- Unsafe zones (stairs, loading docks, wet floors)

Implementation: Add `"exclusion_zones"` to the topological map JSON:
```json
{
    "exclusion_zones": [
        {
            "name": "glass_wall_conference_room",
            "type": "line",
            "points": [[2.0, 3.0], [2.0, 6.0]],
            "buffer_m": 0.5
        },
        {
            "name": "server_room",
            "type": "rectangle",
            "min": [5.0, 1.0], "max": [8.0, 3.0]
        }
    ]
}
```

The safety monitor checks if the robot's projected path enters any exclusion zone on every nav cycle. The global planner avoids routing through them. These zones can be defined manually or learned from contact events (see detection strategy #4 above).

### Implementation

The depth camera anomaly detection is implementable in Phase 1 as part of the DepthProcessor pipeline: after the ground plane removal step, check for "void regions" (contiguous areas of invalid depth surrounded by valid depth at similar distances). These are likely glass surfaces. Contact-learned obstacles and no-bot-zones are added in Phase 5 as part of spatial memory.

---

## 9. 4D LiDAR Integration

The Go2 EDU has a built-in **Unitree L1 4D LiDAR** that is not yet accessed in the codebase.

### LiDAR Specifications

| Parameter | Value |
|-----------|-------|
| Coverage | 360 degrees |
| Range | ~25 meters |
| Update rate | ~20 Hz |
| Points per scan | ~21,600 |
| Access | DDS topic `rt/utlidar/cloud` as `sensor_msgs/PointCloud2` |

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

## 10. Compute Budget (Orin Nano 40 TOPS)

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
| **Global planner (occupancy A*)** | **5-200ms** | **CPU** | Only on new goal or replan; outside the 10Hz hot path |
| **Depth-to-map correction** | **periodic** | **CPU** | Bounded scan match every 2 seconds |
| **Odometry subscription** | **~0ms** | **DDS callback** | Async, negligible processing |
| **Safety monitor** | **~0.1ms** | **CPU** | Simple distance comparisons |
| **Total nav_core** | **~17ms** | **CPU only** | **Fits comfortably in 10 Hz (100ms) loop** |

### Key Insight

Nav_core is **CPU-only**. It leaves the GPU entirely available for vision_core's YOLO TensorRT inference. The 10 Hz navigation loop and the vision pipeline can run concurrently without resource contention.

---

## 11. Architecture Overview

```
                        +-----------------------+
                        |   conscious_agent     |
                        |   (HOCON front-man)   |
                        +-----------+-----------+
                                    |
              +----------+----------+----------+----------+
              |          |          |          |          |
         robot_macros  learn_face  memory   nav_planner
              |                               |
              |                    |
              |              NavCore (singleton, background thread)
              |                    |
              |         +----------+----------+-----------+
              |         |          |          |           |
              |    LocalPlanner  Global   Safety     Odometry
              |     (VFH+)     Planner   Monitor    Provider
              |         |         (A*)       |
              |         |          |          |
              |    ObstacleGrid  Occupancy +  |
              |         |       semantic map  |
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
3. **GlobalPlanner** produces collision-free metric targets from the static
   **occupancy map**, overlaying the current ObstacleGrid during replans
4. **LocalPlanner** (VFH+) combines ObstacleGrid + waypoint direction into velocity commands
5. **SafetyMonitor** filters every velocity command before it reaches the robot
6. **Go2Macros.move(vx, vy, vyaw)** sends the command to the Unitree SDK
7. **SpatialMemory** (Phase 5) captures and stores spatial snapshots at key positions, enabling the robot to remember and update its understanding of the environment across sessions

---

## 12. Module Decomposition

### New Files

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `coded_tools/unigo2/nav_core.py` | Main navigation engine: state machine, local planner, global-planner integration, safety, recovery, and odometry | ~1500-2000 |
| `coded_tools/unigo2/metric_navigation.py` | Occupancy A*, route smoothing, dynamic overlays, and conservative scan-to-map matching | ~500 |
| `maps/build_occupancy_map.py` | Offline floor-plan-to-occupancy generator; runtime does not require Pillow | ~200 |
| `coded_tools/unigo2/depth_processor.py` | Depth camera access (pyrealsense2), obstacle grid generation, LiDAR processing, ground plane detection | ~600-800 |
| `coded_tools/unigo2/nav_planner.py` | Neuro SAN CodedTool wrapper for navigation commands and queries | ~150-200 |
| `maps/cail_lab.json` | Initial topological map for the CAIL lab | ~50 |
| `tests/test_nav_core.py` | Unit tests: local planner, global planner, state machine, safety monitor | ~400-600 |
| `tests/test_depth_processor.py` | Unit tests: depth frame processing, obstacle grid generation | ~200-300 |

### Modified Files

| File | Change |
|------|--------|
| `registries/conscious_agent.hocon` | Add the `nav_planner` tool registration and navigation-event instructions |
| `requirements.txt` | Add `pyrealsense2>=2.50` (Linux only) and `scipy>=1.10` |
| `apps/conscious_assistant/scene_observer.py` | Optional: enrich observations with navigation spatial data |

---

## 13. Key Data Structures

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

## 14. Navigation State Machine

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
| NAVIGATING | IDLE | Goal reached (within 0.15m tolerance) |
| NAVIGATING | AVOIDING | Obstacle detected in avoidance zone (0.20m - 0.60m) |
| AVOIDING | NAVIGATING | Obstacle cleared, path to goal open |
| AVOIDING | STUCK | Unable to clear obstacle for 10 seconds |
| STUCK | NAVIGATING | After recovery_stand() + successful replan |
| STUCK | IDLE | Navigation cancelled |
| Any | E_STOP | Confirmed path obstacle within 0.20m safety distance |
| E_STOP | IDLE | Explicit `resume()` call |
| Any | IDLE | `stop()` called |

---

## 15. Navigation Loop (10 Hz)

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

## 16. CodedTool Integration (HOCON)

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

### Agent Instructions Addition

Add to the conscious_agent instructions:

```
If the user asks you to go somewhere, move to a location, or navigate, use your nav_planner tool.
Available navigation commands: navigate_to (go to a named place), move_forward (go straight),
turn (rotate), stop (cancel navigation), status, destinations, and obstacles.
When navigating, waypoint, obstacle, arrival, and failure events are delivered automatically.
```

### Tools List Update

```hocon
"tools": ["commit_to_memory", "recall_memory", "list_topics", "reorganize_memory",
          "robot_macros", "learn_face", "nav_planner"]
```

---

## 17. Graceful Degradation

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

## 18. Environment Variables

Following existing patterns from `vision_core.py` (`_env_flag()`, `_env_float()`):

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `NAV_ENABLED` | bool | `False` | Master enable for navigation subsystem |
| `NAV_LOOP_HZ` | int | `10` | Navigation loop frequency |
| `NAV_SAFETY_DISTANCE` | float | `0.20` | Confirmed stop distance in the swept path corridor (meters) |
| `NAV_AVOIDANCE_DISTANCE` | float | `0.75` | Slowdown and local-steering start distance for path-corridor obstacles (meters) |
| `NAV_CENTER_ONLY_CLOSE_CONFIRM_S` | float | `0.2` | Seconds a grid-disputed close center-depth reading must persist before it can abort navigation |
| `NAV_CENTER_ONLY_CLOSE_CONFIRM_READINGS` | int | `3` | Readings a grid-disputed close center-depth hazard must persist |
| `NAV_CENTER_ONLY_GRID_MARGIN` | float | `0.15` | Grid-clearance margin for identifying center-depth disagreement |
| `NAV_PATH_OBSTACLE_CONFIRM_S` | float | `0.3` | Seconds an avoidance-band path obstacle must persist before planner/safety use it |
| `NAV_PATH_OBSTACLE_CONFIRM_READINGS` | int | `3` | Nav-loop readings an avoidance-band path obstacle must persist before planner/safety use it |
| `NAV_PATH_OBSTACLE_CENTER_DEPTH_MARGIN` | float | `0.15` | Raw center-depth agreement margin for avoidance-band path obstacles |
| `NAV_MAX_LINEAR_SPEED` | float | `0.40` | Maximum planned forward speed (m/s) |
| `NAV_MAX_YAW_RATE` | float | `0.08` | Maximum yaw correction while translating (rad/s) |
| `NAV_PIVOT_YAW_RATE` | float | `0.50` | In-place yaw rate for planned map turns (rad/s) |
| `NAV_DEPTH_CAMERA_SOURCE` | str | `"auto"` | Depth camera device identifier |
| `NAV_USE_LIDAR` | bool | `True` | Attempt to subscribe to LiDAR DDS topic |
| `NAV_MAP_FILE` | str | repo `maps/cail_lab.json` on robot, `""` in simulation | Path to topological map JSON file |
| `NAV_SIMULATION_MODE` | bool | `False` | Desktop testing with synthetic obstacles |
| `NAV_GOAL_TOLERANCE` | float | `0.15` | Distance to consider goal reached (meters) |
| `NAV_STUCK_TIMEOUT` | float | `6.0` | Seconds without progress before guarded physical recovery |
| `NAV_METRIC_BLOCKED_REPLAN_DELAY` | float | `1.0` | Persistent-obstacle delay before proactive metric replan |
| `NAV_METRIC_REPLAN_COOLDOWN` | float | `3.0` | Minimum time between proactive metric replans |
| `NAV_METRIC_LOCALIZATION_INTERVAL` | float | `2.0` | Depth-to-map correction interval |
| `NAV_PATH_CORRIDOR_HALF_WIDTH` | float | `0.27` | Robot half-width plus swept-path clearance (meters) |
| `NAV_WALL_CLEARANCE` | float | `0.55` | Target clearance from a reliable one-sided wall (meters) |
| `NAV_OBSTACLE_MEMORY_SECONDS` | float | `0.8` | Lifetime of odometry-aligned steering geometry |
| `NAV_OBSTACLE_TELEMETRY_SECONDS` | float | `1.0` | Directional-clearance and wall-fit log interval |

---

## 19. Dependencies

### New Python Packages

Add to `requirements.txt`:

```
# Navigation dependencies (nav_core)
pyrealsense2>=2.50; sys_platform == "linux"   # Intel RealSense depth camera (Jetson only)
# Metric planning uses NumPy only; Pillow is needed only to regenerate the map.
```

### Notes

- `scipy` is likely already present as a transitive dependency of `ultralytics`
- `pyrealsense2` is Linux-only and required only when using an Intel RealSense depth camera
- For macOS development, all sensor inputs are mockable - no hardware-specific packages needed
- `numpy` and `opencv-python` are already in `requirements.txt`

---

## 20. Phased Development Plan

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

### Phase 5: 3D Space Scanning & Persistent Spatial Memory (Week 7-9)

**Goal**: The robot can scan a space, store a persistent spatial representation, and update it as the environment changes.

**Why this matters**: Phases 1-4 give the robot reactive obstacle avoidance and pre-defined topological navigation. But for true autonomy, the robot must **build its own understanding of a space** and remember it across power cycles. Without spatial memory, every boot starts from zero.

**Approach: Hybrid Spatial Memory** (not full metric SLAM)

Full SLAM (e.g., RTABMap, ORB-SLAM3) continuously builds and maintains a dense 3D map. This is compute-intensive and requires 2-4GB RAM on top of existing vision workloads — tight on Orin Nano's 8GB. Instead, we use a lighter hybrid approach:

#### 5a. Exploration & Scanning

When commanded to explore (or during idle patrol), the robot:
1. Walks along edges of the topological graph (or performs a systematic sweep in unknown areas)
2. At regular intervals (~every 1-2 meters or when turning), captures a **spatial snapshot**:
   - Downsampled depth frame → compressed 2D occupancy grid (the same ObstacleGrid format used by local planner)
   - Robot pose from odometry (x, y, yaw)
   - YOLO detections visible at this position (landmark candidates)
3. At "interesting" locations (room entrances, intersections, dead ends), creates a new **MapNode** in the topological graph

#### 5b. Persistent Storage Format

```
maps/
├── cail_lab.json                  # Topological graph (nodes + edges)
├── spatial_data/
│   ├── charging_station.npz       # Compressed occupancy grid at this node
│   ├── main_desk_area.npz         # Compressed occupancy grid at this node
│   ├── kitchen.npz                # Compressed occupancy grid at this node
│   └── landmarks.json             # Visual landmarks tied to nodes
```

Each spatial snapshot (`.npz` file) stores:
```python
{
    "grid": np.ndarray,        # 80x80 float32 occupancy grid
    "resolution": 0.05,        # meters per cell
    "pose": [x, y, yaw],       # robot pose when captured
    "timestamp": 1714200000.0, # when captured
    "landmarks": [             # YOLO objects visible at this position
        {"class": "couch", "bearing": 0.3, "distance": 2.1},
        {"class": "potted plant", "bearing": -0.5, "distance": 1.8}
    ]
}
```

Total storage per node: ~30KB (compressed). A 50-node map uses ~1.5MB. Trivial on disk.

#### 5c. Map Updates (Incremental, Not Full Rebuild)

When the robot revisits a known location:

1. **Localization check**: Compare current depth scan to stored spatial signature using normalized cross-correlation or IoU on the occupancy grids
2. **Change detection**: If similarity is below threshold (e.g., < 0.7):
   - Update the stored grid with the new scan
   - Check if new objects block previously-clear edges → mark edges as non-traversable
   - Check if previously-blocked edges are now clear → restore traversability
   - Log the change for agent awareness ("I noticed the hallway to the kitchen is now blocked by a cart")
3. **Landmark update**: Add/remove visual landmarks based on current YOLO detections

This runs in ~15ms per comparison (numpy correlation on 80x80 grids). No GPU needed.

#### 5d. Auto-Discovery of New Nodes

During exploration, if the robot reaches a position that is far (>2m) from any existing MapNode:
1. Create a new MapNode at the current position
2. Connect it to the nearest existing node with an edge
3. Capture spatial snapshot
4. Optionally name it based on visible landmarks ("near_the_couch") or leave as auto-generated ("node_17")
5. Save updated map to disk

This means the topological map **grows organically** as the robot explores, rather than requiring manual creation.

#### 5e. Compute Budget for Scanning

| Operation | Time | When |
|-----------|------|------|
| Capture spatial snapshot | ~12ms | Every 1-2m during exploration |
| Save snapshot to disk | ~5ms | Async, non-blocking |
| Compare to stored snapshot | ~15ms | On revisiting a node |
| Update topological graph | ~1ms | On change detection |
| **Total per node visit** | **~33ms** | **Negligible** |

**Deliverables**:
- `SpatialMemory` class in nav_core: capture, store, compare, update
- `maps/spatial_data/` directory structure
- Exploration command in nav_planner: `{"command": "explore", "area": "unknown"}` or `{"command": "scan_area"}`
- Auto-discovery of new nodes during exploration
- Change detection and edge traversability updates
- Agent notification of spatial changes

### Phase 6: Additional Advanced Features (Week 10+)

**Potential additions** (prioritize based on need):
- **Step/stair detection**: Depth camera ground plane discontinuity analysis
- **Follow person**: Use YOLO person detection + depth to follow at a fixed distance
- **Return to charger**: Navigate to charging station on low battery (if battery state available via SDK LowState topic)
- **Multi-floor maps**: Separate topological graphs per floor, connected by stair/elevator nodes
- **Visual place recognition**: Use vision_core embeddings to recognize revisited locations (loop closure without full SLAM)

---

## 21. Testing Strategy

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

## 22. Open Source Repos Reference

These repositories serve as **implementation references**, not direct integrations:

| Repository | What to learn from it |
|------------|----------------------|
| **[BotBrain](https://github.com/botbotrobotics/BotBrain)** | Go2 + Orin Nano + dual RealSense proven combination; RTABMap SLAM configuration; Nav2 parameter tuning for quadrupeds |
| **[isaac-go2-ros2](https://github.com/Zhefan-Xu/isaac-go2-ros2)** | Isaac Sim setup for Go2; RL agent integration; sensor simulation for testing |
| **[ABS](https://github.com/LeCAR-Lab/ABS)** | Safety-supervisor architecture; reach-avoid value network concept; sim-to-real deployment on quadrupeds |
| **[Nav2](https://github.com/ros-navigation/navigation2)** | Layered planner architecture; VFH/DWA algorithm reference implementations; behavior tree patterns for complex navigation |
| **[unitree_ros2](https://github.com/unitreerobotics)** | Official ROS2 wrapper for Unitree SDK; DDS topic names and message types; SportModeState field definitions |
