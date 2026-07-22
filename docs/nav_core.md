# Nav Core - Autonomous Navigation for CAIL-E

Navigation module for the Unitree Go2 EDU robot. Provides obstacle avoidance,
path planning, and safety monitoring as an autonomous background process.

For full architecture details, see [nav_core_design.md](nav_core_design.md).

---

## Quick Start

### Desktop / Simulation (no robot, no cameras)

```bash
# Run tests
PYTHONPATH=. python -m unittest discover -s tests -p "test_nav_core.py" -v
PYTHONPATH=. python -m unittest discover -s tests -p "test_depth_processor.py" -v

# Interactive check in simulation mode
NAV_SIMULATION_MODE=1 python -c "
from coded_tools.unigo2.nav_core import NavCore
nav = NavCore.get_instance()
print(nav.get_status_summary())
print(nav.list_destinations())
nav.shutdown()
"
```

### On the Robot (Jetson Orin Nano)

```bash
# 1. Connect depth camera to a USB-C port on the Orin Nano
# 2. Verify detection
python -c "
from coded_tools.unigo2.depth_processor import DepthProcessor
dp = DepthProcessor()
print('Backend:', dp.backend)
print(dp.get_obstacle_summary())
"

# 3. Start navigation with the default map
export NAV_ENABLED=1
python -c "
from coded_tools.unigo2.nav_core import NavCore
nav = NavCore.get_instance()
print(nav.list_destinations())
nav.navigate_to('kitchen')
"
```

---

## How It Works

NavCore runs a **background thread at 10 Hz**. Each cycle:

1. Read obstacle grid from depth camera
2. Get robot pose from odometry
3. Safety pre-check (confirmed stop if a path-corridor obstacle is <= 0.10 m)
4. Local planner computes velocity (VFH+ algorithm)
5. Safety monitor filters the command
6. Send velocity to Go2Macros

The agent only sets goals ("go to kitchen"). All real-time obstacle avoidance
happens autonomously in the nav loop -- no LLM round-trips in the control path.

Unitree SportModeState translation and heading are aligned to the map and used
as the authoritative pose. Wireless-controller input pauses autonomous motion;
when control is released, NavCore projects the corrected pose onto the current
route edge, preserves waypoint progress, and continues. `set_location` is only
for unmeasured moves, such as physically carrying the robot to a known place.

The filtered obstacle grid guides local planning, while raw fresh path clearance
independently gates every autonomous motion command. Stale depth data cannot
authorize movement. When two sufficiently long, parallel corridor walls are
visible, their heading and center offset add a small bounded steering correction;
one-sided or inconsistent geometry is ignored.

Robot initialization disables both SportClient `FreeAvoid` and the Unitree
obstacle-avoidance service so firmware steering cannot conflict with NavCore.

```
Agent says "go to kitchen"
        |
        v
  GlobalPlanner: Dijkstra on topological map
        |  [charging_station -> main_desk_area -> kitchen]
        v
  NavCore loop (10 Hz):
    DepthCamera -> ObstacleGrid -> LocalPlanner (VFH+) -> SafetyMonitor -> Go2Macros.move()
```

---

## Depth Camera Setup

The depth camera connects via **USB-C** to the Nvidia Orin Nano. It is completely
independent from the front RGB camera (which uses DDS over ethernet).

### Supported cameras

| Camera | Library | Notes |
|--------|---------|-------|
| Intel RealSense D435i / D455 | `pyrealsense2` | Recommended. Auto-detected. |
| Generic USB depth camera | OpenCV | Fallback. Must output single-channel frames. |

### Auto-detection priority

1. **RealSense** (`pyrealsense2`): scans for connected devices, caches intrinsics
2. **OpenCV V4L2**: scans `/dev/video*`, accepts only single-channel frames (skips the front RGB camera which is network-attached, not V4L2)
3. **Simulation**: if `NAV_SIMULATION_MODE=1`, uses synthetic depth data

### Installing pyrealsense2 on Jetson

```bash
# pyrealsense2 is Linux-only (included in requirements.txt with platform guard)
pip install pyrealsense2
```

### Verify depth camera

```bash
# Check if RealSense is detected
python -c "
import pyrealsense2 as rs
ctx = rs.context()
for d in ctx.query_devices():
    print(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
"
```

---

## Topological Maps

Maps are JSON files that define named locations and connections between them.

### Map format

```json
{
  "name": "My Lab",
  "nodes": [
    {"name": "entrance", "x": 0.0, "y": 0.0, "description": "Front door"},
    {"name": "kitchen",  "x": 3.0, "y": 4.0, "description": "Break area"}
  ],
  "edges": [
    {"from": "entrance", "to": "kitchen", "distance": 5.0}
  ]
}
```

- **x, y**: approximate position in meters (relative to any consistent origin)
- **distance**: edge weight for Dijkstra path planning
- Edges are bidirectional

### Loading a map

```bash
# The default robot map is loaded at startup from maps/cail_lab.json.
# Set NAV_MAP_FILE only when overriding the map path.

# Or programmatically
from coded_tools.unigo2.nav_core import NavCore
nav = NavCore.get_instance()
nav._topo_map.load_from_file("maps/cail_lab.json")
```

### Included map

`maps/cail_lab.json` defines the CAIL Lab in San Francisco with 5 locations:
charging_station, main_desk_area, kitchen, entrance, demo_area.

---

## Agent Integration (Neuro SAN)

One CodedTool exposes navigation to the conscious agent:

### nav_planner

Commands the robot to navigate.

| Command | Parameters | Example |
|---------|-----------|---------|
| `navigate_to` | `target`: destination name | "Go to the kitchen" |
| `move_forward` | `distance`: max meters (0.1-10) | "Move forward 2 meters" |
| `move_until_obstacle` | `distance`: stop distance in meters | "Move forward until something is 0.75 meters ahead" |
| `turn` | `target`: left/right/around, `distance`: degrees | "Turn left 90 degrees" |
| `stop` | — | "Stop moving" |
| `status` | — | State, position, heading, goal, and obstacle summary |
| `destinations` | — | List of mapped locations |
| `obstacles` | — | Current obstacle-sensor summary |

`NavCore` also pushes waypoint, obstacle, arrival, and failure events to the
agent. Queries are intended for explicit questions and diagnostics, not progress polling.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NAV_ENABLED` | `false` | Master enable for navigation |
| `NAV_SIMULATION_MODE` | `false` | Use synthetic depth data (no hardware) |
| `NAV_MAP_FILE` | repo `maps/cail_lab.json` on robot, `""` in simulation | Path to topological map JSON |
| `NAV_LOOP_HZ` | `10` | Navigation loop frequency |
| `NAV_MAX_LINEAR_SPEED` | `0.40` | Max planned forward speed (m/s) |
| `NAV_MAX_YAW_RATE` | `0.08` | Max yaw correction while translating (rad/s) |
| `NAV_PIVOT_YAW_RATE` | `0.50` | In-place yaw rate for planned turns (rad/s) |
| `NAV_SAFETY_DISTANCE` | `0.10` | Confirmed stop distance in the path corridor (meters) |
| `NAV_AVOIDANCE_DISTANCE` | `0.30` | Start slowing down and locally steering for path-corridor obstacles (meters) |
| `NAV_PATH_OBSTACLE_CONFIRM_S` | `0.3` | Seconds an avoidance-band path obstacle must persist before affecting planning |
| `NAV_PATH_OBSTACLE_CONFIRM_READINGS` | `3` | Nav-loop readings an avoidance-band path obstacle must appear in before affecting planning |
| `NAV_PATH_OBSTACLE_CENTER_DEPTH_MARGIN` | `0.15` | Raw center-depth agreement margin for avoidance-band path obstacles |
| `NAV_FORWARD_SPEED` | `0.45` | Continuous guarded forward command speed |
| `NAV_FORWARD_STOP_DISTANCE` | `0.50` | Center-depth watchdog stop distance |
| `NAV_FORWARD_MAX_SECONDS` | `15` | Timeout for move_until_obstacle |
| `NAV_DEPTH_CAMERA_SOURCE` | `auto` | `auto`, `realsense`, or OpenCV device index |
| `NAV_GRID_RESOLUTION` | `0.05` | Obstacle grid cell size (meters) |
| `NAV_CAMERA_MOUNT_HEIGHT` | `0.30` | Depth camera height from ground (meters) |
| `NAV_GOAL_TOLERANCE` | `0.15` | Distance to consider goal reached (meters) |

---

## Safety

Every velocity command passes through the SafetyMonitor before reaching the motors.

| Condition | Action |
|-----------|--------|
| Confirmed path-corridor obstacle within 0.20 m | Stop |
| Ground plane missing (cliff/step) | Stop |
| No progress for 10 seconds | Stop, report stuck |
| All occupied (surrounded) | Stop |

To resume after e-stop: `nav.resume()` or send a new navigation command.

---

## Module Structure

```
coded_tools/unigo2/
  depth_processor.py    # Depth camera -> ObstacleGrid pipeline
  nav_core.py           # Navigation engine (singleton, background thread)
  nav_planner.py        # CodedTool: navigation commands and queries

maps/
  cail_lab.json         # CAIL Lab topological map

tests/
  test_depth_processor.py  # 13 tests
  test_nav_core.py         # 29 tests
```

---

## Running Tests

```bash
# All nav tests (no hardware needed)
PYTHONPATH=. python -m unittest discover -s tests -p "test_nav_core.py" -v
PYTHONPATH=. python -m unittest discover -s tests -p "test_depth_processor.py" -v

# Both at once
PYTHONPATH=. python -m unittest test_nav_core test_depth_processor -v
```

Tests run with pure numpy fallbacks -- no cv2, pyrealsense2, or scipy required.
