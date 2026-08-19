# neuro-san-robotics
Neuro SAN Robotics

## Start the robot

The examples below use CAIL-E, the original unit in San Francisco. A sibling
robot differs only by its `ROBOT_NAME` and `ROBOT_HOME`; see
[Robot identity](#robot-identity).

1. Physically start the robot with a short press followed by a long press
on its power button.
2. Find the robot's IP address and ssh into it:
    ```shell
    ssh unitree@10.194.17.116
    ```
    **Note 1**: its IP address might change. If it does, you'll have to plug in a monitor
    on its back, log in and check its network settings.  
    **Note 2**: Ask around for the password.
3. Navigate to the following folder:
   ```shell
   cd exp/neuro-san-robotics-hormoz
   ```
4. Setup the environment:
   ```shell
   source setmyenv.sh
   ```
5. Start the app
   ```shell
   python apps/conscious_assistant/interface_flask.py
   ```
You can now navigate to https://10.194.17.116:5001 (check the IP address)
to interact with the robot.

The Flask process starts one local Neuro SAN event service on port `8188` before
serving the browser. Flask only transports browser input/output, TTS playback,
voice transcription, and the latest camera image. The native service owns agent
events and periodic autonomous turns; `NavCore` remains the independent real-time
navigation controller. A lightweight Python observer service captures the scene
periodically, updates the UI image, and sends compact `observation:` events to
the agent. Stop the Flask process to stop the service it started.

### Ambient listening mode

The web UI has an **Ambient Off / Ambient On** control beside the microphone.
When enabled, the browser keeps a WebRTC microphone stream connected to OpenAI's
Realtime transcription service. Server voice-activity detection produces complete
utterances, and each is queued to the native agent as an `ambient:` event. There
is no LLM pre-filter and no acknowledgement or automatic speech for these events.
All raw **Heard:** text appears in the Thoughts pane. The robot remains silent unless
the transcript clearly addresses or refers to it; when it responds, the relevant
addressed speech is also promoted into chat like a push-to-talk transcript.
There is no wake-word matcher: the agent decides it was addressed from its own
name in the persona prompt, which is why `ROBOT_NAME` matters for this to work.
Capture pauses while the robot speaks so it does not transcribe its own TTS output.
The Flask backend mints a short-lived Realtime client token; the standard API key
never leaves the robot, and the browser negotiates its WebRTC session directly
with OpenAI. Transient gateway failures are retried once.

Realtime ambient transcription uses `gpt-4o-transcribe` by default. Override
it with `CONSCIOUS_AMBIENT_TRANSCRIPTION_MODEL` if needed.

### Robot environment (`setmyenv.sh`)

Use this as the robot-side `setmyenv.sh`. Keep secrets such as
`OPENAI_API_KEY` outside this tracked file, for example in an untracked local
shell file or as a manual export.

```shell
source venv/bin/activate

export PYTHONPATH="$HOME/librealsense-2.55.1/build/Release:$PWD:$PWD/coded_tools:${PYTHONPATH:-}"
export AGENT_TOOL_PATH="$PWD/coded_tools"
export AGENT_MANIFEST_FILE="$PWD/registries/manifest.hocon"

export CYCLONEDDS_HOME="$PWD/cyclonedds/install"
export CYCLONEDDS_URI="file://$HOME/cyclonedds.xml"

export LD_LIBRARY_PATH="$PWD/cyclonedds/install/lib:$HOME/librealsense-2.55.1/build:${LD_LIBRARY_PATH:-}"

printf 'PYTHONPATH=%s\n' "$PYTHONPATH"
printf 'AGENT_TOOL_PATH=%s\n' "$AGENT_TOOL_PATH"
printf 'AGENT_MANIFEST_FILE=%s\n' "$AGENT_MANIFEST_FILE"
printf 'CYCLONEDDS_HOME=%s\n' "$CYCLONEDDS_HOME"
printf 'CYCLONEDDS_URI=%s\n' "$CYCLONEDDS_URI"
printf 'LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH"
```

The final `printf` block is intentionally limited to local environment wiring:
after `source setmyenv.sh`, the shell prints only the values that differ by
checkout or local install path. Navigation, camera, network-interface, and
behavior tuning defaults live in code and are logged by the app at startup.

#### Robot identity

One checkout serves every unit. Two variables carry everything that differs
between robots, and both belong in the robot's `setmyenv.sh`. The template in
`.env.example` already includes them with CAIL-E as the default; if your
`setmyenv.sh` predates this, add these two lines to it:

```shell
export ROBOT_NAME="${ROBOT_NAME:-CAIL-E}"
export ROBOT_HOME="${ROBOT_HOME:-the Cognizant AI Lab (CAIL) in San Francisco}"
```

On a sibling robot, set them to that unit's values instead:

```shell
export ROBOT_NAME="BIT-2"
export ROBOT_HOME="the Cognizant AI Lab in Bengaluru"
```

| Variable | Reaches | Notes |
| --- | --- | --- |
| `ROBOT_NAME` | agent persona, speech recogniser, web UI | The name the robot answers to. With no wake-word matcher, this is what makes it recognise being addressed in ambient mode. |
| `ROBOT_HOME` | agent persona, speech recogniser | Rendered mid-sentence as "You live in ...", so keep the leading lowercase article. |

Three independent consumers read these, which is worth knowing when debugging:
`registries/conscious_agent.hocon` resolves `${?ROBOT_NAME}` through pyhocon
without passing through Python, `apps/conscious_assistant/robot_identity.py`
serves the recogniser prompt, and a Flask context processor serves the
templates. All three read the environment when the app starts, so **restart the
app after changing either value**; the manifest reload will not pick it up.

Both have working defaults in code as well, so a robot whose `setmyenv.sh`
predates this still behaves as CAIL-E. The lab branding in the web UI footer
stays hardcoded, since every unit lives in a Cognizant AI Lab.

Site-specific data is separate and not covered by these variables: each lab
needs its own `maps/<lab>.json` and occupancy grid, a `NAV_INITIAL_LOCATION`
naming a real node in it, and its own `face_database/`.

#### Environment variable meanings

| Variable | Value above | Meaning |
| --- | --- | --- |
| `PYTHONPATH` | RealSense binding, repo root, and `coded_tools` | Lets Python import the locally built `pyrealsense2`, project modules, and coded tools. |
| `AGENT_TOOL_PATH` | `$PWD/coded_tools` | Directory where Neuro SAN finds coded tools. |
| `AGENT_MANIFEST_FILE` | `$PWD/registries/manifest.hocon` | Tool/agent manifest used by the assistant runtime. |
| `CYCLONEDDS_HOME` | `$PWD/cyclonedds/install` | Local CycloneDDS install path. |
| `CYCLONEDDS_URI` | `file://$HOME/cyclonedds.xml` | CycloneDDS configuration file. |
| `LD_LIBRARY_PATH` | CycloneDDS and librealsense libs | Lets the runtime loader find DDS and RealSense shared libraries. |

#### Navigation and behavior defaults in code

These are the built-in defaults. Keep them out of `setmyenv.sh` unless a
specific robot really needs a temporary override.

| Setting | Code default | Meaning |
| --- | --- | --- |
| `CONSCIOUS_ENABLE_SCENE_OBSERVER` | enabled on Linux | Keeps the latest camera scene available in the UI and face-learning tools. |
| Native scene observation | every 15 seconds | The native runtime's Python observer service captures the scene and sends its metadata to the agent as an internal event. |
| `GO2_MOVE_LOG_INTERVAL_SECONDS` | `-1` | Suppresses repeated raw `Move(vx, vy, vyaw)` logs. |
| `GO2_USE_SDK_SPECIAL_MOTIONS` | `1` | Uses Unitree SDK special motions when available. |
| `GO2_NETWORK_INTERFACE` / `CYCLONEDDS_NETWORK_INTERFACE` | `eth0` | Unitree SDK communication interface. Override only if the robot network is not on `eth0`. |
| `VISION_CAMERA_SOURCE` | auto RealSense color camera by stable `/dev/v4l/by-id` link | Camera source for visual observation and face detection. Automatic face capture does not silently switch cameras; set an explicit source such as `unitree:eth0` only when intended. |
| `NAV_MAP_FILE` | repo `maps/cail_lab.json` on robot, none in simulation | Topological map for named destinations. Override only for a different map, or set empty to disable map loading. |
| `NAV_DEPTH_CAMERA_SOURCE` | `auto` | Tries RealSense first, then OpenCV depth sources. |
| `NAV_DEPTH_PROCESS_WIDTH` | `640` | Depth-frame processing width. The robot default uses the full RealSense depth width for denser obstacle sampling. |
| `NAV_DEPTH_PROCESS_HEIGHT` | `480` | Depth-frame processing height. The robot default uses the full RealSense depth height for denser obstacle sampling. |
| `NAV_OBSTACLE_SOURCE` | `depth` | Selects collision sensing source: `depth`, `lidar`, or `fused`. The robot default uses the depth camera; LiDAR is opt-in. |
| `NAV_USE_LIDAR` | enabled outside simulation | Enables the onboard Unitree LiDAR service when `NAV_OBSTACLE_SOURCE` includes LiDAR. |
| `NAV_LIDAR_TOPIC` | `rt/utlidar/cloud` | DDS `sensor_msgs/PointCloud2` topic used by the LiDAR perimeter service. Override only if the robot publishes LiDAR on a different topic. |
| `NAV_LIDAR_POINTCLOUD_YAW_OFFSET_RAD` | `1.2217` | Rotates Unitree PointCloud2 points into the robot frame before self-masking and path-corridor checks. |
| `NAV_LIDAR_SELF_MASK_FORWARD` | `0.45` | Ignores LiDAR returns inside the robot footprint up to 0.45 m in front of the LiDAR frame. |
| `NAV_LIDAR_SELF_MASK_REAR` | `0.35` | Ignores LiDAR returns inside the robot footprint up to 0.35 m behind the LiDAR frame. |
| `NAV_LIDAR_SELF_MASK_HALF_WIDTH` | `0.25` | Ignores LiDAR returns inside the robot footprint within 0.25 m left/right of the LiDAR frame. |
| `NAV_LIDAR_MAX_RANGE` | `4.0` | Maximum LiDAR range projected into the local obstacle grid. Longer-range map building should use a separate SLAM layer. |
| `NAV_LIDAR_MAX_SAMPLE_AGE` | `0.75` | LiDAR samples older than 0.75 seconds are ignored. |
| `NAV_MAX_LINEAR_SPEED` | `0.40` | Maximum planned forward speed in meters per second. |
| `NAV_MAX_YAW_RATE` | `0.08` | Maximum yaw correction while translating. |
| `NAV_PIVOT_YAW_RATE` | `0.50` | In-place yaw rate for planned map turns. |
| `NAV_USE_SDK_ODOMETRY` | enabled outside simulation | Uses Unitree SportModeState position and yaw when available, with command integration as a stale-data fallback. |
| `NAV_USE_SDK_TRANSLATION_ODOMETRY` | `1` | Uses measured SportModeState translation as the map position. Disable only for SDKs that do not publish valid translation. |

| `NAV_SPORT_MODE_STATE_TOPIC` | `rt/sportmodestate` | DDS topic for measured Unitree sport-mode position and yaw. |
| `NAV_SDK_ODOMETRY_MAX_AGE` | `0.75` | SDK pose samples older than 0.75 seconds are stale. |
| `NAV_SDK_ODOMETRY_MIN_DELTA_M` | `0.02` | SDK position must change by at least 0.02 m before translation odometry is trusted. |
| `NAV_SDK_ODOMETRY_MIN_DELTA_YAW_RAD` | `0.03` | SDK yaw must change by at least 0.03 rad before it is trusted as moving odometry. |
| `NAV_ODOMETRY_YAW_RATE_RATIO` | `1.00` | Fallback yaw scale used while SDK odometry is stale or unconfirmed. |
| `NAV_SAFETY_DISTANCE` | `0.20` | Confirmed safety stop threshold inside the swept path corridor. |
| `NAV_AVOIDANCE_DISTANCE` | `0.75` | Slowdown and local steering begin for supported path obstacles closer than 0.75 m. |
| `NAV_PIVOT_HARD_STOP_DISTANCE` | `0.00` | Lets close path obstacles use the confirmation window before aborting. |
| `NAV_CLOSE_OBSTACLE_CONFIRM_S` | `0.7` | Close path obstacles must persist for at least 0.7 seconds before aborting. |
| `NAV_CLOSE_OBSTACLE_CONFIRM_READINGS` | `6` | Close path obstacles must also persist for at least 6 nav-loop readings. |
| `NAV_CENTER_ONLY_CLOSE_CONFIRM_S` | `0.2` | A close center-depth reading absent from the obstacle grid must persist this long before it can abort navigation; the robot holds meanwhile. |
| `NAV_CENTER_ONLY_CLOSE_CONFIRM_READINGS` | `3` | A grid-disputed close center-depth reading must also appear in at least 3 readings. |
| `NAV_CENTER_ONLY_GRID_MARGIN` | `0.15` | Clearance margin used to identify center-depth readings that disagree with the projected obstacle grid. |
| `NAV_PATH_OBSTACLE_CONFIRM_S` | `0.3` | Avoidance-band path obstacles must persist briefly before slowing or steering. |
| `NAV_PATH_OBSTACLE_CONFIRM_READINGS` | `3` | Avoidance-band path obstacles must also appear in at least 3 nav-loop readings. |
| `NAV_PATH_OBSTACLE_DISTANCE_TOLERANCE` | `0.15` | Consecutive avoidance-band readings within 0.15 m are treated as the same obstacle track. |
| `NAV_PATH_OBSTACLE_BEARING_TOLERANCE_RAD` | `0.1745` | Consecutive avoidance-band readings within about 10 degrees are treated as the same obstacle track. |
| `NAV_PATH_OBSTACLE_CENTER_DEPTH_MARGIN` | `0.15` | When a depth source is active, avoidance-band projected obstacles must agree with raw center depth within the slowdown distance plus this margin. |
| `NAV_GOAL_TOLERANCE` | `0.15` | Destination is considered reached within 0.15 m. |
| `NAV_PATH_CORRIDOR_HALF_WIDTH` | `0.27` | Robot half-width plus clearance used for swept-path obstacle checks. |
| `NAV_PATH_OBSTACLE_MIN_POINTS` | `6` | Requires at least 6 supporting sensor points before a path obstacle is considered real. |
| `NAV_WALL_CLEARANCE` | `0.55` | Target lateral clearance from a reliably fitted one-sided wall. |
| `NAV_OBSTACLE_MEMORY_SECONDS` | `0.8` | Retains recent wall geometry and aligns it using odometry; current depth remains authoritative for safety. |
| `NAV_OBSTACLE_TELEMETRY_SECONDS` | `1.0` | Interval for directional-clearance and wall-fit log messages; set to `0` to disable. |
| `NAV_STUCK_TIMEOUT` | `6.0` | No-progress interval before guarded physical recovery. |
| `NAV_METRIC_BLOCKED_REPLAN_DELAY` | `1.0` | Persistent-obstacle delay before a live-obstacle route is calculated. |
| `NAV_METRIC_REPLAN_COOLDOWN` | `3.0` | Minimum interval between proactive route replacements. |
| `NAV_METRIC_LOCALIZATION_INTERVAL` | `2.0` | Interval for conservative RealSense-to-floor-plan pose correction. |

The production map references a 10cm occupancy grid generated from the office floor
plan. Named map nodes remain the destination interface, while clearance-aware A*
routes around static walls and fixtures. RealSense obstacles are overlaid for initial
planning and replanning; small, high-confidence scan-to-map corrections limit odometry
drift. Intermediate metric targets are internal and do not generate agent-network
events. Maps without an occupancy grid retain the topological-edge fallback.

Map nodes may still declare `arrival_tolerance_m`, `pass_through_tolerance_m`, and
directional `arrival_landmarks` for compatibility and destination-specific arrival
behavior.

## Setup

### Clone the repo

```shell
git clone https://github.com/cognizant-ai-lab/neuro-san-robotics.git
```

### Set up the python env

#### Install PyEnv

You'll need Python 3.11, NOT later. You can use `pyenv` to manage your Python versions.
```shell
# Install pyenv
brew install pyenv

# Or update it
brew update
brew upgrade pyenv

# Check the latest version of Python 3.11
pyenv install --list | grep " 3\.11"

# Install Python 3.11. Latest version at the time of writing is 3.11.13
pyenv install 3.11.13
```

#### Create a virtual environment

```shell
# Navigate to the newly cloned repo
cd neuro-san-robotics

# Create a virtual environment for Python 3.11
$HOME/.pyenv/versions/3.11.13/bin/python -m venv ./venv

# Activate the virtual environment:
source venv/bin/activate && export PYTHONPATH=`pwd` 

# Check you're using the right Python executable
which python

# Check the version. Must be 3.11.x
python --version
```

#### Install the repo's requirements

Install `neuro-san` and the other requirements:

```shell
pip install -r requirements.txt
python -c 'from importlib.metadata import version; print(version("neuro-san"))'
```

The conscious assistant's native event and periodic execution require
`neuro-san==0.6.76` or later. Older releases can acknowledge an event without
continuing the agent work, so the application refuses to start with them.

### Set up Unitree's SDK

See https://github.com/unitreerobotics/unitree_sdk2_python for more details.

```shell
# Go to the project directly if you're not already there
cd neuro-san-robotics

# Install cmake
brew install cmake

# If you face any issues
# Note: ubuntu 20.04 and later have cmake by default.
# However, just to be sure, do this step before building cyclonedds on linux
# sudo apt install -y cmake build-essential git libssl-dev libxml2-dev flex bison
# And optional ROS2 integration packages
# sudo apt install -y libice-dev libsm-dev libx11-dev


# Clone these 2 repos within the neuro-san-robotics dir
git clone https://github.com/unitreerobotics/unitree_sdk2_python
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x

# Go to cyclonedds and create dirs to build
cd cyclonedds && mkdir build install && cd build

# Build cyclonedds
cmake .. -DCMAKE_INSTALL_PREFIX=../install

# If you face issues with cmak installation on linux
# Note for ubuntu 20.04 or later
# Make sure CMake finds the correct system toolchain (not a Homebrew one):
#cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=OFF

#If you’re on Jetson (ARM), you may also want to disable testing tools:
# cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF

# Sometimes the system’s libc6-dev and flex/bison are ARM builds that mismatch with host compiler expectations.
# If that’s the case, you can skip building ddsperf altogether (it’s just a benchmark tool, not needed for runtime).
# In your CMake command:
# cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DBUILD_DDSPERF=OFF
# This will still install CycloneDDS core libraries, without building the ddsperf tool.

# Install cyclonedds
cmake --build . --target install

# Back to cyclonedds/
cd ..

# Export env variable
export CYCLONEDDS_HOME="$(pwd)/install"

# Install cyclonedds
python -m pip install cyclonedds --no-binary cyclonedds

# Back to the project root
cd ..

# Go to unitree_sdk_python
cd unitree_sdk2_python

# Install unitree_sdk_python
python -m pip install -e .

# Back to the project root
cd ..
```

### Set up navigation sensors on the robot

`nav_core` uses the RealSense/depth obstacle grid for collision sensing by
default. The onboard Unitree LiDAR remains available only as an explicit opt-in
with `NAV_OBSTACLE_SOURCE=lidar` or `NAV_OBSTACLE_SOURCE=fused`.

For RealSense/depth mode on the Jetson/Go2, install the system tools first:

```shell
sudo apt-get update
sudo apt-get install -y \
  git cmake build-essential pkg-config \
  libssl-dev libusb-1.0-0-dev libudev-dev libgtk-3-dev \
  v4l-utils librealsense2-utils librealsense2-dev
```

Verify that the camera is visible to librealsense:

```shell
rs-enumerate-devices
```

You should see an Intel RealSense device, such as `Intel RealSense D435I`.

#### Build `pyrealsense2` for Jetson/aarch64

Jetson/aarch64 usually cannot install `pyrealsense2` from pip because there is
no matching wheel. Build the Python binding against the active Python 3.11 venv:

```shell
cd ~/exp/neuro-san-robotics-hormoz
source venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

SDK_VER="$(dpkg-query -W -f='${Version}' librealsense2-utils 2>/dev/null | sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/')"
echo "SDK_VER=$SDK_VER"

cd ~
git clone --depth 1 --branch "v${SDK_VER}" \
  https://github.com/IntelRealSense/librealsense.git \
  "librealsense-${SDK_VER}"

cd "librealsense-${SDK_VER}"
mkdir -p build && cd build

cmake .. \
  -DBUILD_PYTHON_BINDINGS:bool=true \
  -DPYTHON_EXECUTABLE="$(which python)" \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_TOOLS=OFF \
  -DCMAKE_BUILD_TYPE=Release

make -j"$(nproc)"
```

Add the built binding and librealsense library to the robot shell environment.
If the robot uses a locally built RealSense Python binding, include those local
library paths in `setmyenv.sh`. The depth backend itself defaults to `auto` and
will try RealSense first, so `NAV_DEPTH_CAMERA_SOURCE` is not normally needed.

```shell
SDK_VER="${SDK_VER:-$(dpkg-query -W -f='${Version}' librealsense2-utils 2>/dev/null | sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/')}"
export PYTHONPATH="$HOME/librealsense-${SDK_VER}/build/Release:$PWD:$PYTHONPATH"
export LD_LIBRARY_PATH="$HOME/librealsense-${SDK_VER}/build:$LD_LIBRARY_PATH"
```

If you enable RealSense/depth mode, verify Python can see the camera:

```shell
python - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
print("pyrealsense2:", rs.__file__)
print("devices:", len(ctx.query_devices()))
PY
```

Expected result: `devices: 1`.

#### Verify nav depth before movement

```shell
cd ~/exp/neuro-san-robotics-hormoz
source venv/bin/activate
SDK_VER="${SDK_VER:-$(dpkg-query -W -f='${Version}' librealsense2-utils 2>/dev/null | sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/')}"
export PYTHONPATH="$HOME/librealsense-${SDK_VER}/build/Release:$PWD:$PYTHONPATH"
export LD_LIBRARY_PATH="$HOME/librealsense-${SDK_VER}/build:$LD_LIBRARY_PATH"

python - <<'PY'
from coded_tools.unigo2.depth_processor import DepthProcessor

dp = DepthProcessor()
print("backend:", dp.backend)
grid = dp.get_single_frame_grid()
print("grid:", grid is not None)
if grid:
    print("nearest_obstacle_m:", grid.nearest_obstacle_m)
    print("occupied_cells:", int((grid.grid > 0).sum()))
reading = dp.get_center_depth_reading()
print("center_depth_m:", None if reading is None else reading.distance_m)
dp.stop()
PY
```

Expected result: `backend: realsense`, `grid: True`, and a finite
`center_depth_m` when something is in the camera's center view.

#### Run guarded forward movement

For physical movement, prefer the guarded forward primitive. It keeps a
continuous Go2 walking command active while a raw RealSense center-depth
watchdog stops the robot when something is close enough. This avoids the
dead-reckoned low-speed crawl that can make the Go2 tiptoe or shake.

The movement defaults are built into `NavCore`; do not add them to
`setmyenv.sh` unless a specific robot needs a temporary local override.

To move forward until the center-depth watchdog sees an obstacle at about
`0.50m`:

```shell
python - <<'PY'
from coded_tools.unigo2.nav_core import NavCore

nav = NavCore.get_instance()
try:
    print(nav.move_forward_guarded(stop_distance_m=0.50))
finally:
    nav.shutdown()
PY
```

The Neuro SAN agent command `move_until_obstacle` uses the same guarded
movement path. `move_forward` also uses this path with a time backstop derived
from the requested distance until real odometry is available.

---

## How to run neuro-san agents on the robot

**Step1:** 
- Set env variables:
```bash
export AGENT_TOOL_PATH="coded_tools"
export AGENT_MANIFEST_FILE="registries/manifest.hocon"
```

- Optionally set update period
```bash
export AGENT_MANIFEST_UPDATE_PERIOD_SECONDS=5
```

**Step2:** Interaction:

- Direct interaction
  ```bash
  python -m neuro_san.client.agent_cli --connection direct --agent "unigo2"
  ```

**OR**

- Server and client setup

  On one terminal, start neuro_san server
  ```bash
  python -m neuro_san.service.main_loop.server_main_loop
  ```

  And on another terminal run the CLI client
  ```bash
  python -m neuro_san.client.agent_cli --connection http --agent "unigo2"
  ```

**Step 3**:
- Sample expected interaction
```
Please enter your response ('quit' to terminate):
hello
Sending user_input hello

Response from robot_manager:
Hello! How can I assist you today?
Please enter your response ('quit' to terminate):
stand up
Sending user_input stand up
[13:44:25] Stand up

Response from robot_manager:
The robot has stood up. How else can I assist you today?
Please enter your response ('quit' to terminate):
quit
```

---

For Troubleshooting, refer to [./docs/troubleshooting.md](./docs/troubleshooting.md)
