# neuro-san-robotics
Neuro SAN Robotics

## Start CAIL-E

1. Physically start CAIL-E with a short press followed by a long press
on its power button.
2. Find CAIL-E's IP address and ssh into it:
    ```shell
    ssh unitree@10.194.17.130
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
You can now navigate to https://10.194.17.130:5001 (check the IP address)
to interact with CAIL-E.

### Robot environment (`setmyenv.sh`)

Use this as the robot-side `setmyenv.sh`. Keep secrets such as
`OPENAI_API_KEY` outside this tracked file, for example in an untracked local
shell file or as a manual export.

```shell
source venv/bin/activate

export PYTHONPATH="$HOME/librealsense-2.54.2/build/Release:$PWD:$PWD/coded_tools:${PYTHONPATH:-}"
export AGENT_TOOL_PATH="$PWD/coded_tools"
export AGENT_MANIFEST_FILE="$PWD/registries/manifest.hocon"

export CONSCIOUS_DIRECT_ROBOT_COMMANDS=1
export CONSCIOUS_ROBOT_MOTION_PROBABILITY=0
export CONSCIOUS_ENABLE_IDLE_THINKING=0
export CONSCIOUS_ENABLE_SCENE_AGENT_INPUT=0
export CONSCIOUS_ENABLE_SCENE_OBSERVER=1
export GO2_USE_SDK_SPECIAL_MOTIONS=1
export VISION_CAMERA_SOURCE=unitree:eth0
export GO2_MOVE_LOG_INTERVAL_SECONDS=-1

export GO2_NETWORK_INTERFACE=eth0
export CYCLONEDDS_NETWORK_INTERFACE=eth0
export CYCLONEDDS_HOME="$PWD/cyclonedds/install"
export CYCLONEDDS_URI="file://$HOME/cyclonedds.xml"

export LD_LIBRARY_PATH="$PWD/cyclonedds/install/lib:$HOME/librealsense-2.54.2/build:${LD_LIBRARY_PATH:-}"

export NAV_DEPTH_CAMERA_SOURCE=realsense
export NAV_MAP_FILE="$PWD/maps/cail_lab.json"
export NAV_MAX_LINEAR_SPEED=0.40
export NAV_MAX_YAW_RATE=0.08
export NAV_PIVOT_YAW_RATE=0.50
export NAV_USE_SDK_ODOMETRY=1
export NAV_SPORT_MODE_STATE_TOPIC=rt/sportmodestate
export NAV_SDK_ODOMETRY_MAX_AGE=0.75
export NAV_SDK_ODOMETRY_MIN_DELTA_M=0.02
export NAV_SDK_ODOMETRY_MIN_DELTA_YAW_RAD=0.03
export NAV_ODOMETRY_YAW_RATE_RATIO=1.00
export NAV_SAFETY_DISTANCE=0.20
export NAV_AVOIDANCE_DISTANCE=0.60
export NAV_PIVOT_HARD_STOP_DISTANCE=0.00
export NAV_CLOSE_OBSTACLE_CONFIRM_S=0.7
export NAV_CLOSE_OBSTACLE_CONFIRM_READINGS=6
export NAV_GOAL_TOLERANCE=0.15
export NAV_PATH_CORRIDOR_HALF_WIDTH=0.12
export NAV_PATH_OBSTACLE_MIN_POINTS=6

printf 'PYTHONPATH=%s\n' "$PYTHONPATH"
printf 'AGENT_TOOL_PATH=%s\n' "$AGENT_TOOL_PATH"
printf 'AGENT_MANIFEST_FILE=%s\n' "$AGENT_MANIFEST_FILE"

printf 'CONSCIOUS_DIRECT_ROBOT_COMMANDS=%s\n' "$CONSCIOUS_DIRECT_ROBOT_COMMANDS"
printf 'CONSCIOUS_ROBOT_MOTION_PROBABILITY=%s\n' "$CONSCIOUS_ROBOT_MOTION_PROBABILITY"
printf 'CONSCIOUS_ENABLE_IDLE_THINKING=%s\n' "$CONSCIOUS_ENABLE_IDLE_THINKING"
printf 'CONSCIOUS_ENABLE_SCENE_AGENT_INPUT=%s\n' "$CONSCIOUS_ENABLE_SCENE_AGENT_INPUT"
printf 'CONSCIOUS_ENABLE_SCENE_OBSERVER=%s\n' "$CONSCIOUS_ENABLE_SCENE_OBSERVER"
printf 'GO2_USE_SDK_SPECIAL_MOTIONS=%s\n' "$GO2_USE_SDK_SPECIAL_MOTIONS"
printf 'VISION_CAMERA_SOURCE=%s\n' "$VISION_CAMERA_SOURCE"
printf 'GO2_MOVE_LOG_INTERVAL_SECONDS=%s\n' "$GO2_MOVE_LOG_INTERVAL_SECONDS"

printf 'GO2_NETWORK_INTERFACE=%s\n' "$GO2_NETWORK_INTERFACE"
printf 'CYCLONEDDS_NETWORK_INTERFACE=%s\n' "$CYCLONEDDS_NETWORK_INTERFACE"
printf 'CYCLONEDDS_HOME=%s\n' "$CYCLONEDDS_HOME"
printf 'CYCLONEDDS_URI=%s\n' "$CYCLONEDDS_URI"
printf 'LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH"

printf 'NAV_DEPTH_CAMERA_SOURCE=%s\n' "$NAV_DEPTH_CAMERA_SOURCE"
printf 'NAV_MAP_FILE=%s\n' "$NAV_MAP_FILE"
printf 'NAV_MAX_LINEAR_SPEED=%s\n' "$NAV_MAX_LINEAR_SPEED"
printf 'NAV_MAX_YAW_RATE=%s\n' "$NAV_MAX_YAW_RATE"
printf 'NAV_PIVOT_YAW_RATE=%s\n' "$NAV_PIVOT_YAW_RATE"
printf 'NAV_USE_SDK_ODOMETRY=%s\n' "$NAV_USE_SDK_ODOMETRY"
printf 'NAV_SPORT_MODE_STATE_TOPIC=%s\n' "$NAV_SPORT_MODE_STATE_TOPIC"
printf 'NAV_SDK_ODOMETRY_MAX_AGE=%s\n' "$NAV_SDK_ODOMETRY_MAX_AGE"
printf 'NAV_SDK_ODOMETRY_MIN_DELTA_M=%s\n' "$NAV_SDK_ODOMETRY_MIN_DELTA_M"
printf 'NAV_SDK_ODOMETRY_MIN_DELTA_YAW_RAD=%s\n' "$NAV_SDK_ODOMETRY_MIN_DELTA_YAW_RAD"
printf 'NAV_ODOMETRY_YAW_RATE_RATIO=%s\n' "$NAV_ODOMETRY_YAW_RATE_RATIO"
printf 'NAV_SAFETY_DISTANCE=%s\n' "$NAV_SAFETY_DISTANCE"
printf 'NAV_AVOIDANCE_DISTANCE=%s\n' "$NAV_AVOIDANCE_DISTANCE"
printf 'NAV_PIVOT_HARD_STOP_DISTANCE=%s\n' "$NAV_PIVOT_HARD_STOP_DISTANCE"
printf 'NAV_CLOSE_OBSTACLE_CONFIRM_S=%s\n' "$NAV_CLOSE_OBSTACLE_CONFIRM_S"
printf 'NAV_CLOSE_OBSTACLE_CONFIRM_READINGS=%s\n' "$NAV_CLOSE_OBSTACLE_CONFIRM_READINGS"
printf 'NAV_GOAL_TOLERANCE=%s\n' "$NAV_GOAL_TOLERANCE"
printf 'NAV_PATH_CORRIDOR_HALF_WIDTH=%s\n' "$NAV_PATH_CORRIDOR_HALF_WIDTH"
printf 'NAV_PATH_OBSTACLE_MIN_POINTS=%s\n' "$NAV_PATH_OBSTACLE_MIN_POINTS"
```

The final `printf` block is intentional: after `source setmyenv.sh`, the shell
prints the active values with names so it is clear what will override the app
defaults before Flask starts.

#### Environment variable meanings

| Variable | Value above | Meaning |
| --- | --- | --- |
| `PYTHONPATH` | RealSense binding, repo root, and `coded_tools` | Lets Python import the locally built `pyrealsense2`, project modules, and coded tools. |
| `AGENT_TOOL_PATH` | `$PWD/coded_tools` | Directory where Neuro SAN finds coded tools. |
| `AGENT_MANIFEST_FILE` | `$PWD/registries/manifest.hocon` | Tool/agent manifest used by the assistant runtime. |
| `CONSCIOUS_DIRECT_ROBOT_COMMANDS` | `1` | Allows direct robot commands from the conscious assistant path. |
| `CONSCIOUS_ROBOT_MOTION_PROBABILITY` | `0` | Disables random acknowledgment motions. |
| `CONSCIOUS_ENABLE_IDLE_THINKING` | `0` | Disables passive idle agent turns. |
| `CONSCIOUS_ENABLE_SCENE_AGENT_INPUT` | `0` | Prevents scene observations from injecting autonomous agent prompts. |
| `CONSCIOUS_ENABLE_SCENE_OBSERVER` | `1` | Keeps the camera scene observer enabled. |
| `GO2_USE_SDK_SPECIAL_MOTIONS` | `1` | Uses SDK-backed Go2 special motions when available. |
| `VISION_CAMERA_SOURCE` | `unitree:eth0` | Uses the Unitree front camera over `eth0` for visual observation. |
| `GO2_MOVE_LOG_INTERVAL_SECONDS` | `-1` | Disables repeated raw `Move(vx, vy, vyaw)` command logs. Set a positive number to sample move logs every N seconds. |
| `GO2_NETWORK_INTERFACE` | `eth0` | Network interface used for Go2 SDK communication. |
| `CYCLONEDDS_NETWORK_INTERFACE` | `eth0` | Network interface CycloneDDS should bind to. |
| `CYCLONEDDS_HOME` | `$PWD/cyclonedds/install` | Local CycloneDDS install path. |
| `CYCLONEDDS_URI` | `file://$HOME/cyclonedds.xml` | CycloneDDS configuration file. |
| `LD_LIBRARY_PATH` | CycloneDDS and librealsense libs | Lets the runtime loader find DDS and RealSense shared libraries. |
| `NAV_DEPTH_CAMERA_SOURCE` | `realsense` | Uses RealSense depth for navigation safety. |
| `NAV_MAP_FILE` | `$PWD/maps/cail_lab.json` | Topological map for named destinations. |
| `NAV_MAX_LINEAR_SPEED` | `0.40` | Maximum planned forward speed in meters per second. |
| `NAV_MAX_YAW_RATE` | `0.08` | Maximum yaw correction while translating. This keeps walking from weaving aggressively. |
| `NAV_PIVOT_YAW_RATE` | `0.50` | In-place yaw rate for planned map turns, such as the 90-degree turn from Shrushti's desk toward the kitchen. |
| `NAV_USE_SDK_ODOMETRY` | `1` | Uses Unitree SportModeState as the primary pose source when available. |
| `NAV_SPORT_MODE_STATE_TOPIC` | `rt/sportmodestate` | DDS topic used for measured Unitree sport-mode position and yaw. |
| `NAV_SDK_ODOMETRY_MAX_AGE` | `0.75` | SDK pose samples older than 0.75 seconds are treated as stale. |
| `NAV_SDK_ODOMETRY_MIN_DELTA_M` | `0.02` | SDK position must change by at least 0.02 m before it is trusted as a moving odometry source. |
| `NAV_SDK_ODOMETRY_MIN_DELTA_YAW_RAD` | `0.03` | SDK yaw must change by at least 0.03 rad before it is trusted as a moving odometry source. |
| `NAV_ODOMETRY_YAW_RATE_RATIO` | `1.00` | Fallback yaw scale used only while measured SDK odometry is stale or has not yet proven motion. |
| `NAV_SAFETY_DISTANCE` | `0.20` | Confirmed safety stop threshold: path obstacles at or below 0.20 m are treated as stop conditions. |
| `NAV_AVOIDANCE_DISTANCE` | `0.60` | Slowdown begins when a supported path obstacle is closer than 0.60 m. |
| `NAV_PIVOT_HARD_STOP_DISTANCE` | `0.00` | Keeps the hard-stop bypass below the safety threshold so close path obstacles use the confirmation window. During confirmation, motion is still stopped. |
| `NAV_CLOSE_OBSTACLE_CONFIRM_S` | `0.7` | A borderline close path obstacle must persist for at least 0.7 seconds before the action is aborted. |
| `NAV_CLOSE_OBSTACLE_CONFIRM_READINGS` | `6` | The same close path obstacle must also persist for at least 6 nav-loop readings before aborting. |
| `NAV_GOAL_TOLERANCE` | `0.15` | Destination is considered reached within 0.15 m. |
| `NAV_PATH_CORRIDOR_HALF_WIDTH` | `0.12` | Only depth points within 0.12 m left/right of the robot centerline count as path-corridor obstacles. |
| `NAV_PATH_OBSTACLE_MIN_POINTS` | `6` | Requires at least 6 supporting depth points before a path-corridor obstacle is considered real. |

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
```

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
# This will still install CycloneDDS core libraries — without building the ddsperf tool.

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

### Set up navigation and RealSense depth on the robot

`nav_core` needs a forward-facing depth camera for physical movement. Do not
run navigation movement commands until `DepthProcessor` reports
`backend: realsense` and returns a valid grid.

On the Jetson/Go2, install the system tools first:

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
These exports can also go into `setmyenv.sh`:

```shell
SDK_VER="${SDK_VER:-$(dpkg-query -W -f='${Version}' librealsense2-utils 2>/dev/null | sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/')}"
export PYTHONPATH="$HOME/librealsense-${SDK_VER}/build/Release:$PWD:$PYTHONPATH"
export LD_LIBRARY_PATH="$HOME/librealsense-${SDK_VER}/build:$LD_LIBRARY_PATH"
export NAV_DEPTH_CAMERA_SOURCE=realsense
```

Verify Python can see the RealSense camera:

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
export NAV_DEPTH_CAMERA_SOURCE=realsense

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

Set the movement defaults:

```shell
export NAV_FORWARD_SPEED=0.45
export NAV_FORWARD_STOP_DISTANCE=0.75
export NAV_FORWARD_MAX_SECONDS=15
export NAV_FORWARD_COMMAND_PERIOD=0.20
```

To move forward until the center-depth watchdog sees an obstacle at about
`0.75m`:

```shell
python - <<'PY'
from coded_tools.unigo2.nav_core import NavCore

nav = NavCore.get_instance()
try:
    print(nav.move_forward_guarded(stop_distance_m=0.75))
finally:
    nav.shutdown()
PY
```

The Neuro SAN agent command `move_until_obstacle` uses the same guarded
movement path. `move_forward` also uses this path with a time backstop derived
from the requested distance until real odometry is available.

### Troubleshooting

If you run into the following error:
```shell
      building 'cyclonedds._clayer' extension
      creating build/temp.macosx-14.7-arm64-cpython-313/clayer
      clang -fno-strict-overflow -Wsign-compare -Wunreachable-code -DNDEBUG -g -O3 -Wall -I/Users/754337/workspace/neuro-san-robotics/cyclonedds/install/include -I/private/var/folders/w6/w8ptt70j5ylfrcw1rgjc5pfr0000gq/T/pip-install-_cut3gf5/cyclonedds_7518aeede8314c2c995982c34d1e59f6/clayer -I/Users/754337/workspace/neuro-san-robotics/venv/include -I/Users/754337/.pyenv/versions/3.13.5/include/python3.13 -c clayer/cdrkeyvm.c -o build/temp.macosx-14.7-arm64-cpython-313/clayer/cdrkeyvm.o
      clang -fno-strict-overflow -Wsign-compare -Wunreachable-code -DNDEBUG -g -O3 -Wall -I/Users/754337/workspace/neuro-san-robotics/cyclonedds/install/include -I/private/var/folders/w6/w8ptt70j5ylfrcw1rgjc5pfr0000gq/T/pip-install-_cut3gf5/cyclonedds_7518aeede8314c2c995982c34d1e59f6/clayer -I/Users/754337/workspace/neuro-san-robotics/venv/include -I/Users/754337/.pyenv/versions/3.13.5/include/python3.13 -c clayer/pysertype.c -o build/temp.macosx-14.7-arm64-cpython-313/clayer/pysertype.o
      clayer/pysertype.c:610:10: error: call to undeclared function '_Py_IsFinalizing'; ISO C99 and later do not support implicit function declarations [-Wimplicit-function-declaration]
        610 |     if (!_Py_IsFinalizing()) {
            |          ^
      clayer/pysertype.c:610:10: note: did you mean 'Py_IsFinalizing'?
      /Users/754337/.pyenv/versions/3.13.5/include/python3.13/pylifecycle.h:68:17: note: 'Py_IsFinalizing' declared here
         68 | PyAPI_FUNC(int) Py_IsFinalizing(void);
            |                 ^
      clayer/pysertype.c:1784:48: warning: passing 'const dds_typeid_t *' (aka 'const struct ddsi_typeid *') to parameter of type 'dds_typeid_t *' (aka 'struct ddsi_typeid *') discards qualifiers [-Wincompatible-pointer-types-discards-qualifiers]
       1784 |             ddspy_typeid_ser(&type_obj_stream, type_id);
            |                                                ^~~~~~~
      clayer/typeser.h:19:54: note: passing argument to parameter here
         19 | void ddspy_typeid_ser (dds_ostream_t*, dds_typeid_t *);
            |                                                      ^
      clayer/pysertype.c:1875:48: warning: passing 'const dds_typeid_t *' (aka 'const struct ddsi_typeid *') to parameter of type 'dds_typeid_t *' (aka 'struct ddsi_typeid *') discards qualifiers [-Wincompatible-pointer-types-discards-qualifiers]
       1875 |             ddspy_typeid_ser(&type_obj_stream, type_id);
            |                                                ^~~~~~~
      clayer/typeser.h:19:54: note: passing argument to parameter here
         19 | void ddspy_typeid_ser (dds_ostream_t*, dds_typeid_t *);
            |                                                      ^
      clayer/pysertype.c:1962:48: warning: passing 'const dds_typeid_t *' (aka 'const struct ddsi_typeid *') to parameter of type 'dds_typeid_t *' (aka 'struct ddsi_typeid *') discards qualifiers [-Wincompatible-pointer-types-discards-qualifiers]
       1962 |             ddspy_typeid_ser(&type_obj_stream, type_id);
            |                                                ^~~~~~~
      clayer/typeser.h:19:54: note: passing argument to parameter here
         19 | void ddspy_typeid_ser (dds_ostream_t*, dds_typeid_t *);
            |                                                      ^
      clayer/pysertype.c:2079:48: warning: passing 'const dds_typeid_t *' (aka 'const struct ddsi_typeid *') to parameter of type 'dds_typeid_t *' (aka 'struct ddsi_typeid *') discards qualifiers [-Wincompatible-pointer-types-discards-qualifiers]
       2079 |             ddspy_typeid_ser(&type_obj_stream, type_id);
            |                                                ^~~~~~~
      clayer/typeser.h:19:54: note: passing argument to parameter here
         19 | void ddspy_typeid_ser (dds_ostream_t*, dds_typeid_t *);
            |                                                      ^
      4 warnings and 1 error generated.
      error: command '/usr/bin/clang' failed with exit code 1
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for cyclonedds
Failed to build cyclonedds
```

That's because you're using Python 3.12.x or 3.13. You need to downgrade to Python 3.11.x.

## Test

Run the `hello_world` test:

In one terminal, run the subscriber:
```bash
# Navigate to the project's repo
cd neuro-san-robotics

# Activate the virtual environment
source venv/bin/activate && export PYTHONPATH=`pwd`

# Navigate to the unitree_sdk2_python directory
cd  unitree_sdk2_python

# Run the subscriber
python ./example/helloworld/subscriber.py
```

In another terminal, run the publisher:
```bash
# Navigate to the project's repo
cd neuro-san-robotics

# Activate the virtual environment
source venv/bin/activate && export PYTHONPATH=`pwd`

# Navigate to the unitree_sdk2_python directory
cd  unitree_sdk2_python

# Run the publisher
python ./example/helloworld/publisher.py
```

For more information look at the [Unitree SDK2 Python documentation](https://github.com/unitreerobotics/unitree_sdk2_python).

---

## How to run neuro-san agents on Cailey

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
