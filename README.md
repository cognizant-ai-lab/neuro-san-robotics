# neuro-san-robotics
Neuro SAN Robotics

## Start Cailey

1. Physically start Caily with a short press followed by a long press
on its power button.
2. Find Cailey's IP address and ssh into it:
    ```shell
    ssh unitree@10.194.17.33
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
You can now navigate to https://10.194.17.33:5001 (check the IP address)
to interact with Cailey.

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
