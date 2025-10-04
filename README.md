# neuro-san-robotics
Neuro San Robotics

## Setup

### Clone the repo

```shell
git clone https://github.com/cognizant-ai-lab/neuro-san-robotics.git
```

### Set up the python env
```shell
# Navigate to the newly cloned repo
cd neuro-san-robotics

# Create a dedicated Python virtual environment:
python -m venv .venv

# Activate the virtual environment:
source venv/bin/activate && export PYTHONPATH=`pwd`

# Install the requirements:
pip install -r requirements.txt
```

### Set up Unitree's SDK

```shell
# Go to the project directly if you're not already there
cd neuro-san-robotics

# Install cmake
brew install cmake

# Clone these 2 repos within the neuro-san-robotics dir
git clone https://github.com/unitreerobotics/unitree_sdk2_python
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x

# Go to cyclonedds and create dirs to build
cd cyclonedds && mkdir build install && cd build

# Build cyclonedds
cmake .. -DCMAKE_INSTALL_PREFIX=../install

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

## Test

Runt the `hello_world` test:

```bash
cd unitree_sdk_python
python ./example/helloworld/subscriber.py
python ./example/helloworld/publisher.py
```
