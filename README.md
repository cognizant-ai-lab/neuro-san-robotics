# neuro-san-robotics
Neuro San Robotics


unitree sdk setup

```bash
brew install cmake

# make a dir
mkdir caily

# go to the dir
cd caily

# create venv
python -m venv .venv

# source from venv
source .venv/bin/activate

# clone them both there
git clone https://github.com/unitreerobotics/unitree_sdk2_python
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x

# go to cyclonedds and create dirs to build
cd cyclonedds && mkdir build install && cd build

# build cyclonedds
cmake .. -DCMAKE_INSTALL_PREFIX=../install

# install cyclonedds
cmake --build . --target install

# back to cyclonedds/
cd ..

# export env variable
export CYCLONEDDS_HOME="$(pwd)/install"

# install cyclonedds
cd cyclonedds; python -m pip install cyclonedds --no-binary cyclonedds

# back to caily/
cd ..

# go to unitree_sdk_python
cd unitree_sdk2_python

# install unitree_sdk_python
python -m pip install -e .

# back to caily
cd ..
```


test hello_world:
use the same venv
```bash

cd unitree_sdk_python
python ./example/helloworld/subscriber.py
python ./example/helloworld/publisher.py

```
