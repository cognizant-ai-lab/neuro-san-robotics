# Unitree Go2 Robot Bringup Guide

Unitree Go2 EDU + `neuro-san-robotics` field setup.

**Target platform:** Go2 EDU (Jetson Orin, L4T R35.x / JetPack 5.x, Ubuntu 20.04, aarch64)
**Audience:** someone bringing up a unit from cold, on site, possibly without reliable internet.

---

## How to use this guide

Phases are ordered by dependency, not by convenience. Each phase ends in a **Gate**: a command with an expected output. Do not proceed past a failed gate, because every later failure will be misattributed to the wrong layer.

Conventions used below:

| Marker | Meaning |
| :--- | :--- |
| **Gate** | Verification step with a concrete expected result |
| **Optional** | Skippable without breaking the core app |
| **Hazard** | Can drop your SSH session, brick a service, or damage hardware |

Record these three values before you start. Several later steps depend on them:

```bash
cat /etc/nv_tegra_release     # authoritative L4T version, e.g. R35 (release), REVISION: 3.1
uname -r                      # e.g. 5.10.104-tegra
ip -br addr                   # note the wired interface name and address
```

---

## Phase 0: Pre-flight

Five minutes here saves an hour later.

### 0.1 Physical

- Battery at **50 percent or more**. A Jetson-side compile plus motor idle will drain a half-charged pack faster than expected, and a brownout mid-build corrupts the eMMC filesystem.
- Robot either **suspended on a stand** or **folded on flat, clear ground**. Clear a radius of at least 2 m.
- **LiDAR must rotate freely.** Nothing resting on the head, no packing foam, no strap across the dome.
- Know the panic inputs before powering on: `L2 + B` puts the robot into damping (limp), `L2 + A` sits it down.

### 0.2 Host resources

```bash
df -h /                       # need 15 GB+ free; librealsense alone wants 6 to 8 GB
free -h                       # confirm swap exists before any -j4 compile
date                          # see 0.3
```

If swap is absent and you plan to build librealsense, add temporary swap:

```bash
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

### 0.3 Clock sanity

The Jetson has no battery-backed RTC that survives a long shelf period. If the clock is skewed, every HTTPS call fails on certificate validity, including calls to the OpenAI API. The error surfaces as a TLS or SSL failure and never mentions the clock.

```bash
date                                   # compare against your phone
timedatectl                            # want: System clock synchronized: yes
```

If wrong and you have no internet yet, set it manually and fix properly after Phase 3:

```bash
sudo date -s "2026-08-19 14:30:00"
```

**Gate 0:** disk has 15 GB+ free, clock is within a minute of real time, LiDAR spins freely by hand.

---

## Phase 1: Power on and controller check

### 1.1 Power sequence

For both the remote controller and the robot: **quick press the power button once, then press and hold** until the device powers on.

Order: controller first, then robot. Wait for the robot to complete its stand-up sequence before touching the sticks.

### 1.2 Walk test

Drive the robot a few metres with the remote. You are checking for motor faults, uneven gait, and joint errors before you invest any time in software. If the robot limps, reports a joint error, or refuses to stand, stop and triage the hardware.

### 1.3 Exit obstacle avoidance mode

Press and hold **Y** for about 3 seconds.

This matters for more than convenience. Obstacle avoidance runs in the factory stack and will contest or veto SDK-issued velocity commands, so leaving it on produces a robot that appears to ignore your navigation code intermittently. Turn it off before any SDK testing.

### 1.4 Powering off

Quick press, then hold. **Hazard:** hold the handle securely while powering down. The robot goes limp the instant power is cut and will drop.

**Gate 1:** robot walks cleanly under remote control, obstacle avoidance is off, no joint errors on the controller display.

---

## Phase 2: Network and SSH

### 2.1 Wired connection to the robot

The Go2 internal network is `192.168.123.0/24`. Your laptop needs a **static address on that subnet**, because the robot does not run a DHCP server for you. This is the step most often skipped, and it presents as "SSH just hangs".

On the laptop, set the wired interface to:

- Address: `192.168.123.222`
- Netmask: `255.255.255.0`
- Gateway: leave empty

Then confirm reachability before trying SSH:

```bash
ping -c 3 192.168.123.18       # Jetson (development compute unit)
ping -c 3 192.168.123.161      # motion control board
```

Confirm the actual Jetson address against your unit's documentation. It varies across EDU revisions.

### 2.2 SSH in

```bash
ssh unitree@192.168.123.18
```

Enter the password when prompted. Same password for `sudo` throughout.

### 2.3 The ROS prompt

Login triggers a prompt from `.bashrc`:

```
Foxy(1) / Noetic(2)?
```

**Answer `1` for Foxy.**

Both distributions are installed on the EDU image. Noetic is referenced again in the optional RealSense phase, because the factory `realsense2_camera` packages live there. Answering 1 does not remove Noetic, it only selects what gets sourced into this shell.

### 2.4 Hazard: ROS leaks into your Python environment

Sourcing either ROS distribution sets `PYTHONPATH` to a Python 3.8 site-packages tree. That path stays in front of your Python 3.11 virtualenv and causes import failures and version mismatches that look like broken packages.

Once you reach Phase 7, either clear it in your working shell or handle it in `setmyenv.sh`:

```bash
unset PYTHONPATH
```

Do this **before** activating the venv, not after.

### 2.5 Static address for later Wi-Fi access

After Phase 3, you will want to reach the robot over Wi-Fi as well. Pin it, either by DHCP reservation on the venue router (preferred, no robot-side config) or by a static profile on the robot (see 3.9).

**Gate 2:** `ssh unitree@<jetson-ip>` succeeds, `sudo -v` accepts your password, `echo $PYTHONPATH` shows what you expect.

---

## Phase 3: USB Wi-Fi dongle

### 3.1 Choose the right dongle

Prefer a **RTL8192EU** part (TP-Link TL-WN823N **v3** and equivalents), because the in-tree `rtl8xxxu` driver on the Tegra 5.10 kernel has a reasonable chance of binding it without any compilation.

| Hardware revision | Chipset | Verdict |
| :--- | :--- | :--- |
| v1 / v2 | RTL8192CU | Works, older driver path |
| **v3** | **RTL8192EU** | **Recommended** |

**Constraint worth checking before you travel:** RTL8192EU is **2.4 GHz only** (802.11n). If the venue broadcasts a 5 GHz-only SSID, this dongle will never see the network and no amount of driver work will change that. Confirm the venue has a 2.4 GHz SSID, or bring a dual-band backup.

### 3.2 Identify the device

```bash
lsusb
```

Look for a `2357:` (TP-Link) or `0bda:` (Realtek) vendor ID. Record the full ID pair.

```bash
uname -r          # expect something like 5.10.104-tegra
```

### 3.3 Try the in-tree driver first

```bash
sudo modprobe rtl8xxxu
sudo modprobe rtl8192cu
```

**These commands print nothing on success.** Silence means the module loaded. Output means it failed. Do not read the absence of output as failure.

Verify separately:

```bash
sudo dmesg | tail -30
ip link
```

### 3.4 Decision point

- **A `wlx*` or `wlan*` interface appears in `ip link`:** you are done with driver work. Skip to **3.8**.
- **No interface, or `modprobe` reported "Module not found":** continue to 3.5.

### 3.5 Confirm the blockers

```bash
ls /lib/modules/$(uname -r)/build                                            # kernel headers
ls /lib/modules/$(uname -r)/kernel/drivers/net/wireless/realtek/             # shipped realtek modules
```

The first line is the only test for headers. Newer units (Go2 EDU U4) ship with the Realtek modules present.

**Headers are not a driver.** They are the source and build infrastructure needed to compile one. Having them changes nothing on its own.

### 3.6 Install build prerequisites and headers

Requires a working internet path, so do this over Ethernet or a tethered connection.

```bash
sudo apt install -y build-essential dkms bc libelf-dev libssl-dev flex bison nano
sudo apt install -y nvidia-l4t-kernel-headers
```

**Hazard: version matching.** `5.10.104-tegra` spans the entire R35.x series (JetPack 5.0.2 through 5.1.2), so `uname -r` cannot tell you which one you have. Go2 EDU units ship with JetPack 5.1.1, which is R35.3.1, but confirm rather than assume:

```bash
cat /etc/nv_tegra_release
```

Pull the wrong `public_sources.tbz2` or the wrong header package and the module compiles cleanly, then refuses to load on a vermagic mismatch. Always check before loading:

```bash
modinfo ./8192eu.ko | grep vermagic
```

If the `nvidia-l4t-kernel-headers` package is unavailable via apt (the L4T apt source is sometimes removed from the Unitree image), fetch `public_sources.tbz2` for your exact L4T version from NVIDIA on your laptop and `scp` it over. You need only `kernel/kernel-5.10`, plus the Unitree kernel config:

```bash
zcat /proc/config.gz > /tmp/unitree.config
```

If the header package installs but `/lib/modules/$(uname -r)/build` is still missing, create the symlink by hand:

```bash
ls /usr/src/ | grep linux-headers
sudo ln -sfn /usr/src/linux-headers-<exact-version> /lib/modules/$(uname -r)/build
```

### 3.7 Build the out-of-tree driver

Use the maintained fork, not the vendor tarball that ships with the dongle:

```bash
git clone https://github.com/Mange/rtl8192eu-linux-driver
cd rtl8192eu-linux-driver
```

If the robot has no internet, clone on the laptop and `scp -r` the directory across.

Set the platform in the `Makefile`. **Confirm the exact symbol names rather than guessing**, because a wrong name silently leaves the x86 target selected and the build fails in a confusing place:

```bash
grep -n '^CONFIG_PLATFORM' Makefile
```

For aarch64, the correct pair is:

```make
CONFIG_PLATFORM_I386_PC = n
CONFIG_PLATFORM_ARM_AARCH64 = y
```

(Not `CONFIG_PLATFORM_ARM64`, and not `CONFIG_PLATFORM_ARM_RPI`, which is 32-bit ARM and sets a cross-compile prefix that breaks a native build.)

**Install via DKMS**, which is why `dkms` is in the prerequisites. This survives kernel updates and gives you a clean uninstall path:

```bash
sudo dkms add .
sudo dkms install rtl8192eu/1.0
sudo dkms status                 # expect: rtl8192eu, 1.0, <kernel>, aarch64: installed
```

Plain `make -j4 && sudo make install && sudo depmod -a` also works, but the module then disappears on the next kernel bump with no warning. Prefer DKMS.

Then:

```bash
sudo modprobe 8192eu
ip link
```

Reboot before declaring victory.

### 3.8 Determine which driver actually bound the device

After a reboot, either module could own the interface. `sudo depmod -a` alone can make the in-tree `rtl8xxxu` loadable when it was not before, so your compile may have been incidental.

```bash
readlink -f /sys/class/net/wl*/device/driver
lsmod | grep -E '8192eu|rtl8xxxu'
```

- Path ends in **`rtl8192eu`** or **`8192eu`** with a non-zero use count: your build is doing the work.
- Path ends in **`rtl8xxxu`**: the kernel's own driver picked it up.

### 3.9 Blacklist the loser, not the winner

**Hazard, and a correction to prior practice:** blacklist only the module that is **not** bound. Blacklisting `rtl8xxxu` when `rtl8xxxu` is the driver in use removes your Wi-Fi entirely, and the failure appears only after reboot, which makes it hard to trace back.

Both modules present at boot will race and produce intermittent drops, so do blacklist one of them.

If **8192eu** won:

```bash
printf 'blacklist rtl8xxxu\ninstall rtl8xxxu /bin/false\n' | sudo tee /etc/modprobe.d/rtl-blacklist.conf
sudo update-initramfs -u
```

If **rtl8xxxu** won:

```bash
printf 'blacklist 8192eu\ninstall 8192eu /bin/false\n' | sudo tee /etc/modprobe.d/rtl-blacklist.conf
sudo update-initramfs -u
```

Reboot and confirm the interface still comes up. Surviving a reboot is the definition of working, not working in the current session.

### 3.10 NetworkManager is unmanaged on the Go2

This is the one that costs people an afternoon. The Unitree image ships `managed=false`, so the interface shows in `ip link` while NetworkManager ignores it completely. The symptom is "there is no Wi-Fi in settings" on a robot whose dongle is working perfectly.

**Hazard: read this before running the sed.** Flipping the global flag lets NetworkManager take over **every** unmanaged interface, including the wired interface carrying your SSH session and all robot-internal DDS traffic. You can lose the session and the robot's internal networking in the same instant.

Protect the wired interface first. Substitute your actual wired interface name:

```bash
sudo tee /etc/NetworkManager/conf.d/99-unmanaged-eth.conf >/dev/null <<'EOF'
[keyfile]
unmanaged-devices=interface-name:eth0
EOF
```

Then flip the global flag:

```bash
sudo sed -i 's/managed=false/managed=true/' /etc/NetworkManager/NetworkManager.conf
grep -n managed /etc/NetworkManager/NetworkManager.conf     # verify the edit actually landed
```

If `grep` shows nothing, the key was absent and the `sed` was a no-op. Add it under the `[ifupdown]` section by hand.

```bash
sudo systemctl restart NetworkManager
nmcli device        # wl* should no longer read "unmanaged"; eth0 should still read "unmanaged"
```

### 3.11 Connect

```bash
nmcli device wifi list --rescan yes
```

Use the **literal interface name**. `wl*` is a shell glob against the filesystem, not an interface pattern, and will not resolve here:

```bash
WLAN=$(ls /sys/class/net | grep -m1 '^wl')
nmcli device wifi connect "<SSID>" password "<pw>" ifname "$WLAN"
nmcli connection modify "<SSID>" connection.autoconnect yes
```

Optionally pin the robot's address so SSH targets stay stable:

```bash
nmcli connection modify "<SSID>" ipv4.method manual \
  ipv4.addresses 192.168.1.50/24 ipv4.gateway 192.168.1.1 ipv4.dns 8.8.8.8
```

`netplan` or `wpa_supplicant` equivalents are fine if you prefer them.

**Gate 3:** after a full reboot, `nmcli device` shows the wl interface connected, `ping -c3 1.1.1.1` succeeds, and `ssh` over Wi-Fi works. Re-check `timedatectl` now that NTP is reachable.

---

## Phase 4: Bluetooth (Optional)

An AC600-class USB adapter (Wi-Fi + Bluetooth 4.2 combo, typically RTL8821CU silicon) works out of the box on Go2 EDU U4 units.

Two caveats:

- The **Bluetooth** half binds to `btusb` and works. The **Wi-Fi** half of RTL8821CU is *not* supported by in-tree `rtl8xxxu` on 5.10 and needs its own out-of-tree driver. Do not plan on this adapter for Wi-Fi.
- Audio profile support is constrained by the image's PulseAudio 13.99: A2DP sink works, HSP/HFP support is unreliable. If your use case needs a Bluetooth headset microphone, validate it early rather than assuming.

```bash
bluetoothctl show
```

---

## Phase 5: Audio

```bash
sudo apt install -y alsa-utils espeak-ng ffmpeg
```

Verify a playback device actually exists on the Jetson before writing playback code against it:

```bash
aplay -l                          # expect at least one card
speaker-test -c 2 -t wav -l 1     # audible output
```

If `aplay -l` reports no soundcards, the Jetson has no local ALSA sink and audio must route through the robot's audio service over DDS instead. That is a different code path, so find out now rather than while debugging the app.

**Gate 5:** `speaker-test` produces audible sound, and `espeak-ng "hello"` speaks.

---

## Phase 6: Repository access

`neuro-san-robotics` is **private**, so the robot needs credentials.

Cleanest option, no secrets left on the robot: **SSH agent forwarding**. Disconnect, then reconnect from the laptop with your key loaded:

```bash
# on the laptop
ssh-add -l                        # confirm the key is in the agent
ssh -A unitree@<robot-ip>
```

Then on the robot:

```bash
ssh -T git@github.com             # expect: Hi <user>! You've successfully authenticated
cd ~ && mkdir -p exp && cd exp
git clone git@github.com:cognizant-ai-lab/neuro-san-robotics.git
cd neuro-san-robotics
```

Alternatives if agent forwarding is unavailable: a read-only deploy key generated on the robot, or a fine-grained PAT over HTTPS. Avoid copying your personal private key onto a shared robot.

Everything below assumes the repo is at `~/exp/neuro-san-robotics`. Keep this consistent, because the RealSense phase hardcodes paths.

**Gate 6:** repo cloned, `git log -1` shows the expected head commit.

---

## Phase 7: Python environment

Either `pyenv` or `uv` works. This guide uses `uv`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc          # or: export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Create and **activate** the environment:

```bash
cd ~/exp/neuro-san-robotics
unset PYTHONPATH                  # drop the ROS 3.8 path, see 2.4
uv venv --python 3.11.13
source .venv/bin/activate
python -V                         # must print 3.11.13
```

Set up environment variables:

```bash
cp .env.example setmyenv.sh
nano setmyenv.sh                  # add your OPENAI_API_KEY
```

**Check that every line in `setmyenv.sh` is prefixed with `export`.** A plain `KEY=value` assignment is visible to the sourcing shell only and will not reach any subprocess, which presents as a missing API key despite the file being correct. If the file uses bare assignments, either add `export` or source it with auto-export:

```bash
set -a; source setmyenv.sh; set +a
```

Then install dependencies:

```bash
source setmyenv.sh
uv pip install -r requirements.txt
```

**Gate 7:** `python -c "import os; print(bool(os.environ.get('OPENAI_API_KEY')))"` prints `True` in a **fresh** shell after sourcing.

---

## Phase 8: CycloneDDS and the Unitree SDK

Both repos are cloned inside the project directory.

```bash
cd ~/exp/neuro-san-robotics
git clone https://github.com/unitreerobotics/unitree_sdk2_python
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
```

### 8.1 Build CycloneDDS from source

```bash
cd cyclonedds && mkdir -p build install && cd build

cmake .. -DCMAKE_INSTALL_PREFIX=../install \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DBUILD_DDSPERF=OFF

cmake --build . --target install -j4
cd ..
export CYCLONEDDS_HOME="$(pwd)/install"
```

Sanity check the API generation. The 0.10.x layout uses the older header names:

```bash
ls "$CYCLONEDDS_HOME/include/dds/ddsi/" | grep radmin      # want q_radmin.h, not ddsi_radmin.h
```

### 8.2 Persist CYCLONEDDS_HOME

**This is a runtime dependency, not just a build-time one.** The Python binding resolves `libddsc.so` through `CYCLONEDDS_HOME` on every import, so the variable must be set in every shell that runs the app. Add it to `setmyenv.sh` now:

```bash
export CYCLONEDDS_HOME="$HOME/exp/neuro-san-robotics/cyclonedds/install"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH"
```

Skipping this produces a `CycloneDDSLoaderException` on import in any new shell, hours after the build appeared to succeed.

### 8.3 Install the Python bindings

```bash
uv pip install "cyclonedds==0.10.2" --no-binary cyclonedds
cd unitree_sdk2_python
uv pip install -e . --no-binary cyclonedds
cd ..
```

**Gate 8:** in a **fresh** shell:

```bash
cd ~/exp/neuro-san-robotics && source setmyenv.sh && source .venv/bin/activate
python -c "import cyclonedds; print(cyclonedds.__version__)"
python -c "import unitree_sdk2py; print('sdk ok')"
```

---

## Phase 9: DDS configuration and pub/sub test

The RealSense phase assumes `~/cyclonedds.xml` exists and names the robot-facing wired interface. Create it here so it is not a hidden prerequisite.

```bash
cat > ~/cyclonedds.xml <<'EOF'
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="eth0" priority="default" multicast="default" />
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>
EOF

xmllint --noout ~/cyclonedds.xml && echo XML_OK
```

Substitute your actual robot-side interface for `eth0`. Add to `setmyenv.sh` if the app expects it:

```bash
export CYCLONEDDS_URI="file://$HOME/cyclonedds.xml"
```

### 9.1 Pub/sub test

Run the publisher and subscriber examples from the `unitree_sdk2_python` README, in two SSH sessions. Both need the network interface argument, because `ChannelFactoryInitialize` binds to a named interface and will silently find no peers otherwise.

```bash
# session A
cd ~/exp/neuro-san-robotics/unitree_sdk2_python && python example/helloworld/subscriber.py
# session B
cd ~/exp/neuro-san-robotics/unitree_sdk2_python && python example/helloworld/publisher.py
```

**Gate 9:** the subscriber prints messages produced by the publisher. If nothing arrives, the interface name is wrong, not the SDK.

TLS setup is handled automatically by a script in the repo, so no manual step is required.

---

## Phase 10: RealSense depth camera (Optional)

This is a long path for an optional capability. Skip it unless depth navigation is in scope for the demo.

**Assumptions**

- Repo at `~/exp/neuro-san-robotics`, venv at `.venv` (Python 3.11, uv-managed)
- CycloneDDS 0.10.x built at `<repo>/cyclonedds/install`, bindings pinned to `cyclonedds==0.10.2`
- `~/cyclonedds.xml` exists (Phase 9), robot interface `eth0`
- Camera: D435i (`8086:0b3a`)

### 10.1 apt hygiene (Optional, non-blocking)

The Open Robotics signing key expired 2025-06-01, so the two Tsinghua ROS mirrors throw `EXPKEYSIG F42ED6FBAB17C654`. Every repo you actually need reports `Hit:`, so this blocks nothing. To silence it:

```bash
grep -rn tsinghua /etc/apt/sources.list /etc/apt/sources.list.d/
```

If the entries carry `[signed-by=...ros-archive-keyring.gpg]`:

```bash
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
```

Otherwise (legacy path, works on focal, emits a deprecation warning):

```bash
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```

**Do not disable those repos.** `ros-noetic-realsense2-camera` and friends came from them.

### 10.2 Pre-flight

```bash
lsusb | grep -i intel          # expect 8086:0b3a
lsusb -t                       # the D435i must show 5000M, not 480M
pgrep -af realsense            # must be empty
ls ~/cyclonedds.xml
xmllint --noout ~/cyclonedds.xml && echo XML_OK
```

Interpreting that output:

- **`5000M` on a 10000M root hub:** proper USB3 path, so the depth resolutions `DepthProcessor` requests will open.
- **`480M`:** you are on USB2. Swap the cable or the port before going further, or depth streams fail to open with an error that never mentions USB.
- **Five Video interfaces plus one HID on Dev 2:** the full D435i, meaning depth, the IR pair, RGB, and the HID interface for the IMU. All present.
- **`pgrep` empty:** nothing holds the device.

If `pgrep` is not empty, a `realsense2_camera_node` holds the device exclusively. Stop it, and check whether it is started by a factory service that will bring it back after reboot:

```bash
systemctl list-units --type=service | grep -iE 'realsense|ros'
```

### 10.3 Build dependencies

```bash
sudo apt-get install -y git cmake build-essential pkg-config \
  libssl-dev libusb-1.0-0-dev libudev-dev libgtk-3-dev \
  libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev v4l-utils
```

### 10.4 Clone and install udev rules

```bash
export SDK_VER=2.55.1
cd ~ && git clone --depth 1 --branch "v${SDK_VER}" \
  https://github.com/IntelRealSense/librealsense.git "librealsense-${SDK_VER}"
cd "librealsense-${SDK_VER}"
```

**Unplug the camera**, then:

```bash
sudo ./scripts/setup_udev_rules.sh
ls /etc/udev/rules.d/ | grep -i realsense      # expect a realsense rules file
```

**Replug the camera.** Skipping this leaves the device root-only, and `query_devices()` returns 0 with no useful error.

### 10.5 CMake build

Start this step over from a clean `build/` on any failure.

```bash
cd ~/exp/neuro-san-robotics && source .venv/bin/activate
python -V                                      # must be 3.11.x

cd ~/librealsense-2.55.1 && mkdir -p build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -DPython_ROOT_DIR="$(python -c 'import sys; print(sys.base_prefix)')" \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release
make -j4
```

Non-obvious constraints:

- **Confirm CMake reports Python 3.11, not 3.8.** Foxy's `.bashrc` setup puts a 3.8 interpreter in play, and if CMake latches onto it the resulting binding is unusable. `-DPython_ROOT_DIR` is included above pre-emptively rather than as a recovery step, because a wrong pick means a full rebuild.
- **`-j4`, not `$(nproc)`.** librealsense OOMs the Jetson otherwise. This is also why Phase 0 checks for swap.
- **`FORCE_RSUSB_BACKEND=ON`** keeps you clear of kernel patching, which you do not want to attempt on the Tegra kernel. It also routes the IMU through libusb, so no `hid_sensor_*` modules are needed. The `uvcvideo` bindings visible in `lsusb -t` are fine, because librealsense detaches the kernel driver on open.
- **Never `sudo make install`.** It would shadow the ROS-managed librealsense 2.50.0 in `/opt/ros/noetic` and put the factory ROS stack at risk. The version-qualified sonames (`.so.2.50` versus `.so.2.55`) keep the two isolated only as long as yours stays in its build tree.

Success looks like:

```
Linking CXX shared library ../../Release/pyrealsense2.cpython-311-aarch64-linux-gnu.so
```

### 10.6 Locate artifacts and set paths

```bash
find ~/librealsense-2.55.1/build \( -name 'pyrealsense2*.so' -o -name 'librealsense2.so*' \)
```

Set the paths from what `find` actually reports rather than assuming. Both artifacts normally land in `build/Release`:

```bash
export SDK_VER=2.55.1
export PYTHONPATH="$HOME/librealsense-2.55.1/build/Release:$HOME/exp/neuro-san-robotics:$PYTHONPATH"
export LD_LIBRARY_PATH="$HOME/librealsense-2.55.1/build/Release:$LD_LIBRARY_PATH"
```

Add these to `setmyenv.sh`. Use the absolute repo path, not `$PWD`, because `$PWD` resolves to whatever directory you happened to be in when sourcing. Leave everything else in `.env.example` alone unless absolutely necessary.

Then, in a fresh shell:

```bash
cd ~/exp/neuro-san-robotics && source setmyenv.sh      # expect no WARN
```

### 10.7 Gates

**Gate 10a: tools see the camera**

```bash
RS_BIN="$(find ~/librealsense-2.55.1/build -name rs-enumerate-devices -type f | head -1)"
echo "$RS_BIN" && "$RS_BIN"
```

Want: a D435i entry with serial and firmware. Failure here is udev rules or USB, not Python. An empty `RS_BIN` just means `BUILD_TOOLS` did not take, so skip ahead.

**Gate 10b: Python sees the camera**

```bash
python - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
print("pyrealsense2:", rs.__file__)
print("devices:", len(ctx.query_devices()))
PY
```

Want: `devices: 1`, and a file path inside `librealsense-2.55.1/build/Release`. A path elsewhere means something earlier on `PYTHONPATH` wins. `devices: 0` means udev rules.

**Gate 10c: navigation depth backend**

```bash
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

Want: `backend: realsense`, `grid: True`, and a finite `center_depth_m` near the true distance to a test object. This is the first gate that opens a stream, so a failure here after 10b passes points at stream configuration or USB bandwidth, not installation.

Leave `NAV_OBSTACLE_SOURCE` and `NAV_DEPTH_CAMERA_SOURCE` unset. Depth is the default and the backend auto-detects.

---

## Phase 11: Run the application

```bash
cd ~/exp/neuro-san-robotics
source setmyenv.sh
source .venv/bin/activate
python apps/conscious_assistant/interface_flask.py
```

To reach the interface from your laptop, the Flask app must bind `0.0.0.0` rather than `127.0.0.1`. Confirm what it is listening on:

```bash
ss -ltnp | grep python
```

Then browse to `http://<robot-ip>:<port>` from the laptop.

**Gate 11:** the interface loads from the laptop, a spoken or typed prompt reaches the agent, and the robot responds. Verify obstacle avoidance is still off (Phase 1.3) before issuing any motion command.

---

## Quick troubleshooting index

| Symptom | Likely cause | Section |
| :--- | :--- | :--- |
| SSH hangs, no response | Laptop not on `192.168.123.0/24` | 2.1 |
| No Wi-Fi networks in settings | NetworkManager `managed=false` | 3.10 |
| Wi-Fi worked, gone after reboot | Blacklisted the module that was actually bound | 3.9 |
| Module compiles, will not load | vermagic / L4T header mismatch | 3.6 |
| Cannot see the venue SSID at all | RTL8192EU is 2.4 GHz only | 3.1 |
| TLS or certificate errors on API calls | System clock skew | 0.3 |
| Missing API key despite correct file | `setmyenv.sh` lacks `export` | Phase 7 |
| Import errors, wrong Python version | ROS 3.8 `PYTHONPATH` leaking | 2.4 |
| `CycloneDDSLoaderException` on import | `CYCLONEDDS_HOME` not persisted | 8.2 |
| Pub/sub sees no peers | Wrong network interface name | 9.1 |
| `devices: 0` from pyrealsense2 | udev rules not installed or camera not replugged | 10.4 |
| Depth streams fail to open | USB2 path (`480M`) | 10.2 |
| librealsense build killed | `-j` too high, or no swap | 10.5, 0.2 |
