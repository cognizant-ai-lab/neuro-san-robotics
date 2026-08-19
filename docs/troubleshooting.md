# Troubleshooting Guide

## cyclonedds build issue

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

## cyclonedds.xml pins name="eth0", but eth0 had no IP address. 

CycloneDDS enumerates interfaces by their addresses, so an address-less eth0 doesn't exist as far as it's concerned even with the link up at 1000Mb/s.
Hence eth0: does not match an available interface → channel factory init error.

The cause was NetworkManager's profile: "Wired connection 1" was set to ipv4.method auto (DHCP), and the robot's internal link has no DHCP server, so it timed out and left the interface bare.

### The fix

```bash
sudo nmcli con mod "Wired connection 1" \
  ipv4.method manual \
  ipv4.addresses 192.168.123.99/24 \
  ipv4.gateway "" \
  ipv4.never-default yes
```

```bash
sudo nmcli con up "Wired connection 1"
```

Why each argument matters:
- ipv4.method manual — stop asking for DHCP; the robot link has no server.
- ipv4.addresses 192.168.123.99/24 — the Go2's internal subnet. .161 is the sport service; .99 is a free host octet for this companion computer.
- ipv4.gateway "" — no gateway on this link; it's point-to-point to the robot.
- ipv4.never-default yes — critical. Without it eth0 can steal the default route and kill internet on the Wi-Fi dongle.

nmcli con mod writes straight to the profile on disk, and autoconnect was already yes, so this survives reboot. No extra persistence step needed.

### Verification

```bash
ip -br addr show eth0        # eth0 UP 192.168.123.99/24
ping -c2 192.168.123.161     # Go2 sport service replies
```

And the DDS layer itself, without touching the robot:

```bash
PYTHONPATH="$PWD" LD_LIBRARY_PATH="$PWD/cyclonedds/install/lib" \
.venv/bin/python -c "
from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelFactoryInitialize
ChannelFactoryInitialize(0,'eth0'); print('DDS OK')"
```

The SDK writes a Cyclone trace to /tmp/cdds.LOG — grep interfaces: /tmp/cdds.LOG shows exactly what Cyclone can see, which is the fastest way to confirm this class of failure:

interfaces: lo udp/127.0.0.1(q1) eth0 udp/192.168.123.99(q9) docker0 ... wlan0 ... selected interfaces: eth0 (index 3 priority 0)

Before the fix, eth0 was simply absent from that line.

What was not needed

The cyclonedds.xml / CYCLONEDDS_URI change did nothing for this. ChannelFactoryInitialize builds its DDS config inline at unitree_sdk2_python/unitree_sdk2py/core/channel.py:210-218 and passes it to Domain(id, config), which overrides CYCLONEDDS_URI. The interface name comes from go2_macros.py:9-13 (GO2_NETWORK_INTERFACE → CYCLONEDDS_NETWORK_INTERFACE → "eth0"). No XML file influences it.