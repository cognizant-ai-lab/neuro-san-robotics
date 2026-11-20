# Robot performance check: step by step, from zero

Goal: find out **why the robot is slow when we talk to it**. You run one
measuring script on the robot, copy its output, and paste it to Claude.
You do not need to understand Linux for this. Just follow the steps in order.

Time needed: about 15 minutes.

## Words you will see

| Word | Meaning |
|------|---------|
| Terminal | The window where you type commands. On the Mac: press `Cmd + Space`, type `Terminal`, press Enter. |
| Command | One line of text you type (or paste) into the terminal, then press Enter. |
| `ssh` | "Remote control": your Mac's terminal types commands on the robot's computer. |
| `Ctrl + C` | Stops whatever is currently running in the terminal. |
| The app | The program on the robot that listens and talks (`interface_flask.py`). |

Tips:

- Copy each command exactly. Paste into the terminal with `Cmd + V`, then press Enter.
- When you type a password, **nothing appears on screen**. That is normal. Type it and press Enter.
- Lines starting with `#` are explanations. You do not need to type them.

## Before you start: fill in these three things

The values below come from the project README and may be different for your robot.
Ask a colleague if you are unsure.

| What | Value used in this guide | Yours |
|------|--------------------------|-------|
| Robot IP address | `10.194.17.116` | at home: you find it in Step 1b |
| Project folder on the robot | `exp/neuro-san-robotics-hormoz` | |
| Password for user `unitree` | ask a colleague | |

Wherever you see `10.194.17.116` or `exp/neuro-san-robotics-hormoz` below,
replace it with your value.

## Step 1: Get the robot ready

1. Put the robot **on the floor with about 2 metres of free space** around it.
   The test only asks it questions, but the robot decides for itself and may still gesture.
2. Switch it on: one **short press**, then one **long press** on the power button.
3. Wait about 2 minutes until it stands up.
4. Make sure your Mac is in the **same network (WiFi)** as the robot.
   If the robot is in a new place (for example at home), do Step 1b first.

## Step 1b: Connect the robot to a new WiFi (only once per place)

The robot only knows the office WiFi, and it joins any WiFi it knows by itself.
To teach it your home WiFi, use a monitor and keyboard once.

### 1. Monitor and keyboard

Plug a monitor (a TV with HDMI works too), a USB keyboard and a mouse into the robot's back
(a USB-C docking station works too) and log in as `unitree`. You see a normal Linux desktop.

### 2. Join your home WiFi

1. Click the icons at the **top right** of the screen, open the WiFi entry, pick your
   **normal** home WiFi and type its password. Not the guest WiFi: guest networks stop
   devices from talking to each other, so `ssh` from the Mac would never work.
2. Same menu, **Wi-Fi Settings**, then the **gear** next to your connected WiFi.
   Write down the **IPv4 Address** (for example `192.168.178.57`) and the **Hardware Address**.

The robot remembers your home WiFi from now on, and still knows the office WiFi.

If your WiFi does not show up in the list: the robot's WiFi stick may only see
2.4 GHz networks. Enable 2.4 GHz in your router.

### 3. Give the robot a fixed address in the FRITZ!Box

Otherwise the FRITZ!Box may hand out a new address some day, and `ssh` stops working.

1. On the Mac, open `http://fritz.box` and log in.
2. **Heimnetz > Netzwerk**, tab **Netzwerkverbindungen**. Find the robot: the line with the
   IPv4 or hardware address you wrote down.
3. Click the pencil (edit) next to it.
4. Tick **"Diesem Netzwerkgerät immer die gleiche IPv4-Adresse zuweisen"**, click **Übernehmen**.

This address, for example `192.168.178.57`, is **the robot's IP address at home**.
Write it into the table above and use it everywhere this guide says `10.194.17.116`.

### 4. Log in from the Mac without a password

Monitor and keyboard are no longer needed. On the **Mac** (with the Mac on your home WiFi),
replace `192.168.178.57` with the robot's address:

```shell
# a short name: afterwards "ssh robot" is enough
printf 'Host robot\n    HostName 192.168.178.57\n    User unitree\n' >> ~/.ssh/config

# a key instead of the password (press Enter at every question of ssh-keygen)
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519
ssh-copy-id robot

# test: must print the robot's name without asking for a password
ssh robot hostname
```

From now on `robot:` replaces `unitree@10.194.17.116:` in the `scp` commands, and
Claude Code on the Mac can run commands on the robot with `ssh robot "..."`.
Commands with `sudo` still need the password, typed by you.

## Step 2: Copy the measuring scripts to the robot

The scripts are on your Mac, not yet on the robot. Open a terminal **on the Mac** and run:

```shell
cd ~/Dev/neuro-san-robotics

ssh unitree@10.194.17.116 "mkdir -p exp/neuro-san-robotics-hormoz/tests/scenarios"

scp scripts/cpu_probe.sh scripts/talk_driver.py unitree@10.194.17.116:exp/neuro-san-robotics-hormoz/scripts/

scp tests/scenarios/robot_talk.json unitree@10.194.17.116:exp/neuro-san-robotics-hormoz/tests/scenarios/
```

Each command asks for the robot password.
The first time, it also asks `Are you sure you want to continue connecting?` Type `yes` and press Enter.

## Step 3: Terminal 1, start the app

Use the same terminal window. We call it **Terminal 1**.

```shell
ssh unitree@10.194.17.116
```

If it asks `Foxy(1) / Noetic(2)?`, type `1` and press Enter. (It asks this at every login.)

The start of the line now changes to something like `unitree@...`. From now on,
everything you type in this window runs **on the robot**.

```shell
cd exp/neuro-san-robotics-hormoz
source setmyenv.sh
export NAV_MAP_FILE=""     # only when NOT in the office, see "Settings for home"
python apps/conscious_assistant/interface_flask.py
```

Lots of text scrolls by. That is fine. **Leave this window open and do not type in it.**
Wait about 1 minute.

Check that it works: open `https://10.194.17.116:5001` in your browser.
The browser warns about the certificate. Choose "Advanced" and then "Proceed".
You should see the robot's page.

## Step 4: Terminal 2, run the measurement

Open a **second** terminal window (`Cmd + N` in Terminal). We call it **Terminal 2**.

```shell
ssh unitree@10.194.17.116
cd exp/neuro-san-robotics-hormoz
source setmyenv.sh
```

Install the two helper tools (only needed the first time):

```shell
pip install py-spy "python-socketio[client]"
```

Now the measurement:

```shell
sudo -v
bash scripts/cpu_probe.sh 2>&1 | tee ~/probe_result.txt
```

- `sudo -v` asks for the password once. It lets the script look inside the app.
- The second command takes about **1 minute**. The robot is asked three questions
  (its name, a joke, 2 + 2) and should answer out loud.

## Step 5: Read the result

The last line is one word:

| Last line | Meaning |
|-----------|---------|
| `RED` | The script caught the problem. **This is what we want.** Lines containing `RED:` say what was too slow or too busy. |
| `GREEN` | The script did not see the problem this time. Run the second command from Step 4 again, and tell Claude if it stays green even though the robot feels slow. |
| `no process matching ...` | The app is not running. Check Terminal 1 (Step 3). |

You do not need to understand the rest of the output.

## Step 6: Send the result to Claude

Select **everything** the script printed in Terminal 2, copy it (`Cmd + C`),
and paste it into Claude Code together with this line:

```
/diagnosing-bugs Here is the cpu_probe output from the robot:
```

Claude uses it to find the slow part and fix it. After a fix, you repeat
Steps 2 to 5. When the last line is `GREEN` and the robot feels fast, it is done.

## Step 7: Tidy up

1. Terminal 1: press `Ctrl + C` to stop the app. Then type `exit`.
2. Terminal 2: type `exit`.
3. Switch the robot off if you are finished.

## Settings for home

None of these are required for the measurement, except the first one.

| Idea | Why | How |
|------|-----|-----|
| **Switch off the office map** | The robot's map describes the office. At home it would think it stands at the office charging station and could try to walk to places that do not exist. Without a map it still talks, turns and walks forward, and avoids obstacles. | Type `export NAV_MAP_FILE=""` after `source setmyenv.sh` and before starting the app (already in Step 3). |
| Browser warning after an IP change | The app makes a new security certificate for the new address when it starts, so the browser asks again. | Nothing to fix. Choose "Advanced" and "Proceed" once more. |
| Make it quieter | The speaker is loud for a living room. | On the robot: `python scripts/set_audio_volume.py --help` shows the options. |
| Tell it where it lives | It otherwise says it lives in the lab. | `export ROBOT_HOME="a flat in Berlin"` before starting the app. Keep it lowercase at the start, it is used mid-sentence. |

Two things to keep in mind when measuring at home:

- The robot needs **internet**: every answer comes from OpenAI over your WiFi.
  A slow or far-away WiFi makes answers slow even when the robot itself is fine.
  Put the robot close to the router for the measurement.
- So a result from home is not exactly the same as one from the office. The CPU
  numbers are comparable, the reaction times only roughly. Tell Claude where you measured.

## If something goes wrong

| What you see | What to do |
|--------------|-----------|
| `ssh: connect to host ... timed out` | Wrong IP address or different network. The IP can change: plug a monitor into the robot's back, log in, and look at the network settings. |
| `Permission denied` | Wrong password. Remember: typing shows nothing. |
| `No such file or directory` | The project folder on the robot has a different name. Type `ls` and `ls exp` to see what exists, or ask a colleague. |
| `python: command not found` or `No module named ...` | You forgot `source setmyenv.sh` in that terminal window. |
| `No module named 'socketio'` | Run the `pip install` line from Step 4 again. |
| No "hottest" lines in the output | `py-spy` could not look inside the app. Run `sudo -v` again, then the measurement again. |
| The robot starts moving unexpectedly | Terminal 1: `Ctrl + C`. Keep the remote control in reach. |
| The robot answers `No map loaded` | Expected at home, because the office map is switched off. |
| Anything else | Copy the error text and paste it to Claude. Nothing in this guide can damage the robot's software. |
