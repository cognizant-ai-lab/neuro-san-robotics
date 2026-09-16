# Environment variables guide

Every setting in `setmyenv.sh`, and what it is for. `.env.example` is the
template you copy to `setmyenv.sh`; it stays deliberately short, and this file
holds the reasoning.

```shell
cp .env.example setmyenv.sh
# edit setmyenv.sh, then:
source setmyenv.sh
```

Upgrading an existing robot? Nothing above the "new in this version" line in
`.env.example` has changed, and every setting below it defaults to how the
robot already behaves. Copy the new template, put your API key back, and the
robot runs exactly as before.

---

## Robot platform

### `RS_VER`

RealSense release to look for, default `2.55.1`. The Python bindings are built
from source on Jetson because there is no aarch64 wheel, so this points at
`$HOME/librealsense-$RS_VER/build`. `setmyenv.sh` warns if it finds no
`pyrealsense2*.so` there.

### `PYTHONPATH`, `LD_LIBRARY_PATH`

Put the RealSense build, the repo root, `coded_tools/` and the CycloneDDS
libraries on the loader's path. Snapshotted once via `SETMYENV_BASE_*` so
re-sourcing the file does not append duplicates.

### `CYCLONEDDS_HOME`, `CYCLONEDDS_URI`

Where the locally built CycloneDDS lives, and its config file. This is the
transport the Unitree SDK rides on. It is also why the robot is pinned to
Python 3.11. See [Python version](#python-version).

### `GO2_NETWORK_INTERFACE` / `CYCLONEDDS_NETWORK_INTERFACE`

Interface the Unitree SDK talks over, default `eth0`. Override only if the
robot network is not on `eth0`.

### `AGENT_TOOL_PATH`, `AGENT_MANIFEST_FILE`

Where neuro-san finds the coded tools and the agent network manifest.

---

## Robot identity

### `ROBOT_NAME`, `ROBOT_HOME`

Who and where this robot is. The name reaches the agent persona, the speech
recogniser's prompt and the web UI; the home lab reaches the persona and the
recogniser.

`ROBOT_HOME` is read mid-sentence ("You live in ..."), so keep the leading
lowercase article. Both are read when the agent starts, so restart the app
after changing them.

```shell
export ROBOT_NAME="BIT-2"
export ROBOT_HOME="the Cognizant AI Lab in Bengaluru"
```

---

## Navigation

### `NAV_MAP_FILE`

The map belongs to the building, not the robot, so every robot in a lab points
at the same file. There is no default in code: leave it empty and the robot
reports it has no map and offers only `move_forward` and `turn`, which is the
right answer at a site nobody has mapped yet.

### `NAV_INITIAL_LOCATION`

Node in that map where the robot is parked at startup. Must name a real node in
`NAV_MAP_FILE`; the robot warns and starts unanchored if it does not.

---

## Web UI and TLS

### `ROBOT_HOST_IP`, `TLS_CERT_DIR`

The address browsers use to reach the Flask UI, embedded as an IP SAN in the
TLS certificate. DHCP rotates it, so it is auto-detected from the default route
rather than hardcoded; export it beforehand to override.

The certificate is refreshed by the app itself on every start, so an address
change usually needs nothing. To inspect or force it:

```shell
python scripts/setup_tls_certs.py --check
python scripts/setup_tls_certs.py --force
python scripts/setup_tls_certs.py --fresh   # forget previous networks
```

Addresses accumulate rather than replace each other. A robot that moves between
a few known routers ends up with a certificate covering all of them and stops
regenerating; only a router it has never seen triggers a new one. The list of
remembered addresses is bounded, so a network handing out a fresh lease every
day cannot grow it without limit.

`ROBOT_HOST_IP` is captured once when `setmyenv.sh` is sourced and then sticks
for the life of that shell, so it goes stale as soon as the robot changes
network. The certificate therefore covers both it *and* the address the robot
is actually on, which is why a stale value no longer leaves you serving a
certificate for somewhere you used to be.

Browsers withhold microphone access from anything but `localhost` over plain
HTTP, so without a certificate voice input will not work from another machine.

---

## Choosing a setup

Three choices, and they are independent. Pick each one separately:

| | What it decides | Variable |
|---|---|---|
| **Agents** | which LLM the agent networks think with | `AGENT_LLM_CLASS` |
| **Speech out** | whether the robot's voice is a hosted model or Piper on the robot | `GO2_TTS_ENGINE` |
| **Speech in** | whether transcription is a hosted model or Whisper on the robot | `GO2_STT_ENGINE` |

A fourth, `GO2_AUDIO_PROVIDER`, only says *which cloud* hosted speech calls. It
is never consulted once both speech settings are local, and the startup summary
leaves it out in that case rather than printing a value that decides nothing.

### Common setups

`-` means leave it unset.

| Setup | `AGENT_LLM_CLASS` | `GO2_TTS_ENGINE` | `GO2_STT_ENGINE` | Also needs |
|---|---|---|---|---|
| Everything on public OpenAI | - | - | - | `OPENAI_API_KEY` |
| Everything on Azure | `azure-openai` | - | - | Azure credentials + all 5 deployments |
| **Azure agents, speech on the robot** | `azure-openai` | `piper` | `local` | Azure credentials + chat deployment only |
| Azure agents, speech on public OpenAI | `azure-openai` | - | - | Azure credentials + `OPENAI_API_KEY` + `GO2_AUDIO_PROVIDER="openai"` |
| Hosted voice, listening on the robot | either | - | `local` | credentials for the hosted half |
| Robot voice, hosted listening | either | `piper` | - | credentials for the hosted half |
| Fully offline | - | `piper` | `local` | no API keys at all |

The bolded row is the usual Azure case, because Azure's speech models are
region-limited and frequently absent from the resource serving chat.

### Azure agents with speech on the robot

The complete set. Nothing else is required, and no `OPENAI_API_KEY`:

```shell
export AGENT_LLM_CLASS="azure-openai"
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export OPENAI_API_VERSION="2024-10-21"
export AZURE_OPENAI_DEPLOYMENT_NAME="<your-gpt-5.1-or-5.4-deployment>"

export GO2_TTS_ENGINE="piper"
export GO2_STT_ENGINE="local"
```

Naming `piper` and `local` also installs what they need when `setmyenv.sh` is
sourced: the Piper voice, and the Whisper weights.

Re-source `setmyenv.sh` after editing it, in the shell you start the app from.
The app reads the environment of the process that launches it, so an edit that
has not been sourced leaves it running on the old values.

Set these by **editing `setmyenv.sh`**, not by exporting them beforehand. The
file assigns both engines outright, so `GO2_TTS_ENGINE=piper source setmyenv.sh`
is overwritten by the file's own value. That is deliberate: it is what stops a
value left over in your shell from quietly changing what the robot does. It
does mean the file is the only place to set them.

Two things worth knowing about this setup:

- The `unigo2` network will report `model_name: gpt-4o-2024-08-06` while
  pointing at your chat deployment. Harmless, since Azure routes on the
  deployment name, but it reads oddly in logs. Set
  `AGENT_LLM_MODEL_NAME_LIGHT` to what the deployment really serves if that
  matters.
- Do not set `GO2_TTS_ENGINE` back to `auto` here. With
  `AZURE_OPENAI_ENDPOINT` set, `auto` makes the speech layer believe Azure is
  available, so every utterance would call a text-to-speech deployment that
  does not exist and wait out `GO2_OPENAI_TIMEOUT_SECONDS` before falling back
  to Piper. Naming the engines explicitly is what avoids that.

### Switching one thing later

Each row above changes independently, so moving one part does not disturb the
others:

| To change | Edit | Effect |
|---|---|---|
| Speech out → the robot | `GO2_TTS_ENGINE="piper"` | stops calling out for the voice |
| Speech out → hosted | `GO2_TTS_ENGINE="auto"` | hosted first, Piper if it fails |
| Speech in → the robot | `GO2_STT_ENGINE="local"` | stops calling out for transcription |
| Speech in → hosted | `GO2_STT_ENGINE="auto"` | hosted first, local Whisper if it fails |
| Hosted speech → public OpenAI | `GO2_AUDIO_PROVIDER="openai"` | speech leaves Azure, agents stay |
| Hosted speech → Azure | `GO2_AUDIO_PROVIDER="azure"` | needs the three Azure speech deployments |
| Agents → Azure | `AGENT_LLM_CLASS="azure-openai"` | every agent network at once |
| Agents → public OpenAI | unset `AGENT_LLM_CLASS` | back to `OPENAI_API_KEY` |

`auto` versus naming an engine is worth restating: `auto` tries hosted first
and falls back, which costs a failed request and its timeout before every
utterance at a site with no hosted model. Naming the engine skips that
entirely. Use `auto` where a hosted model genuinely exists.

---

## Model providers

Two separate APIs are in play, and they are configured independently.

1. **Agents**: the reasoning LLM behind every agent network. neuro-san owns
   this. It reads the `AZURE_OPENAI_*` variables itself, and each agent
   registry under `registries/` reads the `AGENT_LLM_*` variables below.
2. **Speech**: text-to-speech, push-to-talk transcription and ambient
   listening. neuro-san has no audio support at all, so this repo calls those
   endpoints directly and they need their own settings: the `GO2_*` variables.

They are deliberately separable. Azure's audio models are region-limited, so a
customer whose chat deployment lives in one resource often cannot host
`gpt-4o-mini-tts` or `whisper` alongside it. Any combination works: both on
Azure, both on public OpenAI, or agents on Azure with speech on OpenAI.

Changing model later is only ever an edit to `setmyenv.sh` plus a restart. No
HOCON file needs touching.

### Public OpenAI (default)

```shell
export OPENAI_API_KEY="sk-..."
```

Prefer exporting it outside the script, or keeping it in an untracked local
file, rather than committing it.

### Azure OpenAI

#### Credentials

Shared by the agents and, by default, by speech too.

```shell
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export OPENAI_API_VERSION="2024-10-21"
```

Using Entra ID instead of a resource key? Set `AZURE_OPENAI_AD_TOKEN` and leave
the key unset.

`OPENAI_API_VERSION` is **not** discoverable from Azure. It is a constant you
pick from Microsoft's published API version list, and the Python SDK refuses to
build a client without it.

It is **not the model's version**. `gpt-5.1` has a version of `2025-11-13`,
which is the model snapshot you choose in the Deployments blade; the
api-version is the REST contract, one of `2024-10-21`, `2025-04-01-preview`
and so on. Putting a model version here returns `404 Resource not found` on
every call, because Azure does not recognise it as an api-version. Newer
models generally need a recent preview: `2025-04-01-preview` works for the
GPT-5 series.

The endpoint is the resource **root**, with no path: the SDK appends
`/openai/deployments/<name>/...` itself. Both host forms work --
`https://<resource>.openai.azure.com` and
`https://<resource>.services.ai.azure.com` -- but a Foundry *project* URL
(`.../api/projects/<name>`) or a `/openai/v1` suffix will not, because the SDK
appends its own path on top of yours. `2024-10-21` is a safe GA value; use `preview` only
if you need something that has not reached GA. If an audio model returns 404 on
a deployment you can see in the portal, a too-old api-version is the usual
cause, because those models landed after the older ones.

#### Agents

Setting the class switches every agent network to Azure at once.

```shell
export AGENT_LLM_CLASS="azure-openai"
export AZURE_OPENAI_DEPLOYMENT_NAME="<your-chat-deployment>"
```

`AGENT_LLM_MODEL_NAME` is optional. Azure routes on the deployment name above;
this is carried for logs and token accounting. Any name works, including one
neuro-san has never heard of.

The `unigo2` "look alive" network runs on a cheaper model. Leave
`AZURE_OPENAI_DEPLOYMENT_NAME_LIGHT` and `AGENT_LLM_MODEL_NAME_LIGHT` unset to
run it on the same deployment as the assistant; `setmyenv.sh` fills the former
in for you.

#### Speech

```shell
export GO2_AUDIO_PROVIDER="auto"
export GO2_AZURE_TTS_DEPLOYMENT="<gpt-4o-mini-tts deployment>"
export GO2_AZURE_TRANSCRIBE_DEPLOYMENT="<whisper deployment>"
export GO2_AZURE_REALTIME_DEPLOYMENT="<gpt-4o-transcribe deployment>"
```

`GO2_AUDIO_PROVIDER` is `auto` by default, which follows
`AZURE_OPENAI_ENDPOINT`. Set `openai` to keep speech on public OpenAI while the
agents run on Azure, useful when the chat region carries no audio models. Set
`azure` to force it.

### What to deploy on Azure

Five models are in play. Azure offers all five under the same names this repo
uses on public OpenAI, so nothing needs substituting:

| Purpose | Model to deploy | Version | Variable |
|---|---|---|---|
| Conscious assistant | `gpt-5.1` | latest | `AZURE_OPENAI_DEPLOYMENT_NAME` |
| `unigo2` "look alive" | `gpt-4o` | `2024-08-06` | `AZURE_OPENAI_DEPLOYMENT_NAME_LIGHT` |
| Text to speech | `gpt-4o-mini-tts` | latest | `GO2_AZURE_TTS_DEPLOYMENT` |
| Push-to-talk STT | `whisper` | latest | `GO2_AZURE_TRANSCRIBE_DEPLOYMENT` |
| Ambient STT | `gpt-4o-transcribe` | latest | `GO2_AZURE_REALTIME_DEPLOYMENT` |

The `gpt-4o` version matters: Azure lets you pick it at deploy time, and
`2024-08-06` is the snapshot this repo has always used for that network.

#### Finding the deployment names

Deployment names are chosen by whoever deployed the model, so there is no
canonical name for `gpt-5.1`, so you have to look up what yours was called. In
the portal: Microsoft Foundry → your resource → **Deployments**. The Name
column is the deployment name; the Model column says what it serves.

From the CLI, which shows the mapping in one go:

```shell
az cognitiveservices account list \
  --query "[?kind=='OpenAI'].{name:name,rg:resourceGroup,loc:location}" -o table

az cognitiveservices account deployment list --name <resource> --resource-group <rg> \
  --query "[].{deployment:name, model:properties.model.name, version:properties.model.version}" \
  -o table
```

Endpoint and key come from the same resource:

```shell
az cognitiveservices account show --name <resource> --resource-group <rg> \
  --query properties.endpoint -o tsv
az cognitiveservices account keys list --name <resource> --resource-group <rg> \
  --query key1 -o tsv
```

#### Do not use GPT-5.6 or GPT-6 for the assistant

GPT-5.6 (sol / terra / luna) and GPT-6 Astra serve tool calling only over the
Responses API, which the bundled neuro-san does not speak. Every agent here
calls tools on every turn, so those deployments fail outright.

This is not a version to chase. neuro-san added Responses support in 0.7, which
needs Python 3.12, and this robot is held at 3.11 by CycloneDDS. `gpt-5.1`
through `gpt-5.5` are all fine.

---

## Navigation sensing

### `NAV_OBSTACLE_SOURCE`

What the robot avoids obstacles with.

| Value | Sensing |
|---|---|
| `depth` (default) | the front depth camera only |
| `fused` | depth camera and LiDAR together |
| `lidar` | LiDAR only |

The camera sees through a narrow cone in front of the robot; the LiDAR sees all
round. That difference matters most in **free navigation** -- moving without a
map, where nothing but live sensing keeps the robot off the furniture, and
anything approaching from the side is invisible to the camera until it is
already in the way.

`fused` is the recommendation for a new robot: it adds the LiDAR without giving
up the camera as a second opinion.

The template leaves this at `depth` so a robot already in service keeps the
sensing it was commissioned with until someone has run the check below on it.

### Verify the LiDAR before trusting it

Two things go wrong here and neither raises an error. Either no data arrives --
and the robot quietly navigates on depth alone -- or data arrives **rotated**,
which is the dangerous one: the obstacle map is turned, so the robot swerves
around empty floor and walks into a real wall.

```shell
python scripts/test_lidar.py --seconds 5
```

Want `backend: lidar:rt/utlidar/cloud`, a nearest-obstacle line that matches
where you are standing, and a picture whose walls match the room. Exit status is
0 only when usable data arrived, so it works as a bring-up gate.

Scans are merged over a few seconds and written as a PNG under
`~/lidar_checks` (`NAV_LIDAR_CHECK_DIR` moves it). A single rotation is a thin
scatter of points that no one can read; merged, the shape of the room appears.
The last ten snapshots are kept and older ones pruned.

Phase 10.9 of [the bring-up guide](bringup_guide.md) covers this in sequence.

### `NAV_LIDAR_POINTCLOUD_YAW_OFFSET_RAD`

How the LiDAR is bolted on, in radians. Defaults to 70 degrees, which is correct
for the units this repo was built against. A differently mounted one needs its
own, and it can be measured rather than guessed:

```shell
python scripts/test_lidar.py --calibrate
```

Put one unmistakable object a metre directly in front of the nose, closer than
anything else. With the rotation switched off, the bearing the LiDAR reports for
it *is* the mounting angle, so the offset that corrects it is that bearing
negated. The script prints the `export` line to paste in, then redraws the room
with it applied so you can confirm the object now sits straight ahead.

This works because coverage is 360 degrees. Nothing is missing from the scan; the
only question is which direction each point is filed under, and one known
direction is enough to pin that down.

Do not reach for `NAV_LIDAR_ANGLE_OFFSET_RAD` instead. That one applies to 2D
scan messages rather than the point cloud the Go2 publishes -- and because the
point-cloud offset falls back to it, setting it to `0` silently cancels the
70-degree default and rotates the map.

### What free navigation can do

With `NAV_MAP_FILE` empty the robot navigates relative to itself:

- **Works:** `move_forward`, `move_until_obstacle`, `turn`, `stop`, `status`,
  `obstacles`
- **Needs a map:** `navigate_to`, `set_location`, `destinations`

Without one the robot says so rather than guessing: *"No map loaded. Only
relative navigation (move_forward, turn) is available."*

### Other LiDAR settings

Rarely changed, but worth knowing they exist when something looks wrong:

| Variable | Default | Purpose |
|---|---|---|
| `NAV_USE_LIDAR` | on outside simulation | second switch; can disable LiDAR while `NAV_OBSTACLE_SOURCE` still says `fused` |
| `NAV_LIDAR_TOPIC` | `rt/utlidar/cloud` | DDS topic the Go2 publishes on |
| `NAV_LIDAR_MIN_HEIGHT` / `MAX_HEIGHT` | `-0.25` / `0.80` m | rejects the floor and the ceiling |
| `NAV_LIDAR_MIN_RANGE` / `MAX_RANGE` | `0.05` / `4.0` m | usable range |
| `NAV_LIDAR_SELF_MASK_FORWARD` / `REAR` / `HALF_WIDTH` | `0.45` / `0.35` / `0.25` m | stops the robot seeing its own body |
| `NAV_LIDAR_MAX_SAMPLE_AGE` | `0.75` s | how stale a scan may be before it is ignored |

---

## Speech engines and offline fallbacks

The robot does not need a hosted speech model to talk or listen. Both fall back
to engines running on the robot, which matters for a region with no audio
models deployed, and for a robot that is simply offline.

### `GO2_TTS_ENGINE`

Which engine speaks, and in what order engines are tried.

| Value | Meaning |
|---|---|
| unset / `auto` | try each engine in order, skipping ones not installed |
| `piper` | that engine only; its failures are raised, not hidden |
| `piper,espeak` | try exactly these, in this order |

Naming a single engine is a statement that you want to know when it breaks, so
it is a chain of one rather than a preference. `auto` is the forgiving mode,
where a timeout or a missing model quietly moves to the next engine.

Default order: hosted → piper → `say` (macOS) → espeak. Hosted is tried
first whenever a key is configured, which is what `auto` meant before
these engines were made swappable, and still means now.

Only engines that exist are listed. pocket-tts and Qwen3-TTS are candidates
but are not implemented; naming one would fail as though it were a typo.

`setmyenv.sh` sets this and `GO2_STT_ENGINE` explicitly rather than falling
back to whatever the shell already had. `source` runs in your current shell, so
a value exported earlier in that terminal would otherwise survive and change
what the robot does, including whether it downloads a model. If a robot
reports an engine you did not choose, check for a stale export:

```shell
echo "$GO2_TTS_ENGINE $GO2_STT_ENGINE"
unset GO2_TTS_ENGINE GO2_STT_ENGINE
```

### Piper, the offline voice

`requirements.txt` installs the Piper binary but not the voice it speaks with,
and the voice is a ~114 MB download rather than a Python package.

```shell
python scripts/install_piper_voice.py
python scripts/install_piper_voice.py --check
```

`setmyenv.sh` installs it automatically when `GO2_TTS_ENGINE` names `piper`,
since declaring that engine is the same as saying the robot depends on it.
Otherwise it warns while the voice is missing and leaves the download to you,
because sourcing an env file should not block on 114 MB. `GO2_PIPER_AUTO_INSTALL=1`
opts into fetching it regardless.

Without the voice, offline speech falls through to `espeak-ng`, which is
intelligible but markedly worse.

`GO2_PIPER_MODEL` and `GO2_PIPER_CONFIG` move where the voice lives. The
default is `~/piper_models/`, which on the robot is `/home/unitree/piper_models`
and on a laptop is under your own home directory, so the same setting works in
both places.

### `GO2_STT_ENGINE`

Mirrors `GO2_TTS_ENGINE`, for listening.

| Value | Meaning |
|---|---|
| `auto` (default) | hosted first, falling back to local |
| `openai` | hosted only; failures surface |
| `local` | skip the hosted call entirely |

Use `local` at a site with no hosted recogniser. On `auto` the robot tries the
hosted call first and only falls back once it fails, putting that wait in front
of every utterance.

This governs both halves: the push-to-talk button and ambient listening.

### The local recogniser

Whisper, via `faster-whisper`, the same model family as the hosted `whisper-1`
it stands in for.

```shell
export GO2_STT_MODEL="base"    # tiny, base (default), small, medium
export GO2_STT_DEVICE="auto"   # auto -> cpu, or "cuda"
export GO2_STT_LANGUAGE="en"
```

It defaults to the **CPU**. The Orin already carries YOLO under TensorRT and
deepface, and on a 16 GB Orin NX memory is the binding constraint rather than
TOPS, so a background recogniser stays off the GPU unless you ask.

Weights download on first use. `setmyenv.sh` pre-fetches them when
`GO2_STT_ENGINE="local"`, so the download does not land inside whichever
request needs it first:

```shell
python scripts/install_stt_model.py
python scripts/install_stt_model.py --check
```

On `auto`, the fallback is used **only if the weights are already cached**.
That keeps a working robot from stalling a request behind a long download the
one time a hosted call fails.

### What a region with no audio models can do

Speech out and speech in degrade differently, and it is worth knowing which
before promising a deployment.

- **Text to speech** degrades cleanly. Set `GO2_TTS_ENGINE="piper"` and the
  robot never calls out at all. Set it explicitly: on `auto` the robot still
  tries the hosted call first and waits out `GO2_OPENAI_TIMEOUT_SECONDS` before
  falling back, which puts that delay in front of every utterance.
- **Speech to text** degrades too, onto a Whisper running on the robot. Set
  `GO2_STT_ENGINE="local"`, for the same reason.

Ambient listening works locally as well. The browser asks the robot how to
listen before it opens a microphone, so a site with no realtime recogniser goes
straight to listening locally rather than negotiating a session that cannot
succeed. Local ambient listening segments on silence rather than on a timer, so
it hears a whole sentence and reacts when you stop speaking. Expect it to be
slower to respond than the hosted realtime path, which transcribes while you
are still talking.

### Older `tts` / `tts-hd` deployments

Not worth chasing as a middle ground: both are still Preview on Azure while
`gpt-4o-mini-tts` is GA, and Piper runs locally with no latency or egress. If
you do use one, blank `GO2_OPENAI_INSTRUCTIONS` (they reject style guidance)
and pick a classic voice such as `alloy` or `nova`, since `coral` is not one of
theirs.

---

## Speech behaviour

### Text-to-speech output

| Variable | Default | Purpose |
|---|---|---|
| `GO2_OPENAI_MODEL` | `gpt-4o-mini-tts` | hosted TTS model |
| `GO2_OPENAI_VOICE` | `coral` | hosted TTS voice |
| `GO2_OPENAI_INSTRUCTIONS` | friendly, conversational | style guidance; blank to omit |
| `GO2_OPENAI_TIMEOUT_SECONDS` | `20` | before falling back to an offline engine |
| `GO2_TTS_DEVICE` | `auto` | ALSA output: `auto`, `usb`, `onboard`, `pulse`, or a device name |
| `GO2_TTS_VOLUME` | `100` | output volume percent |

### When the speaker is busy

`aplay: audio open error: Device or resource busy` means something already has
the sound card open. `auto` and `usb` resolve to the hardware directly
(`plughw:2,0`), which fails while a sound server owns the device -- PulseAudio
claims USB speakers on boot on many images.

```shell
fuser -v /dev/snd/*     # who holds it
aplay -l                # "Subdevices: 0/1" means the card is taken
```

If PulseAudio is the holder, route through it instead of fighting it:

```shell
export GO2_TTS_DEVICE="pulse"
```

### Ambient listening and barge-in

| Variable | Default | Purpose |
|---|---|---|
| `CONSCIOUS_AMBIENT_TRANSCRIPTION_MODEL` | `gpt-4o-transcribe` | hosted realtime model |
| `CONSCIOUS_BARGE_IN_MIN_WORDS` | `2` | words needed before a transcript may interrupt |
| `CONSCIOUS_SELF_ECHO_TAIL_SECONDS` | `1.5` | how long after playback a transcript may still be the robot |
| `CONSCIOUS_SELF_ECHO_OVERLAP` | `0.6` | word overlap that counts as the robot hearing itself |
| `CONSCIOUS_DUCK_RELEASE_SECONDS` | `2.5` | restore volume after a duck with no transcript behind it |
| `CONSCIOUS_ACKNOWLEDGE_USER_INPUT` | `0` | speak an acknowledgement before answering |

The microphone stays open while the robot talks, which is what makes barge-in
possible. The server drops transcripts that turn out to be the robot hearing
itself.

---

## Python version

The robot is pinned to **Python 3.11, not later**. CycloneDDS is built from
source at `releases/0.10.x` against the active virtualenv, and the Unitree SDK
rides on it. That ceiling is why neuro-san stays on 0.6.x: 0.7 requires Python
3.12.
