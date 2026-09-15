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
Python 3.11 — see [Python version](#python-version).

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

After an IP rotation, refresh the certificate:

```shell
python scripts/setup_tls_certs.py
```

Browsers withhold microphone access from anything but `localhost` over plain
HTTP, so without a certificate voice input will not work from another machine.

---

## Model providers

Two separate APIs are in play, and they are configured independently.

1. **Agents** — the reasoning LLM behind every agent network. neuro-san owns
   this. It reads the `AZURE_OPENAI_*` variables itself and takes the model
   from `registries/llm_config.hocon`, which reads the `AGENT_LLM_*`
   variables below.
2. **Speech** — text-to-speech, push-to-talk transcription and ambient
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
build a client without it. `2024-10-21` is a safe GA value; use `preview` only
if you need something that has not reached GA. If an audio model returns 404 on
a deployment you can see in the portal, a too-old api-version is the usual
cause — those models landed after the older ones.

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
agents run on Azure — useful when the chat region carries no audio models. Set
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
canonical name for `gpt-5.1` — you have to look up what yours was called. In
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
| `pocket,piper,espeak` | try exactly these, in this order |

Naming a single engine is a statement that you want to know when it breaks, so
it is a chain of one rather than a preference. `auto` is the forgiving mode,
where a timeout or a missing model quietly moves to the next engine.

Default order: hosted → pocket → piper → `say` (macOS) → espeak.

### Piper, the offline voice

`requirements.txt` installs the Piper binary but not the voice it speaks with,
and the voice is a ~114 MB download rather than a Python package.

```shell
python scripts/install_piper_voice.py
python scripts/install_piper_voice.py --check
```

`setmyenv.sh` installs it automatically when `GO2_TTS_ENGINE` names `piper`,
since declaring that engine is the same as saying the robot depends on it.
Otherwise it warns while the voice is missing and leaves the download to you —
sourcing an env file should not block on 114 MB. `GO2_PIPER_AUTO_INSTALL=1`
opts into fetching it regardless.

Without the voice, offline speech falls through to `espeak-ng`, which is
intelligible but markedly worse.

Related: `GO2_PIPER_MODEL` and `GO2_PIPER_CONFIG` move where the voice lives.

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

Whisper, via `faster-whisper` — the same model family as the hosted `whisper-1`
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
it hears a whole sentence and reacts when you stop speaking — expect it to be
slower to respond than the hosted realtime path, which transcribes while you
are still talking.

### Older `tts` / `tts-hd` deployments

Not worth chasing as a middle ground: both are still Preview on Azure while
`gpt-4o-mini-tts` is GA, and Piper runs locally with no latency or egress. If
you do use one, blank `GO2_OPENAI_INSTRUCTIONS` — they reject style guidance —
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
| `GO2_TTS_DEVICE` | `auto` | ALSA output: `auto`, `usb`, `onboard`, or a device name |
| `GO2_TTS_VOLUME` | `100` | output volume percent |

### Ambient listening and barge-in

| Variable | Default | Purpose |
|---|---|---|
| `CONSCIOUS_AMBIENT_TRANSCRIPTION_MODEL` | `gpt-4o-transcribe` | hosted realtime model |
| `CONSCIOUS_BARGE_IN_MIN_WORDS` | `2` | words needed before a transcript may interrupt |
| `CONSCIOUS_SELF_ECHO_TAIL_SECONDS` | `1.5` | how long after playback a transcript may still be the robot |
| `CONSCIOUS_SELF_ECHO_OVERLAP` | `0.6` | word overlap that counts as the robot hearing itself |
| `CONSCIOUS_DUCK_RELEASE_SECONDS` | `2.5` | restore volume after a duck with no transcript behind it |
| `CONSCIOUS_ACKNOWLEDGE_USER_INPUT` | `0` | speak an acknowledgement before answering |

The microphone stays open while the robot talks — that is what makes barge-in
possible. The server drops transcripts that turn out to be the robot hearing
itself.

---

## Python version

The robot is pinned to **Python 3.11, not later**. CycloneDDS is built from
source at `releases/0.10.x` against the active virtualenv, and the Unitree SDK
rides on it. That ceiling is why neuro-san stays on 0.6.x: 0.7 requires Python
3.12.
