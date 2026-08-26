# Conscious Agent

The **Conscious Agent** is a basic multi-agent system that is called from the
[conscious_assistant.py](../../apps/conscious_assistant/conscious_assistant.py) Flask app.

## Note

- The native runtime periodically captures the scene without invoking an LLM.
- The flask app will store memory items in a file locally. You can turn this feature off by changing the flag in
 [list_topics.py](../../coded_tools/kwik_agents/list_topics.py)

---

## File

[conscious_agent.hocon](../../registries/conscious_agent.hocon)

---

## Prerequisites

- This agent is **disabled by default**. To test it:
    - Manually enable it in the `manifest.hocon` file.
    - Make sure to install the requirements for this app using the following command:  
      ```sh
      pip install -r apps/conscious_assistant/requirements.txt
      ```
    - run the application with the command:  
      ```sh
      python -m apps.conscious_assistant.interface_flask
      ```

---

## Description

Once you run the [conscious_assistant.py](../../apps/conscious_assistant/conscious_assistant.py) Flask app, it will provid
 you with a link, which you can open in your browser to play around with the conscious assistant. This assistant is running
 constantly in the background and "thinking". It may even initiated a dialog. When you chat with it, it will remember facts
 about what you said, and store them in memory, which is saved in a local file.

The hocon file includes an example of calling a coded_tool in a non-default path. This is for the memory operations, for
which the kwik_agents coded tools are reused here.

---

## Conversation model

Ambient listening is on by default: the browser streams room audio to a realtime
transcription session and the agent decides, per utterance, whether it was being
addressed. The microphone stays live **while the robot is speaking**, so you can
talk over it.

### Interrupting the robot

Interruption runs in two stages, because voice-activity detection fires on any
sound in the room and cannot tell you apart from the robot's own voice:

1. **Duck.** The moment the realtime session reports speech, the browser sends a
   `barge_in` event and the server drops the speaker volume. This is instant and
   reversible.
2. **Cancel or restore.** When the transcript arrives, the server decides. A
   transcript that matches what the robot is currently saying is discarded as
   self-echo and the volume comes back. Anything else is a real interruption:
   playback is killed mid-word, queued speech is dropped, and the answer the
   agent was still composing is discarded.

Holding the mic button skips straight to stage 2 -- reaching for the mic is an
unambiguous interruption, so it cancels outright.

### Why not just mute the microphone

Muting during playback is what made the robot impossible to interrupt. Browser
echo cancellation cannot replace it either: `echoCancellation` only cancels audio
the *browser* plays, and the robot's speech comes out of an ALSA device on the
Jetson, entirely outside the browser's audio graph. The server-side self-echo
filter is what makes an open microphone safe.

### Microphone capture

How the room is captured depends on the microphone, not the robot, so it is set
by environment.

| Variable | Default | Effect |
|----------|---------|--------|
| `CONSCIOUS_MIC_NOISE_REDUCTION` | `near_field` | `near_field` for a worn or handheld mic, `far_field` for one covering the whole room |
| `CONSCIOUS_VAD_THRESHOLD` | `0.45` | How loud speech must be to open an utterance (0.0-1.0) |
| `CONSCIOUS_VAD_PREFIX_PADDING_MS` | `400` | Audio kept from before speech was detected |
| `CONSCIOUS_VAD_SILENCE_MS` | `700` | Silence needed to close an utterance |

Getting the profile wrong is expensive: `far_field` on a worn mic lifts distant
sound, which is exactly the robot's own speaker and motors, so the recogniser
spends its effort on noise and mishears the person wearing it. If robot noise
still opens utterances on its own, raise `CONSCIOUS_VAD_THRESHOLD` before
reaching for anything else.

Note that the browser's own `echoCancellation` cannot help here. It only cancels
audio the browser itself plays, and the robot speaks through an ALSA device on
the Jetson -- acoustically present in the room, invisible to the browser. That
is why the self-echo filter above exists.

### Tuning

| Variable | Default | Effect |
|----------|---------|--------|
| `CONSCIOUS_SELF_ECHO_TAIL_SECONDS` | `1.5` | How long after playback a transcript can still be self-echo |
| `CONSCIOUS_SELF_ECHO_OVERLAP` | `0.6` | Word overlap needed to call a transcript self-echo |
| `CONSCIOUS_SELF_ECHO_NOVEL_WORDS` | `2` | Consecutive words the robot never said that mark a real person |
| `CONSCIOUS_BARGE_IN_MIN_WORDS` | `2` | Words required before a transcript may cut the robot off |
| `CONSCIOUS_DUCK_RELEASE_SECONDS` | `2.5` | When to undo a duck no transcript confirmed |
| `CONSCIOUS_SUPERSEDED_SPEECH_GRACE_SECONDS` | `1.5` | How long after a barge-in agent speech is treated as the interrupted turn's |
| `CONSCIOUS_ACKNOWLEDGE_USER_INPUT` | `0` | Speak a filler phrase ("On it") before answering |
| `GO2_TTS_DUCK_VOLUME` | `20` | Mixer percentage held while ducked |

Raise `CONSCIOUS_BARGE_IN_MIN_WORDS` if background conversation keeps cutting the
robot off. If the robot cuts *itself* off and answers its own sentence, raise
`CONSCIOUS_SELF_ECHO_NOVEL_WORDS` to 3; if it ignores real interruptions, drop
it to 1.

---
