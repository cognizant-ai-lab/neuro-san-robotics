# neuro-san-robotics

A voice-driven assistant running on a Unitree Go2 robot: it listens, decides what to say or do, and speaks its answer aloud.

## Language

### Speaking

**Answer**:
Everything the robot says in reply to one voice turn, from its first word to its last.
_Avoid_: Utterance, response

**Time to first word**:
How long the user waits, after they finish speaking, until the robot's first word of the answer is audible.
_Avoid_: First-audio latency, TTFA

**Acknowledgement**:
A short utterance without content ("Mhm", "Moment") that tells the user they were heard, spoken before the answer. Not part of the answer.
_Avoid_: Filler, backchannel

**Time to acknowledgement**:
How long the user waits, after they finish speaking, until the acknowledgement is audible.

**Mid-answer gap**:
A silence of the robot's voice between the first and last word of one answer, long enough that the user may think the robot has finished.
_Avoid_: Pause, underrun (an underrun is the log symptom, not the concept)
