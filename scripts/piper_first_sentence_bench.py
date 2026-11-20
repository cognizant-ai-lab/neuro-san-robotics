#!/usr/bin/env python3
"""Time how long Piper takes to synthesise a reply's first sentence (#16).

Loads the configured Piper voice once (as the Flask app does) and, for each
test reply, prints its length and the time until Piper yields its first
chunk -- the synthesis part of the turn_timing step
``speak_start -> audio``. Nothing is played; runs without the Flask app.

    GO2_TTS_ENGINE=piper python scripts/piper_first_sentence_bench.py
    GO2_TTS_ENGINE=piper python scripts/piper_first_sentence_bench.py "One sentence." "Another one."

Pass one sentence per argument: chars counts the whole argument, phonemes only
the first sentence Piper synthesised.

The turn_timing line reports ``first_sentence_phonemes``; the phoneme column
here maps it back to characters.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coded_tools.unigo2 import tts_go2  # noqa: E402  pylint: disable=wrong-import-position

SENTENCES = (
    "Yes.",
    "I can see a chair next to the door.",
    "There is a red backpack on the floor, about two metres to my left, next to the table.",
    "I walked over to the kitchen and checked the counter, the shelves and the area by the "
    "window, but I could not find your keys anywhere, so they might be in another room.",
    "Sure, I will go and have a look in the hallway, then come back and tell you what I found "
    "there, including anything that looks out of place, any people I see, and whether the "
    "front door is open or closed.",
)


def first_chunk(voice, text: str) -> tuple[float, int]:
    """Seconds until Piper yields its first chunk, and that chunk's phoneme count."""
    started = time.monotonic()
    chunk = next(iter(voice.synthesize(text)))
    return time.monotonic() - started, len(chunk.phonemes)


def main(argv: list[str]) -> int:
    voice = tts_go2._piper_voice()  # pylint: disable=protected-access
    first_chunk(voice, "Warming up.")  # the first inference pays one-off setup
    for text in argv or SENTENCES:
        seconds, phonemes = first_chunk(voice, text)
        print(f"{len(text):>4} chars {phonemes:>4} phonemes  first chunk {seconds * 1000:>6.0f} ms  "
              f"{text[:50]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
